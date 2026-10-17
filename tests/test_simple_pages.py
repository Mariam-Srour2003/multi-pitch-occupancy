"""Playgrounds and the simple pages over them.

No camera is contacted and no model loads: picture-taking and counting are faked, and every
file and database is a temporary one.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pitch_occupancy import cameras, playgrounds
from pitch_occupancy.api import cameras_page, simple_pages
from pitch_occupancy.config import settings
from pitch_occupancy.db.schema import connect, initialise

SECRET = "s3cretPW"
FRAME = np.full((360, 640, 3), 90, np.uint8)


class CountingPipeline:
    """Reports a scripted number of people per camera, as the detector would."""

    burst = (1, 0.0)

    def __init__(self, people: dict[str, int]) -> None:
        self.people = people
        self.calls: list[str] = []

    def observe_minute(self, camera, frames, previous=None, *, polygon=None, venue=None,
                       slot_key=None):
        self.calls.append(camera)
        n = self.people.get(camera, 0)
        verdict = SimpleNamespace(state="C1_EMPTY" if n == 0 else "C2_ACTIVE_PLAY",
                                  confidence=0.9, people=n, ball=False, trace=())
        return SimpleNamespace(verdict=verdict, evidence_index=0, instances=())


@pytest.fixture
def env(tmp_path, monkeypatch):
    cam_file = tmp_path / "cameras.json"
    cam_file.write_text(json.dumps({"site": {
        name: f"rtsp://admin:{SECRET}@10.0.0.{i}:554/1/1"
        for i, name in enumerate(("a", "b", "c", "dead"), start=5)
    }}), encoding="utf-8")
    monkeypatch.setattr(cameras, "CONFIG_PATH", cam_file)
    monkeypatch.setattr(settings, "db_path", tmp_path / "db" / "pitch.db")
    monkeypatch.setattr(playgrounds, "CONFIG_PATH", tmp_path / "playgrounds.json")
    monkeypatch.setattr(playgrounds, "SPEED_PATH", tmp_path / "speed.json")
    monkeypatch.setattr(playgrounds, "play_min", lambda: 5)
    pipeline = CountingPipeline({"a": 3, "b": 5, "c": 4})
    monkeypatch.setattr(playgrounds, "_pipeline", lambda: pipeline)
    monkeypatch.setattr(cameras_page, "_pipeline", lambda: pipeline)

    def grab(url, timeout_s):
        if "10.0.0.8" in url:  # the "dead" camera
            return None, "The camera did not answer.", 3000
        return FRAME, None, 120

    monkeypatch.setattr(playgrounds, "_grab", grab)
    app = FastAPI()
    app.include_router(cameras_page.router)
    app.include_router(simple_pages.router)
    return SimpleNamespace(client=TestClient(app), tmp=tmp_path, pipeline=pipeline)


def _new(name="Playground 1", cams=("site/a",), every=30, counting="add", **kw) -> dict:
    return playgrounds.save_playground({"name": name, "cameras": list(cams),
                                        "every_seconds": every, "counting": counting, **kw})


# --- settings ------------------------------------------------------------------------------

def test_a_playground_is_saved_with_its_cameras_and_timing(env) -> None:
    p = _new("Big pitch", ("site/a", "site/b"), every=10)
    assert p["id"] == "big-pitch" and p["cameras"] == ["site/a", "site/b"]
    assert [x["id"] for x in playgrounds.load()] == ["big-pitch"]
    assert _new("Big pitch", ("site/c",))["id"] == "big-pitch-2", "names may repeat, ids not"


@pytest.mark.parametrize("values, problem", [
    ({"name": "", "cameras": ["site/a"], "every_seconds": 30}, "name"),
    ({"name": "x", "cameras": [], "every_seconds": 30}, "at least one camera"),
    ({"name": "x", "cameras": ["site/zz"], "every_seconds": 30}, "Unknown camera"),
    ({"name": "x", "cameras": ["site/a", "site/a"], "every_seconds": 30}, "twice"),
    ({"name": "x", "cameras": ["site/a"], "every_seconds": 7}, "how often"),
    ({"name": "x", "cameras": ["site/a"], "every_seconds": 30, "counting": "avg"}, "combined"),
    ({"name": "x", "cameras": ["site/a"], "every_seconds": 30,
      "hours": {"from": "9", "to": "10:00"}}, "From must look like"),
    ({"name": "x", "cameras": ["site/a"], "every_seconds": 30,
      "hours": {"from": "10:00", "to": "10:00"}}, "the same time"),
])
def test_bad_settings_are_refused_with_a_reason(env, values, problem) -> None:
    with pytest.raises(ValueError, match=problem):
        playgrounds.save_playground(values)
    assert not playgrounds.CONFIG_PATH.exists()


def test_a_camera_belongs_to_one_playground(env) -> None:
    _new("One", ("site/a",))
    with pytest.raises(ValueError, match="one playground only"):
        _new("Two", ("site/a", "site/b"))


def test_hours_include_ones_that_cross_midnight() -> None:
    day = {"hours": {"from": "08:00", "to": "23:00"}}
    night = {"hours": {"from": "20:00", "to": "02:00"}}

    def at(h, m=0):
        return datetime(2026, 10, 17, h, m)

    assert playgrounds.in_hours(day, at(8)) and not playgrounds.in_hours(day, at(23))
    assert playgrounds.in_hours(night, at(23, 30)) and playgrounds.in_hours(night, at(1))
    assert not playgrounds.in_hours(night, at(12))
    assert playgrounds.in_hours({"hours": None}, at(4))


# --- what the computer can do --------------------------------------------------------------

def test_capacity_adds_each_camera_once_per_check(env) -> None:
    (env.tmp / "speed.json").write_text(json.dumps({"seconds_per_picture": 0.5}))
    _new("One", ("site/a", "site/b"), every=10)        # 12 pictures a minute
    _new("Two", ("site/c",), every=30)                 # 2 a minute
    cap = playgrounds.capacity()
    assert cap["need_per_min"] == 14.0
    assert cap["can_per_min"] == pytest.approx(60 / 0.5 * playgrounds.HEADROOM)
    assert cap["cameras_at"][30] == int(cap["can_per_min"] * 30 // 60)


def test_settings_the_computer_cannot_keep_up_with_are_refused(env) -> None:
    """At 5 s a picture the computer manages 8.4 pictures a minute (with headroom)."""
    (env.tmp / "speed.json").write_text(json.dumps({"seconds_per_picture": 5.0}))
    with pytest.raises(ValueError, match="pictures a minute"):
        _new("Too fast", ("site/a", "site/b"), every=5)   # 24 a minute
    slow = _new("Slow", ("site/a", "site/b"), every=60)   # 2 a minute
    _new("Other", ("site/c",), every=10)                  # 6 more: 8 of 8.4
    playgrounds.set_enabled(slow["id"], False)
    _new("Third", ("site/dead",), every=60)               # 7 of 8.4 while Slow is off
    with pytest.raises(ValueError, match="pictures a minute"):
        playgrounds.set_enabled(slow["id"], True)         # would be 9 of 8.4
    assert not next(p for p in playgrounds.load() if p["id"] == slow["id"])["enabled"]


# --- one check -----------------------------------------------------------------------------

def test_cameras_seeing_different_parts_are_added(env) -> None:
    """3 people on one camera and 5 on the other are 8 - playing, because 8 >= 5."""
    record = playgrounds.check_playground(_new("Big", ("site/a", "site/b")))
    assert record["people"] == 8 and record["playing"] == 1
    assert record["cameras_read"] == 2 and record["cameras_total"] == 2


def test_cameras_seeing_the_same_area_use_the_biggest_count(env) -> None:
    record = playgrounds.check_playground(_new("Same", ("site/a", "site/b"),
                                               counting="largest"))
    assert record["people"] == 5, "the same people seen twice are not counted twice"
    assert record["playing"] == 1


def test_a_camera_without_a_picture_is_left_out_not_counted_as_zero(env) -> None:
    record = playgrounds.check_playground(_new("Half", ("site/a", "site/dead")))
    assert record["people"] == 3 and record["playing"] == 0
    assert record["cameras_read"] == 1 and record["cameras_total"] == 2
    dead = next(c for c in record["cameras"] if c["camera"] == "site/dead")
    assert not dead["readable"] and dead["problem"]


def test_no_picture_at_all_is_no_count(env) -> None:
    record = playgrounds.check_playground(_new("Dark", ("site/dead",)))
    assert record["people"] is None and record["playing"] is None


def test_a_check_stores_the_playground_and_each_camera(env) -> None:
    record = playgrounds.check_playground(_new("Big", ("site/a", "site/b")))
    conn = connect(settings.db_path)
    rows = conn.execute("SELECT * FROM camera_checks WHERE playground_check_id = ?",
                        (record["check_id"],)).fetchall()
    conn.close()
    assert sorted((r["camera_id"], r["people"]) for r in rows) == [("a", 3), ("b", 5)]
    assert all(r["grab_ms"] == 120 for r in rows)


def test_an_older_database_gets_the_new_columns(tmp_path) -> None:
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE camera_checks (check_id INTEGER PRIMARY KEY, venue_id TEXT,
        camera_id TEXT, checked_at TEXT, readable INTEGER, people INTEGER, state TEXT,
        label TEXT, confidence REAL, picture_path TEXT, problem TEXT)""")
    conn.commit()
    conn.close()
    conn = connect(db)
    initialise(conn)
    columns = {r[1] for r in conn.execute("PRAGMA table_info(camera_checks)")}
    conn.close()
    assert {"playground_check_id", "grab_ms", "detect_ms"} <= columns


def test_checks_fall_on_the_playgrounds_own_grid() -> None:
    now = datetime(2026, 10, 17, 10, 0, 7).astimezone()
    assert playgrounds._due_after(now, 30).strftime("%H:%M:%S") == "10:00:30"
    assert playgrounds._due_after(now, 5).strftime("%H:%M:%S") == "10:00:10"


# --- looking back --------------------------------------------------------------------------

def _row(conn, pid, when, people, playing):
    conn.execute(
        """INSERT INTO playground_checks (playground_id, checked_at, people, playing,
               cameras_read, cameras_total, counting, duration_ms)
           VALUES (?, ?, ?, ?, 1, 1, 'add', 1)""", (pid, when, people, playing))


def test_records_group_by_an_hour_and_a_half_and_estimate_time_played(env) -> None:
    p = _new("Big", ("site/a",), every=60)
    conn = connect(settings.db_path)
    initialise(conn)
    for minute, n in ((5, 0), (6, 7), (7, 9), (25, 2)):           # 09:00-10:30
        _row(conn, p["id"], f"2026-10-17T10:{minute:02d}:00+03:00", n, int(n >= 5))
    _row(conn, p["id"], "2026-10-17T11:00:00+03:00", None, None)  # 10:30-12:00, no picture
    conn.commit()
    rows = playgrounds.records(conn, date(2026, 10, 17), period_minutes=90)
    conn.close()
    assert [(r["start"], r["end"]) for r in rows] == [("09:00", "10:30"), ("10:30", "12:00")]
    first, second = rows
    assert first["max_people"] == 9 and first["playing_checks"] == 2
    assert first["minutes_playing"] == 2, "two playing checks, one minute apart"
    assert second["max_people"] is None and second["checks_without_picture"] == 1


# --- pages and endpoints -------------------------------------------------------------------

@pytest.mark.parametrize("path, text", [
    ("/", "/api/v1/playgrounds"), ("/playgrounds", "+ New playground"),
    ("/cameras/add", "+ Add camera"), ("/records", "Played for about"),
    ("/more", "Pitch boundaries"),
])
def test_every_simple_page_renders_with_the_short_menu(env, path, text) -> None:
    r = env.client.get(path)
    assert r.status_code == 200 and text in r.text
    nav = r.text[r.text.index("<nav>"):r.text.index("</nav>")]
    labels = ("Home", "Playgrounds", "Cameras", "Records", "More")
    assert [label for label in labels if f">{label}</a>" in nav] == list(labels)
    assert nav.count("</a>") == 5


@pytest.mark.parametrize("old", ["/live", "/cameras/test"])
def test_the_first_versions_addresses_lead_home(env, old) -> None:
    r = env.client.get(old, follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/"


def test_making_changing_and_deleting_a_playground_through_the_api(env) -> None:
    r = env.client.post("/api/v1/playgrounds", json={
        "name": "Pitch 1", "cameras": ["site/a", "site/b"], "every_seconds": 30})
    assert r.status_code == 200 and r.json()["id"] == "pitch-1"
    listed = env.client.get("/api/v1/playgrounds").json()
    assert listed["playgrounds"][0]["cameras"] == ["site/a", "site/b"]
    owned = {c["ref"]: c["playground"] for c in listed["cameras"]}
    assert owned["site/a"] == "pitch-1" and owned["site/c"] is None
    assert SECRET not in json.dumps(listed)
    r = env.client.put("/api/v1/playgrounds/pitch-1", json={
        "name": "Pitch 1", "cameras": ["site/c"], "every_seconds": 60, "counting": "largest",
        "hours": {"from": "08:00", "to": "22:00"}})
    assert r.status_code == 200 and r.json()["hours"] == {"from": "08:00", "to": "22:00"}
    assert env.client.post("/api/v1/playgrounds/pitch-1/enabled",
                           json={"enabled": False}).json()["enabled"] is False
    bad = env.client.post("/api/v1/playgrounds", json={
        "name": "x", "cameras": ["site/c"], "every_seconds": 30})
    assert bad.status_code == 422 and "one playground only" in bad.json()["detail"]
    assert env.client.delete("/api/v1/playgrounds/pitch-1").status_code == 200
    assert env.client.delete("/api/v1/playgrounds/pitch-1").status_code == 404


def test_look_now_returns_pictures_and_latest_shows_each_camera(env, monkeypatch) -> None:
    monkeypatch.setattr(playgrounds, "_picture", lambda frame, counted: "PICTURE")
    _new("Big", ("site/a", "site/dead"))
    r = env.client.post("/api/v1/playgrounds/big/look").json()
    assert r["people"] == 3
    assert {c["camera"]: c["picture"] for c in r["cameras"]} == {
        "site/a": "PICTURE", "site/dead": None}
    latest = env.client.get("/api/v1/playgrounds/big/latest").json()
    assert latest["latest"]["people"] == 3
    assert [(c["camera"], c["readable"], c["people"]) for c in latest["cameras"]] == [
        ("site/a", True, 3), ("site/dead", False, None)]


def test_records_endpoint_and_spreadsheet(env) -> None:
    playgrounds.check_playground(_new("Big", ("site/a", "site/b")))
    today = datetime.now().astimezone().date().isoformat()
    d = env.client.get(f"/api/v1/records?day={today}").json()
    assert d["period_minutes"] == 90 and d["rows"][0]["max_people"] == 8
    csv_text = env.client.get(f"/api/v1/records.csv?day={today}").text
    assert csv_text.splitlines()[0].startswith("day,start,end,playground,max_people")
    assert ",Big,8," in csv_text and ",yes," in csv_text
    assert env.client.get("/api/v1/records?day=yesterday").status_code == 422


def test_a_playground_name_cannot_break_out_of_its_page_script(env) -> None:
    p = _new("x</script><b>y", ("site/a",))
    r = env.client.get(f"/playground/{p['id']}")
    assert r.status_code == 200 and "x</script><b>y" not in r.text
    assert env.client.get("/playground/nope").status_code == 404


def test_each_camera_still_has_its_own_live_page(env) -> None:
    r = env.client.get("/live/site/a")
    assert r.status_code == 200 and SECRET not in r.text
    assert env.client.get("/live/site/nope").status_code == 404


def test_the_runner_checks_a_playground_when_asked_and_reports_it(env, monkeypatch) -> None:
    import threading
    import time

    done = threading.Event()

    def fake_check(p, **kwargs):
        done.set()
        return {"checked_at": "2026-10-17T10:00:00+03:00", "people": 6, "playing": 1,
                "cameras_read": 1, "cameras_total": 1, "duration_ms": 5}

    monkeypatch.setattr(playgrounds, "check_playground", fake_check)
    p = _new("Big", ("site/a",))
    playgrounds.set_enabled(p["id"], False)       # off: only "now" may run it
    try:
        playgrounds.run_now(p["id"])
        assert done.wait(5), "the runner did not check the playground it was asked to"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            st = playgrounds.status()["playgrounds"].get(p["id"], {})
            if st.get("last"):
                break
            time.sleep(0.05)
        assert st["last"]["people"] == 6 and not st["checking"]
    finally:
        playgrounds.stop()
    with pytest.raises(KeyError):
        playgrounds.run_now("nope")
