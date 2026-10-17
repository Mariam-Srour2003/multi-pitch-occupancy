"""The simple front end and the automatic camera checks behind its Records page.

No camera is contacted and no model loads: the capture opener and the pipeline are faked as in
`test_cameras_api.py`, and every file and database is a temporary one.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pitch_occupancy import cameras, monitor
from pitch_occupancy.api import cameras_page, simple_pages
from pitch_occupancy.config import settings
from pitch_occupancy.db.schema import connect, initialise
from pitch_occupancy.vision import roi

SECRET = "s3cretPW"


class FakeCapture:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def isOpened(self) -> bool:  # noqa: N802
        return self.ok

    def read(self):
        return (True, np.full((360, 640, 3), 90, np.uint8)) if self.ok else (False, None)

    def grab(self) -> bool:
        return self.ok

    def get(self, prop) -> float:
        return 25.0

    def set(self, *args) -> bool:
        return True

    def release(self) -> None:
        pass


class CountingPipeline:
    """Reports a scripted number of people per camera, as the detector path would."""

    model_key = "scripted"
    burst = (1, 0.0)

    def __init__(self, people: dict[str, int]) -> None:
        self.people = people

    def observe_minute(self, camera, frames, previous=None, *, polygon=None, venue=None,
                       slot_key=None):
        n = self.people[camera]
        state = "C1_EMPTY" if n == 0 else "C3_MAINTENANCE_NON_SPORTING"
        verdict = SimpleNamespace(state=state, confidence=0.9, people=n, ball=False,
                                  trace=(f"{n} people",))
        return SimpleNamespace(verdict=verdict, evidence_index=0, instances=())

    def fuse(self, observations):
        n = sum(o.verdict.people for o in observations.values())
        return SimpleNamespace(state="C1_EMPTY", confidence=0.9, people_inside=n,
                               ball_seen=False, trace=())


@pytest.fixture
def env(tmp_path, monkeypatch):
    cam_file = tmp_path / "cameras.json"
    cam_file.write_text(json.dumps({"site": {
        "gate": f"rtsp://admin:{SECRET}@10.0.0.5:554/1/1",
        "pitch": f"rtsp://admin:{SECRET}@10.0.0.6:554/1/1",
        "dead": f"rtsp://admin:{SECRET}@10.0.0.7:554/1/1",
    }}), encoding="utf-8")
    monkeypatch.setattr(cameras, "CONFIG_PATH", cam_file)
    monkeypatch.setattr(settings, "db_path", tmp_path / "db" / "pitch.db")
    monkeypatch.setattr(settings, "evidence_dir", tmp_path / "evidence")
    monkeypatch.setattr(monitor, "CONFIG_PATH", tmp_path / "monitor.json")
    monkeypatch.setattr(cameras, "open_capture",
                        lambda url, timeout_ms=5000: FakeCapture(ok="10.0.0.7" not in url))
    monkeypatch.setattr(cameras, "_tcp_connect", lambda host, port, timeout: None)
    monkeypatch.setattr(roi, "resolve_with_key", lambda camera, venue=None, slot_key=None:
                        (None, None))
    pipeline = CountingPipeline({"gate": 0, "pitch": 7, "dead": 0})
    monkeypatch.setattr(cameras_page, "_pipeline", lambda: pipeline)
    app = FastAPI()
    app.include_router(cameras_page.router)
    app.include_router(simple_pages.router)
    return SimpleNamespace(client=TestClient(app), tmp=tmp_path, pipeline=pipeline)


def _rows(db) -> list[sqlite3.Row]:
    conn = connect(db)
    try:
        return conn.execute("SELECT * FROM camera_checks ORDER BY check_id").fetchall()
    finally:
        conn.close()


# --- settings ------------------------------------------------------------------------------

def test_settings_default_to_off_and_refuse_nonsense(env) -> None:
    assert monitor.load_settings() == monitor.DEFAULTS
    with pytest.raises(ValueError, match="every_minutes"):
        monitor.save_settings({"every_minutes": 7})
    with pytest.raises(ValueError, match="true or false"):
        monitor.save_settings({"enabled": "yes"})
    assert not monitor.CONFIG_PATH.exists(), "a refused change wrote nothing"
    saved = monitor.save_settings({"enabled": True, "every_minutes": 15})
    assert saved["enabled"] and saved["every_minutes"] == 15
    assert monitor.load_settings()["every_minutes"] == 15


def test_a_broken_settings_file_reads_as_off(env) -> None:
    monitor.CONFIG_PATH.write_text("{not json", encoding="utf-8")
    assert monitor.load_settings()["enabled"] is False


def test_only_the_chosen_cameras_are_checked(env) -> None:
    assert len(monitor.selected_cameras()) == 3
    assert monitor.selected_cameras({"cameras": ["site/gate"]}) == [("site", "gate")]


# --- one round -----------------------------------------------------------------------------

def test_a_round_stores_one_row_per_camera_with_the_people_count(env) -> None:
    stored = monitor.check_once()
    by_cam = {r["camera_id"]: r for r in stored}
    assert set(by_cam) == {"gate", "pitch", "dead"}
    assert by_cam["pitch"]["people"] == 7 and by_cam["pitch"]["readable"]
    assert by_cam["gate"]["people"] == 0
    assert not by_cam["dead"]["readable"] and by_cam["dead"]["people"] is None
    assert by_cam["dead"]["problem"], "an unreadable camera says why"
    rows = _rows(settings.db_path)
    assert len(rows) == 3 and all(r["picture_path"] is None for r in rows), (
        "no picture is kept unless switched on")


def test_one_failing_camera_does_not_stop_the_round(env) -> None:
    def scorer(venue, camera):
        if camera == "gate":
            raise RuntimeError("boom")
        return {"readable": True, "people": 2, "label": "In use, not a game"}

    stored = monitor.check_once(scorer=scorer)
    assert len(stored) == 3
    gate = next(r for r in stored if r["camera_id"] == "gate")
    assert not gate["readable"] and "RuntimeError" in gate["problem"]


def test_pictures_are_kept_only_when_switched_on(env) -> None:
    jpeg = base64.b64encode(b"\xff\xd8fake\xff\xd9").decode()

    def scorer(venue, camera):
        return {"readable": True, "people": 1, "snapshot": jpeg}

    stored = monitor.check_once(scorer=scorer, conf={**monitor.DEFAULTS,
                                                     "save_pictures": True})
    paths = [r["picture_path"] for r in stored]
    assert all(paths) and all((env.tmp / "evidence" / "checks") in
                              __import__("pathlib").Path(p).parents for p in paths)


# --- looking back --------------------------------------------------------------------------

def _insert(conn, camera: str, when: str, people: int | None, readable: bool = True) -> None:
    conn.execute(
        """INSERT INTO camera_checks (venue_id, camera_id, checked_at, readable, people, state,
               label, confidence, picture_path, problem)
           VALUES ('site', ?, ?, ?, ?, NULL, ?, 0.9, NULL, NULL)""",
        (camera, when, int(readable), people, "Empty" if not people else "In use, not a game"))


def test_records_group_checks_into_hour_and_a_half_periods(tmp_path) -> None:
    conn = connect(tmp_path / "x.db")
    initialise(conn)
    day = "2026-10-17"
    for when, n in (("10:05", 0), ("10:40", 3), ("11:29", 5),   # 09:00-10:30 and 10:30-12:00
                    ("12:10", 0), ("12:50", 0)):                 # 12:00-13:30
        _insert(conn, "gate", f"{day}T{when}:00+03:00", n)
    _insert(conn, "gate", f"{day}T12:55:00+03:00", None, readable=False)
    _insert(conn, "gate", "2026-10-16T10:00:00+03:00", 9)        # another day: excluded
    conn.commit()

    rows = monitor.records(conn, date(2026, 10, 17), period_minutes=90)
    spans = [(r["start"], r["end"]) for r in rows]
    assert spans == [("09:00", "10:30"), ("10:30", "12:00"), ("12:00", "13:30")]
    first, second, third = rows
    assert first["people_seen"] is False and first["max_people"] == 0
    assert second["people_seen"] is True and second["max_people"] == 5
    assert second["avg_people"] == 4.0 and second["checks_with_people"] == 2
    assert third["checks"] == 3 and third["readable"] == 2 and third["people_seen"] is False


def test_the_last_period_of_the_day_ends_at_midnight(tmp_path) -> None:
    conn = connect(tmp_path / "x.db")
    initialise(conn)
    _insert(conn, "gate", "2026-10-17T23:59:00+03:00", 1)
    conn.commit()
    (row,) = monitor.records(conn, date(2026, 10, 17), period_minutes=90)
    assert (row["start"], row["end"]) == ("22:30", "24:00")


def test_records_refuse_an_unknown_period(tmp_path) -> None:
    conn = connect(tmp_path / "x.db")
    initialise(conn)
    with pytest.raises(ValueError):
        monitor.records(conn, date(2026, 10, 17), period_minutes=7)


# --- pages and endpoints -------------------------------------------------------------------

@pytest.mark.parametrize("path, text", [
    ("/", "/api/v1/records/latest"), ("/cameras/add", "+ Add camera"),
    ("/records", "automatic checking"), ("/more", "Pitch boundaries"),
])
def test_every_simple_page_renders_with_the_short_menu(env, path, text) -> None:
    r = env.client.get(path)
    assert r.status_code == 200 and text in r.text
    nav = r.text[r.text.index("<nav>"):r.text.index("</nav>")]
    assert [label for label in ("Home", "Cameras", "Records", "More") if f">{label}</a>" in nav] \
        == ["Home", "Cameras", "Records", "More"]
    assert nav.count("</a>") == 4, "four menu items and no more"


@pytest.mark.parametrize("old", ["/live", "/cameras/test"])
def test_the_first_versions_addresses_lead_home(env, old) -> None:
    r = env.client.get(old, follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/"


def test_the_records_page_groups_by_an_hour_and_a_half(env) -> None:
    html = env.client.get("/records").text
    assert "const PERIOD = 90;" in html and "__PERIOD__" not in html


def test_each_camera_has_its_own_live_page(env) -> None:
    r = env.client.get("/live/site/pitch")
    assert r.status_code == 200 and '"camera": "pitch"' in r.text
    assert SECRET not in r.text
    assert env.client.get("/live/site/nope").status_code == 404


def test_a_camera_name_cannot_break_out_of_the_live_page_script(env, monkeypatch) -> None:
    monkeypatch.setattr(cameras, "list_cameras", lambda path=None: [
        {"venue": "site", "camera_id": "x<script>y", "url": "", "kind": "rtsp"}])
    r = env.client.get("/live/site/x%3Cscript%3Ey")
    assert r.status_code == 200 and "x<script>y" not in r.text
    assert r"x\u003cscript>y" in r.text


def test_the_monitor_switch_and_run_now(env, monkeypatch) -> None:
    started = []
    monkeypatch.setattr(monitor, "start_background", lambda: started.append(1))
    r = env.client.put("/api/v1/monitor", json={"enabled": True, "every_minutes": 90})
    assert r.status_code == 200
    d = r.json()
    assert d["settings"]["enabled"] and d["settings"]["every_minutes"] == 90
    assert d["cameras_selected"] == 3 and started
    assert env.client.put("/api/v1/monitor", json={"every_minutes": 3}).status_code == 422


def test_records_endpoint_and_spreadsheet(env) -> None:
    monitor.check_once()
    today = datetime.now().astimezone().date().isoformat()
    d = env.client.get(f"/api/v1/records?day={today}&period=90").json()
    assert d["totals"]["checks"] == 3 and d["totals"]["unreadable_checks"] == 1
    assert d["totals"]["max_people"] == 7 and d["totals"]["periods_with_people"] == 1
    assert {r["camera"] for r in d["rows"]} == {"gate", "pitch", "dead"}
    assert "site/pitch" in d["cameras"]
    one = env.client.get(f"/api/v1/records?day={today}&camera=site/pitch").json()
    assert [r["camera"] for r in one["rows"]] == ["pitch"]
    csv_text = env.client.get(f"/api/v1/records.csv?day={today}").text
    assert csv_text.splitlines()[0].startswith("day,start,end,venue,camera,people_seen")
    assert ",pitch,yes,7," in csv_text
    assert env.client.get("/api/v1/records?day=yesterday").status_code == 422
    assert env.client.get("/api/v1/records?camera=nope").status_code == 422


def test_latest_lists_every_saved_camera_even_before_any_check(env) -> None:
    d = env.client.get("/api/v1/records/latest").json()
    assert [c["camera_id"] for c in d["cameras"]] == ["dead", "gate", "pitch"]
    assert all(c["checked_at"] is None for c in d["cameras"])
    monitor.check_once()
    d = env.client.get("/api/v1/records/latest").json()
    pitch = next(c for c in d["cameras"] if c["camera_id"] == "pitch")
    assert pitch["people"] == 7 and pitch["checked_at"]


def test_a_picture_is_served_only_from_the_checks_folder(env) -> None:
    conn = connect(settings.db_path)
    initialise(conn)
    outside = env.tmp / "secret.jpg"
    outside.write_bytes(b"x")
    conn.execute(
        """INSERT INTO camera_checks (venue_id, camera_id, checked_at, readable, people,
               picture_path) VALUES ('site', 'gate', '2026-10-17T10:00:00+03:00', 1, 1, ?)""",
        (str(outside),))
    conn.commit()
    check_id = conn.execute("SELECT MAX(check_id) FROM camera_checks").fetchone()[0]
    conn.close()
    assert env.client.get(f"/api/v1/records/picture/{check_id}").status_code == 404
    assert env.client.get("/api/v1/records/picture/99999").status_code == 404
