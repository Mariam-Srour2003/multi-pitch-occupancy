"""Playgrounds: settings, capacity, one check, the runner, saved pictures, sessions, and the
second opinion.

No camera is contacted and no model loads: picture-taking and counting are faked
(`playground_env`), and every file and database is a temporary one.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime

import pytest
from playground_env import make_env
from playground_env import new_playground as _new

from pitch_occupancy import playgrounds
from pitch_occupancy.config import settings
from pitch_occupancy.db.schema import connect, initialise


@pytest.fixture
def env(tmp_path, monkeypatch):
    return make_env(tmp_path, monkeypatch)


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
    assert not playgrounds.store.CONFIG_PATH.exists()


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
    cap = playgrounds.capacity.capacity()
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
    assert playgrounds.runner._due_after(now, 30).strftime("%H:%M:%S") == "10:00:30"
    assert playgrounds.runner._due_after(now, 5).strftime("%H:%M:%S") == "10:00:10"


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


def test_the_runner_checks_a_playground_when_asked_and_reports_it(env, monkeypatch) -> None:
    import threading
    import time

    done = threading.Event()

    def fake_check(p, **kwargs):
        done.set()
        return {"checked_at": "2026-10-17T10:00:00+03:00", "people": 6, "playing": 1,
                "cameras_read": 1, "cameras_total": 1, "duration_ms": 5}

    monkeypatch.setattr(playgrounds.check, "check_playground", fake_check)
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


# --- more ways to combine, saved pictures, sessions ------------------------------------------

@pytest.mark.parametrize("counting, counts, expected", [
    ("add", [3, 5], (8, 1)), ("largest", [3, 5], (5, 1)), ("average", [3, 4, 8], (5, 1)),
    ("average", [3, 4], (4, 0)), ("vote", [6, 2], (6, 1)), ("vote", [6, 2, 1], (6, 0)),
    ("vote", [7], (7, 1)), ("add", [], (None, None)),
])
def test_every_way_of_combining_cameras(counting, counts, expected) -> None:
    assert playgrounds.combine(counting, counts, 5) == expected


def test_a_check_with_save_keeps_each_cameras_picture(env) -> None:
    record = playgrounds.check_playground(_new("Big", ("site/a", "site/b")), save=True)
    paths = [c["picture_path"] for c in record["cameras"]]
    assert all(paths) and all((playgrounds.snapshots.SNAPSHOT_DIR / p).is_file() for p in paths)
    conn = connect(settings.db_path)
    stored = {r[0] for r in conn.execute("SELECT picture_path FROM camera_checks")}
    conn.close()
    assert stored == set(paths)
    r = env.client.get("/snapshots/" + paths[0])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert env.client.get("/snapshots/../playgrounds.json").status_code == 404


def test_saved_pictures_stay_under_the_cap(env, monkeypatch) -> None:
    p = _new("Big", ("site/a",))
    playgrounds.check_playground(p, save=True)
    one = playgrounds.snapshot_usage()["bytes"]
    monkeypatch.setattr(playgrounds.snapshots, "SNAPSHOT_CAP_BYTES", int(one * 2.5))
    for _ in range(6):
        playgrounds.check_playground(p, save=True)
    usage = playgrounds.snapshot_usage()
    assert usage["bytes"] <= one * 2.5 and usage["files"] >= 1


def test_the_runner_saves_one_check_in_save_every(env) -> None:
    p = _new("Big", ("site/a",), save_every=3)
    playgrounds.runner._counter.pop(p["id"], None)
    for _ in range(7):
        playgrounds.runner._run_one(p)
    assert playgrounds.snapshot_usage()["files"] == 3, "checks 1, 4 and 7"


def test_sessions_run_every_hour_and_a_half_through_half_past_nine() -> None:
    at = lambda h, m: datetime(2026, 10, 18, h, m).astimezone()  # noqa: E731
    assert playgrounds.session_bounds(at(9, 30))[0].strftime("%H:%M") == "09:30"
    assert playgrounds.session_bounds(at(21, 40))[0].strftime("%H:%M") == "21:30"
    start, end = playgrounds.session_bounds(at(23, 59))
    assert (start.strftime("%H:%M"), end.strftime("%d %H:%M")) == ("23:00", "19 00:30")


def test_each_session_gets_one_answer(env) -> None:
    p = _new("Big", ("site/a",), every=60)
    conn = connect(settings.db_path)
    initialise(conn)
    off = datetime(2026, 10, 18, 21, 30).astimezone().strftime("%z")
    off = off[:3] + ":" + off[3:]
    for minute, n in enumerate([6, 7, 2, 8]):  # 21:30 session: 3 of 4 playing
        _row(conn, p["id"], f"2026-10-18T21:{31 + minute}:00{off}", n, int(n >= 5))
    for minute, n in enumerate([0, 6, 1, 0, 2, 0, 0]):  # 23:00 session: 1 of 7
        _row(conn, p["id"], f"2026-10-18T23:{10 + minute}:00{off}", n, int(n >= 5))
    for minute in range(3):  # 00:30 session next day: nobody
        _row(conn, p["id"], f"2026-10-19T00:{40 + minute}:00{off}", 0, 0)
    conn.commit()
    rows = playgrounds.sessions.sessions(conn, datetime(2026, 10, 18, 21, 30).astimezone(),
                                datetime(2026, 10, 19, 2, 0).astimezone())
    conn.close()
    assert [r["verdict"] for r in rows] == ["Playing", "Mostly empty", "Empty"]
    assert rows[0]["avg_people"] == 5.8 and rows[0]["minutes_playing"] == 3
    d = env.client.get("/api/v1/sessions", params={"start": "2026-10-18T21:30",
                                                   "end": "2026-10-19T01:00"}).json()
    assert len(d["columns"]) == 3 and d["play_min"] == 5
    detail = env.client.get("/api/v1/sessions/detail",
                            params={"playground": p["id"], "start": "2026-10-18T21:30"}).json()
    assert len(detail["checks"]) == 4
    assert env.client.get("/results").status_code == 200
    assert "Playing" in env.client.get("/api/v1/sessions.csv", params={
        "start": "2026-10-18T21:30", "end": "2026-10-19T01:00"}).text


# --- the free vision-model second opinion -------------------------------------------------


def _assist_on(env, monkeypatch, tmp_path, state: str, confidence: float) -> list:
    import json as _json

    from pitch_occupancy import config as app_config
    from pitch_occupancy.assist import budget, providers
    from pitch_occupancy.assist import config as assist_config

    cfg = tmp_path / "assist.json"
    cfg.write_text(_json.dumps({"enabled": True, "providers": ["groq"]}), encoding="utf-8")
    monkeypatch.setattr(assist_config, "CONFIG_PATH", cfg)
    monkeypatch.setattr(app_config, "env_value",
                        lambda name: "k" if name == "GROQ_API_KEY" else "")
    monkeypatch.setattr(budget, "_minute", {})
    sent: list = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            content = _json.dumps({"state": state, "people_on_pitch": 14,
                                   "confidence": confidence, "reason": "a game"})
            return _json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def urlopen(request, timeout):
        sent.append(_json.loads(request.data))
        return Reply()

    monkeypatch.setattr(providers.urllib.request, "urlopen", urlopen)
    return sent


def test_a_doubtful_check_is_upgraded_by_a_confident_second_opinion(env, monkeypatch,
                                                                   tmp_path) -> None:
    sent = _assist_on(env, monkeypatch, tmp_path, "playing", 0.9)
    record = playgrounds.check_playground(_new("Half", ("site/a",)))  # 3 people: doubtful
    playgrounds.wait_for_second_opinions()
    assert len(sent) == 1
    conn = connect(settings.db_path)
    row = conn.execute("SELECT playing, detector_playing, playing_source FROM "
                       "playground_checks WHERE check_id = ?", (record["check_id"],)).fetchone()
    opinions = conn.execute("SELECT COUNT(*) FROM assist_opinions").fetchone()[0]
    conn.close()
    assert tuple(row) == (1, 0, "assist") and opinions == 1


def test_a_second_opinion_never_turns_a_game_off(env, monkeypatch, tmp_path) -> None:
    _assist_on(env, monkeypatch, tmp_path, "empty", 1.0)
    record = playgrounds.check_playground(_new("Big", ("site/a", "site/b")))  # 8 people
    playgrounds.wait_for_second_opinions()
    conn = connect(settings.db_path)
    playing = conn.execute("SELECT playing FROM playground_checks WHERE check_id = ?",
                           (record["check_id"],)).fetchone()[0]
    conn.close()
    assert playing == 1


def test_a_broken_second_opinion_never_costs_the_count(env, monkeypatch, tmp_path) -> None:
    _assist_on(env, monkeypatch, tmp_path, "playing", 0.9)
    from pitch_occupancy.assist import providers

    def down(request, timeout):
        raise OSError("network down")

    monkeypatch.setattr(providers.urllib.request, "urlopen", down)
    record = playgrounds.check_playground(_new("Half", ("site/a",)))
    playgrounds.wait_for_second_opinions()
    assert record["people"] == 3 and record["playing"] == 0


def test_a_spot_check_from_the_live_page_spends_no_quota(env, monkeypatch, tmp_path) -> None:
    sent = _assist_on(env, monkeypatch, tmp_path, "playing", 0.9)
    playgrounds.check_playground(_new("Half", ("site/a",)), with_pictures=True)
    playgrounds.wait_for_second_opinions()
    assert sent == []
