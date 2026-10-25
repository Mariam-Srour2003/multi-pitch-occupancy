"""The playground path in the operator tools: alerts, reports, live status and setup checks.

Playgrounds are written straight into the (redirected) playgrounds file and their checks
straight into a temporary database, so no camera, detector or runner thread is involved;
`NOW` is naive local time, as the alert checker uses it, and stored stamps carry the local
offset, as `playgrounds.check` writes them.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import types
from datetime import date, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pitch_occupancy import alerts, cameras, playgrounds, reports, setup_check, status
from pitch_occupancy.api import reports_page, setup_api
from pitch_occupancy.api.routes import get_conn
from pitch_occupancy.assist import config as assist_config
from pitch_occupancy.config import settings
from pitch_occupancy.db.schema import connect, initialise

NOW = datetime(2026, 10, 10, 10, 30)
DAY = NOW.date()


def _stamp(at: datetime) -> str:
    return at.astimezone().isoformat(timespec="seconds")


def _playgrounds(*items: dict) -> None:
    base = {"cameras": ["site/a"], "every_seconds": 30, "counting": "add", "enabled": True}
    playgrounds.store.CONFIG_PATH.write_text(json.dumps(
        {"playgrounds": [{**base, **p} for p in items]}), encoding="utf-8")


def _check(conn, pid: str, at: datetime, *, people: int | None = 6, playing: int | None = 1,
           cams: dict[str, bool] | None = None, source: str | None = None) -> None:
    cams = cams if cams is not None else {"site/a": people is not None}
    cur = conn.execute(
        """INSERT INTO playground_checks (playground_id, checked_at, people, playing,
               cameras_read, cameras_total, counting, playing_source)
           VALUES (?, ?, ?, ?, ?, ?, 'add', ?)""",
        (pid, _stamp(at), people, playing, sum(cams.values()), len(cams), source))
    for ref, readable in cams.items():
        venue, cam = ref.split("/")
        conn.execute(
            """INSERT INTO camera_checks (venue_id, camera_id, checked_at, readable, people,
                   problem, playground_check_id) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (venue, cam, _stamp(at), int(readable), people if readable else None,
             None if readable else "The camera did not answer.", cur.lastrowid))
    conn.commit()


@pytest.fixture
def db():
    conn = connect(settings.db_path, same_thread=False)
    initialise(conn)
    yield conn
    conn.close()


@pytest.fixture
def quiet_slots(tmp_path, monkeypatch):
    """No recording schedule and a live slot worker, so only playground events remain."""
    schedule = tmp_path / "schedule.json"
    schedule.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(alerts, "SCHEDULE_PATH", schedule)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    fake = types.ModuleType("pitch_occupancy.status")
    fake.read_status = lambda **kw: {"alive": True, "state": "running"}
    monkeypatch.setitem(sys.modules, "pitch_occupancy.status", fake)


def _types(events) -> list[tuple[str, str]]:
    return sorted((e.type, e.key) for e in events)


# --- alerts ---------------------------------------------------------------------------------


def test_stalled_playground_is_serious_and_explains_its_dark_cameras(db, quiet_slots):
    _playgrounds({"id": "north", "name": "North"})
    for minutes in (12, 11, 10):  # three dark checks, but the last is ten minutes old
        _check(db, "north", NOW - timedelta(minutes=minutes), people=None, playing=None)
    events = alerts.check(NOW)
    assert _types(events) == [("playground_stalled", "playground_stalled:north")]
    assert events[0].severity == "serious" and "North" in events[0].text
    assert events[0].details["minutes_since"] == 10


def test_a_recent_check_is_not_stalled_and_the_floor_applies(db, quiet_slots):
    _playgrounds({"id": "north", "name": "North", "every_seconds": 5})
    _check(db, "north", NOW - timedelta(seconds=100))  # 20 intervals, but under two minutes
    assert alerts.check(NOW) == []
    _playgrounds({"id": "north", "name": "North", "every_seconds": 60})
    assert alerts.check(NOW + timedelta(seconds=90)) != []  # 190 s > 3 x 60 s


def test_off_out_of_hours_and_never_checked_playgrounds_are_quiet(db, quiet_slots):
    _playgrounds({"id": "off", "name": "Off", "enabled": False},
                 {"id": "night", "name": "Night", "cameras": ["site/b"],
                  "hours": {"from": "18:00", "to": "23:00"}},
                 {"id": "new", "name": "New", "cameras": ["site/c"]})
    for pid in ("off", "night"):
        _check(db, pid, NOW - timedelta(hours=5))
    assert alerts.check(NOW) == []


def test_stall_counts_from_when_the_hours_began(db, quiet_slots):
    _playgrounds({"id": "day", "name": "Day", "hours": {"from": "10:29", "to": "20:00"}})
    _check(db, "day", NOW - timedelta(days=1))  # yesterday's last check
    assert alerts.check(NOW) == []  # hours began a minute ago
    assert [e.type for e in alerts.check(NOW + timedelta(minutes=5))] == ["playground_stalled"]


def test_dark_camera_needs_both_checks_in_a_row_and_minutes(db, quiet_slots):
    _playgrounds({"id": "big", "name": "Big", "cameras": ["site/a", "site/b"]})
    _check(db, "big", NOW - timedelta(minutes=30), cams={"site/a": True, "site/b": True})
    for minutes in (20, 15, 1):
        _check(db, "big", NOW - timedelta(minutes=minutes), cams={"site/a": True,
                                                                  "site/b": False})
    events = alerts.check(NOW)
    assert _types(events) == [("playground_no_picture", "playground_no_picture:big:site/b")]
    ev = events[0]
    assert ev.severity == "warning" and ev.details["minutes_dark"] == 20
    assert "did not answer" in ev.text and "Big" in ev.text

    cfg = alerts.AlertConfig(channels={}, camera_silent_minutes=25)
    assert alerts.check(NOW, config=cfg) == []  # dark for 20 minutes only
    cfg = alerts.AlertConfig(channels={}, playground_dark_checks=4)
    assert alerts.check(NOW, config=cfg) == []  # three dark checks only


def test_playground_dark_checks_setting_and_example_config():
    from pitch_occupancy.config import CONFIGS_DIR

    assert alerts.parse_config({"playground_dark_checks": 5}).playground_dark_checks == 5
    with pytest.raises(ValueError, match="playground_dark_checks"):
        alerts.parse_config({"playground_dark_checks": 0})
    cfg = alerts.load_config(CONFIGS_DIR / "alerts.example.json")
    assert {"playground_no_picture", "playground_stalled"} <= cfg.channels["slack"].events


def test_daily_summary_adds_the_playground_sessions(db, quiet_slots):
    _playgrounds({"id": "north", "name": "North"})
    yesterday = NOW - timedelta(days=1)
    _check(db, "north", yesterday.replace(hour=10, minute=0))
    _check(db, "north", yesterday.replace(hour=10, minute=1), source="assist")
    _check(db, "north", yesterday.replace(hour=14, minute=0), people=None, playing=None)
    cfg = alerts.AlertConfig(channels={}, daily_summary_at=NOW.time())
    summary = [e for e in alerts.check(NOW, config=cfg) if e.type == "daily_summary"][0]
    assert "playgrounds: 2 session(s) - 1 playing" in summary.text
    assert "1 without pictures" in summary.text
    assert summary.details["playground_totals"]["assist_upgrades"] == 1


# --- reports --------------------------------------------------------------------------------


def _week_of_checks(db) -> None:
    _playgrounds({"id": "north", "name": "North"}, {"id": "south", "name": "South",
                                                    "cameras": ["site/b"], "every_seconds": 60})
    monday = datetime(2026, 10, 5, 10, 0)
    for i in range(4):  # Monday 09:30 session: 4 playing checks, one upgraded
        _check(db, "north", monday + timedelta(minutes=i), source="assist" if i == 0 else None)
    _check(db, "north", monday + timedelta(hours=3), people=1, playing=0)  # 12:30 session
    _check(db, "north", monday + timedelta(days=2), people=None, playing=None)
    _check(db, "south", monday + timedelta(days=1), people=2, playing=0,
           cams={"site/b": True})
    _check(db, "north", monday + timedelta(days=7))  # the next week


def test_playground_summary_by_day(db):
    _week_of_checks(db)
    out = reports.playground_summary(db, start=date(2026, 10, 5), end=date(2026, 10, 11))
    rows = {(r["period"], r["playground_id"]): r for r in out["rows"]}
    mon = rows[("2026-10-05", "north")]
    assert mon["sessions"] == 2 and mon["playing"] == 1 and mon["empty"] == 0
    assert mon["some_people_no_game"] == 1
    assert mon["minutes_playing"] == 2 and mon["assist_upgrades"] == 1  # 4 x 30 s
    assert rows[("2026-10-07", "north")]["no_pictures"] == 1
    assert rows[("2026-10-07", "north")]["checks_without_picture"] == 1
    assert ("2026-10-12", "north") not in rows
    t = out["totals"]
    assert t["sessions"] == 4 and t["checks"] == 7 and t["hours_playing"] == 0.03


def test_playground_summary_by_week_and_csv(db):
    _week_of_checks(db)
    out = reports.playground_summary(db, start=date(2026, 10, 5), end=date(2026, 10, 12),
                                     group="week", playground_id="north")
    assert [(r["period"], r["sessions"]) for r in out["rows"]] == [
        ("2026-10-05", 3), ("2026-10-12", 1)]
    lines = list(csv.DictReader(io.StringIO(reports.playground_summary_to_csv(out))))
    assert lines[-1]["period"] == "total" and lines[-1]["sessions"] == "4"
    assert set(reports.PLAYGROUND_VERDICTS) == set(playgrounds.sessions.VERDICTS)
    with pytest.raises(ValueError):
        reports.playground_summary(db, start=DAY, end=DAY, group="month")


@pytest.fixture
def client(db):
    app = FastAPI()
    app.include_router(reports_page.router)
    app.include_router(setup_api.router)
    app.dependency_overrides[get_conn] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_playground_report_api_and_page(db, client):
    _week_of_checks(db)
    q = {"start": "2026-10-05", "end": "2026-10-11", "group": "week"}
    r = client.get("/api/v1/reports/playgrounds", params=q)
    assert r.status_code == 200 and r.json()["totals"]["sessions"] == 4
    r = client.get("/api/v1/reports/playgrounds", params={**q, "format": "csv"})
    assert r.headers["content-type"].startswith("text/csv")
    assert "playground-sessions-by-week_2026-10-05_2026-10-11.csv" in \
        r.headers["content-disposition"]
    assert client.get("/api/v1/reports/playgrounds",
                      params={**q, "group": "month"}).status_code == 422

    page = client.get("/reports").text  # playground checks but no slot recordings
    assert "No recordings yet" not in page and "Playground sessions" in page
    assert '<section id="pgReports">' in page and '<div id="reports" class="hide">' in page
    assert '<option value="south">South</option>' in page


# --- live status and setup checks -----------------------------------------------------------


def test_playground_runner_reports_ages_and_overdue(db, client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")  # no worker heartbeat
    _playgrounds({"id": "north", "name": "North"}, {"id": "off", "name": "Off",
                                                    "enabled": False, "cameras": ["site/b"]})
    now = datetime.now()
    _check(db, "north", now - timedelta(minutes=10))
    state = status.playground_runner()
    north = {p["id"]: p for p in state["playgrounds"]}["north"]
    assert state["enabled"] == 1 and north["overdue"] is True and 590 < north["age_s"] < 700
    assert {p["id"]: p for p in state["playgrounds"]}["off"]["last_checked_at"] is None
    body = client.get("/api/v1/live-status").json()
    assert set(body) == {"worker", "playgrounds"} and body["worker"] is None
    assert body["playgrounds"]["playgrounds"][0]["id"] == "north"


def _cameras_file(tmp_path, monkeypatch, names=("a",)) -> None:
    path = tmp_path / "cams.json"
    path.write_text(json.dumps({"site": {n: f"rtsp://10.0.0.{i}/1"
                                         for i, n in enumerate(names, 5)}}), encoding="utf-8")
    monkeypatch.setattr(cameras, "CONFIG_PATH", path)


def test_setup_flags_playgrounds_without_cameras_and_a_stopped_runner(tmp_path, monkeypatch):
    step = setup_check._step_playgrounds
    assert step().state == "optional"
    _playgrounds({"id": "north", "name": "North", "enabled": False})
    assert step().state == "optional" and "switched off" in step().detail
    _playgrounds({"id": "north", "name": "North", "cameras": ["site/a", "site/b"]})
    s = step()
    assert (s.state, s.required) == ("todo", True) and "no camera is saved" in s.detail
    _cameras_file(tmp_path, monkeypatch, ("a",))
    assert step().state == "warning" and "site/b" in step().detail
    _cameras_file(tmp_path, monkeypatch, ("a", "b"))
    monkeypatch.setattr(playgrounds, "status", lambda: {"running": False, "playgrounds": {}})
    assert step().state == "warning" and "not running" in step().detail
    monkeypatch.setattr(playgrounds, "status", lambda: {"running": True, "playgrounds": {}})
    assert step().state == "done"


def test_setup_flags_a_second_opinion_without_a_key(monkeypatch):
    step = setup_check._step_assist
    assert step().state == "optional"
    assist_config.CONFIG_PATH.write_text(json.dumps({"enabled": True, "providers": ["groq"]}),
                                         encoding="utf-8")
    monkeypatch.setattr("pitch_occupancy.config.env_value", lambda name: "")
    s = step()
    assert (s.state, s.required) == ("warning", True) and "GROQ_API_KEY" in s.detail
    monkeypatch.setattr("pitch_occupancy.config.env_value",
                        lambda name: "k" if name == "GROQ_API_KEY" else "")
    assert step().state == "done" and "groq" in step().detail
