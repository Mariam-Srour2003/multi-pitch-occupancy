"""Reports from recorded data: the daily summary, busy hours and a recorded day re-cut.

The database is built by hand so every expected number can be worked out on paper. One
pitch, two cameras, Monday 2026-07-06:

* s1 08:00-09:00 - camera A sees an empty half, camera B a match: strongest activity wins,
  so the pitch is playing. Minutes 40-49 are missing from both cameras (a gap).
  Verdict USED, play 1.0 over 50 decided minutes.
* s2 09:00-10:30 - both halves empty, except camera B undecided at 09:00 (so that minute is
  undecided: EMPTY needs both halves). Verdict NOT USED; reconciliation says no-show.
* s3 10:30-11:00 - camera A play, camera B undecided throughout: playing. The model said
  REVIEW, a person overrode it to USED; reconciliation says unbooked play.
* s4 11:00-12:00 - scheduled, never checked: no samples, no verdict.
"""

from __future__ import annotations

import csv
import io
from datetime import date, time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pitch_occupancy import reports
from pitch_occupancy.api.reports_page import router
from pitch_occupancy.api.routes import get_conn
from pitch_occupancy.db.schema import connect, initialise
from pitch_occupancy.timetable import SlotStatus

DAY = date(2026, 7, 6)  # a Monday
P, E, U = "C2_ACTIVE_PLAY", "C1_EMPTY", "UNCERTAIN"


def _sample(conn, slot, cam, minute, pred, start_hhmm):
    h, m = map(int, start_hhmm.split(":"))
    total = h * 60 + m + minute
    # captured_at deliberately the processing time (UTC, much later), as the worker writes it
    conn.execute(
        "INSERT INTO frame_samples (slot_id, camera_id, captured_at, minute_index, predicted,"
        " confidence) VALUES (?, ?, ?, ?, ?, ?)",
        (slot, cam, f"2026-07-07T23:{total % 60:02d}:00+00:00", minute, pred, 0.9))


def _evaluation(conn, slot, status, play, empty, n, *, override=None):
    conn.execute(
        "INSERT INTO slot_evaluations (slot_id, status, reason, play_ratio, empty_ratio,"
        " maintenance_ratio, n_samples, mean_confidence, evaluated_at, model_key,"
        " is_overridden, override_status, override_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (slot, status, "test", play, empty, 0.0, n, 0.9, "2026-07-06T12:00:00", "m",
         1 if override else 0, override, "ops" if override else None))


def _build(conn) -> None:
    initialise(conn)
    conn.execute("INSERT INTO venues VALUES ('v1', 'Venue')")
    conn.execute("INSERT INTO fields VALUES ('f1', 'v1', 'Pitch 1')")
    conn.execute("INSERT INTO fields VALUES ('f2', 'v1', 'Pitch 2')")
    conn.execute("INSERT INTO cameras VALUES ('cA', 'f1', 'A', NULL, NULL)")
    conn.execute("INSERT INTO cameras VALUES ('cB', 'f1', 'B', NULL, NULL)")
    slots = [("s1", "08:00", "09:00"), ("s2", "09:00", "10:30"), ("s3", "10:30", "11:00"),
             ("s4", "11:00", "12:00")]
    for sid, a, b in slots:
        conn.execute("INSERT INTO rental_slots VALUES (?, 'f1', ?, ?, ?)",
                     (sid, DAY.isoformat(), a, b))
    for m in range(60):
        if 40 <= m < 50:
            continue  # the gap
        _sample(conn, "s1", "cA", m, E, "08:00")
        _sample(conn, "s1", "cB", m, P, "08:00")
    for m in range(90):
        _sample(conn, "s2", "cA", m, E, "09:00")
        _sample(conn, "s2", "cB", m, U if m == 0 else E, "09:00")
    for m in range(30):
        _sample(conn, "s3", "cA", m, P, "10:30")
        _sample(conn, "s3", "cB", m, U, "10:30")
    _evaluation(conn, "s1", "USED", 1.0, 0.0, 50)
    _evaluation(conn, "s2", "NOTUSED", 0.0, 1.0, 89)
    _evaluation(conn, "s3", "REVIEW", 1.0, 0.0, 30, override="USED")
    for sid, anomaly in (("s1", "CONSISTENT"), ("s2", "NO_SHOW_OR_OVERRECORDED"),
                         ("s3", "UNBOOKED_USAGE")):
        conn.execute("INSERT INTO reconciliations (slot_id, anomaly, severity, explanation,"
                     " created_at) VALUES (?, ?, 'warning', 'x', '2026-07-06')", (sid, anomaly))
    conn.commit()


@pytest.fixture
def conn(tmp_path: Path):
    c = connect(tmp_path / "reports.db", same_thread=False)
    _build(c)
    yield c
    c.close()


# --- daily summary ----------------------------------------------------------------------------


def test_daily_summary_counts_hours_and_anomalies(conn):
    out = reports.daily_summary(conn, start=DAY, end=DAY, field_id=None)
    assert len(out["rows"]) == 1
    row = out["rows"][0]
    assert row["field_name"] == "Pitch 1"
    assert row["scheduled"] == 4
    # the override wins: s3 counts as used, not review
    assert (row["used"], row["not_used"], row["review"]) == (2, 1, 0)
    assert row["overridden"] == 1
    assert row["not_checked"] == 1
    assert row["hours_scheduled"] == 4.0  # 60 + 90 + 30 + 60 minutes
    # minutes seen playing: 50 (s1, gap excluded) + 0 + 30 = 80
    assert row["hours_played"] == round(80 / 60, 2)
    assert row["hours_observed"] == round((50 + 89 + 30) / 60, 2)
    assert row["utilisation"] == round(80 / 240, 4)
    assert row["no_shows"] == 1 and row["unbooked_play"] == 1
    assert out["totals"]["scheduled"] == 4 and out["totals"]["used"] == 2


def test_daily_summary_filters_by_pitch_and_dates(conn):
    assert reports.daily_summary(conn, start=DAY, end=DAY, field_id="f2")["rows"] == []
    later = date(2026, 7, 7)
    out = reports.daily_summary(conn, start=later, end=later, field_id=None)
    assert out["rows"] == [] and out["totals"]["utilisation"] is None


def test_summary_csv_has_totals(conn):
    text = reports.summary_to_csv(reports.daily_summary(conn, start=DAY, end=DAY))
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["day"] for r in rows] == [DAY.isoformat(), "total"]
    assert rows[0]["utilisation_pct"] == "33.3"
    assert rows[0]["no_shows"] == "1"


# --- per-minute fusion ------------------------------------------------------------------------


def test_minute_fusion_strongest_activity_and_empty_needs_agreement(conn):
    minutes = {m.at.time(): m for m in reports.minute_states(conn, start=DAY, end=DAY)}
    # A empty + B play -> playing
    assert minutes[time(8, 0)].state == P
    # A empty + B undecided -> undecided, not empty
    assert minutes[time(9, 0)].state == U
    assert minutes[time(9, 1)].state == E
    # A play + B undecided -> playing
    assert minutes[time(10, 30)].state == P
    # the gap is absent, not invented
    assert time(8, 45) not in minutes
    assert len(minutes) == 50 + 90 + 30
    # placed on the slot's local clock, not on captured_at (processing time, next day)
    assert all(m.at.date() == DAY for m in minutes.values())


# --- busy hours -------------------------------------------------------------------------------


def test_busy_hours_not_observed_is_not_zero(conn):
    grid = reports.busy_hours(conn, start=DAY, end=DAY, field_id="f1")
    cells = {(c["weekday"], c["hour"]): c for c in grid["cells"]}
    assert len(cells) == 7 * 24
    assert cells[(0, 8)]["share"] == 1.0 and cells[(0, 8)]["observed_min"] == 50
    # 09:00 undecided is left out of the denominator
    assert cells[(0, 9)]["share"] == 0.0 and cells[(0, 9)]["observed_min"] == 59
    assert cells[(0, 9)]["undecided_min"] == 1
    assert cells[(0, 10)]["share"] == 0.5
    # nobody watched 11:00 or Tuesday: not observed, which is not 0%
    assert cells[(0, 11)]["share"] is None and cells[(0, 11)]["observed"] is False
    assert cells[(1, 9)]["share"] is None

    rows = list(csv.DictReader(io.StringIO(reports.busy_hours_to_csv(grid))))
    by = {(r["weekday"], r["hour"]): r for r in rows}
    assert by[("Monday", "09:00")]["playing_pct"] == "0.0"
    assert by[("Monday", "11:00")]["playing_pct"] == ""


# --- a recorded day, re-cut -------------------------------------------------------------------


def test_recorded_day_recut_into_90_minute_slots(conn):
    rec = reports.recorded_day(conn, field_id="f1", day=DAY, first_slot=time(8, 0),
                               slot_minutes=90)
    assert rec.start == time(8, 0)
    assert rec.duration_s == 3 * 3600  # 08:00 to the end of 10:59
    labels = [r.window.label for r in rec.results]
    assert labels == ["08:00-09:30", "09:30-11:00"]
    first, second = rec.results
    # 50 playing, 29 empty, 1 undecided, 10 unseen (the gap) - the gap lowers the seen share
    assert first.playing_s == 50 * 60 and first.empty_s == 29 * 60
    assert first.uncertain_s == 60
    assert first.unseen_s == pytest.approx(10 * 60)
    assert first.seen_share == pytest.approx(79 / 90)
    assert first.status is SlotStatus.USED  # 50/79 = 63% playing
    assert second.empty_s == 60 * 60 and second.playing_s == 30 * 60
    assert second.status is SlotStatus.NOT_USED
    # the minute timeline covers the gap as "not recorded"
    assert len(rec.timeline) == 180
    codes = rec.to_dict()["timeline"]
    assert codes[45][1] == "N" and codes[0][1] == "P" and codes[60][1] == "U"
    assert rec.to_dict()["counts"] == {"slots": 2, "USED": 1, "NOT_USED": 1, "REVIEW": 0}

    text = reports.recorded_day_to_csv(rec)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["slot"] for r in rows] == labels
    assert rows[0]["result"] == "USED" and rows[0]["unseen_min"] == "10.0"


def test_recorded_day_first_slot_before_recording(conn):
    rec = reports.recorded_day(conn, field_id="f1", day=DAY, first_slot=time(6, 45),
                               slot_minutes=90)
    assert [r.window.label for r in rec.results] == ["06:45-08:15", "08:15-09:45",
                                                     "09:45-11:15"]
    first, last = rec.results[0], rec.results[-1]
    # only 08:00-08:15 of the first slot was recorded: review, never "not used"
    assert first.window.coverage == pytest.approx(15 / 90)
    assert first.status is SlotStatus.REVIEW
    assert "outside the video" in first.reason
    # recording ends at 11:00, a quarter of an hour before the last slot does
    assert last.window.covered_end_s == 3 * 3600


def test_recorded_day_without_recordings(conn):
    rec = reports.recorded_day(conn, field_id="f2", day=DAY, first_slot=time(8, 0),
                               slot_minutes=90)
    assert rec.results == [] and rec.timeline == [] and rec.start is None


# --- API --------------------------------------------------------------------------------------


@pytest.fixture
def client(conn):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_conn] = lambda: conn
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_summary_and_csv(client):
    r = client.get("/api/v1/reports/summary", params={"start": DAY, "end": DAY})
    assert r.status_code == 200
    assert r.json()["totals"]["used"] == 2
    r = client.get("/api/v1/reports/summary",
                   params={"start": DAY, "end": DAY, "field_id": "f1", "format": "csv"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "filename=" in r.headers["content-disposition"]
    assert r.text.splitlines()[0].startswith("day,field_id")
    assert client.get("/api/v1/reports/summary",
                      params={"start": "2026-07-08", "end": DAY}).status_code == 422


def test_api_busy_hours(client):
    r = client.get("/api/v1/reports/busy-hours", params={"start": DAY, "end": DAY})
    cells = {(c["weekday"], c["hour"]): c for c in r.json()["cells"]}
    assert cells[(0, 10)]["share"] == 0.5 and cells[(0, 11)]["share"] is None
    r = client.get("/api/v1/reports/busy-hours",
                   params={"start": DAY, "end": DAY, "format": "csv"})
    assert r.headers["content-type"].startswith("text/csv")
    assert "busy-hours" in r.headers["content-disposition"]


def test_api_day(client):
    q = {"field_id": "f1", "day": DAY, "first_slot": "08:00", "slot_minutes": 90}
    d = client.get("/api/v1/reports/day", params=q).json()
    assert [s["status"] for s in d["slots"]] == ["USED", "NOT_USED"]
    assert d["start"] == "08:00" and len(d["timeline"]) == 180
    r = client.get("/api/v1/reports/day", params={**q, "format": "csv"})
    assert r.headers["content-type"].startswith("text/csv")
    assert "recorded-day_f1_2026-07-06_0800_90min.csv" in r.headers["content-disposition"]
    assert client.get("/api/v1/reports/day",
                      params={**q, "first_slot": "8 o'clock"}).status_code == 422


def test_reports_page(client):
    r = client.get("/reports")
    assert r.status_code == 200
    assert "Cut a recorded day into slots" in r.text
    assert 'value="f1"' in r.text and "Pitch 2" in r.text
    assert "No recordings yet" not in r.text
    assert "C2_ACTIVE_PLAY" not in r.text.split("<main>")[1].split("<script>")[0]


def test_reports_page_empty_database(tmp_path):
    c = connect(tmp_path / "empty.db", same_thread=False)
    initialise(c)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_conn] = lambda: c
    try:
        r = TestClient(app).get("/reports")
    finally:
        c.close()
    assert r.status_code == 200
    assert "No recordings yet" in r.text and "Start recording from the" in r.text
