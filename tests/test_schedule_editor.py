"""Editing the capture schedule (WP6-T6).

This is the API's only write path and the most dangerous endpoint in the project, because the
schedule decides whether an hour is *observed at all*. A dropped entry is not a wrong verdict,
it is no verdict and no footage, discovered weeks later when someone disputes a booking.

So the tests are about the three protections: that validation is the scheduler's own, that a
rejected edit leaves no trace, and that a write can be undone.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api import schedule_editor as se
from pitch_occupancy.api.app import app
from pitch_occupancy.scheduler import load_schedule

GOOD = {
    "venue_id": "venue_01", "start": "10:00", "duration_minutes": 60,
    "cameras": ["camera_A", "camera_B"], "field_id": "field_01", "days": ["mon", "tue"],
}


@pytest.fixture()
def schedule(tmp_path, monkeypatch):
    """A schedule file in a temp directory, so no test can touch the committed one."""
    path = tmp_path / "slots_schedule.json"
    path.write_text(json.dumps([GOOD], indent=2), encoding="utf-8")
    monkeypatch.setattr(se, "DEFAULT_SCHEDULE", path)
    return path


@pytest.fixture()
def client(schedule):
    return TestClient(app)


# --- validation is the scheduler's own ----------------------------------------------------


def test_a_valid_schedule_passes(schedule) -> None:
    assert se.validate([GOOD], schedule) is None


@pytest.mark.parametrize("broken, fragment", [
    ({**GOOD, "start": "25:00"}, "not HH:MM"),
    ({**GOOD, "cameras": []}, "never be observed"),
    ({**GOOD, "duration_minutes": 0}, "must be positive"),
    ({**GOOD, "days": ["funday"]}, "unknown day"),
])
def test_the_scheduler_s_own_rules_are_the_ones_enforced(broken, fragment, schedule) -> None:
    """Not a second copy of them. Two validators agree until they do not, and the one that
    matters is the one the scheduler actually runs at 10:00."""
    problem = se.validate([broken], schedule)
    assert problem and fragment in problem, problem


def test_a_missing_required_field_is_rejected(schedule) -> None:
    assert se.validate([{"venue_id": "v", "start": "10:00"}], schedule) is not None


def test_two_slots_at_the_same_time_on_one_venue_are_rejected(schedule) -> None:
    """They share a slot id, so one verdict would overwrite the other's."""
    problem = se.validate([GOOD, dict(GOOD)], schedule)
    assert problem and "two slots starting" in problem


def test_the_error_names_the_proposal_rather_than_a_temporary_file(schedule) -> None:
    """An operator reading "tmpx8f2b.json entry 0" learns nothing about their own edit."""
    problem = se.validate([{**GOOD, "start": "nope"}], schedule)
    assert "tmp" not in problem.lower()
    assert "the proposed schedule" in problem


# --- a rejected edit leaves no trace ------------------------------------------------------


def test_validation_writes_nothing(schedule) -> None:
    before = schedule.read_text(encoding="utf-8")
    se.validate([{**GOOD, "start": "25:00"}], schedule)
    assert schedule.read_text(encoding="utf-8") == before
    assert not se.history_dir(schedule).exists()


def test_a_rejected_write_leaves_the_schedule_and_the_history_untouched(schedule) -> None:
    """The check happens before the backup, so a bad edit does not even leave a version
    behind - otherwise every typo would fill the history with copies of the same file."""
    before = schedule.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        se.write_schedule([{**GOOD, "cameras": []}], schedule)
    assert schedule.read_text(encoding="utf-8") == before
    assert not se.history_dir(schedule).exists()


def test_no_staging_file_is_left_behind(schedule) -> None:
    se.write_schedule([GOOD], schedule)
    assert not list(schedule.parent.glob("*.json.new"))


# --- a write can be undone ------------------------------------------------------------------


def test_a_write_keeps_the_previous_version(schedule) -> None:
    original = schedule.read_text(encoding="utf-8")
    backup = se.write_schedule([{**GOOD, "start": "11:00"}], schedule)
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == original
    assert json.loads(schedule.read_text(encoding="utf-8"))[0]["start"] == "11:00"


def test_the_written_file_is_one_the_scheduler_can_read(schedule) -> None:
    """The property that matters more than any assertion about JSON shape."""
    se.write_schedule([{**GOOD, "start": "09:30"}], schedule)
    assert len(load_schedule(schedule)) == 1


def test_history_is_pruned_rather_than_growing_without_bound(schedule, monkeypatch) -> None:
    monkeypatch.setattr(se, "KEEP_VERSIONS", 3)
    for minute in range(6):
        se.write_schedule([{**GOOD, "start": f"10:{minute:02d}"}], schedule)
    assert len(list(se.history_dir(schedule).glob("*.json"))) <= 3


# --- the endpoints ----------------------------------------------------------------------------


def test_reading_returns_the_entries_and_whether_they_load(client) -> None:
    body = client.get("/api/v1/schedule").json()
    assert body["problem"] is None
    assert body["entries"][0]["start"] == "10:00"


def test_a_schedule_that_does_not_load_is_still_returned_with_its_problem(client, schedule) -> None:
    """An operator cannot repair a file the editor refuses to display."""
    schedule.write_text(json.dumps([{**GOOD, "start": "25:00"}]), encoding="utf-8")
    body = client.get("/api/v1/schedule").json()
    assert body["entries"], "the broken entries must still come back"
    assert "not HH:MM" in body["problem"]


def test_unparseable_json_is_reported_rather_than_raising(client, schedule) -> None:
    schedule.write_text("{not json", encoding="utf-8")
    body = client.get("/api/v1/schedule").json()
    assert "not valid JSON" in body["problem"]


def test_the_validate_endpoint_writes_nothing(client, schedule) -> None:
    before = schedule.read_text(encoding="utf-8")
    response = client.post("/api/v1/schedule/validate",
                           json={"entries": [{**GOOD, "start": "11:11"}]})
    assert response.json()["ok"] is True
    assert schedule.read_text(encoding="utf-8") == before


def test_putting_a_valid_schedule_replaces_it(client, schedule) -> None:
    body = client.put("/api/v1/schedule",
                      json={"entries": [{**GOOD, "start": "08:00"}]}).json()
    assert body["ok"] and body["slots"] == 1
    assert body["previous_version"].endswith(".json")
    assert json.loads(schedule.read_text(encoding="utf-8"))[0]["start"] == "08:00"


def test_putting_an_invalid_schedule_is_a_422_carrying_the_scheduler_s_words(client) -> None:
    """422, not 500: the request was understood and refused, and the operator needs to see
    what they have to satisfy."""
    response = client.put("/api/v1/schedule",
                          json={"entries": [{**GOOD, "days": ["funday"]}]})
    assert response.status_code == 422
    assert "unknown day" in response.json()["detail"]


def test_the_response_says_when_the_change_takes_effect(client) -> None:
    """An operator who edits at 09:55 needs to know whether the 10:00 slot is affected."""
    body = client.put("/api/v1/schedule", json={"entries": [GOOD]}).json()
    assert "next checks what is due" in body["note"]


def test_bookings_still_have_no_write_path(client) -> None:
    """The asymmetry is deliberate: the schedule is this system's own configuration, a
    booking sheet is the facility's financial record. Adding a write path there would need a
    very good reason, and this test is where that argument would have to be made."""
    from pitch_occupancy.bookings import CsvBookingSource

    assert {m for m in dir(CsvBookingSource) if not m.startswith("_")} == {
        "read", "path", "source"
    }
