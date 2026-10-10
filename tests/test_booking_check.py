"""The booking-data quality check: every rule, the verdict, and that no one is named.

The check collects problems instead of raising, so each test feeds rows with one known defect
and asserts the issue that comes back - and that clean rows produce none.
"""

from __future__ import annotations

import json
from datetime import date, time

from pitch_occupancy.booking_check import check
from pitch_occupancy.booking_sources import rows_from_csv_text
from pitch_occupancy.bookings import EXAMPLE_PATH
from pitch_occupancy.scheduler import Schedule, ScheduledSlot

TODAY = date(2026, 8, 1)


def row(n=None, **kw):
    base = {"field_id": "f1", "date": "2026-07-11", "start": "10:00", "end": "11:00",
            "customer_ref": "R", "status": "confirmed", "entered_by": "a.demir"}
    base.update(kw)
    if n is not None:
        base["row"] = n
    return base


def codes(report, severity=None):
    return [i.code for i in report.issues if severity is None or i.severity == severity]


def test_clean_data_is_good():
    rep = check([row(), row(start="11:00", end="12:00")], today=TODAY)
    assert rep.verdict == "good" and rep.issues == []
    assert rep.total_rows == rep.valid_rows == 2
    assert (rep.date_from, rep.date_to) == ("2026-07-11", "2026-07-11")
    assert rep.by_status["confirmed"] == 2 and rep.pitches == 1


def test_the_committed_example_has_no_errors():
    rows = rows_from_csv_text(EXAMPLE_PATH.read_text(encoding="utf-8"))
    rep = check(rows, today=TODAY)
    assert "error" not in {i.severity for i in rep.issues}
    assert rep.valid_rows == rep.total_rows == 8


def test_missing_values():
    rep = check([row(2, field_id=""), row(3, start="", end="")], today=TODAY)
    missing = [i for i in rep.issues if i.code == "missing_value"]
    assert {(i.rows, i.field) for i in missing} == {((2,), "field_id"), ((3,), "start"),
                                                     ((3,), "end")}
    assert rep.valid_rows == 0 and rep.verdict == "has errors"


def test_unparseable_date_and_time():
    rep = check([row(2, date="11/31/2026"), row(3, start="ten")], today=TODAY)
    assert codes(rep, "error") == ["bad_date", "bad_time"]
    assert "'11/31/2026'" in rep.issues[0].message


def test_end_not_after_start():
    rep = check([row(2, start="23:00", end="01:00")], today=TODAY)
    (issue,) = rep.issues
    assert issue.code == "end_not_after_start" and "midnight" in issue.message


def test_unknown_status_suggests_the_closest():
    rep = check([row(2, status="canceled"), row(3, status="no show", start="12:00",
                                                 end="13:00"),
                 row(4, status="Confrimed", start="14:00", end="15:00"),
                 row(5, status="zzz", start="16:00", end="17:00")], today=TODAY)
    msgs = {i.rows[0]: i.message for i in rep.issues if i.code == "unknown_status"}
    assert "'cancelled'" in msgs[2] and "status_map" in msgs[2]
    assert "'no_show'" in msgs[3]
    assert "'confirmed'" in msgs[4]
    assert "Did you mean" not in msgs[5]


def test_exact_duplicates_are_one_issue_not_also_a_double_booking():
    rep = check([row(2), row(3), row(4)], today=TODAY)
    assert codes(rep) == ["duplicate"]
    assert rep.issues[0].rows == (2, 3, 4)


def test_different_bookings_for_the_same_slot():
    rep = check([row(2), row(3, status="cancelled", customer_ref="X")], today=TODAY)
    assert codes(rep) == ["same_slot"]


def test_overlapping_sold_bookings_are_a_double_booking():
    rep = check([row(2), row(3, start="10:30", end="11:30"),
                 row(4, start="10:30", end="11:30", field_id="f2")], today=TODAY)
    (issue,) = rep.issues
    assert issue.code == "double_booking" and issue.rows == (2, 3)


def test_a_cancelled_booking_does_not_double_book():
    rep = check([row(2), row(3, start="10:30", end="11:30", status="cancelled")],
                today=TODAY)
    assert "double_booking" not in codes(rep)


def test_back_to_back_is_not_an_overlap():
    rep = check([row(2), row(3, start="11:00", end="12:00")], today=TODAY)
    assert rep.issues == []


def test_opening_hours():
    hours = {"*": ("08:00", "22:00"), "sat": ("09:00", "18:00")}  # 2026-07-11 is a Saturday
    rep = check([row(2, start="17:30", end="18:30"), row(3, start="12:00", end="13:00"),
                 row(4, start="07:00", end="08:00", status="blocked")],
                schedule=hours, today=TODAY)
    assert [(i.code, i.rows) for i in rep.issues] == [("outside_hours", (2,))]
    assert rep.verdict == "usable with warnings"


def test_camera_schedule_reports_unwatched_bookings_as_info():
    sched = Schedule((ScheduledSlot(venue_id="v", start=time(10), duration_minutes=60,
                                    cameras=("A",), field_id="f1"),))
    rep = check([row(2), row(3, start="18:00", end="19:00"), row(4, field_id="f9")],
                schedule=sched, today=TODAY)
    found = {(i.code, i.rows[:1]) for i in rep.issues}
    assert ("not_watched", (3,)) in found
    assert ("pitch_not_scheduled", (4,)) in found
    assert {i.severity for i in rep.issues} == {"info"}


def test_days_with_no_bookings_are_reported_as_a_gap():
    rep = check([row(date="2026-07-01"), row(date="2026-07-02"), row(date="2026-07-06")],
                today=TODAY)
    (gap,) = [i for i in rep.issues if i.code == "empty_days"]
    assert gap.severity == "info" and "3 days" in gap.message and "2026-07-03" in gap.message


def test_future_bookings_are_counted_as_info():
    rep = check([row(date="2026-07-11"), row(date="2026-09-01")], today=TODAY)
    assert (rep.future_rows, rep.past_rows) == (1, 1)
    assert [i for i in rep.issues if i.code == "future"][0].severity == "info"


def test_very_short_and_very_long_bookings_are_warnings():
    rep = check([row(2, start="10:00", end="10:05"), row(3, start="12:00", end="19:00")],
                today=TODAY)
    assert codes(rep, "warning") == ["very_short", "very_long"]
    assert rep.verdict == "usable with warnings"


def test_sold_without_reference_is_info():
    rep = check([row(2, customer_ref="")], today=TODAY)
    assert codes(rep) == ["no_customer_ref"]
    assert rep.verdict == "good"


def test_no_rows_is_an_error():
    rep = check([], today=TODAY)
    assert rep.verdict == "has errors" and codes(rep) == ["empty"]


def test_many_issues_of_one_kind_are_capped():
    rows = [row(i, date=f"bad{i}") for i in range(2, 252)]
    rep = check(rows, today=TODAY)
    bad = [i for i in rep.issues if i.code == "bad_date"]
    assert len(bad) == 101 and "150 more" in bad[-1].message


def test_nothing_is_grouped_or_reported_by_entered_by():
    staff = ["s.yilmaz", "a.demir", "groundstaff"]
    rows = [row(2, entered_by=staff[0], status="bogus"),
            row(3, entered_by=staff[1], start="09:30", end="10:30"),
            row(4, entered_by=staff[2], date="nope"),
            row(5, entered_by=staff[0], start="10:00", end="10:05", field_id="f2"),
            row(6, entered_by=staff[1], field_id="f2"),
            row(7, entered_by=staff[1], field_id="f2")]
    out = json.dumps(check(rows, schedule={"*": ("12:00", "13:00")}, today=TODAY).to_dict())
    for name in staff:
        assert name not in out
    assert "entered_by" not in out
    assert "staff" not in out.lower()
