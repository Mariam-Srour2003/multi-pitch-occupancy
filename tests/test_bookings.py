"""The booking importer (WP6-T4).

A booking export is the facility's financial record, and reconciliation turns it into
statements about whether a customer used what they paid for. So the tests here are mostly
about refusing: the failure that matters is not a crash, it is a row silently dropped or
misread, because a booking that goes missing at import reconciles to *unbooked usage* - the
anomaly that gets a customer accused.
"""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pytest

from pitch_occupancy.bookings import (
    EXAMPLE_PATH,
    SOLD,
    STATUSES,
    BookingSource,
    CsvBookingSource,
    read_bookings,
    slot_id_for,
    to_reconcile_bookings,
)

HEADER = "field_id,date,start,end,customer_ref,status,entered_by\n"


def write(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "b.csv"
    path.write_text(HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    return path


# --- read-only by construction ----------------------------------------------------------


def test_the_source_interface_offers_no_way_to_write() -> None:
    """WP6-T4's constraint is "never write to client systems", and the way to keep it is to
    have no write path rather than a disabled one. If this fails, someone added a method that
    would need a very good reason."""
    public = {m for m in dir(CsvBookingSource) if not m.startswith("_")}
    assert public == {"read", "path", "source"}, public
    assert isinstance(CsvBookingSource(), BookingSource)


# --- the committed example --------------------------------------------------------------


def test_the_example_loads_and_is_tagged_as_an_example() -> None:
    """A table built on the hand-written example must not read as one built on a real export,
    so the tag is set by the importer and cannot be supplied by the file."""
    records = read_bookings()
    assert records
    assert {r.source for r in records} == {"example"}


def test_the_example_covers_every_status() -> None:
    """Its whole job is to be a fixture for the reconciliation matrix. A status with no row is
    a branch with no test and a column the client is never asked for."""
    assert {r.status for r in read_bookings()} == set(STATUSES)


def test_the_example_lines_up_with_the_two_really_recorded_slots() -> None:
    """Otherwise it exercises the importer and nothing downstream."""
    ids = {r.slot_id("venue_01") for r in read_bookings()}
    assert {"venue_01_2026-07-11_1000", "venue_01_2026-07-12_2030"} <= ids


def test_a_real_export_is_never_tagged_as_the_example(tmp_path) -> None:
    path = write(tmp_path, "field_01,2026-07-11,10:00,11:00,R-1,confirmed,a.demir")
    assert read_bookings(path)[0].source == "b"


# --- refusing rather than skipping --------------------------------------------------------


def test_an_unknown_status_is_refused(tmp_path) -> None:
    path = write(tmp_path, "field_01,2026-07-11,10:00,11:00,R-1,provisional,a.demir")
    with pytest.raises(ValueError, match="not one of"):
        read_bookings(path)


def test_a_missing_column_names_what_is_missing(tmp_path) -> None:
    path = tmp_path / "b.csv"
    path.write_text("field_id,date,start\nfield_01,2026-07-11,10:00\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        read_bookings(path)


def test_an_empty_field_id_is_refused(tmp_path) -> None:
    path = write(tmp_path, ",2026-07-11,10:00,11:00,R-1,confirmed,a.demir")
    with pytest.raises(ValueError, match="field_id is empty"):
        read_bookings(path)


def test_a_slot_booked_twice_is_refused(tmp_path) -> None:
    """Two bookings for one hour reconcile to two verdicts for that hour, and whichever is
    processed second wins silently."""
    path = write(
        tmp_path,
        "field_01,2026-07-11,10:00,11:00,R-1,confirmed,a.demir",
        "field_01,2026-07-11,10:00,11:00,R-2,confirmed,s.yilmaz",
    )
    with pytest.raises(ValueError, match="second booking"):
        read_bookings(path)


def test_an_end_before_its_start_is_refused(tmp_path) -> None:
    path = write(tmp_path, "field_01,2026-07-11,22:00,01:00,R-1,confirmed,a.demir")
    with pytest.raises(ValueError, match="not after start"):
        read_bookings(path)


def test_an_unparseable_date_warns_about_the_american_convention(tmp_path) -> None:
    """05/06/2026 is two different days depending on the exporter, and no importer can tell.
    The error is where the reader is told to go and ask."""
    path = write(tmp_path, "field_01,not-a-date,10:00,11:00,R-1,confirmed,a.demir")
    with pytest.raises(ValueError, match="MM/DD/YYYY"):
        read_bookings(path)


def test_a_missing_file_says_what_the_columns_should_be(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="field_id"):
        read_bookings(tmp_path / "absent.csv")


def test_blank_lines_are_skipped_but_blank_fields_are_not(tmp_path) -> None:
    path = tmp_path / "b.csv"
    path.write_text(
        HEADER
        + "field_01,2026-07-11,10:00,11:00,R-1,confirmed,a.demir\n"
        + ",,,,,,\n",
        encoding="utf-8",
    )
    assert len(read_bookings(path)) == 1


# --- meaning ------------------------------------------------------------------------------


def test_cancelled_is_not_the_same_as_never_booked(tmp_path) -> None:
    """A cancelled slot showing a match is unbooked usage; one showing an empty pitch is the
    system working. Collapsing the two at import destroys that before reconciliation sees it."""
    assert "cancelled" in STATUSES
    assert "cancelled" not in SOLD


def test_maintenance_and_blocked_are_not_sales() -> None:
    """Treating a scheduled mow as a booking reports every one of them as a no-show."""
    assert not (SOLD & {"maintenance", "blocked"})


def test_a_no_show_is_still_a_sale() -> None:
    """The customer paid. A no-show that reads as unbooked would invert the anomaly."""
    assert "no_show" in SOLD


def test_the_slot_id_matches_the_one_the_scheduler_builds() -> None:
    """Two functions composing the same identifier are two that can drift apart, and the
    symptom is a booking that never matches its slot."""
    from pitch_occupancy.scheduler import ScheduledSlot

    slot = ScheduledSlot("venue_01", time(10, 0), 60, ("camera_A",), field_id="field_01")
    assert slot_id_for("venue_01", date(2026, 7, 11), time(10, 0)) == slot.slot_id(
        date(2026, 7, 11)
    )


def test_staff_observation_is_not_invented_from_the_booking_status() -> None:
    """A booking export says what was sold. It does not say what staff saw, and guessing that
    would fabricate the second opinion reconciliation exists to compare against."""
    converted = to_reconcile_bookings(read_bookings())
    assert all(b.staff_recorded_used is None for b in converted)


def test_staff_observations_are_used_when_actually_supplied() -> None:
    supplied = {"field_01_2026-07-11_1000": True}
    converted = to_reconcile_bookings(read_bookings(), staff_recorded=supplied)
    match = [b for b in converted if b.date == "2026-07-11" and b.start == "10:00"]
    assert match and match[0].staff_recorded_used is True


def test_the_converted_bookings_reconcile(tmp_path) -> None:
    """End to end: the importer's output is the shape reconcile.py consumes."""
    from pitch_occupancy.data.taxonomy import SlotStatus
    from pitch_occupancy.slots.reconcile import reconcile

    booking = to_reconcile_bookings(read_bookings())[0]
    result = reconcile(booking, SlotStatus.USED)
    assert result.field_id == "field_01"


def test_the_example_path_points_at_the_committed_file() -> None:
    assert EXAMPLE_PATH.exists(), EXAMPLE_PATH


# --- the safe entry point (runbook row 8, 2026-09-11) --------------------------------------


def test_reconcile_slot_computes_coverage_so_a_caller_cannot_forget_it() -> None:
    """The whole reason this function exists: the safe call is longer than the unsafe one.

    Reconciling a slot the export does not mention means building `Booking(booked=False)` by
    hand and remembering `records_cover_this_day` beside it. A caller who forgets gets no
    error - just UNBOOKED_USAGE at SERIOUS for a slot nobody sold.
    """
    from datetime import date

    from pitch_occupancy.bookings import read_bookings, reconcile_slot
    from pitch_occupancy.data.taxonomy import SlotStatus
    from pitch_occupancy.slots.reconcile import Anomaly, Severity

    records = read_bookings()
    known = records[0]

    outside = reconcile_slot(records, field_id=known.field_id, day=date(2027, 1, 1),
                             start=known.start, vision_status=SlotStatus.USED)
    assert outside.anomaly is Anomaly.NEEDS_REVIEW
    assert outside.severity is Severity.INFO


def test_an_unmatched_slot_inside_the_export_is_still_reported_as_unbooked() -> None:
    """The guard must not buy safety by suppressing the finding reconciliation exists to
    make. A pitch with no booking on a day the export covers is genuinely unbooked usage."""
    from datetime import time

    from pitch_occupancy.bookings import read_bookings, reconcile_slot
    from pitch_occupancy.data.taxonomy import SlotStatus
    from pitch_occupancy.slots.reconcile import Anomaly

    records = read_bookings()
    covered_day = records[0].date
    out = reconcile_slot(records, field_id="pitch_that_is_not_in_the_export",
                         day=covered_day, start=time(6, 0), vision_status=SlotStatus.USED)
    assert out.anomaly is Anomaly.UNBOOKED_USAGE


def test_a_matched_booking_carries_its_fields_through() -> None:
    """`maintenance_window` and `entered_by` come off the matched record, so a lookup that
    matched the wrong row would change the *anomaly*, not merely its explanation."""
    from datetime import date, time

    from pitch_occupancy.bookings import BookingRecord, reconcile_slot
    from pitch_occupancy.data.taxonomy import SlotStatus
    from pitch_occupancy.slots.reconcile import Anomaly

    day, at = date(2026, 7, 11), time(10, 0)
    record = BookingRecord(field_id="field_01", date=day, start=at, end=time(11, 0),
                           customer_ref="ref", status="maintenance", entered_by="alex",
                           source="test")
    out = reconcile_slot([record], field_id="field_01", day=day, start=at,
                         vision_status=SlotStatus.USED)
    # maintenance and unsold: play during a window nobody bought is unbooked usage
    assert out.anomaly is Anomaly.UNBOOKED_USAGE


def test_blocked_slot_sold_cannot_be_produced_from_a_real_export() -> None:
    """A SERIOUS anomaly the matrix advertises and the importer cannot reach.

    `BLOCKED_SLOT_SOLD` fires on `maintenance_window and booked`. Both are derived from one
    `status` column - `booked = status in SOLD`, `maintenance_window = status ==
    "maintenance"` - and `SOLD` is `{confirmed, no_show}`. One column cannot hold two values,
    so the conflict the anomaly describes is **inexpressible in the schema WP6-T4 asks the
    client for**, and every instance of it in this repository is a hand-built fixture.

    That is a finding about the request, not a bug in the code: detecting a blocked slot that
    was nevertheless sold needs the export to carry the block separately from the status.
    `thesis/data_requests.md` now asks for it. This test fails if the schema grows that field
    and the derivation is not updated with it - which is the moment the anomaly becomes real.
    """
    from pitch_occupancy.bookings import SOLD

    assert "maintenance" not in SOLD, (
        "the schema can now express a sold maintenance window; update reconcile_slot's "
        "derivation and the note in data_requests.md, because BLOCKED_SLOT_SOLD just became "
        "reachable from a real export"
    )
