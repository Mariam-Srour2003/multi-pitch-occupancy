"""Reconciliation converts a prediction into a claim about someone's records, so the
tests that matter most are the ones asserting when it must stay silent."""

from __future__ import annotations

import pytest

from pitch_occupancy.data.taxonomy import SlotStatus
from pitch_occupancy.slots.reconcile import Anomaly, Booking, Severity, reconcile

USED, NOTUSED, REVIEW = SlotStatus.USED, SlotStatus.NOTUSED, SlotStatus.REVIEW


def booking(**kw) -> Booking:
    base = dict(field_id="field_01", date="2026-07-11", start="10:00", booked=True)
    return Booking(**{**base, **kw})


# --- the matrix -------------------------------------------------------------


def test_booked_recorded_and_played_is_consistent() -> None:
    r = reconcile(booking(staff_recorded_used=True), USED)
    assert r.anomaly is Anomaly.CONSISTENT
    assert not r.is_anomaly


def test_recorded_used_but_nothing_observed() -> None:
    r = reconcile(booking(staff_recorded_used=True), NOTUSED)
    assert r.anomaly is Anomaly.NO_SHOW_OR_OVERRECORDED
    assert r.severity is Severity.WARNING


def test_played_but_not_recorded() -> None:
    r = reconcile(booking(staff_recorded_used=False), USED)
    assert r.anomaly is Anomaly.PLAYED_NOT_RECORDED


def test_unbooked_usage_is_serious() -> None:
    r = reconcile(booking(booked=False), USED)
    assert r.anomaly is Anomaly.UNBOOKED_USAGE
    assert r.severity is Severity.SERIOUS


def test_unbooked_and_unused_is_consistent() -> None:
    assert reconcile(booking(booked=False), NOTUSED).anomaly is Anomaly.CONSISTENT


def test_maintenance_window_sold_is_serious() -> None:
    r = reconcile(booking(maintenance_window=True, staff_recorded_used=True), USED)
    assert r.anomaly is Anomaly.BLOCKED_SLOT_SOLD
    assert r.severity is Severity.SERIOUS


# --- when the system must stay silent ---------------------------------------


def test_review_never_becomes_an_anomaly() -> None:
    """The system may not convert its own uncertainty into someone else's error."""
    for b in (
        booking(staff_recorded_used=True),
        booking(staff_recorded_used=False),
        booking(booked=False),
    ):
        r = reconcile(b, REVIEW)
        assert r.anomaly is Anomaly.NEEDS_REVIEW
        assert not r.is_anomaly


def test_low_confidence_is_downgraded_before_any_rule_runs() -> None:
    """A dirty lens on a wet night must not generate a discrepancy."""
    r = reconcile(booking(booked=False), USED, vision_confidence=0.3, min_confidence=0.6)
    assert r.anomaly is Anomaly.NEEDS_REVIEW
    assert "not reliable enough" in r.explanation


def test_confident_slots_are_still_flagged() -> None:
    r = reconcile(booking(booked=False), USED, vision_confidence=0.95, min_confidence=0.6)
    assert r.anomaly is Anomaly.UNBOOKED_USAGE


def test_missing_staff_record_goes_to_review_not_an_accusation() -> None:
    r = reconcile(booking(staff_recorded_used=None), USED)
    assert r.anomaly is Anomaly.NEEDS_REVIEW


# --- audit properties -------------------------------------------------------


def test_every_anomaly_carries_a_severity_and_explanation() -> None:
    cases = [
        (booking(staff_recorded_used=True), USED),
        (booking(staff_recorded_used=True), NOTUSED),
        (booking(staff_recorded_used=False), USED),
        (booking(booked=False), USED),
        (booking(maintenance_window=True, staff_recorded_used=True), USED),
        (booking(staff_recorded_used=True), REVIEW),
    ]
    for b, status in cases:
        r = reconcile(b, status)
        assert r.explanation
        assert isinstance(r.severity, Severity)


def test_reconciliation_carries_the_slot_identity() -> None:
    r = reconcile(booking(staff_recorded_used=True), USED)
    assert (r.field_id, r.date, r.start) == ("field_01", "2026-07-11", "10:00")


def test_entered_by_is_never_part_of_the_output() -> None:
    """Per-staff attribution is deliberately out of scope (WP1-T4, thesis/ethics.md)."""
    r = reconcile(booking(staff_recorded_used=True, entered_by="alex"), NOTUSED)
    assert "alex" not in r.explanation
    assert not hasattr(r, "entered_by")


def test_only_real_disagreements_count_as_anomalies() -> None:
    consistent = reconcile(booking(staff_recorded_used=True), USED)
    review = reconcile(booking(staff_recorded_used=True), REVIEW)
    flagged = reconcile(booking(booked=False), USED)
    assert not consistent.is_anomaly
    assert not review.is_anomaly
    assert flagged.is_anomaly


@pytest.mark.parametrize("status", [USED, NOTUSED, REVIEW])
def test_vision_status_is_preserved_for_the_evidence_trail(status) -> None:
    assert reconcile(booking(staff_recorded_used=True), status).vision_status is status


# --- a booking export that does not reach the day (runbook row 8) --------------------------


def test_a_day_outside_the_export_is_reviewed_not_called_unbooked() -> None:
    """The failure this guard exists for, and it is not a missing comparison but a
    confidently wrong one.

    An export that stops last month makes every observed slot look unbooked, and unbooked
    usage is SERIOUS. That hands an operator a page of serious anomalies against a facility
    that did nothing wrong - and `authority.py` says a human confirms every one, so the cost
    is a person's time and a staff member's standing.
    """
    booking = Booking(field_id="pitch_1", date="2026-08-01", start="10:00", booked=False)
    stale = reconcile(booking, SlotStatus.USED, records_cover_this_day=False)
    assert stale.anomaly is Anomaly.NEEDS_REVIEW
    assert stale.severity is Severity.INFO
    assert "does not cover this date" in stale.explanation

    covered = reconcile(booking, SlotStatus.USED, records_cover_this_day=True)
    assert covered.anomaly is Anomaly.UNBOOKED_USAGE, (
        "with the export covering the day, genuine unbooked usage must still be reported - "
        "the guard must not suppress the finding the system exists to make"
    )


def test_the_coverage_guard_runs_before_anything_serious() -> None:
    """Ordering is the guard. A blocked slot sold, or unbooked usage, are both SERIOUS and
    both reachable from a booking the export never described."""
    sold_maintenance = Booking(field_id="p", date="2026-08-01", start="10:00",
                               booked=True, maintenance_window=True)
    out = reconcile(sold_maintenance, SlotStatus.USED, records_cover_this_day=False)
    assert out.severity is Severity.INFO, "a serious anomaly escaped the coverage check"


def test_coverage_is_the_span_not_the_booked_days() -> None:
    """A facility with no bookings on a Tuesday still has a Tuesday inside an export that
    spans the week. Treating that as uncovered would suppress the genuine unbooked-usage
    finding, which is the opposite failure and the more expensive one."""
    from datetime import date

    from pitch_occupancy.bookings import coverage, covers, read_bookings

    records = read_bookings()
    span = coverage(records)
    assert span is not None
    first, last = span
    assert covers(records, first) and covers(records, last)
    booked_days = {r.date for r in records}
    gap = next((d for d in _days_between(first, last) if d not in booked_days), None)
    if gap is not None:
        assert covers(records, gap), "a quiet day inside the export read as uncovered"
    assert not covers(records, date(last.year + 1, last.month, last.day))


def test_an_empty_export_covers_nothing() -> None:
    """Not "covers everything". An importer that read zero rows - wrong path, wrong
    delimiter, empty file - must not produce a page of serious anomalies."""
    from datetime import date

    from pitch_occupancy.bookings import coverage, covers

    assert coverage([]) is None
    assert not covers([], date(2026, 7, 11))


def _days_between(first, last):
    from datetime import timedelta

    day = first
    while day <= last:
        yield day
        day += timedelta(days=1)
