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
