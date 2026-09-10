"""Reconciling what the cameras saw against what the records claim (WP6-T5, RQ4).

Three sources describe the same slot:

* the **booking** - what was supposed to happen;
* the **staff record** - what someone wrote down;
* the **vision verdict** - what was actually observed.

Disagreement between them is the operational contribution. Aggregated over weeks the log
surfaces systematic problems: a field with frequent no-shows, slots used off-system.

Three rules keep this defensible as an audit rather than an accusation engine:

1. **REVIEW never becomes an anomaly.** An uncertain verdict routes to a human. The system
   is not permitted to convert its own uncertainty into someone else's error.
2. **Low-confidence or low-contrast slots are downgraded to REVIEW** before any rule runs,
   so a dirty lens on a wet night cannot generate a discrepancy against a member of staff.
3. **Anomalies are per field, never per person.** Attributing discrepancies to individuals
   adds nothing scientifically and a great deal of ethical exposure - see `thesis/ethics.md`
   (WP1-T4). `entered_by` is carried for the operator's own use and is deliberately not
   aggregated here.

Nothing in this module writes to a client system. Booking data is read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pitch_occupancy.data.taxonomy import SlotStatus

__all__ = ["Anomaly", "Severity", "Booking", "Reconciliation", "reconcile"]


class Anomaly(StrEnum):
    CONSISTENT = "CONSISTENT"
    NO_SHOW_OR_OVERRECORDED = "NO_SHOW_OR_OVERRECORDED"
    PLAYED_NOT_RECORDED = "PLAYED_NOT_RECORDED"
    UNBOOKED_USAGE = "UNBOOKED_USAGE"
    #: A pitch closed for maintenance that was sold anyway. **Unreachable from a real
    #: export as the importer stands** (2026-09-11): `bookings.py` derives both `booked`
    #: and `maintenance_window` from one `status` column, and one column cannot hold two
    #: values, so every instance of this in the repository is a hand-built fixture. It
    #: needs the export to carry the block separately - `thesis/data_requests.md` §3 now
    #: asks for that column, and `tests/test_bookings.py` fails if the schema gains it
    #: without the derivation being updated.
    BLOCKED_SLOT_SOLD = "BLOCKED_SLOT_SOLD"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class Severity(StrEnum):
    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    SERIOUS = "serious"


SEVERITY: dict[Anomaly, Severity] = {
    Anomaly.CONSISTENT: Severity.NONE,
    Anomaly.NEEDS_REVIEW: Severity.INFO,
    Anomaly.PLAYED_NOT_RECORDED: Severity.WARNING,
    Anomaly.NO_SHOW_OR_OVERRECORDED: Severity.WARNING,
    Anomaly.BLOCKED_SLOT_SOLD: Severity.SERIOUS,
    Anomaly.UNBOOKED_USAGE: Severity.SERIOUS,
}


@dataclass(frozen=True, slots=True)
class Booking:
    field_id: str
    date: str
    start: str
    booked: bool
    staff_recorded_used: bool | None = None
    maintenance_window: bool = False
    entered_by: str | None = None  # carried, never aggregated - see module docstring


@dataclass(frozen=True, slots=True)
class Reconciliation:
    field_id: str
    date: str
    start: str
    anomaly: Anomaly
    severity: Severity
    explanation: str
    vision_status: SlotStatus

    @property
    def is_anomaly(self) -> bool:
        return self.anomaly not in (Anomaly.CONSISTENT, Anomaly.NEEDS_REVIEW)


def reconcile(
    booking: Booking,
    vision_status: SlotStatus,
    *,
    vision_confidence: float = 1.0,
    min_confidence: float = 0.0,
    records_cover_this_day: bool = True,
) -> Reconciliation:
    """Compare one slot's records against its observed verdict.

    ``records_cover_this_day`` is the caller's answer to *"does the booking export reach
    this date at all?"* - `bookings.covers` computes it. It defaults to True because every
    fixture in this repository is written for the day it describes, and it exists because
    the alternative is not a missing comparison but a **confidently wrong one**: a booking
    export that stops last month makes every observed slot look unbooked, and unbooked usage
    is SERIOUS. A stale export would hand an operator a page of serious anomalies against a
    facility that did nothing wrong, and `authority.py` is explicit that a human confirms
    every one of them - so the cost of that failure is a person's time and a staff member's
    standing, not a wrong cell in a table.
    """

    def result(anomaly: Anomaly, explanation: str) -> Reconciliation:
        return Reconciliation(
            field_id=booking.field_id, date=booking.date, start=booking.start,
            anomaly=anomaly, severity=SEVERITY[anomaly], explanation=explanation,
            vision_status=vision_status,
        )

    # First, and before anything that can return a SERIOUS anomaly. If the records do not
    # reach this day then "not booked" is not a fact about the slot, it is a fact about the
    # export, and the two are indistinguishable downstream.
    if not records_cover_this_day:
        return result(
            Anomaly.NEEDS_REVIEW,
            "the booking export does not cover this date, so an absent booking cannot be "
            "told from an absent record; no discrepancy is raised",
        )

    if vision_confidence < min_confidence:
        return result(
            Anomaly.NEEDS_REVIEW,
            f"vision confidence {vision_confidence:.2f} below {min_confidence:.2f}; "
            f"not reliable enough to raise a discrepancy",
        )
    if vision_status is SlotStatus.REVIEW:
        return result(
            Anomaly.NEEDS_REVIEW,
            "vision verdict is REVIEW; routed to an inspector rather than flagged",
        )

    used = vision_status is SlotStatus.USED

    if booking.maintenance_window and booking.booked:
        return result(
            Anomaly.BLOCKED_SLOT_SOLD,
            "slot was blocked for maintenance yet also booked",
        )

    if not booking.booked:
        if used:
            return result(
                Anomaly.UNBOOKED_USAGE,
                "pitch was in use during a slot with no booking on record",
            )
        return result(Anomaly.CONSISTENT, "unbooked and unused, as expected")

    recorded = booking.staff_recorded_used
    if recorded is None:
        return result(
            Anomaly.NEEDS_REVIEW,
            "booked, but no staff record exists to compare the verdict against",
        )
    if recorded and not used:
        return result(
            Anomaly.NO_SHOW_OR_OVERRECORDED,
            "recorded as used, but no play was observed - a no-show or an over-recorded slot",
        )
    if not recorded and used:
        return result(
            Anomaly.PLAYED_NOT_RECORDED,
            "play was observed but the slot is not recorded as used - a missed check-in",
        )
    return result(
        Anomaly.CONSISTENT,
        "booking, staff record and observation agree",
    )
