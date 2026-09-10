"""Reading the booking sheet (WP6-T4).

Reconciliation compares what the cameras saw against what the facility sold. `slots/reconcile.py`
has done the comparing for some time; what was missing was the half that gets the bookings in,
and until now every booking in this repository has been a hand-written fixture.

**This module only ever reads.** That is not a convention here, it is the interface: a
:class:`BookingSource` exposes :meth:`~BookingSource.read` and nothing else. There is no write
path to comment out, no ``dry_run`` flag to leave at the wrong default, and no method that a
later change could quietly point at a client's production database. The constraint comes from
WP6-T4 - *"READ-ONLY - never write to client systems"* - and from the same reasoning as
`slots/authority.py`: a booking system is the facility's financial record, this project is a
measurement instrument, and an instrument that can edit what it measures is not one.

The normalised schema is the one WP6-T4 names::

    field_id, date, start, end, customer_ref, status, entered_by, source

``entered_by`` is carried and never aggregated, for the reason `reconcile.py` gives at length:
the moment anomalies are grouped by the member of staff who typed the booking, this stops being
a pitch-utilisation tool and becomes workplace monitoring. It is here because reconciliation
needs to hand a human the booking as it was written, and it is carried in a field that nothing
groups by.

**The committed CSV is an example, not data.** `configs/bookings_example.csv` is written by
hand to cover every branch of the reconciliation matrix against the two really-recorded slots.
It exists so the importer has a fixture and so the client can be shown the exact columns being
asked for - which is cheaper than describing them in an email. It is not a measurement, and
:func:`read_bookings` marks every row from it with ``source="example"`` so a table built on it
cannot silently read as a table built on a real export.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, time
from pathlib import Path
from typing import Protocol, runtime_checkable

from pitch_occupancy.slots.reconcile import Booking

__all__ = [
    "BookingRecord",
    "BookingSource",
    "CsvBookingSource",
    "EXAMPLE_PATH",
    "STATUSES",
    "read_bookings",
    "to_reconcile_bookings",
    "slot_id_for",
]

#: The booking's own state, as facility systems record it. `cancelled` is not `not booked`:
#: a cancelled slot that shows a match is unbooked usage, and one that shows an empty pitch is
#: the system working. Collapsing the two at import would destroy that distinction before
#: reconciliation ever sees it.
STATUSES: tuple[str, ...] = ("confirmed", "cancelled", "no_show", "maintenance", "blocked")

#: Statuses that mean the pitch was sold for that hour. `maintenance` and `blocked` are *not*
#: bookings - the facility took the slot off sale - and treating them as bookings would report
#: every scheduled mow as a no-show.
SOLD: frozenset[str] = frozenset({"confirmed", "no_show"})

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "configs" / "bookings_example.csv"

FIELDS: tuple[str, ...] = (
    "field_id", "date", "start", "end", "customer_ref", "status", "entered_by",
)


@dataclass(frozen=True, slots=True)
class BookingRecord:
    """One row of a booking export, normalised.

    ``source`` says where the row came from and is set by the importer, never by the file, so
    an example CSV cannot claim to be a client export by adding a column.
    """

    field_id: str
    date: date_type
    start: time
    end: time
    customer_ref: str
    status: str
    entered_by: str
    source: str

    @property
    def is_sold(self) -> bool:
        return self.status in SOLD

    @property
    def duration_minutes(self) -> int:
        return (
            datetime.combine(self.date, self.end) - datetime.combine(self.date, self.start)
        ).seconds // 60

    def slot_id(self, venue_id: str) -> str:
        return slot_id_for(venue_id, self.date, self.start)


def slot_id_for(venue_id: str, day: date_type, start: time) -> str:
    """The manifest's slot key, built the one way it is built anywhere.

    `scheduler.ScheduledSlot.slot_id` composes exactly this string, and two functions meant to
    agree on an identifier are two that can drift. The scheduler's version stays where it is
    because it is the one a slot is named by; this is the booking side of the same convention,
    and a test asserts they produce the same string.
    """
    return f"{venue_id}_{day.isoformat()}_{start:%H%M}"


def _parse_time(value: str, *, row: int, column: str) -> time:
    text = value.strip()
    for fmt in ("%H:%M", "%H:%M:%S", "%H%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"row {row}: {column} {value!r} is not a time (expected HH:MM)")


def _parse_date(value: str, *, row: int) -> date_type:
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"row {row}: date {value!r} is not a date. Expected YYYY-MM-DD; DD/MM/YYYY is also "
        f"accepted, but note that a US-style MM/DD/YYYY export will be silently misread by "
        f"any importer, so confirm the convention with the facility rather than guessing"
    )


def read_bookings(path: Path | str | None = None, *, source: str | None = None
                  ) -> list[BookingRecord]:
    """Import a booking CSV.

    Every problem raises rather than skipping the row. A booking dropped at import is an hour
    the reconciliation believes was never sold, which is the same output as a genuine
    no-booking - so a silent skip would manufacture an *unbooked usage* anomaly out of a typo,
    and that anomaly is the one that gets a customer accused.

    Args:
        path: the CSV. Defaults to the committed example.
        source: overrides the source tag. Defaults to ``"example"`` for the committed file and
            the file's stem otherwise, so a real export is never labelled as the example.
    """
    target = Path(path) if path is not None else EXAMPLE_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"no booking export at {target}. Expected a CSV with columns "
            f"{', '.join(FIELDS)}; see configs/bookings_example.csv for the shape"
        )
    if source is None:
        source = "example" if target.resolve() == EXAMPLE_PATH.resolve() else target.stem

    with target.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{target.name} is missing column(s): {', '.join(missing)}. "
                f"Required: {', '.join(FIELDS)}"
            )
        records = []
        seen: set[tuple[str, date_type, time]] = set()
        for i, raw in enumerate(reader, start=2):
            if not any((raw.get(c) or "").strip() for c in FIELDS):
                continue  # a blank line, not a row with blank fields
            status = (raw["status"] or "").strip().lower()
            if status not in STATUSES:
                raise ValueError(
                    f"row {i}: status {raw['status']!r} is not one of {', '.join(STATUSES)}"
                )
            day = _parse_date(raw["date"], row=i)
            start = _parse_time(raw["start"], row=i, column="start")
            end = _parse_time(raw["end"], row=i, column="end")
            if end <= start:
                raise ValueError(
                    f"row {i}: end {raw['end']} is not after start {raw['start']}; a slot "
                    f"crossing midnight must be split into two rows"
                )
            field_id = (raw["field_id"] or "").strip()
            if not field_id:
                raise ValueError(f"row {i}: field_id is empty")
            key = (field_id, day, start)
            if key in seen:
                raise ValueError(
                    f"row {i}: a second booking for {field_id} at {day} {start:%H:%M}. "
                    f"Two bookings for one slot reconcile to two different verdicts for the "
                    f"same hour; the export needs deduplicating before import"
                )
            seen.add(key)
            records.append(BookingRecord(
                field_id=field_id,
                date=day,
                start=start,
                end=end,
                customer_ref=(raw["customer_ref"] or "").strip(),
                status=status,
                entered_by=(raw["entered_by"] or "").strip(),
                source=source,
            ))
    return records


def coverage(records: Iterable[BookingRecord]) -> tuple[date_type, date_type] | None:
    """The first and last day this export contains a booking for, or None if it is empty.

    A booking export is a snapshot, and a snapshot has an edge. Reconciliation compares what
    the cameras saw against what the facility sold, so a day beyond that edge has nothing to
    compare against - and the failure is not that the comparison is unavailable, it is that
    it silently succeeds. Every observed slot outside the export looks unbooked, and
    `reconcile.py` calls unbooked usage **SERIOUS**. A month-old export therefore produces a
    page of serious anomalies against a facility that did nothing wrong, and the runbook's
    row 8 has said "not implemented" about exactly this.
    """
    days = sorted(r.date for r in records)
    return (days[0], days[-1]) if days else None


def covers(records: Iterable[BookingRecord], day: date_type) -> bool:
    """Whether the export contains any booking on ``day``.

    Membership of the covered *range* rather than of the booked days: a facility with no
    bookings on a Tuesday still has a Tuesday in an export that spans the week, and treating
    that as "not covered" would suppress the genuine unbooked-usage finding this system
    exists to make. What this catches is the day the export never reached at all.
    """
    span = coverage(records)
    return bool(span) and span[0] <= day <= span[1]


def to_reconcile_bookings(
    records: Iterable[BookingRecord], *, staff_recorded: dict[str, bool] | None = None
) -> list[Booking]:
    """Convert to the shape `slots/reconcile.py` consumes.

    ``staff_recorded`` maps ``"<field>_<date>_<HHMM>"`` to what staff say happened, and is
    left empty by default rather than inferred. A booking export says what was *sold*; it does
    not say what staff observed, and guessing that from ``status`` would invent the second
    opinion that reconciliation exists to compare against.
    """
    staff_recorded = staff_recorded or {}
    out = []
    for r in records:
        key = f"{r.field_id}_{r.date.isoformat()}_{r.start:%H%M}"
        out.append(Booking(
            field_id=r.field_id,
            date=r.date.isoformat(),
            start=f"{r.start:%H:%M}",
            booked=r.is_sold,
            staff_recorded_used=staff_recorded.get(key),
            maintenance_window=r.status == "maintenance",
            entered_by=r.entered_by or None,
        ))
    return out


def reconcile_slot(
    records: Sequence[BookingRecord],
    *,
    field_id: str,
    day: date_type,
    start: time,
    vision_status,
    staff_recorded_used: bool | None = None,
    **kwargs,
):
    """Reconcile one observed slot against an export, coverage included.

    **The intended entry point, and the reason it exists is that the safe call is longer
    than the unsafe one.** Reconciling a slot the export does not mention means building a
    `Booking(booked=False)` by hand and remembering to pass `records_cover_this_day` beside
    it - and a caller who forgets gets no error, just `UNBOOKED_USAGE` at SERIOUS for a slot
    nobody sold. This computes the flag from the records themselves, so forgetting is not
    one of the available outcomes.

    An unmatched slot inside the export's span is genuinely unbooked and is reported as
    such: that is the finding reconciliation exists to make, and suppressing it would trade
    one silent error for another.
    """
    from pitch_occupancy.slots.reconcile import reconcile as _reconcile

    match = next(
        (r for r in records
         if r.field_id == field_id and r.date == day and r.start == start),
        None,
    )
    booking = Booking(
        field_id=field_id,
        date=day.isoformat(),
        start=f"{start:%H:%M}",
        booked=bool(match and match.is_sold),
        staff_recorded_used=staff_recorded_used,
        maintenance_window=bool(match and match.status == "maintenance"),
        entered_by=(match.entered_by or None) if match else None,
    )
    return _reconcile(booking, vision_status,
                      records_cover_this_day=covers(records, day), **kwargs)


@runtime_checkable
class BookingSource(Protocol):
    """A read-only view of a facility's bookings.

    One method. Adding a write here is the change that would need explaining, which is the
    point of stating the interface this narrowly rather than documenting a rule elsewhere.
    """

    def read(self) -> Sequence[BookingRecord]: ...


@dataclass(frozen=True, slots=True)
class CsvBookingSource:
    """The importer WP6-T4 asks for first. SQL and REST go behind this same interface."""

    path: Path | None = None
    source: str | None = None

    def read(self) -> list[BookingRecord]:
        return read_bookings(self.path, source=self.source)
