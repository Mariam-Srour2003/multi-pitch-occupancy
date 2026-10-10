"""Is this booking data good enough to reconcile against?

`bookings.read_bookings` is strict: it stops at the first bad row, because a booking silently
dropped at import becomes an "unbooked usage" anomaly against a customer who paid. That is
the right behaviour for the import and the wrong one for the person fixing the export - they
want the whole list, once, in plain English. This module is that list.

:func:`check` takes normalised rows (as `booking_sources` produces them, or any dicts with the
schema's keys) and collects **every** problem instead of raising. Each issue has a severity:

* **error** - the strict importer would refuse the data, or reconciliation against it would
  be wrong: missing values, unreadable dates or times, an end before its start, a status that
  is not one of ours, the same booking twice, two sold bookings on one pitch at once;
* **warning** - probably a data-entry slip worth a look: a 5-minute or a 9-hour booking, a
  booking outside opening hours;
* **info** - true, and worth knowing before reading the reconciliation: days inside the range
  with no bookings at all (often an export that skipped a day), bookings in the future,
  bookings the cameras are not scheduled to watch.

The verdict is ``good`` (no errors or warnings), ``usable with warnings``, or ``has errors``.

**Nothing here groups, counts or reports by ``entered_by``.** The column is read because it
is part of the schema, and it is not mentioned again. A data-quality report broken down by
the member of staff who typed each booking is a staff-error league table, which is precisely
the workplace monitoring `slots/reconcile.py` refuses to become. Issues name rows and pitches,
never people; a test asserts it.
"""

from __future__ import annotations

import difflib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from pitch_occupancy.bookings import SOLD, STATUSES

__all__ = ["Issue", "BookingReport", "check", "MIN_MINUTES", "MAX_MINUTES"]

#: A booking shorter than this is more likely a typo (10:00-10:05) than a sale.
MIN_MINUTES = 15
#: Longer than this is more likely a wrong end time than a six-hour match.
MAX_MINUTES = 6 * 60
#: Beyond this many issues of one kind, the rest are summarised in one line - 40,000 rows that
#: all use DD/MM/YYYY are one problem, not 40,000.
MAX_PER_RULE = 100

REQUIRED = ("field_id", "date", "start", "end", "status")

#: Words booking systems commonly use for our statuses. Used only to *suggest* a status_map
#: entry; nothing is translated automatically, because a guessed status is a guessed sale.
_SYNONYMS = {
    "booked": "confirmed", "paid": "confirmed", "active": "confirmed", "reserved": "confirmed",
    "canceled": "cancelled", "cancel": "cancelled", "void": "cancelled",
    "noshow": "no_show", "no-show": "no_show", "no show": "no_show", "dna": "no_show",
    "closed": "blocked", "unavailable": "blocked", "repair": "maintenance",
}

_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True, slots=True)
class Issue:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    rows: tuple[int, ...] = ()
    field: str = ""


@dataclass(slots=True)
class BookingReport:
    total_rows: int
    valid_rows: int
    date_from: str | None
    date_to: str | None
    by_status: dict[str, int]
    pitches: int
    future_rows: int
    past_rows: int
    verdict: str
    summary: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        c = Counter(i.severity for i in self.issues)
        return {s: c.get(s, 0) for s in ("error", "warning", "info")}

    def to_dict(self) -> dict:
        out = asdict(self)
        out["issues"] = [asdict(i) | {"rows": list(i.rows)} for i in self.issues]
        out["counts"] = self.counts
        return out


@dataclass(slots=True)
class _Row:
    n: int
    raw: Mapping[str, Any]
    field_id: str = ""
    day: date | None = None
    start: time | None = None
    end: time | None = None
    status: str = ""
    ok: bool = True

    @property
    def minutes(self) -> int | None:
        if self.day is None or self.start is None or self.end is None:
            return None
        return int((datetime.combine(self.day, self.end)
                    - datetime.combine(self.day, self.start)).total_seconds() // 60)


def _text(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _parse_date(text: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_time(text: str) -> time | None:
    for fmt in ("%H:%M", "%H:%M:%S", "%H%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


def _suggest(status: str) -> str | None:
    s = status.lower().strip()
    if s in _SYNONYMS:
        return _SYNONYMS[s]
    close = difflib.get_close_matches(s.replace(" ", "_").replace("-", "_"), STATUSES, n=1,
                                      cutoff=0.5)
    return close[0] if close else None


def _rows_text(rows: Sequence[int]) -> str:
    rows = sorted(rows)
    if len(rows) == 1:
        return f"row {rows[0]}"
    return "rows " + ", ".join(map(str, rows[:-1])) + f" and {rows[-1]}"


def _day_text(d: date) -> str:
    return f"{d:%a} {d.isoformat()}"


# --- opening hours ------------------------------------------------------------------------

def _hours_lookup(schedule: Any) -> tuple[Callable[[str, date], list[tuple[time, time]] | None],
                                          bool]:
    """``(lookup, is_camera_schedule)``. ``lookup(field, day)`` -> open windows, or None.

    Two shapes are accepted. A `scheduler.Schedule` is the cameras' timetable: a booking
    outside it is not wrong, it is simply not going to be compared, so it is reported as
    *info*. A mapping ``{"mon": ["08:00", "23:00"], ..., "*": [...]}`` is opening hours, and a
    sold booking outside those is a *warning*.
    """
    slots = getattr(schedule, "slots", None)
    if slots is not None:
        def by_schedule(field_id: str, day: date) -> list[tuple[time, time]] | None:
            mine = [s for s in slots if s.field_id == field_id]
            if not mine:
                return None
            out = []
            for s in mine:
                if s.runs_on(day):
                    begins, ends = s.window(day)
                    out.append((begins.time(), ends.time() if ends.date() == day else time.max))
            return out
        return by_schedule, True

    hours: dict[str, tuple[time, time]] = {}
    for key, value in dict(schedule).items():
        opens, closes = (v if isinstance(v, time) else time.fromisoformat(str(v))
                         for v in value)
        hours[str(key).lower()[:3] if key != "*" else "*"] = (opens, closes)

    def by_hours(field_id: str, day: date) -> list[tuple[time, time]] | None:
        window = hours.get(_WEEKDAYS[day.weekday()]) or hours.get("*")
        return [window] if window else []
    return by_hours, False


# --- the check ----------------------------------------------------------------------------

def check(rows: Sequence[Mapping[str, Any]], *, schedule: Any = None,
          today: date | None = None) -> BookingReport:
    """Every problem in ``rows``, collected rather than raised.

    Args:
        rows: normalised booking rows; a ``row`` key, when present, is the number quoted in
            messages (a file line number for a CSV), otherwise rows count from 1.
        schedule: optional - a `scheduler.Schedule` (bookings the cameras will not watch are
            reported as info) or opening hours ``{"mon": ("08:00", "23:00"), "*": ...}``
            (sold bookings outside them are warnings).
        today: what "future" is measured from; defaults to today.
    """
    today = today or date.today()
    issues: list[Issue] = []
    parsed: list[_Row] = []

    def err(r: _Row, code: str, fld: str, message: str) -> None:
        r.ok = False
        issues.append(Issue("error", code, f"Row {r.n}: {message}", (r.n,), fld))

    for i, raw in enumerate(rows, start=1):
        r = _Row(n=int(raw.get("row") or i), raw=raw)
        parsed.append(r)
        missing = [f for f in REQUIRED if not _text(raw.get(f))]
        for f in missing:
            err(r, "missing_value", f, f"{f} is empty")
        r.field_id = _text(raw.get("field_id"))

        if "date" not in missing:
            r.day = _parse_date(_text(raw.get("date")))
            if r.day is None:
                err(r, "bad_date", "date",
                    f"date {raw.get('date')!r} is not a date (expected YYYY-MM-DD)")
        for name in ("start", "end"):
            if name in missing:
                continue
            t = _parse_time(_text(raw.get(name)))
            if t is None:
                err(r, "bad_time", name,
                    f"{name} {raw.get(name)!r} is not a time (expected HH:MM)")
            setattr(r, name, t)
        if r.start and r.end and r.end <= r.start:
            err(r, "end_not_after_start", "end",
                f"ends at {r.end:%H:%M}, which is not after its start {r.start:%H:%M}. "
                f"A booking that crosses midnight must be split into two rows")

        status = _text(raw.get("status")).lower()
        r.status = status
        if status and status not in STATUSES:
            hint = _suggest(status)
            err(r, "unknown_status", "status",
                f"status {raw.get('status')!r} is not one of {', '.join(STATUSES)}"
                + (f". Did you mean {hint!r}? A \"status_map\" entry "
                   f"{{\"{raw.get('status')}\": \"{hint}\"}} would translate it" if hint else ""))

        minutes = r.minutes
        if minutes is not None and minutes > 0:
            if minutes < MIN_MINUTES:
                issues.append(Issue("warning", "very_short", f"Row {r.n}: lasts only {minutes} "
                                    f"minutes ({r.start:%H:%M}-{r.end:%H:%M}) - check the end "
                                    "time", (r.n,), "end"))
            elif minutes > MAX_MINUTES:
                issues.append(Issue("warning", "very_long", f"Row {r.n}: lasts {minutes // 60} h "
                                    f"{minutes % 60:02d} min ({r.start:%H:%M}-{r.end:%H:%M}) - "
                                    "check the end time", (r.n,), "end"))

    good = [r for r in parsed if r.ok]

    # Exact duplicates, then different bookings for the same slot, then overlaps.
    def content(r: _Row) -> tuple:
        # entered_by is deliberately left out: the same booking typed by two people is still
        # the same booking, and the comparison should not be about who typed it.
        return (r.field_id, r.day, r.start, r.end, r.status,
                _text(r.raw.get("customer_ref")))

    by_content: dict[tuple, list[_Row]] = defaultdict(list)
    for r in good:
        by_content[content(r)].append(r)
    reported: set[frozenset[int]] = set()
    for group in by_content.values():
        if len(group) > 1:
            g = group[0]
            issues.append(Issue(
                "error", "duplicate",
                f"The same booking appears {len(group)} times ({_rows_text([x.n for x in group])}"
                f"): {g.field_id} on {g.day} {g.start:%H:%M}-{g.end:%H:%M}. Remove the copies",
                tuple(x.n for x in group), ""))
            for a in group:
                for b in group:
                    reported.add(frozenset((a.n, b.n)))

    by_slot: dict[tuple, list[_Row]] = defaultdict(list)
    for r in good:
        by_slot[(r.field_id, r.day, r.start)].append(r)
    for (fid, day, start), group in by_slot.items():
        distinct = {content(r) for r in group}
        if len(distinct) > 1:
            issues.append(Issue(
                "error", "same_slot",
                f"{_rows_text([x.n for x in group])} are different bookings for the same slot "
                f"({fid}, {day} {start:%H:%M}; statuses: "
                f"{', '.join(sorted({x.status for x in group}))}). The importer needs one row "
                f"per pitch and start time - keep the one that is true now",
                tuple(x.n for x in group), "start"))
            for a in group:
                for b in group:
                    reported.add(frozenset((a.n, b.n)))

    by_pitch_day: dict[tuple, list[_Row]] = defaultdict(list)
    for r in good:
        if r.status in SOLD:
            by_pitch_day[(r.field_id, r.day)].append(r)
    for (fid, day), group in by_pitch_day.items():
        group.sort(key=lambda x: (x.start, x.end))
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if b.start >= a.end:
                    break
                pair = frozenset((a.n, b.n))
                if pair in reported:
                    continue
                reported.add(pair)
                issues.append(Issue(
                    "error", "double_booking",
                    f"Double booking on {fid}, {day}: {_rows_text([a.n, b.n])} are both sold "
                    f"and overlap ({a.start:%H:%M}-{a.end:%H:%M} and "
                    f"{b.start:%H:%M}-{b.end:%H:%M})",
                    (a.n, b.n), "start"))

    # Opening hours / camera schedule.
    if schedule is not None:
        lookup, cameras = _hours_lookup(schedule)
        unscheduled_pitches: dict[str, list[int]] = defaultdict(list)
        for r in good:
            if r.status not in SOLD:
                continue  # a cancelled or blocked hour outside opening times is harmless
            windows = lookup(r.field_id, r.day)
            if windows is None:
                unscheduled_pitches[r.field_id].append(r.n)
                continue
            inside = any(o < r.end and r.start < c for o, c in windows) if cameras else any(
                o <= r.start and r.end <= c for o, c in windows)
            if inside:
                continue
            if cameras:
                issues.append(Issue(
                    "info", "not_watched",
                    f"Row {r.n}: {r.field_id} on {_day_text(r.day)} {r.start:%H:%M}-"
                    f"{r.end:%H:%M} is outside the hours the cameras are scheduled to watch, "
                    f"so it will not be compared", (r.n,), "start"))
            else:
                issues.append(Issue(
                    "warning", "outside_hours",
                    f"Row {r.n}: a sold booking on {_day_text(r.day)} {r.start:%H:%M}-"
                    f"{r.end:%H:%M} is outside opening hours", (r.n,), "start"))
        for fid, ns in unscheduled_pitches.items():
            issues.append(Issue(
                "info", "pitch_not_scheduled",
                f"{len(ns)} booking(s) are for {fid!r}, which has no camera schedule - check "
                f"the pitch name matches the one in the schedule", tuple(ns[:20]), "field_id"))

    # Coverage and gaps.
    days = sorted({r.day for r in good})
    date_from = days[0] if days else None
    date_to = days[-1] if days else None
    if days:
        present = set(days)
        gap: list[date] = []
        d = date_from
        while d <= date_to:
            if d not in present:
                gap.append(d)
            elif gap:
                issues.append(_gap_issue(gap))
                gap = []
            d += timedelta(days=1)

    future = sum(1 for r in good if r.day > today)
    past = sum(1 for r in good if r.day <= today)
    if future:
        issues.append(Issue(
            "info", "future",
            f"{future} booking(s) are after today ({today}). They can be checked for mistakes "
            f"now but cannot be compared with the cameras until the day has passed"))

    sold_no_ref = [r.n for r in good if r.status in SOLD and not _text(r.raw.get("customer_ref"))]
    if sold_no_ref:
        issues.append(Issue(
            "info", "no_customer_ref",
            f"{len(sold_no_ref)} sold booking(s) have no customer reference. Not required, "
            f"but it is what lets someone look the booking up later",
            tuple(sold_no_ref[:20]), "customer_ref"))

    if not parsed:
        issues.append(Issue("error", "empty", "There are no booking rows in this data"))

    issues = _cap(issues)
    by_status = Counter(r.status for r in good)
    errors = any(i.severity == "error" for i in issues)
    warnings = any(i.severity == "warning" for i in issues)
    verdict = "has errors" if errors else "usable with warnings" if warnings else "good"
    summary = {
        "good": "The booking data looks clean and consistent.",
        "usable with warnings": "The booking data can be used; a few rows are worth a look.",
        "has errors": "Some rows need fixing before the bookings can be compared with what "
                      "the cameras saw.",
    }[verdict]
    order = {"error": 0, "warning": 1, "info": 2}
    # Row-less issues (summaries, gaps) after the rows they summarise.
    issues.sort(key=lambda i: (order[i.severity], i.rows[0] if i.rows else float("inf")))
    return BookingReport(
        total_rows=len(parsed),
        valid_rows=len(good),
        date_from=date_from.isoformat() if date_from else None,
        date_to=date_to.isoformat() if date_to else None,
        by_status={s: by_status.get(s, 0) for s in STATUSES},
        pitches=len({r.field_id for r in good}),
        future_rows=future,
        past_rows=past,
        verdict=verdict,
        summary=summary,
        issues=issues,
    )


def _gap_issue(gap: list[date]) -> Issue:
    if len(gap) == 1:
        span = _day_text(gap[0])
    else:
        span = f"{len(gap)} days, {_day_text(gap[0])} to {_day_text(gap[-1])}"
    return Issue(
        "info", "empty_days",
        f"No bookings at all on any pitch: {span}. If the facility was open, the export may "
        f"have missed these days - and every game the cameras saw then would look unbooked")


def _cap(issues: list[Issue]) -> list[Issue]:
    seen: Counter[str] = Counter()
    kept: list[Issue] = []
    extra: dict[str, tuple[str, int]] = {}
    for issue in issues:
        seen[issue.code] += 1
        if seen[issue.code] <= MAX_PER_RULE:
            kept.append(issue)
        else:
            sev, n = extra.get(issue.code, (issue.severity, 0))
            extra[issue.code] = (sev, n + 1)
    for code, (sev, n) in extra.items():
        kept.append(Issue(sev, code, f"... and {n} more issue(s) like the ones above "
                                     f"({code.replace('_', ' ')})"))
    return kept
