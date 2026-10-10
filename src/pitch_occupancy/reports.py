"""Reports from what the live system recorded: how much each pitch was played, and when.

The live worker leaves two kinds of rows behind. ``slot_evaluations`` holds one verdict per
rental slot (USED / NOT USED / REVIEW, with the play ratio behind it), and ``frame_samples``
holds the raw evidence: one prediction per camera per minute. A client asks three things of
those rows, and each has a function here:

* :func:`daily_summary` - per pitch per day, how many slots were scheduled, how many were
  used, how many hours were played, and what the booking comparison flagged (no-shows,
  unbooked play). Built from the slot verdicts; **a human override replaces the model's
  verdict**, exactly as the dashboard shows it.
* :func:`busy_hours` - a weekday x hour grid of how often the pitch was being played on.
  Built from the per-minute evidence, so it does not depend on how the day was cut into
  slots. A cell nobody watched is reported as *not observed*, never as 0%.
* :func:`recorded_day` - one recorded day re-cut into the client's own slot length (first
  slot 08:00, 90 minutes, say), with the same per-slot table the video booking-day test
  produces. It reuses :func:`pitch_occupancy.timetable.slot_windows` and
  :func:`~pitch_occupancy.timetable.summarise`, so both reports judge a slot by the same
  rules.

Decisions worth stating:

**Which clock a minute is on.** A stored sample's minute is placed at its rental slot's
start (``slot_date`` + ``start_time``, the facility's local clock) plus ``minute_index``.
``captured_at`` is the UTC time the worker *processed* the frame - for a replayed recording
that is not when the minute happened - so it is only a fallback for a slot with no usable
start, converted to local time.

**How two cameras become one pitch minute.** Stored predictions are classes only - no
people counts survive - so the count-summing fusion of the detector path cannot be redone
here. :func:`pitch_occupancy.slots.fusion.fuse` (strongest activity wins: play beats
maintenance beats empty) is the right rule for classes. One addition: EMPTY needs every
camera that recorded that day to agree. A minute where one half says empty and the other
half was missing or undecided is *undecided*, because the missing half may be where the
match was.

**Missing minutes are unseen, never invented.** A minute with no sample is not in the
timeline and lowers a slot's seen share; it is not counted as empty.

**Hours played are minutes seen playing.** ``play_ratio`` is a share of the slot's
*decided* minutes, so hours played = ``play_ratio`` x decided minutes (``n_samples``, one a
minute), capped at the slot length. Multiplying by the full slot length instead would credit
play to minutes nobody saw.

Everything here only reads. Functions take an open ``sqlite3`` connection.
"""

from __future__ import annotations

import csv
import io
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta

from pitch_occupancy import timetable
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.slots.fusion import fuse

__all__ = [
    "ANOMALY_NAMES", "DayRow", "daily_summary", "summary_to_csv", "BusyCell", "busy_hours",
    "busy_hours_to_csv", "PitchMinute", "minute_states", "RecordedDay", "recorded_day",
    "recorded_day_to_csv", "WEEKDAYS",
]

UNDECIDED = "UNCERTAIN"
_CLASS3_VALUES = frozenset(c.value for c in Class3)
_CODES = {Class3.ACTIVE_PLAY.value: "P", Class3.EMPTY.value: "E",
          Class3.MAINTENANCE_NON_SPORTING.value: "M", UNDECIDED: "U"}

WEEKDAYS: tuple[str, ...] = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                             "Saturday", "Sunday")

#: Reconciliation findings, named the way a client would say them. CONSISTENT is not counted.
ANOMALY_NAMES: dict[str, str] = {
    "NO_SHOW_OR_OVERRECORDED": "no_shows",
    "UNBOOKED_USAGE": "unbooked_play",
    "PLAYED_NOT_RECORDED": "played_not_checked_in",
    "BLOCKED_SLOT_SOLD": "sold_while_closed",
    "NEEDS_REVIEW": "needs_checking",
}


# --- small helpers ----------------------------------------------------------------------------


def _parse_hhmm(value: str | None) -> time | None:
    try:
        return time.fromisoformat((value or "").strip())
    except ValueError:
        return None


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat((value or "").strip())
    except ValueError:
        return None


def _slot_minutes(start: str | None, end: str | None) -> int | None:
    """Slot length in minutes from HH:MM strings; an end at or before the start is next day."""
    s, e = _parse_hhmm(start), _parse_hhmm(end)
    if s is None or e is None:
        return None
    mins = (e.hour * 60 + e.minute) - (s.hour * 60 + s.minute)
    return mins if mins > 0 else mins + 24 * 60


def _hours(minutes: float) -> float:
    return round(minutes / 60, 2)


# --- 1. daily summary -------------------------------------------------------------------------


@dataclass(slots=True)
class DayRow:
    """One pitch on one day (or the totals row, with ``day`` and ``field_id`` = "total")."""

    day: str
    field_id: str
    field_name: str
    scheduled: int = 0
    used: int = 0
    not_used: int = 0
    review: int = 0
    #: Scheduled slots with no verdict at all - nobody looked, which is worth seeing.
    not_checked: int = 0
    overridden: int = 0
    minutes_scheduled: float = 0.0
    minutes_played: float = 0.0
    minutes_observed: float = 0.0
    anomalies: dict[str, int] = field(default_factory=lambda: dict.fromkeys(
        ANOMALY_NAMES.values(), 0))

    @property
    def utilisation(self) -> float | None:
        if self.minutes_scheduled <= 0:
            return None
        return self.minutes_played / self.minutes_scheduled

    def add(self, other: DayRow) -> None:
        for name in ("scheduled", "used", "not_used", "review", "not_checked", "overridden",
                     "minutes_scheduled", "minutes_played", "minutes_observed"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for k, v in other.anomalies.items():
            self.anomalies[k] = self.anomalies.get(k, 0) + v

    def to_dict(self) -> dict[str, object]:
        util = self.utilisation
        return {
            "day": self.day, "field_id": self.field_id, "field_name": self.field_name,
            "scheduled": self.scheduled, "used": self.used, "not_used": self.not_used,
            "review": self.review, "not_checked": self.not_checked,
            "overridden": self.overridden,
            "hours_scheduled": _hours(self.minutes_scheduled),
            "hours_played": _hours(self.minutes_played),
            "hours_observed": _hours(self.minutes_observed),
            "utilisation": None if util is None else round(util, 4),
            **self.anomalies,
        }


def daily_summary(conn: sqlite3.Connection, *, start: date, end: date,
                  field_id: str | None = None) -> dict[str, object]:
    """Per pitch per day between ``start`` and ``end`` (inclusive), plus a totals row.

    Returns ``{"start", "end", "field_id", "rows": [...], "totals": {...}}`` where each row is
    :meth:`DayRow.to_dict`. A slot's status is the human override when there is one, else the
    model's verdict; a scheduled slot with no verdict is counted as ``not_checked``.
    """
    sql = """SELECT rs.slot_id, rs.field_id, rs.slot_date, rs.start_time, rs.end_time,
                    COALESCE(f.name, rs.field_id) AS field_name,
                    se.status, se.is_overridden, se.override_status, se.play_ratio,
                    se.n_samples,
                    (SELECT COUNT(DISTINCT fs.minute_index) FROM frame_samples fs
                      WHERE fs.slot_id = rs.slot_id) AS minutes_sampled,
                    rc.anomaly
             FROM rental_slots rs
             LEFT JOIN fields f ON f.field_id = rs.field_id
             LEFT JOIN slot_evaluations se ON se.slot_id = rs.slot_id
             LEFT JOIN reconciliations rc ON rc.slot_id = rs.slot_id
             WHERE rs.slot_date BETWEEN ? AND ?"""
    params: list[object] = [start.isoformat(), end.isoformat()]
    if field_id:
        sql += " AND rs.field_id = ?"
        params.append(field_id)
    sql += " ORDER BY rs.slot_date, rs.field_id, rs.start_time"

    rows: dict[tuple[str, str], DayRow] = {}
    for r in conn.execute(sql, params).fetchall():
        key = (r["slot_date"], r["field_id"])
        row = rows.get(key)
        if row is None:
            row = rows[key] = DayRow(day=r["slot_date"], field_id=r["field_id"],
                                     field_name=r["field_name"])
        row.scheduled += 1
        length = _slot_minutes(r["start_time"], r["end_time"])
        if length is None:  # a slot with no usable end: its recorded minutes stand in
            length = r["minutes_sampled"] or r["n_samples"] or 0
        row.minutes_scheduled += length
        if r["status"] is None:
            row.not_checked += 1
        else:
            status = r["status"]
            if r["is_overridden"] and r["override_status"]:
                status = r["override_status"]
                row.overridden += 1
            if status == "USED":
                row.used += 1
            elif status == "NOTUSED":
                row.not_used += 1
            else:
                row.review += 1
            decided = float(r["n_samples"] or 0)
            row.minutes_observed += min(decided, length) if length else decided
            played = float(r["play_ratio"] or 0.0) * decided
            row.minutes_played += min(played, length) if length else played
        name = ANOMALY_NAMES.get(r["anomaly"] or "")
        if name:
            row.anomalies[name] += 1

    totals = DayRow(day="total", field_id="total", field_name="All pitches")
    for row in rows.values():
        totals.add(row)
    return {"start": start.isoformat(), "end": end.isoformat(), "field_id": field_id,
            "rows": [r.to_dict() for r in rows.values()], "totals": totals.to_dict()}


_SUMMARY_COLUMNS = (
    "day", "field_id", "field_name", "scheduled", "used", "not_used", "review", "not_checked",
    "overridden", "hours_scheduled", "hours_played", "hours_observed", "utilisation",
    *ANOMALY_NAMES.values(),
)


def summary_to_csv(summary: dict[str, object]) -> str:
    """The per-day table as CSV, totals last. Utilisation is a percentage; blank = unknown."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    header = [c if c != "utilisation" else "utilisation_pct" for c in _SUMMARY_COLUMNS]
    writer.writerow(header)
    for d in [*summary["rows"], summary["totals"]]:  # type: ignore[misc]
        line = []
        for c in _SUMMARY_COLUMNS:
            v = d.get(c)
            if c == "utilisation":
                v = "" if v is None else round(100 * v, 1)
            line.append(v)
        writer.writerow(line)
    return buf.getvalue()


# --- per-minute pitch state (shared by busy hours and the re-cut day) ---------------------------


@dataclass(frozen=True, slots=True)
class PitchMinute:
    """One pitch, one clock minute: the fused state and what each camera said."""

    field_id: str
    at: datetime
    #: A Class3 value, or ``"UNCERTAIN"`` when samples exist but nothing could be decided.
    state: str
    per_camera: tuple[tuple[str, str], ...]


def _sample_clock(row: sqlite3.Row) -> datetime | None:
    """The local clock minute a sample stands for (see the module docstring)."""
    day, start = _parse_date(row["slot_date"]), _parse_hhmm(row["start_time"])
    if day is not None and start is not None:
        return datetime.combine(day, start) + timedelta(minutes=int(row["minute_index"]))
    try:
        at = datetime.fromisoformat(row["captured_at"])
    except (TypeError, ValueError):
        return None
    if at.tzinfo is not None:
        at = at.astimezone().replace(tzinfo=None)
    return at.replace(second=0, microsecond=0)


def _fuse_minute(observed: dict[str, tuple[str, float]], expected: set[str]) -> str:
    """Strongest activity across the cameras' decided predictions; EMPTY needs all of them.

    ``observed`` maps camera -> (prediction, confidence) for this minute; ``expected`` is
    every camera of the pitch that recorded anything that day.
    """
    decided = {cam: (pred, conf) for cam, (pred, conf) in observed.items()
               if pred in _CLASS3_VALUES}
    if not decided:
        return UNDECIDED
    state = fuse(decided).state
    if state is Class3.EMPTY and set(decided) != expected:
        return UNDECIDED
    return state.value


def minute_states(conn: sqlite3.Connection, *, start: date, end: date,
                  field_id: str | None = None) -> list[PitchMinute]:
    """Every recorded pitch-minute whose local clock date is in ``[start, end]``, in order."""
    lo, hi = (start - timedelta(days=1)).isoformat(), (end + timedelta(days=1)).isoformat()
    sql = """SELECT fs.camera_id, fs.minute_index, fs.predicted, fs.confidence,
                    fs.captured_at, rs.field_id, rs.slot_date, rs.start_time
             FROM frame_samples fs JOIN rental_slots rs ON rs.slot_id = fs.slot_id
             WHERE (rs.slot_date BETWEEN ? AND ?
                    OR substr(fs.captured_at, 1, 10) BETWEEN ? AND ?)"""
    params: list[object] = [lo, hi, lo, hi]
    if field_id:
        sql += " AND rs.field_id = ?"
        params.append(field_id)
    grouped: dict[tuple[str, datetime], dict[str, tuple[str, float]]] = defaultdict(dict)
    cameras_by_day: dict[tuple[str, date], set[str]] = defaultdict(set)
    for r in conn.execute(sql, params).fetchall():
        at = _sample_clock(r)
        if at is None or not start <= at.date() <= end:
            continue
        cam = r["camera_id"]
        slot = grouped[(r["field_id"], at)]
        prev = slot.get(cam)
        # The same camera twice in one minute (overlapping slots): a decided answer beats
        # an undecided one, then the more confident.
        cand = (str(r["predicted"]), float(r["confidence"] or 0.0))
        if prev is None or ((cand[0] in _CLASS3_VALUES, cand[1])
                            > (prev[0] in _CLASS3_VALUES, prev[1])):
            slot[cam] = cand
        cameras_by_day[(r["field_id"], at.date())].add(cam)
    out = []
    for (fid, at), obs in sorted(grouped.items()):
        state = _fuse_minute(obs, cameras_by_day[(fid, at.date())])
        out.append(PitchMinute(fid, at, state,
                               tuple((cam, obs[cam][0]) for cam in sorted(obs))))
    return out


# --- 2. busy hours ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BusyCell:
    weekday: int
    hour: int
    #: Minutes with a decided pitch state - the denominator.
    observed_min: int
    playing_min: int
    #: Minutes recorded but undecided. Not in the denominator.
    undecided_min: int

    @property
    def share(self) -> float | None:
        """Playing share of observed minutes; None means *not observed*, not 0%."""
        return self.playing_min / self.observed_min if self.observed_min else None

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        share = self.share
        out["weekday_name"] = WEEKDAYS[self.weekday]
        out["share"] = None if share is None else round(share, 4)
        out["observed"] = self.observed_min > 0
        return out


def busy_hours(conn: sqlite3.Connection, *, start: date, end: date,
               field_id: str | None = None) -> dict[str, object]:
    """A 7 x 24 grid (Monday first) of the share of observed minutes that were playing.

    With every pitch selected, each pitch-minute counts once, so two pitches watched for the
    same hour give that cell 120 observed minutes.
    """
    observed: Counter[tuple[int, int]] = Counter()
    playing: Counter[tuple[int, int]] = Counter()
    undecided: Counter[tuple[int, int]] = Counter()
    for m in minute_states(conn, start=start, end=end, field_id=field_id):
        key = (m.at.weekday(), m.at.hour)
        if m.state == UNDECIDED:
            undecided[key] += 1
            continue
        observed[key] += 1
        if m.state == Class3.ACTIVE_PLAY.value:
            playing[key] += 1
    cells = [BusyCell(wd, h, observed[(wd, h)], playing[(wd, h)], undecided[(wd, h)])
             for wd in range(7) for h in range(24)]
    return {"start": start.isoformat(), "end": end.isoformat(), "field_id": field_id,
            "weekdays": list(WEEKDAYS), "cells": [c.to_dict() for c in cells],
            "observed_min": sum(observed.values()), "playing_min": sum(playing.values())}


def busy_hours_to_csv(grid: dict[str, object]) -> str:
    """One row per weekday x hour. ``playing_pct`` is blank where nothing was observed."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["weekday", "hour", "observed_min", "playing_min", "undecided_min",
                     "playing_pct"])
    for c in grid["cells"]:  # type: ignore[union-attr]
        share = c["share"]
        writer.writerow([c["weekday_name"], f"{c['hour']:02d}:00", c["observed_min"],
                         c["playing_min"], c["undecided_min"],
                         "" if share is None else round(100 * share, 1)])
    return buf.getvalue()


# --- 3. a recorded day, cut into the client's slots -------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordedDay:
    field_id: str
    day: date
    first_slot: time
    slot_minutes: int
    #: The clock time used as t=0 for the slot windows.
    start: time | None
    duration_s: float
    results: list[timetable.SlotResult]
    #: Every minute from t=0 to the last recorded minute: (seconds from t=0, clock, state);
    #: state is a Class3 value, ``"UNCERTAIN"``, or None when nothing was recorded.
    timeline: list[tuple[float, time, str | None]]
    minutes_recorded: int

    def to_dict(self) -> dict[str, object]:
        counts = Counter(r.status.value for r in self.results)
        return {
            "field_id": self.field_id, "day": self.day.isoformat(),
            "first_slot": f"{self.first_slot:%H:%M}", "slot_minutes": self.slot_minutes,
            "start": None if self.start is None else f"{self.start:%H:%M}",
            "duration_s": self.duration_s, "interval_s": 60.0,
            "minutes_recorded": self.minutes_recorded,
            "counts": {"slots": len(self.results), "USED": counts.get("USED", 0),
                       "NOT_USED": counts.get("NOT_USED", 0),
                       "REVIEW": counts.get("REVIEW", 0)},
            "slots": [r.to_dict() for r in self.results],
            # (seconds, letter): P playing, E empty, M maintenance, U undecided, N not recorded
            "timeline": [[t, _CODES.get(s, "N") if s else "N"] for t, _, s in self.timeline],
        }


def recorded_day(conn: sqlite3.Connection, *, field_id: str, day: date, first_slot: time,
                 slot_minutes: int, rules: timetable.Rules | None = None) -> RecordedDay:
    """Cut one pitch's recorded day into ``slot_minutes`` slots starting at ``first_slot``.

    t=0 is the first recorded minute of the day, unless that is twelve hours or more from
    ``first_slot`` (where `timetable.slot_windows` would read the first slot as the previous
    or next day) - then t=0 is ``first_slot`` itself. The recording ends after the last
    recorded minute. Minutes inside that stretch with no sample are unseen.
    """
    if slot_minutes <= 0:
        raise ValueError(f"slot_minutes must be positive, not {slot_minutes}")
    minutes = minute_states(conn, start=day, end=day, field_id=field_id)
    minutes = [m for m in minutes if m.field_id == field_id]
    if not minutes:
        return RecordedDay(field_id, day, first_slot, slot_minutes, None, 0.0, [], [], 0)

    first_rec = minutes[0].at
    slot0 = datetime.combine(day, first_slot)
    t0 = first_rec if abs((first_rec - slot0).total_seconds()) < 12 * 3600 else slot0
    end = minutes[-1].at + timedelta(minutes=1)
    duration_s = (end - t0).total_seconds()
    if duration_s <= 0:
        return RecordedDay(field_id, day, first_slot, slot_minutes, t0.time(), 0.0, [], [],
                           len(minutes))

    samples = [((m.at - t0).total_seconds(), m.state) for m in minutes if m.at >= t0]
    windows = timetable.slot_windows(t0.time(), duration_s, slot_minutes, first_slot)
    results = timetable.summarise(samples, windows, 60.0,
                                  rules=rules or timetable.DEFAULT_RULES)

    by_minute = {m.at: m.state for m in minutes}
    timeline = []
    at = t0
    while at < end:
        timeline.append(((at - t0).total_seconds(), at.time(), by_minute.get(at)))
        at += timedelta(minutes=1)
    return RecordedDay(field_id, day, first_slot, slot_minutes, t0.time(), duration_s,
                       results, timeline, len(minutes))


def recorded_day_to_csv(rec: RecordedDay) -> str:
    """The re-cut day's slot table, in the booking-day test's CSV layout."""
    return timetable.to_csv(rec.results)
