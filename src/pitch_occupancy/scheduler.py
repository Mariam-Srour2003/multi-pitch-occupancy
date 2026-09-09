"""The scheduler: which slots are due, and running one when it is (WP6-T2).

`worker.run_slot` samples, classifies, fuses and aggregates a single slot, and has done for
some time. What was missing is the part that decides *when* — reading a schedule, working out
what is running now, and writing the result to the database at slot end.

Two decisions shape the whole module.

**The clock is a parameter.** Every function that cares about time takes ``now``. A scheduler
whose behaviour can only be observed by waiting an hour is a scheduler with no tests, and
"runs continuously" is the least interesting half of the acceptance criterion — the
interesting half is whether it starts the right slot.

**Deciding and doing are separate.** :func:`due` is pure: a schedule and a timestamp in, a
list of slots out. :func:`run_due` does the work. So "what would run" is answerable without
running anything, which is the same shape as the retention worker's dry run and for the same
reason — the thing that touches the world should be the smaller, later half.

What this is not
----------------

**Not a daemon.** There is no process supervision, no restart policy and no systemd unit;
those are WP7-T2 and are not written. `run_forever` exists and is a plain loop with an
injectable sleep, which is enough to run under `nohup` or a service manager but is not itself
one.

**Not a live deployment.** It has never run against a camera. The source is whatever
`FrameSource` it is handed, and the tests hand it recorded video. WP7-T3's shadow-mode run is
where that changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Callable, Iterable

from pitch_occupancy.config import settings

__all__ = [
    "ScheduledSlot", "Schedule", "load_schedule", "due", "run_due", "run_forever",
    "describe", "sources_from_recordings", "schedule_from_recordings", "live_sources",
]

DEFAULT_SCHEDULE = settings.results_dir.parent / "configs" / "slots_schedule.json"

#: Weekday names as the schedule file writes them, Monday first to match `date.weekday()`.
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True, slots=True)
class ScheduledSlot:
    """One bookable period on one pitch.

    ``days`` is a set of weekday names; an empty set means every day. Dates are not stored -
    a schedule is a recurring pattern, and a specific slot instance is that pattern plus a
    date, which is what :func:`slot_id` produces.

    **`venue_id` and `field_id` are different things**, and conflating them was a
    foreign-key failure waiting to happen. The manifest and the database key a slot instance
    as ``<venue>_<date>_<HHMM>``, while `rental_slots.field_id` references a *pitch* - a
    venue may have several. The venue names the slot; the field owns the cameras.
    """

    venue_id: str
    start: time
    duration_minutes: int
    cameras: tuple[str, ...]
    field_id: str = ""
    days: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        # A field defaults to its venue rather than being required, because a
        # single-pitch site is the common case and demanding both would be ceremony.
        if not self.field_id:
            object.__setattr__(self, "field_id", f"{self.venue_id}_field")

    def runs_on(self, day: Date) -> bool:
        return not self.days or WEEKDAYS[day.weekday()] in self.days

    def window(self, day: Date) -> tuple[datetime, datetime]:
        begins = datetime.combine(day, self.start)
        return begins, begins + timedelta(minutes=self.duration_minutes)

    def slot_id(self, day: Date) -> str:
        """The identifier the database and the manifest already use for a slot instance."""
        return f"{self.venue_id}_{day.isoformat()}_{self.start.strftime('%H%M')}"


@dataclass(frozen=True, slots=True)
class Schedule:
    slots: tuple[ScheduledSlot, ...]

    def __len__(self) -> int:
        return len(self.slots)


def load_schedule(path: Path | None = None) -> Schedule:
    """Read `configs/slots_schedule.json`, rejecting anything ambiguous.

    Validation is strict on purpose. A schedule is the one input that decides whether a slot
    is observed at all, and a silently-dropped entry is a slot with no footage and no
    explanation - the failure mode is a missing hour, discovered later, unrecoverable.
    """
    source = path or DEFAULT_SCHEDULE
    if not source.exists():
        raise FileNotFoundError(
            f"no schedule at {source}. It is a list of "
            '{"venue_id", "start", "duration_minutes", "cameras", "days"} entries.'
        )
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{source.name} must hold a list of slots, not {type(raw).__name__}")

    slots: list[ScheduledSlot] = []
    for i, entry in enumerate(raw):
        where = f"{source.name} entry {i}"
        missing = {"venue_id", "start", "duration_minutes", "cameras"} - set(entry)
        if missing:
            raise ValueError(f"{where} is missing {sorted(missing)}")
        try:
            start = time.fromisoformat(entry["start"])
        except ValueError as exc:
            raise ValueError(f"{where}: start {entry['start']!r} is not HH:MM") from exc
        if int(entry["duration_minutes"]) <= 0:
            raise ValueError(f"{where}: duration_minutes must be positive")
        cameras = tuple(entry["cameras"])
        if not cameras:
            raise ValueError(f"{where}: a slot with no cameras can never be observed")
        days = frozenset(d.lower()[:3] for d in entry.get("days", []))
        unknown = days - set(WEEKDAYS)
        if unknown:
            raise ValueError(f"{where}: unknown day(s) {sorted(unknown)}")
        slots.append(ScheduledSlot(
            venue_id=str(entry["venue_id"]), start=start,
            duration_minutes=int(entry["duration_minutes"]),
            cameras=cameras, field_id=str(entry.get("field_id", "")), days=days,
        ))

    seen: set[tuple[str, time]] = set()
    for slot in slots:
        key = (slot.venue_id, slot.start)
        if key in seen:
            raise ValueError(
                f"{source.name}: {slot.venue_id} has two slots starting at {slot.start} - "
                "one would overwrite the other's verdict"
            )
        seen.add(key)
    return Schedule(tuple(slots))


def due(schedule: Schedule, now: datetime) -> list[ScheduledSlot]:
    """Slots whose window contains ``now``. Pure: no I/O, no clock of its own.

    The boundary is half-open - a slot starting at 10:00 and lasting 60 minutes is due at
    10:00 and not at 11:00 - so consecutive slots on one pitch never both claim a minute.
    """
    out = []
    for slot in schedule.slots:
        if not slot.runs_on(now.date()):
            continue
        begins, ends = slot.window(now.date())
        if begins <= now < ends:
            out.append(slot)
    return out


def run_due(
    schedule: Schedule,
    now: datetime,
    *,
    source_for: Callable[[ScheduledSlot, Date], object],
    classify,
    connection=None,
    model_key: str | None = None,
    on_slot: Callable[[str, object], None] | None = None,
    evidence_dir: Path | None = None,
) -> list[str]:
    """Run every slot due at ``now`` and persist its verdict. Returns the slot ids run.

    ``evidence_dir`` is passed straight to `worker.run_slot` and is off by default: writing
    frames of identifiable people to disk is the caller's decision. Without it the verdict is
    still recorded, but `evidence_paths` is empty and the dashboard's inspector has nothing to
    show - which is what it did until 2026-09-09, because `run_slot` set every path to None.

    ``source_for`` supplies the frames, so this module knows nothing about video files, RTSP
    or a simulator - the same seam `worker.run_slot` already uses, and the reason this can be
    tested against recorded footage without pretending to be live.

    A slot that raises is reported and skipped rather than taking the loop down with it: one
    dead camera must not stop the other pitches being observed.
    """
    from pitch_occupancy.db.store import ensure_slot, record_slot
    from pitch_occupancy.worker import run_slot

    ran: list[str] = []
    for slot in due(schedule, now):
        slot_id = slot.slot_id(now.date())
        try:
            run = run_slot(slot_id, source_for(slot, now.date()), classify,
                           evidence_dir=evidence_dir)
        except Exception as exc:  # noqa: BLE001 - one bad camera must not stop the rest
            if on_slot:
                on_slot(slot_id, exc)
            continue
        if connection is not None:
            begins, ends = slot.window(now.date())
            ensure_slot(
                connection, slot_id=slot_id, venue_id=slot.venue_id,
                field_id=slot.field_id, cameras=slot.cameras,
                slot_date=now.date().isoformat(),
                start_time=f"{begins:%H:%M}", end_time=f"{ends:%H:%M}",
            )
            record_slot(
                connection, slot_id, run.verdict, run.samples,
                model_key=model_key or settings.default_model_key,
                evidence_paths=[e.image_path for e in run.evidence if e.image_path],
            )
        ran.append(slot_id)
        if on_slot:
            on_slot(slot_id, run)
    return ran


def run_forever(
    schedule: Schedule,
    *,
    source_for: Callable[[ScheduledSlot, Date], object],
    classify,
    connection=None,
    clock: Callable[[], datetime] = datetime.now,
    sleep: Callable[[float], None] | None = None,
    interval_s: float = 60.0,
    iterations: int | None = None,
    on_slot: Callable[[str, object], None] | None = None,
) -> list[str]:
    """The loop. Not a daemon - no supervision, no restart policy (WP7-T2).

    ``iterations`` bounds it so a test can run the loop rather than a single tick, and
    ``sleep`` is injectable so that test takes microseconds. A loop whose only observable
    behaviour is that it does not return is a loop with no tests.
    """
    import time as _time

    naps = sleep or _time.sleep
    ran: list[str] = []
    count = 0
    while iterations is None or count < iterations:
        ran += run_due(schedule, clock(), source_for=source_for, classify=classify,
                       connection=connection, on_slot=on_slot)
        count += 1
        if iterations is None or count < iterations:
            naps(interval_s)
    return ran


def describe(schedule: Schedule, now: datetime, days: int = 1) -> Iterable[str]:
    """What would run over the next ``days``, without running anything."""
    for offset in range(days):
        day = (now + timedelta(days=offset)).date()
        for slot in sorted(schedule.slots, key=lambda s: (s.start, s.venue_id)):
            if not slot.runs_on(day):
                continue
            begins, ends = slot.window(day)
            yield (f"{slot.slot_id(day):<34} {begins:%Y-%m-%d %H:%M}-{ends:%H:%M}  "
                   f"{len(slot.cameras)} camera(s)")


def _recording_key(slot: ScheduledSlot, day: Date) -> str:
    """The key `discover_slots` uses, which is not the id the manifest and database use.

    Recordings are grouped as ``slot_YYYYMMDD_HHMM`` from the export's filenames; a slot
    instance elsewhere is ``<field>_YYYY-MM-DD_HHMM``. Two conventions for one thing, so the
    translation lives in one named function rather than being done from memory at three call
    sites.
    """
    return f"slot_{day:%Y%m%d}_{slot.start:%H%M}"


def sources_from_recordings(directory: Path) -> Callable[[ScheduledSlot, Date], object]:
    """A ``source_for`` that serves recorded video, for a dry run against real footage.

    Which is how the acceptance criterion *"verdicts correct on recorded slots"* is checked
    without a camera: the same slots `end_to_end_slots.py` evaluates, reached through the
    scheduler instead of directly.
    """
    from pitch_occupancy.frame_source import VideoSlotSource, discover_slots

    found = discover_slots(directory)

    def source_for(slot: ScheduledSlot, day: Date):
        key = _recording_key(slot, day)
        if key not in found:
            raise FileNotFoundError(
                f"no recording {key} under {directory} for slot {slot.slot_id(day)}"
            )
        return VideoSlotSource(found[key])

    return source_for


def live_sources(
    urls: dict[str, dict[str, str]], **kwargs
) -> Callable[[ScheduledSlot, Date], object]:
    """A ``source_for`` that pulls from real cameras (WP6-T3). **Never run against a camera.**

    ``urls`` maps venue id -> camera id -> RTSP URL. The slot's own ``duration_minutes`` is
    handed to the source, which is the piece that was missing: `RTSPSource` used to raise on
    ``n_minutes`` saying "the scheduler decides", and this is the scheduler deciding. Without
    it the live path could not run at all - `worker.run_slot` reads ``n_minutes`` on its first
    line.

    Extra keyword arguments go through to `RTSPSource`, which is how a test injects a clock, a
    sleep and a capture opener. The default is a real socket and a real hour.
    """
    from pitch_occupancy.frame_source import RTSPSource

    def source_for(slot: ScheduledSlot, day: Date):
        cameras = urls.get(slot.venue_id)
        if not cameras:
            raise KeyError(
                f"no camera URLs for venue {slot.venue_id!r}; have {sorted(urls)}. A slot with "
                f"no reachable camera has no state, and defaulting to one would invent it"
            )
        missing = [c for c in slot.cameras if c not in cameras]
        if missing:
            raise KeyError(
                f"slot {slot.slot_id(day)} expects camera(s) {', '.join(missing)} which have "
                f"no URL; a silently dropped camera is half a pitch reported as the whole one"
            )
        return RTSPSource(
            {c: cameras[c] for c in slot.cameras},
            minutes=slot.duration_minutes,
            **kwargs,
        )

    return source_for


def schedule_from_recordings(
    directory: Path, *, venue_id: str = "venue_01", duration_minutes: int = 60
) -> tuple[Schedule, list[Date]]:
    """Derive a schedule and its dates from recorded slots, for a dry run over real footage.

    Only for exercising the path against footage that already exists. **A real deployment
    reads `configs/slots_schedule.json`**, because the schedule is what says a slot *should*
    have happened - and deriving it from what was recorded would make a missing hour
    invisible, which is precisely what the reconciliation exists to catch.

    The camera keys are `file0`/`file1` rather than `camA`/`camB` deliberately: the export's
    `(1)` suffix does not identify a physical camera and the mapping flips between days. See
    `frame_source.discover_slots`.
    """
    from pitch_occupancy.frame_source import discover_slots

    slots: dict[time, ScheduledSlot] = {}
    days: set[Date] = set()
    for key, cameras in sorted(discover_slots(directory).items()):
        _, _, tail = key.partition("slot_")
        stamp, _, clock_text = tail.partition("_")
        try:
            day = Date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
            start = time(int(clock_text[:2]), int(clock_text[2:4]))
        except ValueError:
            continue
        days.add(day)
        slots.setdefault(start, ScheduledSlot(
            venue_id=venue_id, start=start, duration_minutes=duration_minutes,
            cameras=tuple(sorted(cameras)),
        ))
    return Schedule(tuple(slots[k] for k in sorted(slots))), sorted(days)
