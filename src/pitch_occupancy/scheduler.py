"""The scheduler: which slots are due, and running one when it is.

`worker.run_slot` samples, classifies, fuses and aggregates a single slot, and has done for
some time. What was missing is the part that decides *when* — reading a schedule, working out
what is running now, and writing the result to the database at slot end.

Two decisions shape the whole module.

**The clock is a parameter.** Every function that cares about time takes ``now``. A scheduler
whose behaviour can only be observed by waiting an hour is a scheduler with no tests, and
"runs continuously" is the least interesting half of the job — the interesting half is
whether it starts the right slot.

**Deciding and doing are separate.** :func:`due` is pure: a schedule and a timestamp in, a
list of slots out. :func:`run_due` does the work. So "what would run" is answerable without
running anything, which is the same shape as the retention worker's dry run and for the same
reason — the thing that touches the world should be the smaller, later half.

What this is not
----------------

**Not a daemon.** There is no process supervision or restart policy in this module.
`run_forever` (synchronous, for recorded video) and `run_concurrently` (one thread per running
slot, for live cameras) are plain loops with an injectable sleep, which is enough to run under
a service manager but is not itself one. `docs/service.md` shows how to keep the live worker
running under Task Scheduler, NSSM or systemd, and `status.py` is how a page can tell whether
it is.

**Not a live deployment.** It has never run against a camera. The source is whatever
`FrameSource` it is handed, and the tests hand it recorded video. A shadow-mode run against
real cameras is where that changes.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pitch_occupancy.config import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pitch_occupancy.frame_source import Frame, FrameSource

__all__ = [
    "ScheduledSlot", "Schedule", "load_schedule", "due", "due_instances", "run_due", "run_forever",
    "run_concurrently", "SlotCancelled", "describe", "sources_from_recordings",
    "schedule_from_recordings", "live_sources",
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
    return [slot for slot, _ in due_instances(schedule, now)]


def due_instances(schedule: Schedule, now: datetime) -> list[tuple[ScheduledSlot, Date]]:
    """Like :func:`due`, with the day each slot instance *started* on.

    Yesterday is checked too: a 23:30 slot of 90 minutes is still running at 00:30, and
    looking only at today's window made it stop being due at midnight. Its id and its
    recordings belong to the day it started, so the day travels with the slot.
    """
    out = []
    for day in (now.date(), now.date() - timedelta(days=1)):
        for slot in schedule.slots:
            if not slot.runs_on(day):
                continue
            begins, ends = slot.window(day)
            if begins <= now < ends:
                out.append((slot, day))
    return out


def run_due(
    schedule: Schedule,
    now: datetime,
    *,
    source_for: Callable[[ScheduledSlot, Date], object],
    classify=None,
    pipeline=None,
    connection=None,
    model_key: str | None = None,
    on_slot: Callable[[str, object], None] | None = None,
    evidence_dir: Path | None = None,
) -> list[str]:
    """Run every slot due at ``now`` and persist its verdict. Returns the slot ids run.

    ``pipeline`` is the assembled deployment (`pipeline.assemble`) and is what a
    deployment passes: the detector, the rule and the boundary lookup together. A test spies
    on what reaches `run_slot`, so the deployed path is the one that runs.

    ``classify`` alone still works, for tests that script a classifier and want nothing in
    front of it. With a pipeline, the slot's venue and recording key are passed too, so the
    boundary is resolved for *this* slot's cameras (`roi.resolve`).

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
    if classify is None and pipeline is None:
        raise TypeError("run_due needs a pipeline (or, for a scripted test, a classifier)")

    ran: list[str] = []
    for slot, day in due_instances(schedule, now):
        slot_id = slot.slot_id(day)
        try:
            run = _execute(slot, day, source_for(slot, day), classify=classify,
                           pipeline=pipeline, evidence_dir=evidence_dir)
        except Exception as exc:  # noqa: BLE001 - one bad camera must not stop the rest
            if on_slot:
                on_slot(slot_id, exc)
            continue
        if connection is not None:
            _persist(connection, slot, day, run,
                     model_key or getattr(pipeline, "model_key", None))
        ran.append(slot_id)
        if on_slot:
            on_slot(slot_id, run)
    return ran


def _execute(slot: ScheduledSlot, day: Date, source, *, classify, pipeline, evidence_dir,
             on_minute=None):
    """`worker.run_slot` for one slot instance - the one call both runners make.

    Looked up on the module at call time, so a test that replaces `worker.run_slot` sees every
    call, from `run_due` and from `run_concurrently` alike.
    """
    from pitch_occupancy.worker import run_slot

    slot_id = slot.slot_id(day)
    if pipeline is not None:
        return run_slot(slot_id, source, classify, evidence_dir=evidence_dir,
                        pipeline=pipeline, venue=slot.venue_id,
                        slot_key=_recording_key(slot, day), on_minute=on_minute)
    return run_slot(slot_id, source, classify, evidence_dir=evidence_dir, on_minute=on_minute)


def _persist(connection, slot: ScheduledSlot, day: Date, run, model_key: str | None) -> None:
    """Declare the slot and write its verdict with every sample behind it."""
    from pitch_occupancy.db.store import ensure_slot, record_slot
    from pitch_occupancy.pipeline import resolve_model_key

    begins, ends = slot.window(day)
    ensure_slot(
        connection, slot_id=slot.slot_id(day), venue_id=slot.venue_id,
        field_id=slot.field_id, cameras=slot.cameras, slot_date=day.isoformat(),
        start_time=f"{begins:%H:%M}", end_time=f"{ends:%H:%M}",
    )
    record_slot(
        connection, slot.slot_id(day), run.verdict, run.samples,
        model_key=model_key or resolve_model_key(),
        evidence_paths=[e.image_path for e in run.evidence if e.image_path],
    )


def run_forever(
    schedule: Schedule,
    *,
    source_for: Callable[[ScheduledSlot, Date], object],
    classify=None,
    pipeline=None,
    connection=None,
    clock: Callable[[], datetime] = datetime.now,
    sleep: Callable[[float], None] | None = None,
    interval_s: float = 60.0,
    iterations: int | None = None,
    model_key: str | None = None,
    on_slot: Callable[[str, object], None] | None = None,
    evidence_dir: Path | None = None,
) -> list[str]:
    """The loop. Not a daemon - no supervision, no restart policy.

    ``iterations`` bounds it so a test can run the loop rather than a single tick, and
    ``sleep`` is injectable so that test takes microseconds. A loop whose only observable
    behaviour is that it does not return is a loop with no tests.

    ``model_key`` and ``evidence_dir`` are forwarded to `run_due`, and **were not until
    2026-09-11**. `run_due` has accepted both since it was written; this function simply did
    not pass them on, so every verdict the loop produced was stored with no model key and no
    evidence images, whatever the caller asked for. Nothing caught it because nothing called
    this with either argument - the loop had no entry point until `worker --source live`, and
    adding one is what surfaced it.
    """
    import time as _time

    naps = sleep or _time.sleep
    ran: list[str] = []
    count = 0
    while iterations is None or count < iterations:
        ran += run_due(schedule, clock(), source_for=source_for, classify=classify,
                       pipeline=pipeline, connection=connection, model_key=model_key,
                       on_slot=on_slot, evidence_dir=evidence_dir)
        count += 1
        if iterations is None or count < iterations:
            naps(interval_s)
    return ran


class SlotCancelled(Exception):
    """A running slot was stopped by shutdown before its last minute. No verdict is written.

    Handed to ``on_slot`` like any other failure, so the operator sees which slots were cut
    short; it is an exception rather than a partial `SlotRun` because a partial run aggregated
    as if it were the whole slot is exactly the fabricated verdict this system refuses.
    """


class _ObservedSource:
    """A slot's source, watched: every read updates the heartbeat and honours shutdown.

    Wrapping rather than changing `RTSPSource` keeps both concerns out of the frame reader and
    makes them apply to any source the runner is handed. It has the `FrameSource` interface
    without subclassing it, so importing this module does not import OpenCV. The stop check
    runs before *and* after each read: shutdown wakes a source's interruptible sleep early, and
    a frame read in that moment was taken before its minute arrived - so it is thrown away with
    the slot rather than returned under a label it has not earned.
    """

    def __init__(self, inner: FrameSource, slot_id: str, stop: threading.Event,
                 heartbeat=None) -> None:
        self._inner = inner
        self._slot_id = slot_id
        self._stop = stop
        self._heartbeat = heartbeat

    def cameras(self) -> list[str]:
        return self._inner.cameras()

    @property
    def n_minutes(self) -> int:
        return self._inner.n_minutes

    def _check(self) -> None:
        if self._stop.is_set():
            raise SlotCancelled(f"{self._slot_id} stopped by shutdown before its end")

    def _seen(self, camera_id: str, minute_index: int, got: bool) -> None:
        if self._heartbeat is not None:
            self._heartbeat.frame(self._slot_id, camera_id, minute_index, got=got)

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        self._check()
        frame = self._inner.read(camera_id, minute_index)
        self._check()
        self._seen(camera_id, minute_index, frame is not None)
        return frame

    def read_burst(self, camera_id: str, minute_index: int, *, n: int = 3,
                   spacing_s: float = 1.0) -> list[Frame]:
        self._check()
        frames = self._inner.read_burst(camera_id, minute_index, n=n, spacing_s=spacing_s)
        self._check()
        self._seen(camera_id, minute_index, bool(frames))
        return frames


def run_concurrently(
    schedule: Schedule,
    *,
    source_for: Callable[[ScheduledSlot, Date], object],
    classify=None,
    pipeline=None,
    connection=None,
    clock: Callable[[], datetime] = datetime.now,
    sleep: Callable[[float], None] | None = None,
    interval_s: float = 60.0,
    iterations: int | None = None,
    model_key: str | None = None,
    on_slot: Callable[[str, object], None] | None = None,
    evidence_dir: Path | None = None,
    heartbeat=None,
    stop: threading.Event | None = None,
    max_workers: int | None = None,
    on_minute: Callable[[str, int, object], None] | None = None,
) -> list[str]:
    """The live loop: every due slot in its own thread, the clock ticking while they run.

    **Why this exists (2026-10-10).** `run_forever` calls `run_due`, which runs due slots one
    after another, and against a live `RTSPSource` each `run_slot` call blocks for the slot's
    whole length because it waits for every minute to arrive. So two pitches booked at 08:00
    were recorded one at a time: the second started at 09:30, when `due` no longer returned
    it, and was never recorded at all. A slot on a third pitch at 08:30 was lost the same way,
    because the loop that would have noticed it was blocked inside the first slot. That is
    fine for recorded video - a slot is a minute of seeking - which is why `run_due` and
    `run_forever` stay synchronous for that path; this is what the live worker calls.

    **Each slot instance starts at most once.** A slot is tracked by its id (venue, date and
    start) from the moment it is submitted: while running it is not started again, and once
    finished - recorded, failed or cancelled - it is not retried that day. A failure here is
    structural (a missing camera URL, a bug), since a dead camera is a gap rather than an
    exception, and retrying it every tick would only repeat the same message. Before starting,
    the database is asked whether the slot already has a verdict, so a worker restarted
    mid-evening does not record a finished slot a second time.

    **The database: one connection, one lock.** Every slot thread writes through the same
    ``connection``, and a `threading.Lock` serialises each use of it (the restart check, and
    `ensure_slot` + `record_slot` together). The caller must open it with
    ``connect(path, same_thread=False)``, since sqlite3 otherwise refuses use from another
    thread. A connection per thread was the alternative; it was not chosen because the writes
    are two small transactions per slot, so contention is nil, and one connection means one
    place the file is opened, closed and configured rather than one per thread with a lifetime
    tied to a pool.

    **Shutdown.** New slots stop being started as soon as the loop ends. Then:

    * the loop ends by ``iterations`` - running slots are **allowed to finish** and their
      verdicts are written; the call returns when the last one has. That is the bounded run a
      test or ``--iterations`` asks for.
    * ``KeyboardInterrupt`` (Ctrl+C), or ``stop`` set by another thread - running slots are
      **cancelled**: the next read in each raises `SlotCancelled`, which reaches ``on_slot``,
      and **no verdict is written for them**. A slot cut short is not aggregated as if it were
      whole. A source that sleeps with ``stop.wait`` (the live worker passes that) wakes at
      once; any other finishes its current minute first. ``KeyboardInterrupt`` is re-raised
      once the threads have stopped.

    A slot that raises in its thread is reported through ``on_slot(slot_id, exc)``, exactly as
    `run_due` does, and affects no other slot. ``on_slot`` is called from the slot's thread.

    ``heartbeat`` is a `status.Heartbeat`, or ``None`` to write no status file (the default, so
    a test never touches ``data/``). It is ticked every pass of the loop and told about every
    slot start, read and finish. ``max_workers`` defaults to one thread per scheduled slot,
    which is more than can ever overlap, so a slot never queues behind another.

    Returns the ids of the slots whose verdicts were produced (and written, with a connection).
    """
    import time as _time
    from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
    from concurrent.futures import wait as wait_any

    if classify is None and pipeline is None:
        raise TypeError("run_concurrently needs a pipeline (or, for a test, a classifier)")

    naps = sleep or _time.sleep
    stop = stop or threading.Event()
    db_lock = threading.Lock()
    ran: list[str] = []
    ran_lock = threading.Lock()
    running: dict[str, Future] = {}
    finished: set[str] = set()

    def already_recorded(slot_id: str) -> bool:
        if connection is None:
            return False
        with db_lock:
            row = connection.execute(
                "SELECT 1 FROM slot_evaluations WHERE slot_id = ?", (slot_id,)).fetchone()
        return row is not None

    def report(slot_id: str, outcome: object) -> None:
        if on_slot is not None:
            on_slot(slot_id, outcome)

    def one_slot(slot: ScheduledSlot, day: Date) -> None:
        slot_id = slot.slot_id(day)
        try:
            source = source_for(slot, day)
            if heartbeat is not None:
                heartbeat.slot_started(slot, day, minutes_expected=source.n_minutes,
                                       cameras=source.cameras())
            # Each decided minute as it happens, for a person watching the run; the
            # verdict itself still waits for the whole slot.
            minute_cb = ((lambda m, state: on_minute(slot_id, m, state))
                         if on_minute is not None else None)
            run = _execute(slot, day, _ObservedSource(source, slot_id, stop, heartbeat),
                           classify=classify, pipeline=pipeline, evidence_dir=evidence_dir,
                           on_minute=minute_cb)
            if connection is not None:
                with db_lock:
                    _persist(connection, slot, day, run,
                     model_key or getattr(pipeline, "model_key", None))
        except SlotCancelled as exc:
            if heartbeat is not None:
                heartbeat.slot_finished(slot_id, "CANCELLED")
            report(slot_id, exc)
            return
        except Exception as exc:  # noqa: BLE001 - one bad slot must not stop the rest
            if heartbeat is not None:
                heartbeat.error(f"{slot_id}: {type(exc).__name__}: {exc}")
                heartbeat.slot_finished(slot_id, "FAILED")
            report(slot_id, exc)
            return
        with ran_lock:
            ran.append(slot_id)
        if heartbeat is not None:
            heartbeat.slot_finished(
                slot_id, getattr(run.verdict.status, "value", str(run.verdict.status)))
        report(slot_id, run)

    def reap() -> None:
        for slot_id in [sid for sid, f in running.items() if f.done()]:
            running.pop(slot_id)
            finished.add(slot_id)

    def wait_running() -> None:
        """Wait for the running slots in short steps, so Ctrl+C is seen promptly on Windows
        too, and keep the heartbeat fresh while a long slot drains."""
        last_beat = _time.monotonic()
        while True:
            pending = [f for f in running.values() if not f.done()]
            if not pending:
                break
            wait_any(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            if heartbeat is not None and _time.monotonic() - last_beat >= 30.0:
                heartbeat.tick(clock())
                last_beat = _time.monotonic()
        reap()

    pool = ThreadPoolExecutor(max_workers=max_workers or max(len(schedule.slots), 1),
                              thread_name_prefix="slot")
    try:
        count = 0
        while (iterations is None or count < iterations) and not stop.is_set():
            now = clock()
            reap()
            for slot, day in due_instances(schedule, now):
                slot_id = slot.slot_id(day)
                if slot_id in running or slot_id in finished:
                    continue
                if already_recorded(slot_id):
                    finished.add(slot_id)  # a restart: this slot's verdict is already written
                    continue
                running[slot_id] = pool.submit(one_slot, slot, day)
            if heartbeat is not None:
                heartbeat.tick(now)
            count += 1
            if (iterations is None or count < iterations) and not stop.is_set():
                naps(interval_s)
        wait_running()
    except KeyboardInterrupt:
        stop.set()
        wait_running()
        raise
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
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
    """A ``source_for`` that pulls from real cameras. **Never run against a camera.**

    ``urls`` maps venue id -> camera id -> RTSP URL. The slot's own ``duration_minutes`` is
    handed to the source, which is the piece that was missing: `RTSPSource` used to raise on
    ``n_minutes`` saying "the scheduler decides", and this is the scheduler deciding. Without
    it the live path could not run at all - `worker.run_slot` reads ``n_minutes`` on its first
    line.

    **The source starts at the slot's scheduled start, not when it was built** (2026-10-10).
    `RTSPSource.started_at` defaults to "now", and this function used to take that default, so
    a worker started at 09:00 for an 08:00-09:30 slot recorded ninety minutes from 09:00 -
    running half an hour into the next booking and filing 09:00's pictures under 08:00's
    minutes. With the real start, minute *k* is due at start + *k* minutes, the minutes that
    had already passed are read as gaps (`RTSPSource.wait_for`), and the slot ends at its
    scheduled end. A mostly-missed slot then falls under the capture floor and becomes REVIEW,
    which is the honest answer for an hour that was mostly not watched.

    Extra keyword arguments go through to `RTSPSource`, which is how a test injects a clock, a
    sleep and a capture opener - and how the live worker passes ``sleep=stop.wait`` so Ctrl+C
    wakes a source waiting for its next minute. The default is a real socket and a real hour.
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
        begins, _ = slot.window(day)
        return RTSPSource(
            {c: cameras[c] for c in slot.cameras},
            minutes=slot.duration_minutes,
            started_at=begins,
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
