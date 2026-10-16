"""Live slots run side by side (2026-10-10).

`run_forever` runs due slots one after another, and a live slot blocks for its whole length
because it waits for each minute to arrive. So two pitches booked at 08:00 were recorded one at
a time - the second starting at 09:30, when it was no longer due, and so never recorded - and a
third pitch's 08:30 slot was missed because the loop was stuck inside the first.

These drive `run_concurrently` with a simulated clock that only the scheduler loop moves: each
slot's source waits on that clock for its minute, exactly as `RTSPSource` waits on the real
one. The old serial loop would deadlock here (the first slot waits for a clock that only the
blocked loop could advance), so every wait has a real-time timeout that turns a regression into
a failure rather than a hang. Nothing sleeps for real.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, time, timedelta

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.db.schema import connect, initialise
from pitch_occupancy.frame_source import Frame, FrameSource
from pitch_occupancy.scheduler import Schedule, ScheduledSlot, SlotCancelled, run_concurrently

DAY = date(2026, 7, 11)
BEFORE = datetime(2026, 7, 11, 7, 59)
#: Real seconds a source may wait for the simulated clock before the test calls it a hang.
HANG_S = 20.0


class _WakingEvent(threading.Event):
    """A stop event that wakes the sources waiting on the simulated clock when it is set, as
    ``stop.wait`` wakes a real source's sleep."""

    def __init__(self, cond: threading.Condition) -> None:
        super().__init__()
        self._cond = cond

    def set(self) -> None:
        super().set()
        with self._cond:
            self._cond.notify_all()


class SimClock:
    """Simulated time that moves only when the scheduler loop sleeps.

    The slot threads never move it: they wait for it, as a live source waits for a minute to
    arrive. ``stop`` wakes them too, which is what the live worker's ``sleep=stop.wait`` does
    for a real source.
    """

    def __init__(self, at: datetime = BEFORE) -> None:
        self._now = at
        self._cond = threading.Condition()
        self.stop = _WakingEvent(self._cond)
        self.stop_at: datetime | None = None

    def __call__(self) -> datetime:
        with self._cond:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._cond:
            self._now += timedelta(seconds=seconds)
            if self.stop_at is not None and self._now >= self.stop_at:
                self.stop.set()
            self._cond.notify_all()

    def wait_until(self, moment: datetime) -> None:
        with self._cond:
            if not self._cond.wait_for(
                    lambda: self._now >= moment or self.stop.is_set(), timeout=HANG_S):
                raise TimeoutError(f"the simulated clock never reached {moment:%H:%M}")


class ClockedSource(FrameSource):
    """A slot's frames, each released only when the simulated clock reaches its minute."""

    def __init__(self, clock: SimClock, starts: datetime, minutes: int, cameras=("cam",),
                 fail_at: int | None = None) -> None:
        self._clock = clock
        self._starts = starts
        self._minutes = minutes
        self._cameras = list(cameras)
        self._fail_at = fail_at
        self.read_at: list[datetime] = []

    def cameras(self) -> list[str]:
        return self._cameras

    @property
    def n_minutes(self) -> int:
        return self._minutes

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        if self._fail_at is not None and minute_index == self._fail_at:
            raise RuntimeError("decoder crashed")
        self._clock.wait_until(self._starts + timedelta(minutes=minute_index))
        self.read_at.append(self._clock())
        return Frame(camera_id, minute_index, np.zeros((4, 4, 3), np.uint8), None)


def pitch(venue: str, start: str, minutes: int) -> ScheduledSlot:
    hh, mm = (int(x) for x in start.split(":"))
    return ScheduledSlot(venue, time(hh, mm), minutes, ("cam",), field_id=f"{venue}_field")


THREE = Schedule((pitch("venue_01", "08:00", 90), pitch("venue_02", "08:00", 90),
                  pitch("venue_03", "08:30", 60)))
IDS = {"venue_01_2026-07-11_0800", "venue_02_2026-07-11_0800", "venue_03_2026-07-11_0830"}


class Harness:
    """`source_for` that records every call: which slot, at what simulated time."""

    def __init__(self, clock: SimClock, *, minutes: int | None = None,
                 broken_venue: str | None = None, crash_venue: str | None = None) -> None:
        self.clock = clock
        self.minutes = minutes
        self.broken_venue = broken_venue
        self.crash_venue = crash_venue
        self.calls: list[tuple[str, datetime]] = []
        self.sources: dict[str, ClockedSource] = {}
        self.outcomes: list[tuple[str, object]] = []
        self._lock = threading.Lock()

    def source_for(self, slot: ScheduledSlot, day: date) -> ClockedSource:
        with self._lock:
            self.calls.append((slot.slot_id(day), self.clock()))
        if slot.venue_id == self.broken_venue:
            raise KeyError("no camera URLs for this venue")
        begins, _ = slot.window(day)
        src = ClockedSource(self.clock, begins, self.minutes or slot.duration_minutes,
                            fail_at=2 if slot.venue_id == self.crash_venue else None)
        self.sources[slot.slot_id(day)] = src
        return src

    def on_slot(self, slot_id: str, outcome: object) -> None:
        with self._lock:
            self.outcomes.append((slot_id, outcome))

    def started(self) -> dict[str, datetime]:
        return dict(self.calls)


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "live.db", same_thread=False)
    initialise(conn)
    yield conn
    conn.close()


def run(harness: Harness, schedule: Schedule = THREE, *, until: datetime, conn=None,
        **kwargs) -> list[str]:
    ticks = int((until - harness.clock()).total_seconds() // 60) + 1
    return run_concurrently(
        schedule, source_for=harness.source_for,
        classify=lambda image: (Class3.ACTIVE_PLAY, 0.9),
        connection=conn, clock=harness.clock, sleep=harness.clock.advance,
        interval_s=60.0, iterations=ticks, model_key="test", on_slot=harness.on_slot,
        **kwargs,
    )


def verdicts(conn) -> list[str]:
    return sorted(r[0] for r in conn.execute("SELECT slot_id FROM slot_evaluations"))


# --- the defect -----------------------------------------------------------------------------


def test_two_pitches_at_once_and_a_third_later_are_all_recorded_once(db) -> None:
    clock = SimClock()
    harness = Harness(clock)
    ran = run(harness, until=datetime(2026, 7, 11, 10, 5), conn=db)

    assert sorted(ran) == sorted(IDS)
    assert verdicts(db) == sorted(IDS), "each slot's verdict in the database, once"
    assert sorted(sid for sid, _ in harness.calls) == sorted(IDS), "each slot started once"
    started = harness.started()
    assert started["venue_01_2026-07-11_0800"] == datetime(2026, 7, 11, 8, 0)
    assert started["venue_02_2026-07-11_0800"] == datetime(2026, 7, 11, 8, 0), (
        "the second 08:00 pitch starts at 08:00, not when the first one ends")
    assert started["venue_03_2026-07-11_0830"] == datetime(2026, 7, 11, 8, 30), (
        "the loop kept ticking while two slots ran, and noticed the 08:30 one on time")
    first = harness.sources["venue_01_2026-07-11_0800"]
    assert max(first.read_at) > datetime(2026, 7, 11, 8, 30), "they genuinely overlapped"
    for slot_id, src in harness.sources.items():
        assert len(src.read_at) == src.n_minutes, f"{slot_id} lost minutes"
    assert all(not isinstance(o, BaseException) for _, o in harness.outcomes)


def test_a_slot_that_finishes_early_is_not_started_again_while_still_due(db) -> None:
    """Each source here is three minutes long, so every slot finishes while its window is open
    and `due` keeps returning it, every tick, for over an hour."""
    clock = SimClock()
    harness = Harness(clock, minutes=3)
    ran = run(harness, until=datetime(2026, 7, 11, 9, 20), conn=db)
    assert sorted(ran) == sorted(IDS)
    assert len(harness.calls) == 3, harness.calls
    assert verdicts(db) == sorted(IDS)


def test_a_slot_that_raises_does_not_stop_the_others(db) -> None:
    """One at the start (no source at all) and one mid-slot (its reader crashes). Both are
    reported through on_slot with the exception, and the third pitch is recorded."""
    clock = SimClock()
    harness = Harness(clock, broken_venue="venue_02", crash_venue="venue_03")
    ran = run(harness, until=datetime(2026, 7, 11, 10, 5), conn=db)

    assert ran == ["venue_01_2026-07-11_0800"]
    assert verdicts(db) == ["venue_01_2026-07-11_0800"]
    failures = {sid: o for sid, o in harness.outcomes if isinstance(o, BaseException)}
    assert isinstance(failures["venue_02_2026-07-11_0800"], KeyError)
    assert isinstance(failures["venue_03_2026-07-11_0830"], RuntimeError)
    assert sorted(sid for sid, _ in harness.calls) == sorted(IDS), "a failed slot is not retried"


def test_a_restarted_worker_skips_slots_already_in_the_database(db) -> None:
    first = Harness(SimClock(), minutes=3)
    run(first, until=datetime(2026, 7, 11, 8, 40), conn=db)
    assert verdicts(db) == sorted(IDS)

    # The worker restarts at 08:45, with all three slots still inside their windows.
    again = Harness(SimClock(datetime(2026, 7, 11, 8, 45)), minutes=3)
    ran = run(again, until=datetime(2026, 7, 11, 9, 0), conn=db)
    assert ran == []
    assert again.calls == [], "a slot whose verdict is written is never opened again"


def test_shutdown_cancels_running_slots_and_records_no_verdict_for_them(db) -> None:
    """Ctrl+C, or `stop` set elsewhere: no new slot starts, the running ones stop at their next
    read, and a slot cut short is reported as cancelled rather than aggregated as if whole."""
    clock = SimClock()
    clock.stop_at = datetime(2026, 7, 11, 8, 10)
    harness = Harness(clock)
    ran = run(harness, until=datetime(2026, 7, 11, 10, 5), conn=db, stop=clock.stop)

    assert ran == []
    assert verdicts(db) == [], "never a verdict from ten minutes of a ninety-minute slot"
    cancelled = {sid for sid, o in harness.outcomes if isinstance(o, SlotCancelled)}
    assert cancelled == {"venue_01_2026-07-11_0800", "venue_02_2026-07-11_0800"}
    assert "venue_03_2026-07-11_0830" not in harness.started(), "nothing new starts after stop"


def test_ctrl_c_in_the_loop_cancels_and_reraises(db) -> None:
    clock = SimClock()
    harness = Harness(clock)

    def sleep(seconds: float) -> None:
        clock.advance(seconds)
        if clock() >= datetime(2026, 7, 11, 8, 5):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_concurrently(THREE, source_for=harness.source_for,
                         classify=lambda image: (Class3.EMPTY, 0.9), connection=db,
                         clock=clock, sleep=sleep, on_slot=harness.on_slot,
                         stop=clock.stop)
    assert verdicts(db) == []
    assert {sid for sid, o in harness.outcomes if isinstance(o, SlotCancelled)} == {
        "venue_01_2026-07-11_0800", "venue_02_2026-07-11_0800"}


def test_the_synchronous_runner_is_unchanged_for_recorded_video() -> None:
    """`run_due` still runs due slots in turn in the caller's thread - right for recordings,
    where a slot is a minute of seeking, and what `worker --source video` relies on."""
    from pitch_occupancy.scheduler import run_due

    threads: list[str] = []

    class Quick(FrameSource):
        def cameras(self):
            return ["cam"]

        @property
        def n_minutes(self):
            return 2

        def read(self, camera_id, minute_index):
            threads.append(threading.current_thread().name)
            return Frame(camera_id, minute_index, np.zeros((4, 4, 3), np.uint8), None)

    ran = run_due(THREE, datetime(2026, 7, 11, 8, 35), source_for=lambda s, d: Quick(),
                  classify=lambda image: (Class3.EMPTY, 0.9))
    assert sorted(ran) == sorted(IDS)
    assert set(threads) == {threading.current_thread().name}


def test_each_decided_minute_is_reported_as_it_happens(db) -> None:
    """A live slot is otherwise silent until its verdict; the worker prints these lines."""
    import threading

    seen: list[tuple[str, int]] = []
    lock = threading.Lock()

    def on_minute(slot_id, minute, state) -> None:
        with lock:
            seen.append((slot_id, minute))

    harness = Harness(SimClock())
    run(harness, until=datetime(2026, 7, 11, 10, 5), conn=db, on_minute=on_minute)
    for slot_id, src in harness.sources.items():
        assert sorted(m for s, m in seen if s == slot_id) == list(range(src.n_minutes))
