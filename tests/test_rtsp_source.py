"""The live camera path.

This class existed for weeks with no tests, and both of its defects were the kind that only
appear when something actually runs it: it raised on `n_minutes`, so `worker.run_slot` could
never get past its first line, and it had no pacing, so it would have taken sixty snapshots in
under a second and called that an hour.

So these tests drive it through the real pipeline rather than poking at its methods, and the
clock, the sleep and the socket are all parameters - which is the only reason an hour-long
class can be tested in milliseconds.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.frame_source import RTSPSource
from pitch_occupancy.scheduler import (
    Schedule,
    ScheduledSlot,
    live_sources,
    run_concurrently,
    run_due,
)

START = datetime(2026, 7, 11, 10, 0)


class FakeCapture:
    """One connection attempt."""

    def __init__(self, broken: bool) -> None:
        self.broken = broken
        self.released = 0

    def read(self):
        if self.broken:
            return False, None
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self) -> None:
        self.released += 1


class FakeClock:
    """A clock that advances when the thing under test sleeps.

    Not a detail. A frozen clock makes `wait_for` sleep the full minute on *every* call,
    including the second camera of the same minute - so the first version of these tests
    reported a pacing bug that does not exist. A test double that cannot represent time
    passing cannot test code whose whole job is waiting.
    """

    def __init__(self, at: datetime = START) -> None:
        self.now = at
        self.naps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.naps.append(seconds)
        self.now += timedelta(seconds=seconds)


def source(*, minutes=3, failures=0, clock=None, opened=None, **kwargs):
    """A live source whose stream is broken for its first ``failures`` connection attempts.

    The counter lives out here rather than inside `FakeCapture` because the real class opens
    a *new* connection per retry - that is the point of reconnecting rather than holding the
    stream - so a fake that resets its failure count per capture is a fake that can never
    succeed, which is what the first version of this file did.
    """
    remaining = {"n": failures}
    clock = clock or FakeClock()

    def open_capture(url):
        if opened is not None:
            opened.append(url)
        broken = remaining["n"] > 0
        remaining["n"] -= 1
        return FakeCapture(broken)

    return RTSPSource(
        {"camera_A": "rtsp://cam-a/s", "camera_B": "rtsp://cam-b/s"},
        minutes=minutes,
        started_at=START,
        open_capture=open_capture,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


# --- the defect that stopped it running at all -------------------------------------------


def test_a_live_source_reports_the_slot_length_it_was_given() -> None:
    """`n_minutes` used to raise, and `worker.run_slot`'s first statement reads it."""
    assert source(minutes=45).n_minutes == 45


def test_the_slot_length_is_required_rather_than_defaulted() -> None:
    """A default would make the slot's own length a guess, and a wrong length silently
    changes the capture rate that every verdict's trustworthiness depends on."""
    with pytest.raises(TypeError):
        RTSPSource({"camera_A": "rtsp://x"})


def test_a_zero_length_slot_is_refused() -> None:
    with pytest.raises(ValueError, match="how long the slot is"):
        RTSPSource({"camera_A": "rtsp://x"}, minutes=0)


def test_no_cameras_is_refused() -> None:
    with pytest.raises(ValueError, match="no camera URLs"):
        RTSPSource({}, minutes=60)


def test_it_runs_through_the_real_worker() -> None:
    """The test that would have caught the original defect. Nothing mocked but the socket."""
    from pitch_occupancy.worker import run_slot

    run = run_slot("venue_01_2026-07-11_1000", source(minutes=3),
                   lambda image: (Class3.ACTIVE_PLAY, 0.9))
    assert run.minutes_captured == 3
    assert run.verdict.status.value == "USED"


# --- pacing ---------------------------------------------------------------------------------


def test_it_waits_for_each_minute_to_arrive() -> None:
    """Without this it takes an hour's worth of snapshots in a second."""
    clock = FakeClock()
    live = source(minutes=3, clock=clock)
    for minute in range(3):
        live.read("camera_A", minute)
    assert clock.naps == [60.0, 60.0], "minute 0 is already due; 1 and 2 wait a minute each"


def test_the_second_camera_of_a_minute_does_not_wait_again() -> None:
    """Both halves of a pitch are read at the same minute. Waiting twice would double the
    slot's length and halve its capture rate."""
    clock = FakeClock()
    live = source(minutes=3, clock=clock)
    live.read("camera_A", 1)
    live.read("camera_B", 1)
    assert clock.naps == [60.0]


def test_a_run_that_has_fallen_behind_catches_up_rather_than_stretching() -> None:
    """If classification overran, minute 2 is already past. Sleeping the shortfall would hide
    a slow run by stretching the slot; the missed minutes belong in the capture rate - and
    since 2026-10-10 the long-gone minute is a gap rather than a frame of 10:10 labelled 10:02."""
    clock = FakeClock(START.replace(minute=10))
    live = source(minutes=5, clock=clock)
    assert live.read("camera_A", 2) is None
    assert clock.naps == []


# --- a minute whose moment has gone is a gap (2026-10-10) -------------------------------------


def test_a_minute_more_than_a_minute_late_is_not_read_as_now() -> None:
    """Minute k belongs to [start + k, start + k + 1). Reading the stream after that and
    calling it minute k is a frame from the wrong time with the right label."""
    opened: list[str] = []
    clock = FakeClock(START + timedelta(minutes=5))
    live = source(minutes=10, clock=clock, opened=opened)
    assert live.read("camera_A", 2) is None
    assert live.read("camera_A", 4) is None, "exactly a minute late is already the next minute"
    assert live.read_burst("camera_B", 3) == []
    assert opened == [], "a minute that has gone must not even open the stream"
    assert live.read("camera_A", 5) is not None, "the current minute is still read"
    assert clock.naps == []


def test_a_read_a_few_seconds_late_is_still_its_minute() -> None:
    """The second camera of a minute, or a minute after a slow one, is read late by seconds -
    that is still its minute, and dropping it would cost half the pitch for nothing."""
    clock = FakeClock(START + timedelta(minutes=2, seconds=40))
    live = source(minutes=5, clock=clock)
    assert live.read("camera_A", 2) is not None
    assert live.read_burst("camera_B", 2, n=1)


def test_how_late_is_too_late_can_be_set() -> None:
    clock = FakeClock(START + timedelta(minutes=2, seconds=20))
    assert source(minutes=5, clock=clock, max_late_s=10).read("camera_A", 2) is None


def test_the_start_time_defaults_to_now_but_can_be_pinned() -> None:
    live = source(minutes=3)
    assert live.started_at == START


# --- failure is a gap, never a fabricated frame ------------------------------------------


def test_a_transient_failure_is_retried() -> None:
    live = source(minutes=3, failures=2)  # retries defaults to 2, so 3 attempts
    assert live.read("camera_A", 0) is not None


def test_a_camera_that_never_answers_produces_a_gap_not_an_exception() -> None:
    """A dead camera must weaken the verdict, not crash the slot or invent an observation."""
    live = source(minutes=3, failures=99)
    assert live.read("camera_A", 0) is None


def test_a_gap_lowers_the_capture_rate_and_the_verdict_says_so() -> None:
    """The whole reason a gap is not a fabricated EMPTY: an invented empty minute bills a
    used slot as unused, while a missing one is visible as a degraded verdict."""
    from pitch_occupancy.worker import run_slot

    run = run_slot("venue_01_2026-07-11_1000", source(minutes=4, failures=99),
                   lambda image: (Class3.ACTIVE_PLAY, 0.9))
    assert run.minutes_captured == 0
    assert run.minutes_missed == 4
    assert run.verdict.status.value == "REVIEW"


def test_every_attempt_releases_its_capture() -> None:
    """Leaking a capture per minute leaks sixty an hour, on a Mini-PC, forever."""
    released: list[FakeCapture] = []

    def open_capture(url):
        capture = FakeCapture(broken=True)
        released.append(capture)
        return capture

    live = RTSPSource({"camera_A": "rtsp://x"}, minutes=2, started_at=START,
                      open_capture=open_capture, clock=FakeClock(), sleep=lambda _s: None)
    live.read("camera_A", 0)
    assert len(released) == 3, "one open per attempt"
    assert all(c.released == 1 for c in released)


def test_an_unknown_camera_is_named() -> None:
    with pytest.raises(KeyError, match="camera_Z"):
        source().read("camera_Z", 0)


# --- the scheduler's live wiring -----------------------------------------------------------


def slot(cameras=("camera_A", "camera_B"), minutes=60) -> ScheduledSlot:
    return ScheduledSlot("venue_01", time(10, 0), minutes, cameras, field_id="field_01")


def test_the_scheduler_hands_the_slot_its_own_duration() -> None:
    """"The scheduler decides" is what the old docstring said and what nothing did."""
    source_for = live_sources(
        {"venue_01": {"camera_A": "rtsp://a", "camera_B": "rtsp://b"}},
        clock=FakeClock(), sleep=lambda _s: None,
        open_capture=lambda url: FakeCapture(broken=False),
    )
    live = source_for(slot(minutes=90), date(2026, 7, 11))
    assert live.n_minutes == 90
    assert live.cameras() == ["camera_A", "camera_B"]


def test_a_venue_with_no_urls_is_refused() -> None:
    source_for = live_sources({"venue_02": {"camera_A": "rtsp://a"}})
    with pytest.raises(KeyError, match="venue_01"):
        source_for(slot(), date(2026, 7, 11))


def test_a_slot_whose_camera_has_no_url_is_refused_rather_than_run_on_the_others() -> None:
    """Half a pitch reported as the whole pitch is the failure `slots/fusion.py` exists to
    prevent, and dropping a camera here would reintroduce it upstream of the fusion."""
    source_for = live_sources({"venue_01": {"camera_A": "rtsp://a"}})
    with pytest.raises(KeyError, match="camera_B"):
        source_for(slot(), date(2026, 7, 11))


# --- a slot started late runs to its scheduled end (2026-10-10) -----------------------------
#
# `live_sources` built its `RTSPSource` without `started_at`, so the source started "now": a
# worker started at 09:00 for an 08:00-09:30 slot recorded ninety minutes from 09:00, half an
# hour into the next booking, filing 09:00's pictures under 08:00's minutes.

URLS = {"venue_01": {"camera_A": "rtsp://a/s", "camera_B": "rtsp://b/s"}}
EIGHT = datetime(2026, 7, 11, 8, 0)


def eight_oclock_slot() -> ScheduledSlot:
    return ScheduledSlot("venue_01", time(8, 0), 90, ("camera_A", "camera_B"),
                         field_id="field_01")


def test_the_scheduler_starts_the_source_at_the_slot_start_not_now() -> None:
    clock = FakeClock(datetime(2026, 7, 11, 9, 0))
    live = live_sources(URLS, clock=clock, sleep=clock.sleep,
                        open_capture=lambda url: FakeCapture(broken=False))(
        eight_oclock_slot(), date(2026, 7, 11))
    assert live.started_at == EIGHT


def _late_start(runner):
    clock = FakeClock(datetime(2026, 7, 11, 9, 0))
    opened: list[str] = []

    def open_capture(url):
        opened.append(url)
        return FakeCapture(broken=False)

    source_for = live_sources(URLS, clock=clock, sleep=clock.sleep, open_capture=open_capture)
    runs: list[object] = []
    runner(Schedule((eight_oclock_slot(),)), clock, source_for,
           lambda sid, run: runs.append(run))
    return clock, opened, runs


def _check_late_start(clock, opened, runs) -> None:
    (run,) = runs
    assert not isinstance(run, BaseException), run
    assert run.minutes_captured == 30, "only 09:00-09:29 was still to come"
    assert run.minutes_missed == 60, "08:00-08:59 are missing minutes, not invented ones"
    assert len(opened) == 60, "30 minutes x 2 cameras; no stream opened for a past minute"
    assert {s.minute_index for s in run.samples} == set(range(60, 90))
    assert clock.now == datetime(2026, 7, 11, 9, 29), "the last minute is 09:29"
    assert clock.now < datetime(2026, 7, 11, 9, 30), "it ends at the slot's scheduled end"
    assert run.verdict.status is SlotStatus.REVIEW, "a third of the slot is under the floor"


def test_a_worker_started_an_hour_late_reads_only_the_rest_of_the_slot() -> None:
    def runner(schedule, clock, source_for, on_slot):
        run_due(schedule, clock(), source_for=source_for,
                classify=lambda image: (Class3.ACTIVE_PLAY, 0.9), on_slot=on_slot)

    _check_late_start(*_late_start(runner))


def test_the_concurrent_runner_aligns_a_late_slot_the_same_way() -> None:
    """The live worker's own runner. One tick, so the only thing moving the clock is the
    slot's thread waiting for its minutes."""
    def runner(schedule, clock, source_for, on_slot):
        run_concurrently(schedule, source_for=source_for, clock=clock,
                         classify=lambda image: (Class3.ACTIVE_PLAY, 0.9),
                         iterations=1, on_slot=on_slot)

    _check_late_start(*_late_start(runner))


def test_a_live_frame_never_carries_the_camera_password() -> None:
    """`source_path` ends up in the database's frame_samples.image_path."""
    import numpy as np

    from pitch_occupancy.frame_source import RTSPSource

    class Capture:
        def read(self):
            return True, np.zeros((4, 4, 3), np.uint8)

        def get(self, _prop):
            return 25.0

        def grab(self):
            return True

        def release(self):
            pass

    source = RTSPSource({"cam": "rtsp://admin:hunter2@10.0.0.5:554/Streaming/Channels/101"},
                        minutes=1, open_capture=lambda _url: Capture(), sleep=lambda _s: None)
    single = source.read("cam", 0)
    burst = source.read_burst("cam", 0, n=2, spacing_s=0.0)
    for frame in [single, *burst]:
        assert frame is not None
        assert "hunter2" not in frame.source_path
        assert "10.0.0.5" in frame.source_path
