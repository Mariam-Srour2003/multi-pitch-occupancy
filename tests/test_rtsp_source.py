"""The live camera path (WP6-T3).

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

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.frame_source import RTSPSource
from pitch_occupancy.scheduler import ScheduledSlot, live_sources

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
    a slow run by stretching the slot; the missed minutes belong in the capture rate."""
    clock = FakeClock(START.replace(minute=10))
    live = source(minutes=5, clock=clock)
    live.read("camera_A", 2)
    assert clock.naps == []


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
