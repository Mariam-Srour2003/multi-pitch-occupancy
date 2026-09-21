"""A ball, a flicker, and a ball nobody is touching (2026-09-21).

Reported from use: the ball detector "sometimes tracking a ball and it is not there" and
"sometimes taking it veryyyy well", and a ball that does not change place across frames is
not one being played with. `vision/ball.py` carries the measurements behind each of these.
"""

from __future__ import annotations

import pytest

from pitch_occupancy.vision.ball import MIN_FRAMES, MOVED_DIAMETERS, assess
from pitch_occupancy.vision.counting import PitchCount
from pitch_occupancy.vision.detector import SPORTS_BALL, Detection


def frame(*balls: tuple[int, int, int]) -> PitchCount:
    """A count holding balls at ``(x, y, diameter)``, most confident first."""
    dets = tuple(
        Detection(SPORTS_BALL, (x - d // 2, y - d // 2, x + d // 2, y + d // 2), 0.4 - i * 0.05)
        for i, (x, y, d) in enumerate(balls))
    return PitchCount(people_inside=6, people_total=6, raw_inside=6,
                      ball_seen=bool(dets), ball_confidence=dets[0].conf if dets else 0.0,
                      balls=dets, balls_at_once=len(dets))


def test_a_ball_in_one_frame_of_five_is_a_flicker_not_a_ball() -> None:
    """`Maint day` and `not playing day` each fire once in six frames, at 0.17 and 0.23 -
    a stud, a bin lid, a patch of line paint. `persist` used to accept that."""
    burst = [frame(), frame((400, 300, 12)), frame(), frame(), frame()]
    evidence = assess(burst)
    assert evidence.seen is True, "it was detected, and saying otherwise would be a lie"
    assert evidence.persisted is False
    assert evidence.in_play is False
    assert "flicker" in evidence.why and "1 of 5" in evidence.why


def test_a_ball_that_never_moves_is_furniture() -> None:
    """`maint night`: six sightings of six at 0.30, and 0.00 diameters of movement across
    two seconds. It is genuinely there, lying on the grass beside three people working."""
    still = [frame((400, 300, 7)) for _ in range(5)]
    evidence = assess(still)
    assert evidence.seen and evidence.persisted
    assert evidence.travelled == pytest.approx(0.0)
    assert evidence.in_play is False
    assert "lying still" in evidence.why


def test_a_ball_being_played_with_moves_further_than_it_is_wide() -> None:
    moving = [frame((400 + 40 * i, 300, 12)) for i in range(4)]
    evidence = assess(moving)
    assert evidence.in_play is True
    assert evidence.travelled == pytest.approx(40 / 12, abs=1e-6)
    assert "moving" in evidence.why


def test_the_threshold_is_a_noise_floor_and_half_a_diameter_clears_it() -> None:
    """The first attempt required a whole diameter and turned `playing day 4` - a real
    match whose ball travelled 0.50 - into C3, taking 10 of 13 clips to 9."""
    assert MOVED_DIAMETERS < 0.5, "playing day 4 travelled 0.50 and is a match"
    assert MOVED_DIAMETERS > 0.0, "an exactly-zero test would ride on detector jitter"
    barely = [frame((400, 300, 12)), frame((400 + 6, 300, 12))]  # 0.5 diameters
    assert assess(barely).in_play is True


def test_one_frame_cannot_say_whether_a_ball_is_in_play_and_does_not_pretend() -> None:
    """None, not False. The distinction this project keeps having to restate."""
    evidence = assess([frame((400, 300, 12))], min_frames=1)
    assert evidence.seen and evidence.persisted
    assert evidence.in_play is None, "unmeasurable is not the same as stationary"
    assert evidence.usable is False
    assert "not measurable" in evidence.why


def test_two_balls_at_once_are_reported_because_the_matching_is_then_guesswork() -> None:
    """Sightings are matched by taking the most confident ball per frame. With two genuinely
    in play that is wrong, and the trace has to say so rather than the code pretend."""
    burst = [frame((400, 300, 12), (900, 500, 12)), frame((440, 300, 12), (905, 500, 12))]
    evidence = assess(burst)
    assert evidence.most_at_once == 2
    assert "may not be one ball" in evidence.why


def test_no_ball_at_all_is_not_in_play_and_says_how_many_frames_looked() -> None:
    evidence = assess([frame(), frame(), frame()])
    assert (evidence.seen, evidence.persisted, evidence.in_play) == (False, False, False)
    assert evidence.frames_checked == 3 and "3 frame(s)" in evidence.why


def test_an_empty_burst_is_an_error_not_an_answer() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        assess([])


def test_the_persistence_floor_is_two() -> None:
    assert MIN_FRAMES == 2, "twice is what distinguishes a thing from a flicker"
