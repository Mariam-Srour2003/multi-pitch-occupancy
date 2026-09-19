"""Pitch-level fusion of counts (A36): the sum decides, not the halves.

A16 measured "one to four people means not playing" per camera as wrong on 88 of 278 real
matches, because a camera sees half a pitch. The hand-count audit found 18 of 74 play frames
with fewer than five people inside their own camera's boundary. So `fuse_pitch` sums.
"""

from __future__ import annotations

import pytest

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.slots.fusion import PitchVerdict, fuse_pitch
from pitch_occupancy.vision.counting import PitchCount
from pitch_occupancy.vision.rules import MinuteState, RuleConfig, decide

CFG = RuleConfig()


def count(n: int, *, ball: bool = False, raw: int | None = None,
          bounded: bool = True) -> PitchCount:
    return PitchCount(people_inside=n, people_total=n, raw_inside=raw if raw is not None else n,
                      ball_seen=ball, ball_confidence=0.4 if ball else 0.0, bounded=bounded)


def seen(n: int, *, ball: bool = False, motion: float | None = None, raw: int | None = None):
    return decide(count(n, ball=ball, raw=raw), motion=motion, cfg=CFG)


def test_three_and_three_is_a_match_even_though_neither_half_is() -> None:
    a, b = seen(3), seen(3)
    assert a.state is MinuteState.PEOPLE_NOT_PLAYING and b.state is MinuteState.PEOPLE_NOT_PLAYING
    pitch = fuse_pitch({"camA": a, "camB": b}, CFG)
    assert pitch.state is MinuteState.PLAYING and pitch.class3 is Class3.ACTIVE_PLAY
    assert pitch.people_inside == 6 and pitch.n_scored == 2
    assert not pitch.disagreed, "the halves agreed with each other, and the pitch overruled both"
    assert pitch.winning_camera in {"camA", "camB"}
    assert any("camA: 3 inside" in step for step in pitch.trace)


def test_a_ball_seen_by_either_camera_is_seen_by_the_pitch() -> None:
    pitch = fuse_pitch({"camA": seen(4), "camB": seen(2, ball=True)}, CFG)
    assert pitch.ball_seen is True and pitch.state is MinuteState.PLAYING
    assert pitch.confidence == pytest.approx(0.85), "row 7 with six people"


def test_nobody_on_either_half_is_empty_and_the_height_filter_carries_over() -> None:
    pitch = fuse_pitch({"camA": seen(0), "camB": seen(0, raw=1)}, CFG)
    assert pitch.state is MinuteState.EMPTY
    assert pitch.confidence == pytest.approx(0.85), "one dropped detection across the pitch"


def test_a_camera_without_a_count_is_reported_and_halves_the_confidence() -> None:
    """Row 1 (detector unavailable) and row 2 (no boundary) carry nothing to sum; the pitch
    is decided from the other half at half the confidence, and says so."""
    unavailable = decide(None, motion=None, cfg=CFG)
    pitch = fuse_pitch({"camA": seen(6, ball=True), "camB": unavailable}, CFG)
    assert pitch.state is MinuteState.PLAYING and pitch.n_scored == 1
    assert pitch.confidence == pytest.approx(0.85 * 0.5)
    assert any("could not be scored" in step for step in pitch.trace)
    whole_frame = decide(count(20, bounded=False), motion=None, cfg=CFG)
    assert whole_frame.rule == 2
    pitch = fuse_pitch({"camA": seen(0), "camB": whole_frame}, CFG)
    assert pitch.people_inside == 0, "a whole-frame count is never summed into the pitch"
    assert pitch.state is MinuteState.EMPTY


def test_when_no_camera_can_be_scored_the_pitch_abstains() -> None:
    pitch = fuse_pitch({"camA": decide(None, motion=None, cfg=CFG),
                        "camB": decide(count(3, bounded=False), motion=None, cfg=CFG)}, CFG)
    assert pitch.state is MinuteState.UNCERTAIN and pitch.class3 is None
    assert pitch.winning_camera is None and pitch.count is None
    assert "no camera could be scored" in pitch.trace[0]


def test_a_row_4_abstention_is_still_summed_and_its_motion_is_read_again() -> None:
    """Movement with nobody found on one half is evidence, not silence: its zero joins the
    sum and its motion cue is the pitch's."""
    cfg = RuleConfig(motion_hi=2.0)
    moving_empty = decide(count(0), motion=3.0, cfg=cfg)
    assert moving_empty.rule == 4
    with_people = decide(count(6), motion=0.5, cfg=cfg)
    pitch = fuse_pitch({"camA": moving_empty, "camB": with_people}, cfg)
    assert pitch.state is MinuteState.PLAYING and pitch.n_scored == 2
    assert pitch.motion == 3.0
    both_empty = fuse_pitch({"camA": moving_empty, "camB": decide(count(0), motion=0.1, cfg=cfg)},
                            cfg)
    assert both_empty.state is MinuteState.UNCERTAIN and both_empty.count is not None


def test_disagreement_is_between_the_halves_that_were_scored() -> None:
    pitch = fuse_pitch({"camA": seen(0), "camB": seen(7, ball=True)}, CFG)
    assert pitch.disagreed and pitch.state is MinuteState.PLAYING
    assert pitch.winning_camera == "camB"


def test_no_observations_is_an_error_not_an_empty_pitch() -> None:
    with pytest.raises(ValueError, match="no camera observations"):
        fuse_pitch({}, CFG)


def test_the_verdict_is_a_plain_record() -> None:
    pitch = fuse_pitch({"camA": seen(5, ball=True)}, CFG)
    assert isinstance(pitch, PitchVerdict)
    assert pitch.per_camera == (("camA", MinuteState.PLAYING, pitch.per_camera[0][2]),)
