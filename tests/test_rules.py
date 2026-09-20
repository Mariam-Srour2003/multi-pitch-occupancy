"""The decision table (A36, amended A40), one test per row, on counts written by hand.

`vision/rules.decide` is pure - a count, a motion cue and a config in, a verdict out - so
every row is exercised here without a detector.

**The rule answers three classes and an abstention.** A40 dropped the four-class split from
the prediction path: the corpus holds 6 real `3_people_not_playing` frames and 0 real
`4_maintenance` ones, so nothing could measure the distinction, and a branch the data cannot
evaluate should not be in the deployed path.

**And play must now be shown, not assumed.** More than four people *and* a ball *and*
movement. A36 had it the other way round - play was the default above the head count, on the
reasoning that A17's 0.40 cross-venue ball recall makes absence of a ball weak evidence. The
facility's rule is that a game has a ball in it, so the burden moved, and what that costs is
measured in `results/rule_frame_eval.csv` rather than argued about here.
"""

from __future__ import annotations

import json

import pytest

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision.counting import PitchCount
from pitch_occupancy.vision.rules import (
    RULES_PATH, MinuteState, RuleConfig, decide, from_class3, from_label, to_class3)

EMPTY, PLAY, C3 = (MinuteState.EMPTY, MinuteState.ACTIVE_PLAY,
                   MinuteState.MAINTENANCE_NON_SPORTING)

#: The shipped rule, so these tests describe what is deployed rather than a fixture.
CFG = RuleConfig.load()

#: Motion required *and* checkable - `motion_play_min` is null until WP9-T5, so a test about
#: the motion clause has to fit one itself and say that is what it is doing.
MOVING = RuleConfig(**{**CFG.to_json(), "motion_play_min": 1.0})


def count(n: int, *, ball: bool = False, raw: int | None = None, vehicles: int = 0,
          hi_vis: int = 0, spread: float | None = None, bounded: bool = True) -> PitchCount:
    return PitchCount(people_inside=n, people_total=n, raw_inside=raw if raw is not None else n,
                      ball_seen=ball, ball_confidence=0.3 if ball else 0.0,
                      vehicles_inside=vehicles, hi_vis_people=hi_vis, spread=spread,
                      bounded=bounded)


# --- three classes, not four ----------------------------------------------------------------


def test_the_rule_answers_the_three_reporting_classes_and_an_abstention() -> None:
    assert {s.value for s in MinuteState} == {
        "C1_EMPTY", "C2_ACTIVE_PLAY", "C3_MAINTENANCE_NON_SPORTING", "UNCERTAIN"}
    # A state's value *is* its reporting class, so nothing can map between the two wrongly.
    for state in (EMPTY, PLAY, C3):
        assert to_class3(state) is Class3(state.value)
        assert from_class3(Class3(state.value)) is state
    assert to_class3(MinuteState.UNCERTAIN) is None
    assert MinuteState.UNCERTAIN.decided is False
    assert EMPTY.decided is True


def test_the_labelling_folders_map_in_and_the_two_retired_ones_still_read() -> None:
    assert from_label("3_maintenance_non_sporting") is C3
    # retired on 2026-09-21 when the folders collapsed; every CSV older than that says one
    # of these, and reading an old artefact must not need editing it
    assert from_label("3_people_not_playing") is C3
    assert from_label("4_maintenance") is C3
    assert from_label("1_empty") is EMPTY and from_label("2_playing") is PLAY


# --- the rows -------------------------------------------------------------------------------


def test_row_1_no_detector_is_uncertain_not_empty() -> None:
    verdict = decide(None, motion=None, cfg=CFG)
    assert verdict.state is MinuteState.UNCERTAIN and verdict.rule == 1
    assert verdict.class3 is None and verdict.confidence == 0.0


def test_row_2_no_boundary_abstains_on_the_deployed_path_and_counts_on_the_pages() -> None:
    whole = count(3, bounded=False)
    assert decide(whole, motion=None, cfg=CFG).rule == 2
    reported = decide(whole, motion=None, cfg=CFG, require_boundary=False)
    assert reported.state is C3
    assert any("whole frame" in step for step in reported.trace)


def test_row_3_nobody_and_nothing_moving_is_empty() -> None:
    clean = decide(count(0), motion=0.1, cfg=CFG)
    assert clean.state is EMPTY and clean.rule == 3 and clean.confidence == 1.0
    assert clean.class3 is Class3.EMPTY
    # what the height filter dropped lowers the confidence without changing the class
    assert decide(count(0, raw=2), motion=0.1, cfg=CFG).confidence == pytest.approx(0.7)
    assert decide(count(0, raw=9), motion=0.1, cfg=CFG).confidence == pytest.approx(0.6)


def test_row_4_motion_with_nobody_found_abstains_only_once_a_threshold_exists() -> None:
    assert decide(count(0), motion=5.0, cfg=CFG).state is EMPTY, "no motion_hi fitted"
    fitted = RuleConfig(motion_hi=2.0)
    assert decide(count(0), motion=5.0, cfg=fitted).rule == 4
    assert decide(count(0), motion=5.0, cfg=fitted).state is MinuteState.UNCERTAIN
    assert decide(count(0), motion=1.0, cfg=fitted).state is EMPTY


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_row_5_four_or_fewer_is_c3_with_or_without_a_ball(n: int) -> None:
    assert decide(count(n), motion=5.0, cfg=CFG).rule == 5
    assert decide(count(n), motion=5.0, cfg=CFG).state is C3
    assert decide(count(n, ball=True), motion=5.0, cfg=CFG).state is C3, (
        "a ball does not rescue a small group - the facility's rule, §2.7")


def test_row_5_says_when_it_saw_a_vehicle_without_promising_to_tell_them_apart() -> None:
    """The four-class split is gone, so a vehicle no longer produces its own class. It is
    still worth recording, because it is the one cue for maintenance the corpus has - and
    the trace is where an unevaluable signal belongs."""
    truck = decide(count(2, vehicles=1), motion=5.0, cfg=CFG)
    assert truck.state is C3 and truck.rule == 5
    assert "vehicle" in truck.trace[-1]
    assert truck.confidence >= 0.6


def test_row_6_more_than_four_with_a_ball_and_motion_is_play() -> None:
    verdict = decide(count(5, ball=True), motion=5.0, cfg=MOVING)
    assert verdict.state is PLAY and verdict.rule == 6
    assert verdict.confidence == pytest.approx(0.8)
    assert decide(count(12, ball=True), motion=5.0, cfg=MOVING).confidence == 1.0
    assert any("with a ball" in s for s in verdict.trace)


def test_row_7_a_crowd_with_no_ball_is_not_playing() -> None:
    """The inversion. A36 called this ACTIVE_PLAY at lower confidence; the facility's rule
    is that a game has a ball in it."""
    verdict = decide(count(9), motion=5.0, cfg=MOVING)
    assert verdict.state is C3 and verdict.rule == 7
    assert "no ball seen" in verdict.trace[-1]


def test_row_7_a_crowd_that_is_not_moving_is_not_playing() -> None:
    verdict = decide(count(9, ball=True), motion=0.2, cfg=MOVING)
    assert verdict.state is C3 and verdict.rule == 7
    assert "motion 0.200" in verdict.trace[-1]


def test_row_7_names_every_clause_that_failed_not_just_the_first() -> None:
    verdict = decide(count(9), motion=0.2, cfg=MOVING)
    assert verdict.state is C3
    assert "no ball" in verdict.trace[-1] and "motion" in verdict.trace[-1]


def test_a_clustered_crowd_is_not_playing_when_a_threshold_says_what_clustered_means():
    tight = RuleConfig(**{**MOVING.to_json(), "cluster_max": 0.1})
    assert decide(count(9, ball=True, spread=0.05), motion=5.0, cfg=tight).state is C3
    assert decide(count(9, ball=True, spread=0.5), motion=5.0, cfg=tight).state is PLAY
    # without a fitted threshold the cue is not consulted, rather than guessed
    assert decide(count(9, ball=True, spread=0.05), motion=5.0, cfg=MOVING).state is PLAY


# --- a required cue that cannot be checked is reported, never assumed ------------------------


def test_an_unfitted_motion_threshold_is_announced_rather_than_passing_silently() -> None:
    """`require_motion` is on in the shipped config and `motion_play_min` is null until
    WP9-T5. A requirement that silently never fires is the shape of every guard this project
    has caught not guarding, so the verdict says which clause was skipped."""
    assert CFG.require_motion and CFG.motion_play_min is None
    verdict = decide(count(9, ball=True), motion=0.001, cfg=CFG)
    assert verdict.state is PLAY, "the clause cannot be checked, so it does not block"
    assert any("unfitted" in step for step in verdict.trace)


def test_motion_that_was_never_measured_is_announced_too() -> None:
    verdict = decide(count(9, ball=True), motion=None, cfg=MOVING)
    assert verdict.state is PLAY
    assert any("not measured" in step for step in verdict.trace)


def test_the_requirements_can_be_switched_off_and_the_cost_is_a_decision() -> None:
    """A17 measured cross-venue ball recall at 0.40, so `require_ball` loses genuine matches
    wherever the detector cannot see the ball. The switch exists so that is somebody's call."""
    lenient = RuleConfig(**{**CFG.to_json(), "require_ball": False})
    assert decide(count(9), motion=5.0, cfg=CFG).state is C3
    assert decide(count(9), motion=5.0, cfg=lenient).state is PLAY


# --- the promises ---------------------------------------------------------------------------


def test_another_person_never_lowers_the_playing_confidence() -> None:
    previous = 0.0
    for n in range(5, 15):
        verdict = decide(count(n, ball=True), motion=5.0, cfg=MOVING)
        assert verdict.state is PLAY
        assert verdict.confidence >= previous
        previous = verdict.confidence


def test_a_ball_never_lowers_the_class_and_never_raises_it_below_the_head_count() -> None:
    for n in range(5, 15):
        with_ball = decide(count(n, ball=True), motion=5.0, cfg=MOVING)
        without = decide(count(n), motion=5.0, cfg=MOVING)
        assert with_ball.state is PLAY and without.state is C3
    for n in range(1, 5):
        assert decide(count(n, ball=True), motion=5.0, cfg=MOVING).state is C3


def test_every_verdict_carries_a_readable_trace_and_its_row() -> None:
    verdict = decide(count(7, ball=True), motion=1.2, cfg=MOVING)
    assert verdict.rule == 6 and verdict.people == 7 and verdict.ball is True
    assert verdict.motion == 1.2 and verdict.count is not None
    assert any("7 people inside" in step for step in verdict.trace)
    assert verdict.trace[-1].startswith("row 6:")


# --- the config -------------------------------------------------------------------------------


def test_the_requirements_are_the_facilitys_numbers_and_switches() -> None:
    assert CFG.play_min == 5 and CFG.small_group_max == 4
    assert CFG.require_ball is True and CFG.require_motion is True


def test_the_committed_rule_file_loads_and_is_unfrozen_until_wp9_t5(tmp_path) -> None:
    raw = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    assert "_comment" in raw, "the file explains itself"
    assert "row8_enabled" not in raw, "A40 removed it with the four-class split"
    if CFG.frozen:
        assert CFG.frozen_commit and CFG.tuned_on == "venue_01/camera_A"
    other = tmp_path / "rules.json"
    other.write_text(json.dumps({"_note": "x", "future_key": 1, "motion_play_min": 2.5}),
                     encoding="utf-8")
    loaded = RuleConfig.load(other)
    assert loaded.motion_play_min == 2.5 and loaded.person_conf == 0.25 and not loaded.frozen


def test_the_minimum_height_line_is_a_function_of_the_foot_row() -> None:
    assert CFG.min_height_at(1080) is None
    cfg = RuleConfig(min_height_intercept=10.0, min_height_slope=40.0)
    at = cfg.min_height_at(1000)
    assert at is not None
    assert at(0) == pytest.approx(10.0) and at(500) == pytest.approx(30.0)
