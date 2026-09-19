"""The decision table (A36), one test per row, on counts written by hand.

`vision/rules.decide` is pure - a count, a motion cue and a config in, a verdict out - so
every row of the table registered in `thesis/preregistration.md` A36 is exercised here
without a detector, and the two monotonicity promises the amendment makes are pinned: a
person never lowers the PLAYING confidence, and a ball never lowers the class once five
people stand on the pitch.
"""

from __future__ import annotations

import json

import pytest

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision.counting import PitchCount
from pitch_occupancy.vision.rules import RULES_PATH, MinuteState, RuleConfig, decide

CFG = RuleConfig()  # the shipped defaults: motion, hi-vis and row 8 off


def count(n: int, *, ball: bool = False, raw: int | None = None, vehicles: int = 0,
          hi_vis: int = 0, spread: float | None = None, bounded: bool = True) -> PitchCount:
    return PitchCount(people_inside=n, people_total=n, raw_inside=raw if raw is not None else n,
                      ball_seen=ball, ball_confidence=0.3 if ball else 0.0,
                      vehicles_inside=vehicles, hi_vis_people=hi_vis, spread=spread,
                      bounded=bounded)


# --- the rows -----------------------------------------------------------------------------


def test_row_1_no_detector_is_uncertain_not_empty() -> None:
    verdict = decide(None, motion=None, cfg=CFG)
    assert verdict.state is MinuteState.UNCERTAIN and verdict.rule == 1
    assert verdict.class3 is None and verdict.confidence == 0.0


def test_row_2_no_boundary_abstains_on_the_deployed_path_and_counts_on_the_pages() -> None:
    whole = count(3, bounded=False)
    assert decide(whole, motion=None, cfg=CFG).rule == 2
    reported = decide(whole, motion=None, cfg=CFG, require_boundary=False)
    assert reported.state is MinuteState.PEOPLE_NOT_PLAYING
    assert any("whole frame" in step for step in reported.trace)


def test_row_3_nobody_inside_is_empty_and_the_height_filter_lowers_its_confidence() -> None:
    clean = decide(count(0), motion=None, cfg=CFG)
    assert clean.state is MinuteState.EMPTY and clean.rule == 3
    assert clean.confidence == 1.0 and clean.class3 is Class3.EMPTY
    dropped_two = decide(count(0, raw=2), motion=None, cfg=CFG)
    assert dropped_two.state is MinuteState.EMPTY
    assert dropped_two.confidence == pytest.approx(0.7)
    floor = decide(count(0, raw=9), motion=None, cfg=CFG)
    assert floor.confidence == pytest.approx(0.6), "clipped at the floor"


def test_row_4_motion_without_people_abstains_only_when_a_threshold_exists() -> None:
    off = decide(count(0), motion=5.0, cfg=CFG)
    assert off.state is MinuteState.EMPTY, "no motion_hi fitted: the cue is not consulted"
    fitted = RuleConfig(motion_hi=2.0)
    assert decide(count(0), motion=5.0, cfg=fitted).rule == 4
    assert decide(count(0), motion=5.0, cfg=fitted).state is MinuteState.UNCERTAIN
    assert decide(count(0), motion=1.0, cfg=fitted).state is MinuteState.EMPTY


def test_row_5_a_vehicle_or_hi_vis_in_a_small_group_is_maintenance_at_a_fixed_confidence():
    truck = decide(count(2, vehicles=1), motion=None, cfg=CFG)
    assert truck.state is MinuteState.MAINTENANCE and truck.rule == 5
    assert truck.confidence == 0.5
    assert truck.class3 is Class3.MAINTENANCE_NON_SPORTING
    bibs = decide(count(3, hi_vis=1), motion=None, cfg=CFG)
    assert bibs.state is MinuteState.MAINTENANCE
    referee_in_a_match = decide(count(9, hi_vis=1), motion=None, cfg=CFG)
    assert referee_in_a_match.state is MinuteState.PLAYING, "hi-vis needs a small group"


def test_row_6_four_or_fewer_is_not_playing_with_or_without_a_ball() -> None:
    for n in (1, 2, 3, 4):
        assert decide(count(n), motion=None, cfg=CFG).rule == 6
        with_ball = decide(count(n, ball=True), motion=None, cfg=CFG)
        assert with_ball.state is MinuteState.PEOPLE_NOT_PLAYING
    one = decide(count(1), motion=None, cfg=CFG)
    four = decide(count(4), motion=None, cfg=CFG)
    assert one.confidence == pytest.approx(0.9) and four.confidence == pytest.approx(0.6)
    assert decide(count(4, ball=True), motion=None, cfg=CFG).confidence == pytest.approx(0.4), (
        "a ball in a small group is a kickabout risk and lowers the confidence, not the class")


def test_row_7_five_or_more_and_a_ball_is_playing_at_the_highest_confidence() -> None:
    verdict = decide(count(5, ball=True), motion=None, cfg=CFG)
    assert verdict.state is MinuteState.PLAYING and verdict.rule == 7
    assert verdict.confidence == pytest.approx(0.8)
    assert decide(count(12, ball=True), motion=None, cfg=CFG).confidence == 1.0


def test_row_9_five_or_more_without_a_ball_is_still_playing_at_lower_confidence() -> None:
    verdict = decide(count(5), motion=None, cfg=CFG)
    assert verdict.state is MinuteState.PLAYING and verdict.rule == 9
    assert verdict.confidence == pytest.approx(0.6)
    assert decide(count(10), motion=None, cfg=CFG).confidence == pytest.approx(0.85)


def test_row_8_needs_every_cue_and_the_switch() -> None:
    """Still, clustered, no ball - and only when its cost has been measured and it was
    switched on. Any missing piece falls through to row 9."""
    still = dict(motion=0.1)
    clustered = count(6, spread=0.05)
    assert decide(clustered, cfg=CFG, **still).rule == 9, "switched off by default"
    on = RuleConfig(row8_enabled=True, motion_lo=0.5, cluster_max=0.1)
    assert decide(clustered, cfg=on, **still).rule == 8
    assert decide(clustered, cfg=on, **still).state is MinuteState.PEOPLE_NOT_PLAYING
    assert decide(clustered, cfg=on, motion=None).rule == 9, "no motion cue: not consulted"
    assert decide(count(6, spread=0.5), cfg=on, **still).rule == 9, "spread out: not clustered"
    assert decide(count(6, spread=0.05, ball=True), cfg=on, **still).rule == 7
    assert decide(clustered, cfg=RuleConfig(row8_enabled=True, motion_lo=0.5), **still).rule == 9


# --- the promises -------------------------------------------------------------------------


def test_another_person_never_lowers_the_playing_confidence() -> None:
    for ball in (False, True):
        previous = 0.0
        for n in range(5, 15):
            verdict = decide(count(n, ball=ball), motion=None, cfg=CFG)
            assert verdict.state is MinuteState.PLAYING
            assert verdict.confidence >= previous
            previous = verdict.confidence


def test_a_ball_never_lowers_the_class_at_five_or_more_and_never_raises_it_below() -> None:
    for n in range(5, 15):
        assert decide(count(n, ball=True), motion=None, cfg=CFG).state is MinuteState.PLAYING
        assert (decide(count(n, ball=True), motion=None, cfg=CFG).confidence
                >= decide(count(n), motion=None, cfg=CFG).confidence)
    for n in range(1, 5):
        small = decide(count(n, ball=True), motion=None, cfg=CFG)
        assert small.state is MinuteState.PEOPLE_NOT_PLAYING


def test_every_verdict_carries_a_readable_trace_and_its_row() -> None:
    verdict = decide(count(7, ball=True), motion=1.2, cfg=CFG)
    assert verdict.rule == 7 and verdict.people == 7 and verdict.ball is True
    assert verdict.motion == 1.2 and verdict.count is not None
    assert any("7 people inside" in step for step in verdict.trace)
    assert verdict.trace[-1].startswith("row 7:")


# --- the config ---------------------------------------------------------------------------


def test_the_requirements_are_the_facilitys_numbers() -> None:
    assert CFG.play_min == 5 and CFG.small_group_max == 4


def test_the_committed_rule_file_loads_and_is_unfrozen_until_wp9_t5(tmp_path) -> None:
    cfg = RuleConfig.load()
    assert cfg.play_min == 5 and cfg.small_group_max == 4
    assert cfg.detector in {"yolov8n", "yolo11n", "yolov8s", "yolo11s",
                            "yolov8n-seg", "yolo11n-seg", "yolo11s-seg"}
    raw = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    assert "_comment" in raw, "the file explains itself"
    if cfg.frozen:
        assert cfg.frozen_commit and cfg.tuned_on == "venue_01/camera_A"
    # unknown and comment keys are ignored, missing ones default
    other = tmp_path / "rules.json"
    other.write_text(json.dumps({"_note": "x", "play_min": 5, "future_key": 1, "motion_hi": 2.5}),
                     encoding="utf-8")
    loaded = RuleConfig.load(other)
    assert loaded.motion_hi == 2.5 and loaded.person_conf == 0.25 and not loaded.frozen


def test_the_minimum_height_line_is_a_function_of_the_foot_row() -> None:
    assert CFG.min_height_at(1080) is None
    cfg = RuleConfig(min_height_intercept=10.0, min_height_slope=40.0)
    at = cfg.min_height_at(1000)
    assert at is not None
    assert at(0) == pytest.approx(10.0) and at(500) == pytest.approx(30.0)
