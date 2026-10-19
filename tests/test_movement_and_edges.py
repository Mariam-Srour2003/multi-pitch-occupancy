"""Per-person movement, the touchline margin, the dark-pitch guard and the burst early exit."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pitch_occupancy.vision import movement
from pitch_occupancy.vision.counting import PitchCount, count_inside, persist
from pitch_occupancy.vision.detector import DETECTORS, PERSON, Detection, Detector
from pitch_occupancy.vision.pitch_classifier import DetectorFirstClassifier, brightness
from pitch_occupancy.vision.rules import MinuteState, RuleConfig, decide

CFG = RuleConfig.load()
SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
FRAME = np.full((100, 200, 3), 120, np.uint8)


def person(x: int, y: int = 60, h: int = 40) -> Detection:
    return Detection(PERSON, (x - 5, y - h, x + 5, y), 0.9)


def crowd(n: int, dx: int = 0) -> list[Detection]:
    return [person(20 + 25 * i + dx) for i in range(n)]


def test_people_who_step_further_than_part_of_their_height_moved() -> None:
    still = movement.assess([crowd(5), crowd(5)])
    moving = movement.assess([crowd(5), crowd(5, dx=10)])
    assert (still.matched, still.moved) == (5, 0)
    assert (moving.matched, moving.moved) == (5, 5)
    assert movement.assess([crowd(5)]) is None


def test_a_match_too_far_away_is_a_different_person_not_a_sprint() -> None:
    far = movement.assess([[person(10)], [person(190)]])
    assert far.matched == 0 and far.fraction is None


def test_persist_reports_movement_over_the_burst() -> None:
    counts = [count_inside(crowd(6, dx=d), SQUARE, (100, 200)) for d in (0, 0, 0)]
    persisted = persist(counts)
    assert persisted.people_matched == 12 and persisted.people_moved == 0


def standing(n: int, matched: int, moved: int) -> PitchCount:
    return PitchCount(people_inside=n, people_total=n, raw_inside=n, ball_seen=False,
                      people_matched=matched, people_moved=moved)


def test_a_crowd_standing_still_is_not_playing_once_the_clause_is_on() -> None:
    cfg = replace(CFG, still_max_fraction=0.2)
    assert decide(standing(8, 16, 1), motion=None, cfg=cfg).state is \
        MinuteState.MAINTENANCE_NON_SPORTING
    assert decide(standing(8, 16, 10), motion=None, cfg=cfg).state is MinuteState.ACTIVE_PLAY
    # Too few matches to judge: the clause does not fire on a guess.
    assert decide(standing(8, 3, 0), motion=None, cfg=cfg).state is MinuteState.ACTIVE_PLAY
    # Off, it never fires.
    assert decide(standing(8, 16, 0), motion=None,
                  cfg=replace(CFG, still_max_fraction=None)).state is MinuteState.ACTIVE_PLAY


#: A boundary inside the frame: its edges are touchlines. Where a boundary runs off the
#: frame, the pitch carries on past what the camera sees and nothing is trimmed there.
INNER = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def test_people_on_the_touchline_are_not_counted_as_on_the_pitch() -> None:
    edge = [person(100, y=89)] + [person(100, y=50)]
    whole = count_inside(edge, INNER, (100, 200))
    trimmed = count_inside(edge, INNER, (100, 200), edge_margin=0.05)
    assert count_inside(edge, SQUARE, (100, 200), edge_margin=0.05).people_inside == 2
    assert whole.people_inside == 2
    assert trimmed.people_inside == 1 and trimmed.edge_people == 1


def test_an_edge_margin_that_would_erase_the_boundary_is_not_applied() -> None:
    tiny = [[0.5, 0.5], [0.51, 0.5], [0.51, 0.51], [0.5, 0.51]]
    found = count_inside([person(101, y=51)], tiny, (100, 200), edge_margin=0.9)
    assert found.edge_people == 0


def test_a_dark_pitch_with_nobody_found_is_uncertain_not_empty() -> None:
    empty = PitchCount(people_inside=0, people_total=0, raw_inside=0, ball_seen=False)
    cfg = replace(CFG, dark_luma=20.0)
    assert decide(empty, motion=None, cfg=cfg, brightness=5.0).state is MinuteState.UNCERTAIN
    assert decide(empty, motion=None, cfg=cfg, brightness=90.0).state is MinuteState.EMPTY
    assert decide(empty, motion=None, cfg=cfg).state is MinuteState.EMPTY
    assert brightness(np.zeros((50, 80, 3), np.uint8), SQUARE) == 0.0


def scripted(*answers):
    det = Detector(DETECTORS["yolov8n"])
    queue = list(answers)
    calls: list[dict] = []

    def detect(image_bgr, **kwargs):
        calls.append(kwargs)
        return queue.pop(0)

    det.detect = detect  # type: ignore[method-assign]
    det.calls = calls  # type: ignore[attr-defined]
    return det


def test_a_burst_stops_detecting_once_two_frames_settle_an_empty_pitch() -> None:
    det = scripted([], [], crowd(9))
    obs = DetectorFirstClassifier(det, CFG).observe("cam", [FRAME] * 3, SQUARE)
    assert len(det.calls) == 2
    assert obs.verdict.state is MinuteState.EMPTY


def test_a_burst_that_could_still_be_a_game_reads_every_frame() -> None:
    det = scripted(crowd(6), crowd(6), crowd(6))
    DetectorFirstClassifier(det, CFG).observe("cam", [FRAME] * 3, SQUARE)
    assert len(det.calls) == 3
    det = scripted([], crowd(6), crowd(6))
    DetectorFirstClassifier(det, CFG).observe("cam", [FRAME] * 3, SQUARE)
    assert len(det.calls) == 3


def test_the_pitch_decision_keeps_the_dark_and_the_standing_clauses() -> None:
    from pitch_occupancy.slots.fusion import fuse_pitch

    cfg = replace(CFG, dark_luma=20.0, still_max_fraction=0.1)
    empty = PitchCount(people_inside=0, people_total=0, raw_inside=0, ball_seen=False)
    dark = decide(empty, motion=None, cfg=cfg, brightness=5.0)
    lit = decide(empty, motion=None, cfg=cfg, brightness=90.0)
    assert fuse_pitch({"a": dark}, cfg).state is MinuteState.UNCERTAIN
    assert fuse_pitch({"a": dark, "b": lit}, cfg).state is MinuteState.UNCERTAIN
    assert fuse_pitch({"a": lit}, cfg).state is MinuteState.EMPTY
    still = decide(standing(4, 8, 0), motion=None, cfg=cfg)
    still_b = decide(standing(4, 8, 0), motion=None, cfg=cfg)
    assert fuse_pitch({"a": still, "b": still_b}, cfg).state is \
        MinuteState.MAINTENANCE_NON_SPORTING
    moving = decide(standing(4, 8, 6), motion=None, cfg=cfg)
    assert fuse_pitch({"a": moving, "b": moving}, cfg).state is MinuteState.ACTIVE_PLAY
