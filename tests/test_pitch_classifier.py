"""The detector-first classifier on scripted detections (A36, WP9-T3).

No model runs: `Detector.detect` is replaced per test with a script, so what is pinned is the
arithmetic between the detector and the rule - persistence over the burst, the ball seen in
any frame, the frame the verdict rests on, the two motion cues kept apart, and the legacy
contract's honest failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.pipeline import Pipeline
from pitch_occupancy.vision.detector import DETECTORS, PERSON, SPORTS_BALL, Detection, Detector
from pitch_occupancy.vision.pitch_classifier import DetectorFirstClassifier
from pitch_occupancy.vision.rules import MinuteState, RuleConfig

SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
FRAME = np.zeros((100, 200, 3), np.uint8)


def people(n: int, *, ball: bool = False) -> list[Detection]:
    out = [Detection(PERSON, (10 + 15 * i, 10, 20 + 15 * i, 60), 0.9) for i in range(n)]
    if ball:
        out.append(Detection(SPORTS_BALL, (150, 40, 158, 48), 0.35))
    return out


def scripted(*per_frame):
    """A detector whose successive `detect` calls return the given lists (None = broken)."""
    det = Detector(DETECTORS["yolov8n"])
    answers = list(per_frame)
    calls: list[dict] = []

    def detect(image_bgr, **kwargs):
        calls.append(kwargs)
        return answers.pop(0) if answers else []

    det.detect = detect  # type: ignore[method-assign]
    det.calls = calls  # type: ignore[attr-defined]
    return det


def test_a_burst_is_counted_frame_by_frame_and_the_median_persists() -> None:
    clf = DetectorFirstClassifier(scripted(people(6), people(5), people(6, ball=True)),
                                  RuleConfig())
    obs = clf.observe("camA", [FRAME, FRAME, FRAME], SQUARE)
    assert obs.verdict.state is MinuteState.ACTIVE_PLAY and obs.verdict.rule == 6
    assert obs.verdict.people == 6 and obs.verdict.ball is True
    assert obs.frames == 3 and obs.evidence_index == 0, "the first frame with the median count"
    assert obs.motion_burst is not None and obs.motion_minute is None
    assert obs.verdict.polygon == SQUARE
    assert len(obs.instances) == 6 + 0, "the counted people of the evidence frame; no ball there"


def test_a_shadow_in_one_frame_of_three_does_not_survive() -> None:
    clf = DetectorFirstClassifier(scripted(people(1), [], []), RuleConfig())
    obs = clf.observe("camA", [FRAME, FRAME, FRAME], SQUARE)
    assert obs.verdict.state is MinuteState.EMPTY
    assert obs.verdict.confidence == pytest.approx(0.85), "raw_inside 1 lowers the confidence"


def test_a_broken_detector_is_uncertain_and_never_a_class() -> None:
    clf = DetectorFirstClassifier(scripted(None), RuleConfig())
    obs = clf.observe("camA", [FRAME], SQUARE)
    assert obs.verdict.state is MinuteState.UNCERTAIN and obs.verdict.rule == 1
    with pytest.raises(RuntimeError, match="detector unavailable"):
        DetectorFirstClassifier(scripted(None), RuleConfig())(FRAME, polygon=SQUARE)


def test_the_legacy_call_counts_the_whole_frame_without_a_boundary_and_says_so() -> None:
    clf = DetectorFirstClassifier(scripted(people(3)), RuleConfig())
    assert clf(FRAME) == (Class3.MAINTENANCE_NON_SPORTING, pytest.approx(0.7))
    clf = DetectorFirstClassifier(scripted(people(7, ball=True)), RuleConfig())
    assert clf(FRAME, polygon=SQUARE) == (Class3.ACTIVE_PLAY, pytest.approx(0.9))


def test_the_rule_passes_its_detector_settings_through() -> None:
    det = scripted(people(0))
    clf = DetectorFirstClassifier(det, RuleConfig(imgsz=960, tiles=2, person_conf=0.3,
                                                  ball_conf=0.05))
    clf.observe("camA", [FRAME], SQUARE)
    assert det.calls[0]["imgsz"] == 960 and det.calls[0]["tiles"] == 2
    assert det.calls[0]["confidence"] == pytest.approx(0.05), "the lower of the two floors"


def test_the_minute_motion_cue_is_recorded_beside_the_burst_cue_not_instead() -> None:
    clf = DetectorFirstClassifier(scripted([], []), RuleConfig())
    bright = np.full((100, 200, 3), 200, np.uint8)
    obs = clf.observe("camA", [FRAME, FRAME], SQUARE, previous=bright)
    assert obs.motion_burst == pytest.approx(0.0)
    assert obs.motion_minute is not None and obs.motion_minute > 0
    assert obs.verdict.motion == pytest.approx(0.0), "the rule reads the burst cue"


# --- through the seam ---------------------------------------------------------------------


def pipeline_with(det: Detector, **cfg) -> Pipeline:
    rules = RuleConfig(**cfg)
    clf = DetectorFirstClassifier(det, rules)
    return Pipeline("yolov8n", "detector", clf, rules=rules, classifier=clf,
                    boundaries=lambda cam, **_: (SQUARE, "stored"))


def test_the_pipeline_observes_fuses_and_reports_the_boundary() -> None:
    pipe = pipeline_with(scripted(people(3, ball=True), people(3)))
    assert pipe.burst == (3, 1.0)
    a = pipe.observe_minute("camA", [FRAME])
    b = pipe.observe_minute("camB", [FRAME])
    assert a.verdict.state is MinuteState.MAINTENANCE_NON_SPORTING, "three is three, ball or not"
    assert a.verdict.boundary_key == "stored" and a.verdict.trace[0].startswith("boundary stored")
    pitch = pipe.fuse({"camA": a, "camB": b})
    assert pitch.state is MinuteState.ACTIVE_PLAY and pitch.people_inside == 6
    assert "detector yolov8n" in pipe.describe() and "UNFROZEN" in pipe.describe()


def test_the_pipeline_abstains_without_a_boundary_on_the_deployed_path() -> None:
    det = scripted(people(9, ball=True))
    pipe = pipeline_with(det)
    pipe.boundaries = lambda cam, **_: (None, None)
    obs = pipe.observe_minute("camZ", [FRAME])
    assert obs.verdict.state is MinuteState.UNCERTAIN and obs.verdict.rule == 2
    assert det.calls == [], "no boundary, no detector run"
    pipe.require_boundary = False
    verdict = pipe.classify_frame(FRAME, camera_id="camZ")
    assert verdict.state is MinuteState.ACTIVE_PLAY
    assert any("whole frame" in step for step in verdict.trace)
