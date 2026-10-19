"""`worker.run_slot` given an assembled pipeline.

The pipeline supplies the detector-first classifier, its rule and the per-camera boundary
lookup in one object, so a caller cannot pass one and forget the others. With a pipeline the
boundary is mandatory: a camera without one never reaches the detector, and is recorded as
UNCERTAIN rather than dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.frame_source import Frame, FrameSource
from pitch_occupancy.pipeline import Pipeline
from pitch_occupancy.vision.detector import DETECTORS, PERSON, SPORTS_BALL, Detection, Detector
from pitch_occupancy.vision.pitch_classifier import DetectorFirstClassifier
from pitch_occupancy.vision.rules import RuleConfig
from pitch_occupancy.worker import run_slot

PLAY = Class3.ACTIVE_PLAY
SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


class FakeSource(FrameSource):
    """Serves scripted frames; ``None`` entries simulate a camera outage."""

    def __init__(self, script: dict[str, list[object]]) -> None:
        self._script = script

    def cameras(self) -> list[str]:
        return sorted(self._script)

    @property
    def n_minutes(self) -> int:
        return max(len(v) for v in self._script.values())

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        seq = self._script[camera_id]
        if minute_index >= len(seq) or seq[minute_index] is None:
            return None
        return Frame(camera_id, minute_index, np.zeros((40, 80, 3), np.uint8), "x.mp4")


def pipeline(boundaries, *, require_boundary: bool = True, people: int = 6) -> Pipeline:
    """A pipeline whose detector finds ``people`` and a ball, and records each polygon-free
    call so a test can see which cameras reached it."""
    det = Detector(DETECTORS["yolov8n"])
    calls: list[int] = []

    def detect(image_bgr, **_kwargs):
        calls.append(1)
        out = [Detection(PERSON, (2 + 8 * i, 5, 8 + 8 * i, 30), 0.9) for i in range(people)]
        return out + [Detection(SPORTS_BALL, (60, 20, 64, 24), 0.4)]

    det.detect = detect  # type: ignore[method-assign]
    rules = RuleConfig(burst_frames=1)
    clf = DetectorFirstClassifier(det, rules)
    built = Pipeline("yolov8n", clf, rules=rules, classifier=clf, boundaries=boundaries,
                     require_boundary=require_boundary)
    built.calls = calls  # type: ignore[attr-defined]
    return built


def test_a_pipeline_supplies_the_classifier_and_the_boundary_and_abstains_without_one():
    """One camera has a boundary and one does not. The first is counted inside it; the
    second is recorded as UNCERTAIN every minute and never reaches the detector - and because
    the first camera observed every minute, the pitch still gets its verdict."""
    built = pipeline(lambda cam, *, venue=None, slot_key=None: (
        (SQUARE, "k") if cam == "camA" else (None, None)))
    source = FakeSource({"camA": [object()] * 3, "camB": [object()] * 3})
    run = run_slot("s", source, None, pipeline=built)

    assert len(built.calls) == 3, "camB must never reach the detector"
    assert run.uncertain_cameras == ("camB",)
    assert run.minutes_uncertain == 0, "camA observed every minute, so none was lost"
    assert run.minutes_captured == 3 and run.minutes_missed == 0
    camb = [s for s in run.samples if s.camera_id == "camB"]
    assert len(camb) == 3
    assert all(s.predicted == "UNCERTAIN" and s.confidence == 0.0 for s in camb)
    assert run.verdict.status is SlotStatus.USED


def test_when_no_camera_has_a_boundary_every_minute_is_uncertain_and_missed():
    built = pipeline(lambda cam, **_: (None, None))
    run = run_slot("s", FakeSource({"camA": [object()] * 4}), None, pipeline=built)
    assert built.calls == []
    assert run.minutes_uncertain == 4 and run.minutes_missed == 4
    assert run.minutes_captured == 0 and run.capture_rate == 0.0
    assert run.verdict.status is SlotStatus.REVIEW
    assert "captured" in run.verdict.reason


def test_the_interactive_setting_counts_the_whole_frame_instead():
    """`require_boundary=False` is what the pages use."""
    built = pipeline(lambda cam, **_: (None, None), require_boundary=False)
    run = run_slot("s", FakeSource({"camA": [object()] * 2}), None, pipeline=built)
    assert len(built.calls) == 2
    assert run.minutes_uncertain == 0 and run.verdict.status is SlotStatus.USED


def test_an_explicit_lookup_wins_over_the_pipelines():
    built = pipeline(lambda cam, **_: (None, None))
    run = run_slot("s", FakeSource({"camA": [object()] * 2}), None,
                   pipeline=built, polygon_for=lambda c: SQUARE)
    assert run.minutes_uncertain == 0, "the explicit lookup supplied a boundary"
    assert len(built.calls) == 2


def test_the_slot_context_reaches_the_boundary_lookup():
    seen: list[tuple] = []

    def lookup(cam, *, venue=None, slot_key=None):
        seen.append((cam, venue, slot_key))
        return SQUARE, "k"

    run_slot("s", FakeSource({"file0": [object()]}), None, pipeline=pipeline(lookup),
             venue="venue_01", slot_key="slot_20260711_1000")
    assert seen == [("file0", "venue_01", "slot_20260711_1000")]


def test_a_bare_classifier_still_runs_without_a_pipeline():
    """The seam a scripted test uses: one frame per camera-minute, classes fused."""
    seen: list[object] = []

    def classify(image, *, polygon=None):
        seen.append(polygon)
        return PLAY, 0.9

    run = run_slot("s", FakeSource({"camA": [object()] * 2}), classify,
                   polygon_for=lambda c: SQUARE)
    assert seen == [SQUARE, SQUARE]
    assert run.verdict.status is SlotStatus.USED


def test_run_slot_needs_a_classifier_or_a_pipeline():
    with pytest.raises(TypeError, match="classifier"):
        run_slot("s", FakeSource({"camA": [object()]}), None)
