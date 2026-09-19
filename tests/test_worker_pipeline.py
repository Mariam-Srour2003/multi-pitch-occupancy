"""`worker.run_slot` given an assembled pipeline (A36, WP9-T1).

The pipeline supplies the classifier, both gates and the per-camera boundary lookup in one
object, so a caller cannot pass one and forget the others - which is what `scheduler.run_due`
did for as long as the gates existed. With a pipeline the boundary is mandatory: a camera
without one contributes no observation, and is recorded as UNCERTAIN rather than dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.frame_source import Frame, FrameSource
from pitch_occupancy.pipeline import Pipeline
from pitch_occupancy.worker import run_slot

PLAY = Class3.ACTIVE_PLAY
TRIANGLE = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
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
        return Frame(camera_id, minute_index, np.zeros((4, 4, 3), np.uint8), "x.mp4")


def test_a_pipeline_supplies_the_classifier_and_the_boundary_and_abstains_without_one():
    """One camera has a boundary and one does not. The first is scored inside it; the second
    is recorded as UNCERTAIN every minute and never reaches the classifier - and because the
    first camera observed every minute, the pitch still gets its verdict."""
    seen: list[object] = []

    def classify(image, *, polygon=None):
        seen.append(polygon)
        return PLAY, 0.9

    pipeline = Pipeline(
        "stub", "probe", classify,
        boundaries=lambda cam, *, venue=None, slot_key=None: (
            (TRIANGLE, "k") if cam == "camA" else (None, None)),
    )
    source = FakeSource({"camA": [object()] * 3, "camB": [object()] * 3})
    run = run_slot("s", source, None, pipeline=pipeline)

    assert seen == [TRIANGLE] * 3, "camB must never reach the classifier"
    assert run.uncertain_cameras == ("camB",)
    assert run.minutes_uncertain == 0, "camA observed every minute, so none was lost"
    assert run.minutes_captured == 3 and run.minutes_missed == 0
    camb = [s for s in run.samples if s.camera_id == "camB"]
    assert len(camb) == 3
    assert all(s.predicted == "UNCERTAIN" and s.confidence == 0.0 for s in camb)
    assert run.verdict.status is SlotStatus.USED


def test_when_no_camera_has_a_boundary_every_minute_is_uncertain_and_missed():
    pipeline = Pipeline("stub", "probe", lambda image, *, polygon=None: (PLAY, 0.9),
                        boundaries=lambda cam, **_: (None, None))
    run = run_slot("s", FakeSource({"camA": [object()] * 4}), None, pipeline=pipeline)
    assert run.minutes_uncertain == 4 and run.minutes_missed == 4
    assert run.minutes_captured == 0 and run.capture_rate == 0.0
    assert run.verdict.status is SlotStatus.REVIEW
    assert "captured" in run.verdict.reason


def test_the_interactive_setting_scores_the_whole_frame_instead():
    """`require_boundary=False` is what the pages use; here it means the old behaviour."""
    seen: list[object] = []

    def classify(image, *, polygon=None):
        seen.append(polygon)
        return PLAY, 0.9

    pipeline = Pipeline("stub", "probe", classify, require_boundary=False,
                        boundaries=lambda cam, **_: (None, None))
    run = run_slot("s", FakeSource({"camA": [object()] * 2}), None, pipeline=pipeline)
    assert seen == [None, None]
    assert run.minutes_uncertain == 0 and run.verdict.status is SlotStatus.USED


def test_explicit_gates_and_lookups_win_over_the_pipelines():
    """The older keyword arguments still work and still win, so a test can hand this a
    scripted gate while the pipeline carries the real one."""

    class Recording:
        def __init__(self) -> None:
            self.calls = 0

        def apply(self, state, previous, frame, polygon):
            self.calls += 1
            return state, None

    class Never:
        def apply(self, *args):
            raise AssertionError("the pipeline's gate must not run when one is given")

    explicit = Recording()
    pipeline = Pipeline("stub", "probe", lambda image, *, polygon=None: (PLAY, 0.9),
                        motion_gate=Never(), boundaries=lambda cam, **_: (None, None))
    run = run_slot("s", FakeSource({"camA": [object()] * 2}),
                   lambda image, *, polygon=None: (PLAY, 0.9),
                   pipeline=pipeline, motion_gate=explicit, polygon_for=lambda c: SQUARE)
    assert explicit.calls == 2
    assert run.minutes_uncertain == 0, "the explicit lookup supplied a boundary"


def test_the_slot_context_reaches_the_boundary_lookup():
    seen: list[tuple] = []

    def lookup(cam, *, venue=None, slot_key=None):
        seen.append((cam, venue, slot_key))
        return SQUARE, "k"

    pipeline = Pipeline("stub", "probe", lambda image, *, polygon=None: (PLAY, 0.9),
                        boundaries=lookup)
    run_slot("s", FakeSource({"file0": [object()]}), None, pipeline=pipeline,
             venue="venue_01", slot_key="slot_20260711_1000")
    assert seen == [("file0", "venue_01", "slot_20260711_1000")]


def test_run_slot_needs_a_classifier_or_a_pipeline():
    with pytest.raises(TypeError, match="classifier"):
        run_slot("s", FakeSource({"camA": [object()]}), None)
