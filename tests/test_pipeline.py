"""The one seam every surface classifies through (A36, WP9-T1).

A35 found the review page running a different system from the worker; the scheduler turned
out to be running a third. These pin that `pipeline.assemble` is the single place inference
is put together, that the boundary is resolved the way production names cameras, and that
a missing boundary is an abstention on the deployed path and a labelled whole-frame answer
on the interactive one.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3, Class4
from pitch_occupancy.pipeline import Pipeline, assemble, default_gates, reset_shared, shared
from pitch_occupancy.vision.rules import (
    FrameVerdict, MinuteState, from_class3, from_class4, to_class3)

MIDDLE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
FRAME = np.zeros((8, 8, 3), np.uint8)


def scripted(state: Class3 = Class3.ACTIVE_PLAY, confidence: float = 0.9):
    """A classifier that records the boundary it was handed."""
    calls: list[object] = []

    def classify(image, *, polygon=None):
        calls.append(polygon)
        return state, confidence

    classify.calls = calls  # type: ignore[attr-defined]
    return classify


def found(polygon, key: str = "stored"):
    """A boundary lookup that finds ``polygon`` for every camera, or nothing."""

    def lookup(camera, *, venue=None, slot_key=None):
        return (polygon, key) if polygon is not None else (None, None)

    return lookup


class StubProbe:
    backbone = "dinov2"
    n_train = 7

    def __call__(self, image, *, polygon=None):
        return Class3.EMPTY, 0.8


# --- the vocabulary -----------------------------------------------------------------------


def test_minute_states_are_the_reporting_classes_plus_an_abstention() -> None:
    """`Sample.predicted` must read the same whether it came from a prediction or from a
    report, so the three decided values *are* `Class3`'s values (A40). They were `Class4`'s
    until 2026-09-20; the corpus has 0 real maintenance frames, so the fourth branch could
    never be measured and left the prediction path."""
    assert {s.value for s in MinuteState} - {"UNCERTAIN"} == {c.value for c in Class3}
    assert not MinuteState.UNCERTAIN.decided
    assert MinuteState.EMPTY.decided


def test_both_c3_labelling_folders_still_map_in_and_the_abstention_maps_out() -> None:
    assert from_class3(Class3.MAINTENANCE_NON_SPORTING) is MinuteState.MAINTENANCE_NON_SPORTING
    assert from_class3(Class3.ACTIVE_PLAY) is MinuteState.ACTIVE_PLAY
    assert from_class4(Class4.MAINTENANCE) is MinuteState.MAINTENANCE_NON_SPORTING
    assert from_class4(Class4.PEOPLE_NOT_PLAYING) is MinuteState.MAINTENANCE_NON_SPORTING
    assert to_class3(MinuteState.MAINTENANCE_NON_SPORTING) is Class3.MAINTENANCE_NON_SPORTING
    assert to_class3(MinuteState.UNCERTAIN) is None


def test_an_uncertain_verdict_has_no_two_value_form() -> None:
    """The `Classifier` contract cannot say "declined", and inventing a class would be
    the one thing an abstention exists not to do."""
    verdict = FrameVerdict(MinuteState.UNCERTAIN, 0.0)
    assert verdict.class3 is None and not verdict.decided
    with pytest.raises(ValueError, match="UNCERTAIN"):
        verdict.as_pair()
    assert FrameVerdict(MinuteState.ACTIVE_PLAY, 0.7).as_pair() == (Class3.ACTIVE_PLAY, 0.7)


# --- assemble -----------------------------------------------------------------------------


def test_assemble_dispatches_a_backbone_key_to_the_probe_path(monkeypatch) -> None:
    import pitch_occupancy.vision.classifier as classifier_module

    monkeypatch.setattr(classifier_module, "load_classifier", lambda key: StubProbe())
    pipeline = assemble("dinov2")
    assert pipeline.kind == "probe" and pipeline.model_key == "dinov2"
    assert pipeline.n_train == 7
    assert pipeline.motion_gate is not None and pipeline.person_gate is not None
    assert pipeline.require_boundary, "the deployed path requires a boundary by default"
    assert "probe dinov2" in pipeline.describe() and "7 development frames" in pipeline.describe()


def test_assemble_can_leave_the_gates_off(monkeypatch) -> None:
    import pitch_occupancy.vision.classifier as classifier_module

    monkeypatch.setattr(classifier_module, "load_classifier", lambda key: StubProbe())
    pipeline = assemble("dinov2", gates=False)
    assert pipeline.gates() == (None, None)
    assert "gates: none" in pipeline.describe()


def test_assemble_refuses_an_unknown_key() -> None:
    """A deployment that fell back to some other model would report verdicts no table
    describes."""
    with pytest.raises(KeyError, match="unknown model key"):
        assemble("not-a-model")


def test_the_shared_pipeline_is_built_once_and_reports_rather_than_requires(monkeypatch):
    """The API surfaces share one; a page a person is reading labels a whole-frame answer
    rather than refusing to give one."""
    import pitch_occupancy.vision.classifier as classifier_module

    built: list[str] = []

    def load(key):
        built.append(key)
        return StubProbe()

    monkeypatch.setattr(classifier_module, "load_classifier", load)
    reset_shared()
    try:
        first, second = shared(), shared()
        assert first is second
        assert built == [first.model_key]
        assert not first.require_boundary
    finally:
        reset_shared()


def test_default_gates_are_the_deployed_ones() -> None:
    from pitch_occupancy.vision.motion import MotionGate
    from pitch_occupancy.vision.people import PersonGate

    motion, person = default_gates()
    assert isinstance(motion, MotionGate) and isinstance(person, PersonGate)


# --- the boundary -------------------------------------------------------------------------


def test_polygon_for_binds_the_slot_context_to_every_lookup() -> None:
    """`worker.run_slot` takes a one-argument lookup; the venue and recording key it needs
    to resolve `file0` are bound in here, once, rather than remembered at the call site."""
    seen: list[tuple] = []

    def lookup(camera, *, venue=None, slot_key=None):
        seen.append((camera, venue, slot_key))
        return MIDDLE, "k"

    pipeline = Pipeline("m", "probe", scripted(), boundaries=lookup)
    polygon_for = pipeline.polygon_for(venue="venue_01", slot_key="slot_20260711_1000")
    assert polygon_for("file0") == MIDDLE
    assert seen == [("file0", "venue_01", "slot_20260711_1000")]


# --- one frame ----------------------------------------------------------------------------


def test_classify_frame_passes_the_resolved_boundary_and_records_which_it_was() -> None:
    classify = scripted()
    pipeline = Pipeline("m", "probe", classify, boundaries=found(MIDDLE, "slot_x_camA"))
    verdict = pipeline.classify_frame(FRAME, camera_id="file0")
    assert verdict.state is MinuteState.ACTIVE_PLAY
    assert verdict.probed is Class3.ACTIVE_PLAY and not verdict.gated
    assert classify.calls == [MIDDLE]
    assert verdict.boundary_key == "slot_x_camA" and verdict.polygon == MIDDLE
    assert any("keeps 25%" in step for step in verdict.trace), verdict.trace
    assert verdict.model_key == "m"


def test_a_given_polygon_wins_over_the_lookup() -> None:
    def never(camera, **_):
        raise AssertionError("the lookup must not run when a polygon is given")

    classify = scripted()
    pipeline = Pipeline("m", "probe", classify, boundaries=never)
    verdict = pipeline.classify_frame(FRAME, camera_id="cam", polygon=MIDDLE)
    assert classify.calls == [MIDDLE] and verdict.boundary_key == "given"


def test_without_a_boundary_the_deployed_path_abstains_and_never_runs_the_model() -> None:
    """Scoring the neighbouring pitch and the car park as this one is the failure the
    boundary exists to prevent (A19: 0.74 false-play against 0.38), so the deployed path
    declines rather than doing it."""
    classify = scripted()
    pipeline = Pipeline("m", "probe", classify, boundaries=found(None), require_boundary=True)
    verdict = pipeline.classify_frame(FRAME, camera_id="cam9")
    assert verdict.state is MinuteState.UNCERTAIN and verdict.confidence == 0.0
    assert classify.calls == []
    assert "no boundary" in verdict.trace[0] and "cam9" in verdict.trace[0]


def test_without_a_boundary_the_interactive_path_scores_the_whole_frame_and_says_so() -> None:
    classify = scripted()
    pipeline = Pipeline("m", "probe", classify, boundaries=found(None), require_boundary=False)
    verdict = pipeline.classify_frame(FRAME, camera_id="cam9")
    assert verdict.state is MinuteState.ACTIVE_PLAY
    assert classify.calls == [None]
    assert verdict.boundary_key is None
    assert any("whole frame" in step for step in verdict.trace), verdict.trace


def test_the_gates_run_in_the_seam_and_speak_in_the_trace() -> None:
    from pitch_occupancy.vision.people import Counted

    class SilentMotion:
        def apply(self, state, previous, frame, polygon):
            return state, 0.5  # a cue, no overrule

    class NobodyThere:
        def inspect(self, state, frame, polygon):
            return Class3.EMPTY, Counted(people=0, ball=False)

    pipeline = Pipeline("m", "probe", scripted(), motion_gate=SilentMotion(),
                        person_gate=NobodyThere(), boundaries=found(MIDDLE))
    verdict = pipeline.classify_frame(FRAME, camera_id="cam")
    assert verdict.state is MinuteState.EMPTY and verdict.gated
    assert verdict.probed is Class3.ACTIVE_PLAY
    assert verdict.people == 0 and verdict.ball is False and verdict.motion == 0.5
    assert any("person gate -> EMPTY" in step for step in verdict.trace), verdict.trace
    assert any("0 inside the boundary" in step for step in verdict.trace), verdict.trace


def test_a_gate_that_did_not_run_leaves_its_fields_none() -> None:
    """None is "not consulted", which is not zero - `vision/explain.py`'s rule, kept."""
    pipeline = Pipeline("m", "probe", scripted(), boundaries=found(MIDDLE))
    verdict = pipeline.classify_frame(FRAME, camera_id="cam")
    assert verdict.people is None and verdict.ball is None and verdict.motion is None
