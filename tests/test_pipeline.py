"""The one seam every surface classifies through.

These pin that `pipeline.assemble` is the single place inference is put together (one
detector, one rule), that an unknown or retired model key falls back to the default detector
instead of crashing, that the boundary is resolved the way production names cameras, and that
a missing boundary is an abstention on the deployed path and a labelled whole-frame answer on
the interactive one.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3, Label
from pitch_occupancy.pipeline import (
    Pipeline,
    assemble,
    reset_shared,
    resolve_model_key,
    shared,
)
from pitch_occupancy.vision.detector import DETECTORS, PERSON, Detection, Detector
from pitch_occupancy.vision.pitch_classifier import DetectorFirstClassifier
from pitch_occupancy.vision.rules import (
    FrameVerdict,
    MinuteState,
    RuleConfig,
    from_class3,
    from_label,
    to_class3,
)

MIDDLE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
FRAME = np.zeros((100, 100, 3), np.uint8)


def counting_detector(n: int = 6) -> Detector:
    """``n`` people standing in the middle of the frame, and a record of every call."""
    det = Detector(DETECTORS["yolov8n"])
    calls: list[int] = []

    def detect(image_bgr, **_kwargs):
        calls.append(1)
        return [Detection(PERSON, (40 + i, 30, 45 + i, 60), 0.9) for i in range(n)]

    det.detect = detect  # type: ignore[method-assign]
    det.calls = calls  # type: ignore[attr-defined]
    return det


def scripted(n: int = 6, *, boundaries=None, require_boundary: bool = True) -> Pipeline:
    rules = RuleConfig(burst_frames=1)
    clf = DetectorFirstClassifier(counting_detector(n), rules)
    kwargs = {"boundaries": boundaries} if boundaries is not None else {}
    return Pipeline("yolov8n", clf, rules=rules, classifier=clf,
                    require_boundary=require_boundary, **kwargs)


def found(polygon, key: str = "stored"):
    """A boundary lookup that finds ``polygon`` for every camera, or nothing."""

    def lookup(camera, *, venue=None, slot_key=None):
        return (polygon, key) if polygon is not None else (None, None)

    return lookup


# --- the vocabulary -----------------------------------------------------------------------


def test_minute_states_are_the_reporting_classes_plus_an_abstention() -> None:
    """`Sample.predicted` must read the same whether it came from a prediction or from a
    report, so the three decided values *are* `Class3`'s values."""
    assert {s.value for s in MinuteState} - {"UNCERTAIN"} == {c.value for c in Class3}
    assert not MinuteState.UNCERTAIN.decided
    assert MinuteState.EMPTY.decided


def test_the_labels_map_in_legacy_folders_included_and_the_abstention_maps_out() -> None:
    assert from_class3(Class3.MAINTENANCE_NON_SPORTING) is MinuteState.MAINTENANCE_NON_SPORTING
    assert from_class3(Class3.ACTIVE_PLAY) is MinuteState.ACTIVE_PLAY
    assert from_label(Label.MAINTENANCE_NON_SPORTING) is MinuteState.MAINTENANCE_NON_SPORTING
    # the two pre-collapse folder names still read, because old CSVs still say them
    assert from_label("4_maintenance") is MinuteState.MAINTENANCE_NON_SPORTING
    assert from_label("3_people_not_playing") is MinuteState.MAINTENANCE_NON_SPORTING
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


@pytest.fixture
def no_weights(monkeypatch):
    """`Detector.load` without touching weights or torch: the scripted detector."""
    loaded: list[str] = []

    def load(key="yolov8n"):
        loaded.append(key)
        det = counting_detector()
        det.spec = DETECTORS[key]
        return det

    monkeypatch.setattr(Detector, "load", staticmethod(load))
    return loaded


def test_assemble_builds_the_detector_path_with_the_rule(no_weights) -> None:
    pipeline = assemble("yolov8n")
    assert pipeline.model_key == "yolov8n" and no_weights == ["yolov8n"]
    assert isinstance(pipeline.classifier, DetectorFirstClassifier)
    assert pipeline.rules is not None and pipeline.rules.detector == "yolov8n"
    assert pipeline.require_boundary, "the deployed path requires a boundary by default"
    assert pipeline.describe().startswith("detector yolov8n")


def test_another_detector_key_carries_into_the_rule(no_weights) -> None:
    other = next(k for k in DETECTORS if k != resolve_model_key(None))
    pipeline = assemble(other)
    assert pipeline.model_key == other and pipeline.rules.detector == other


@pytest.mark.parametrize("key", ["dinov2", "not-a-model"])
def test_an_unknown_or_retired_key_falls_back_to_the_default_detector(no_weights, key, caplog):
    """Old database rows and saved schedules can still name a model that no longer exists;
    that must not take the worker down."""
    with caplog.at_level(logging.WARNING, logger="pitch_occupancy.pipeline"):
        pipeline = assemble(key)
    assert pipeline.model_key == resolve_model_key(None) and pipeline.model_key in DETECTORS
    assert any(key in r.getMessage() for r in caplog.records), caplog.records


def test_with_no_key_the_default_is_the_detector_the_rule_file_names(no_weights) -> None:
    from pitch_occupancy.config import settings

    named = RuleConfig.load(settings.rules_path).detector
    assert resolve_model_key(None) == (named if named in DETECTORS else "yolov8n")


def test_the_shared_pipeline_is_built_once_and_reports_rather_than_requires(no_weights):
    """The API surfaces share one; a page a person is reading labels a whole-frame answer
    rather than refusing to give one."""
    reset_shared()
    try:
        first, second = shared(), shared()
        assert first is second
        assert no_weights == [first.model_key]
        assert not first.require_boundary
    finally:
        reset_shared()


# --- the boundary -------------------------------------------------------------------------


def test_polygon_for_binds_the_slot_context_to_every_lookup() -> None:
    """`worker.run_slot` takes a one-argument lookup; the venue and recording key it needs
    to resolve `file0` are bound in here, once, rather than remembered at the call site."""
    seen: list[tuple] = []

    def lookup(camera, *, venue=None, slot_key=None):
        seen.append((camera, venue, slot_key))
        return MIDDLE, "k"

    pipeline = scripted(boundaries=lookup)
    polygon_for = pipeline.polygon_for(venue="venue_01", slot_key="slot_20260711_1000")
    assert polygon_for("file0") == MIDDLE
    assert seen == [("file0", "venue_01", "slot_20260711_1000")]


# --- one frame ----------------------------------------------------------------------------


def test_classify_frame_counts_inside_the_resolved_boundary_and_records_which_it_was() -> None:
    pipeline = scripted(boundaries=found(MIDDLE, "slot_x_camA"))
    verdict = pipeline.classify_frame(FRAME, camera_id="file0")
    assert verdict.people == 6
    assert verdict.boundary_key == "slot_x_camA"
    assert any("keeps 25%" in step for step in verdict.trace), verdict.trace
    assert verdict.model_key == "yolov8n"


def test_a_given_polygon_wins_over_the_lookup() -> None:
    def never(camera, **_):
        raise AssertionError("the lookup must not run when a polygon is given")

    pipeline = scripted(boundaries=never)
    verdict = pipeline.classify_frame(FRAME, camera_id="cam", polygon=MIDDLE)
    assert verdict.boundary_key == "given" and verdict.people == 6


def test_without_a_boundary_the_deployed_path_abstains_and_never_runs_the_detector() -> None:
    """Counting the neighbouring pitch and the car park as this one is the failure the
    boundary exists to prevent, so the deployed path declines rather than doing it."""
    pipeline = scripted(boundaries=found(None), require_boundary=True)
    verdict = pipeline.classify_frame(FRAME, camera_id="cam9")
    assert verdict.state is MinuteState.UNCERTAIN and verdict.confidence == 0.0
    assert pipeline.classifier.detector.calls == []
    assert verdict.rule == 2
    assert "no boundary" in verdict.trace[0] and "cam9" in verdict.trace[0]


def test_without_a_boundary_the_interactive_path_counts_the_whole_frame() -> None:
    pipeline = scripted(boundaries=found(None), require_boundary=False)
    verdict = pipeline.classify_frame(FRAME, camera_id="cam9")
    assert verdict.decided and verdict.people == 6
    assert pipeline.classifier.detector.calls
    assert verdict.boundary_key is None


def test_a_still_has_no_motion_cue() -> None:
    """None is "not measured", which is not zero."""
    pipeline = scripted(boundaries=found(MIDDLE))
    verdict = pipeline.classify_frame(FRAME, camera_id="cam")
    assert verdict.motion is None
