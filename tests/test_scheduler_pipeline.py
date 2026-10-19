"""The pipeline reaches `run_slot` through the scheduler.

These assert on what reaches `run_slot` - the assembled pipeline (detector, rule and boundary
lookup) and the slot context the boundary lookup needs - because the handover is the part a
deployment cannot see go missing.
"""

from __future__ import annotations

from datetime import datetime, time
from types import SimpleNamespace

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.frame_source import Frame, FrameSource
from pitch_occupancy.pipeline import Pipeline
from pitch_occupancy.scheduler import Schedule, ScheduledSlot, run_due, run_forever
from pitch_occupancy.vision.detector import DETECTORS, PERSON, SPORTS_BALL, Detection, Detector
from pitch_occupancy.vision.pitch_classifier import DetectorFirstClassifier
from pitch_occupancy.vision.rules import RuleConfig

MIDDLE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


def slot(start: str = "10:00", minutes: int = 60) -> ScheduledSlot:
    hh, mm = (int(x) for x in start.split(":"))
    return ScheduledSlot("venue_01", time(hh, mm), minutes, ("camera_A",), field_id="field_01")


class Stub(FrameSource):
    def __init__(self, minutes: int = 3) -> None:
        self._minutes = minutes

    def cameras(self) -> list[str]:
        return ["camera_A"]

    @property
    def n_minutes(self) -> int:
        return self._minutes

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        return Frame(camera_id, minute_index, np.zeros((100, 100, 3), np.uint8), "x.mp4")


def pipeline(boundary=MIDDLE, *, require: bool = True) -> Pipeline:
    """A detector-first pipeline whose detector sees six people and a ball mid-pitch."""
    detector = Detector(DETECTORS["yolov8n"])
    detector.detect = lambda image, **_: (
        [Detection(PERSON, (40 + i, 30, 45 + i, 60), 0.9) for i in range(6)]
        + [Detection(SPORTS_BALL, (48, 48, 52, 52), 0.5)])
    rules = RuleConfig(burst_frames=1)
    clf = DetectorFirstClassifier(detector, rules)
    return Pipeline(
        "yolov8n", clf, rules=rules, classifier=clf,
        require_boundary=require,
        boundaries=lambda cam, *, venue=None, slot_key=None: (
            (boundary, "k") if boundary is not None else (None, None)),
    )


def test_run_due_hands_the_pipeline_and_the_slot_context_to_run_slot(monkeypatch) -> None:
    import pitch_occupancy.worker as worker

    seen: dict[str, object] = {}

    def spy(slot_id, source, classify, **kwargs):
        seen.update(kwargs, slot_id=slot_id, classify=classify)
        return SimpleNamespace(verdict=None, samples=[], evidence=[])

    monkeypatch.setattr(worker, "run_slot", spy)
    assembled = pipeline()
    ran = run_due(Schedule((slot("10:00", 60),)), datetime(2026, 7, 11, 10, 5),
                  source_for=lambda sl, day: Stub(), pipeline=assembled)
    assert ran == ["venue_01_2026-07-11_1000"]
    assert seen["pipeline"] is assembled
    assert seen["venue"] == "venue_01"
    assert seen["slot_key"] == "slot_20260711_1000", "file0 resolves only through this"
    assert seen["classify"] is None, "the pipeline's classifier is used, not a second one"


def test_a_scripted_classifier_alone_still_works_and_gets_no_context(monkeypatch) -> None:
    """The older shape, for tests that want nothing in front of the classifier."""
    import pitch_occupancy.worker as worker

    seen: dict[str, object] = {}

    def spy(slot_id, source, classify, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(verdict=None, samples=[], evidence=[])

    monkeypatch.setattr(worker, "run_slot", spy)
    run_due(Schedule((slot(),)), datetime(2026, 7, 11, 10, 5),
            source_for=lambda sl, day: Stub(), classify=lambda image: (Class3.EMPTY, 0.9))
    assert "pipeline" not in seen and "slot_key" not in seen


def test_run_due_refuses_to_run_with_neither_a_pipeline_nor_a_classifier() -> None:
    with pytest.raises(TypeError, match="pipeline"):
        run_due(Schedule((slot(),)), datetime(2026, 7, 11, 10, 5),
                source_for=lambda sl, day: Stub())


def test_a_camera_with_no_boundary_reads_uncertain_through_the_scheduler() -> None:
    """The boundary rule end to end: no boundary, no verdict. Every minute abstains, the capture
    floor sees an empty slot, and the verdict is REVIEW - never a confident NOTUSED built
    from a car park."""
    runs: list[object] = []
    run_due(Schedule((slot("10:00", 60),)), datetime(2026, 7, 11, 10, 5),
            source_for=lambda sl, day: Stub(minutes=3),
            pipeline=pipeline(boundary=None),
            on_slot=lambda sid, run: runs.append(run))
    (run,) = runs
    assert run.minutes_uncertain == 3 and run.minutes_missed == 3
    assert run.uncertain_cameras == ("camera_A",)
    assert run.minutes_captured == 0
    assert run.verdict.status is SlotStatus.REVIEW
    assert {s.predicted for s in run.samples} == {"UNCERTAIN"}
    assert all(s.confidence == 0.0 for s in run.samples)


def test_with_a_boundary_the_same_slot_is_decided() -> None:
    runs: list[object] = []
    run_due(Schedule((slot("10:00", 60),)), datetime(2026, 7, 11, 10, 5),
            source_for=lambda sl, day: Stub(minutes=3),
            pipeline=pipeline(boundary=MIDDLE),
            on_slot=lambda sid, run: runs.append(run))
    (run,) = runs
    assert run.minutes_uncertain == 0 and run.minutes_captured == 3
    assert run.verdict.status is SlotStatus.USED


def test_the_loop_forwards_the_pipeline(monkeypatch) -> None:
    import pitch_occupancy.scheduler as sched

    seen: dict[str, object] = {}

    def spy(schedule, now, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(sched, "run_due", spy)
    assembled = pipeline()
    run_forever(Schedule((slot(),)), source_for=lambda sl, day: Stub(),
                pipeline=assembled, clock=lambda: datetime(2026, 7, 11, 10, 5),
                sleep=lambda _: None, iterations=1)
    assert seen["pipeline"] is assembled
