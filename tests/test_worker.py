"""The scheduler's slot loop.

Orchestration bugs are the quiet kind: a dropped camera or a misaligned minute still
produces a plausible verdict. These tests pin the behaviour that keeps a verdict honest
about how much evidence it actually rests on."""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.frame_source import Frame, FrameSource
from pitch_occupancy.worker import run_slot

PLAY, EMPTY = Class3.ACTIVE_PLAY, Class3.EMPTY


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


def classifier_from(plan: dict[str, list[Class3]]):
    """Returns a classifier that reads its answer from the call order per camera."""
    state = {k: iter(v) for k, v in plan.items()}
    order: list[str] = []

    def classify(image: np.ndarray) -> tuple[Class3, float]:
        cam = order.pop(0)
        return next(state[cam]), 0.9

    return classify, order


def constant(cls: Class3, conf: float = 0.9):
    return lambda image: (cls, conf)


# --- happy path -------------------------------------------------------------


def test_a_played_slot_is_used() -> None:
    src = FakeSource({"camA": [1] * 60, "camB": [1] * 60})
    run = run_slot("s1", src, constant(PLAY))
    assert run.verdict.status is SlotStatus.USED
    assert run.minutes_captured == 60
    assert run.minutes_missed == 0


def test_an_empty_slot_is_notused() -> None:
    src = FakeSource({"camA": [1] * 60, "camB": [1] * 60})
    assert run_slot("s1", src, constant(EMPTY)).verdict.status is SlotStatus.NOTUSED


def test_one_sample_recorded_per_camera_per_minute() -> None:
    src = FakeSource({"camA": [1] * 10, "camB": [1] * 10})
    run = run_slot("s1", src, constant(PLAY))
    assert len(run.samples) == 20
    assert {(s.camera_id, s.minute_index) for s in run.samples} == {
        (c, m) for c in ("camA", "camB") for m in range(10)
    }


def test_evidence_is_selected_for_the_verdict() -> None:
    src = FakeSource({"camA": [1] * 60})
    run = run_slot("s1", src, constant(PLAY))
    assert len(run.evidence) == 3
    assert all(e.predicted is PLAY for e in run.evidence)


# --- outages ----------------------------------------------------------------


def test_a_minute_with_no_camera_is_missed_not_invented() -> None:
    """A gap must weaken the verdict, never become a fabricated EMPTY."""
    script = [1] * 60
    for m in range(10, 20):
        script[m] = None
    src = FakeSource({"camA": list(script)})
    run = run_slot("s1", src, constant(PLAY))
    assert run.minutes_missed == 10
    assert run.minutes_captured == 50
    assert run.capture_rate == pytest.approx(50 / 60)


def test_a_slot_still_evaluates_when_one_camera_fails_entirely() -> None:
    """Half a pitch is degraded evidence, not no evidence."""
    src = FakeSource({"camA": [1] * 60, "camB": [None] * 60})
    run = run_slot("s1", src, constant(PLAY))
    assert run.verdict.status is SlotStatus.USED
    assert run.minutes_captured == 60
    assert {s.camera_id for s in run.samples} == {"camA"}


def test_a_total_outage_yields_review_not_notused() -> None:
    """No footage is not evidence a pitch was unused."""
    src = FakeSource({"camA": [None] * 60, "camB": [None] * 60})
    run = run_slot("s1", src, constant(PLAY))
    assert run.verdict.status is SlotStatus.REVIEW
    assert run.minutes_captured == 0
    assert run.capture_rate == 0.0


def test_capture_rate_is_one_when_nothing_is_missed() -> None:
    src = FakeSource({"camA": [1] * 5})
    assert run_slot("s1", src, constant(PLAY)).capture_rate == 1.0


# --- fusion through the loop ------------------------------------------------


def test_play_on_either_camera_carries_the_minute() -> None:
    """The goalkeeper-only half: camB sees a match, camA sees an empty half."""
    src = FakeSource({"camA": [1] * 60, "camB": [1] * 60})

    def classify(image: np.ndarray) -> tuple[Class3, float]:
        classify.n += 1  # type: ignore[attr-defined]
        return (EMPTY, 0.95) if classify.n % 2 else (PLAY, 0.80)  # type: ignore[attr-defined]

    classify.n = 0  # type: ignore[attr-defined]
    run = run_slot("s1", src, classify)
    assert run.verdict.status is SlotStatus.USED


def test_on_minute_callback_sees_every_captured_minute() -> None:
    seen: list[int] = []
    src = FakeSource({"camA": [1] * 12})
    run_slot("s1", src, constant(PLAY), on_minute=lambda m, s: seen.append(m))
    assert seen == list(range(12))
