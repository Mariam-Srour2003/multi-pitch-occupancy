"""`worker.run_slot` on the detector-first pipeline (A36, WP9-T3).

The worker reads a burst per camera-minute, the pipeline counts each camera inside its
boundary, and the pitch is decided from the *sum* - so two halves showing three people each
make a match, the per-camera samples say what each half saw, and an abstained minute counts
against the capture floor rather than toward any verdict.
"""

from __future__ import annotations

import numpy as np

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.frame_source import Frame, FrameSource
from pitch_occupancy.pipeline import Pipeline
from pitch_occupancy.vision.detector import DETECTORS, PERSON, SPORTS_BALL, Detection, Detector
from pitch_occupancy.vision.pitch_classifier import DetectorFirstClassifier
from pitch_occupancy.vision.rules import RuleConfig
from pitch_occupancy.worker import run_slot

SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


class BurstSource(FrameSource):
    """Serves a scripted number of people per camera per minute; a frame's first pixel
    carries the count so the scripted detector can read it back."""

    def __init__(self, plan: dict[str, list[int | None]], *, burst: int = 3) -> None:
        self.plan = plan
        self.burst = burst
        self.bursts_asked: list[tuple[str, int, int]] = []

    def cameras(self) -> list[str]:
        return sorted(self.plan)

    @property
    def n_minutes(self) -> int:
        return max(len(v) for v in self.plan.values())

    def _frame(self, camera_id: str, minute_index: int) -> Frame | None:
        seq = self.plan[camera_id]
        if minute_index >= len(seq) or seq[minute_index] is None:
            return None
        image = np.zeros((40, 80, 3), np.uint8)
        image[0, 0, 0] = seq[minute_index]
        return Frame(camera_id, minute_index, image, "x.mp4")

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        return self._frame(camera_id, minute_index)

    def read_burst(self, camera_id: str, minute_index: int, *, n: int = 3,
                   spacing_s: float = 1.0) -> list[Frame]:
        self.bursts_asked.append((camera_id, minute_index, n))
        frame = self._frame(camera_id, minute_index)
        return [frame] * n if frame is not None else []


def counting_detector() -> Detector:
    """Reads the person count off the frame's first pixel; a ball when the count is >= 5."""
    det = Detector(DETECTORS["yolov8n"])

    def detect(image_bgr, **kwargs):
        n = int(image_bgr[0, 0, 0])
        out = [Detection(PERSON, (2 + 8 * i, 5, 8 + 8 * i, 30), 0.9) for i in range(n)]
        if n >= 5:
            out.append(Detection(SPORTS_BALL, (60, 20, 64, 24), 0.4))
        return out

    det.detect = detect  # type: ignore[method-assign]
    return det


def pipeline(*, boundary_for=None, **cfg) -> Pipeline:
    rules = RuleConfig(**cfg)
    clf = DetectorFirstClassifier(counting_detector(), rules)
    return Pipeline("yolov8n", "detector", clf, rules=rules, classifier=clf,
                    boundaries=boundary_for or (lambda cam, **_: (SQUARE, "k")))


def test_two_halves_of_three_make_a_match_the_worker_records_as_used() -> None:
    source = BurstSource({"camA": [3] * 4, "camB": [3] * 4})
    run = run_slot("s", source, None, pipeline=pipeline())
    assert run.verdict.status is SlotStatus.USED
    assert run.minute_states == ("2_playing",) * 4
    assert run.people_counts == (6, 6, 6, 6), "the pitch count is the sum"
    assert run.ball_minutes == (), "three and three: neither half saw a ball"
    # what each half saw is kept, in the three-class value the dashboard reads
    assert {s.predicted for s in run.samples} == {Class3.MAINTENANCE_NON_SPORTING.value}
    assert source.bursts_asked and all(n == 3 for _, _, n in source.bursts_asked)


def test_an_empty_pitch_with_a_ball_in_the_other_half_is_used_only_if_people_are() -> None:
    source = BurstSource({"camA": [0] * 4, "camB": [0] * 4})
    run = run_slot("s", source, None, pipeline=pipeline())
    assert run.verdict.status is SlotStatus.NOTUSED
    assert run.minute_states == ("1_empty",) * 4
    assert run.people_counts == (0, 0, 0, 0)


def test_a_camera_without_a_boundary_abstains_and_the_other_half_decides_at_half_confidence():
    source = BurstSource({"camA": [6] * 4, "camB": [6] * 4})
    run = run_slot("s", source, None,
                   pipeline=pipeline(boundary_for=lambda cam, **_: (
                       (SQUARE, "k") if cam == "camA" else (None, None))))
    assert run.verdict.status is SlotStatus.USED
    assert run.uncertain_cameras == ("camB",)
    assert run.minutes_uncertain == 0, "camA decided every minute"
    camb = [s for s in run.samples if s.camera_id == "camB"]
    assert all(s.predicted == "UNCERTAIN" for s in camb)
    assert run.verdict.mean_confidence < 0.5, "half a pitch, half the confidence"


def test_a_minute_with_no_frames_is_missed_and_a_slot_mostly_missed_is_review() -> None:
    source = BurstSource({"camA": [6, None, None, None]})
    run = run_slot("s", source, None, pipeline=pipeline())
    assert run.minutes_captured == 1 and run.minutes_missed == 3
    assert run.verdict.status is SlotStatus.REVIEW


def test_the_ball_is_recorded_per_minute_and_the_confidence_reflects_it() -> None:
    source = BurstSource({"camA": [7] * 3})
    run = run_slot("s", source, None, pipeline=pipeline())
    assert run.ball_minutes == (0, 1, 2)
    assert run.verdict.status is SlotStatus.USED
    assert run.verdict.mean_confidence > 0.8, "row 7"


def test_a_single_frame_burst_is_still_a_burst_of_one() -> None:
    source = BurstSource({"camA": [5] * 3}, burst=1)
    run = run_slot("s", source, None, pipeline=pipeline(burst_frames=1))
    assert run.verdict.status is SlotStatus.USED
    assert source.bursts_asked == [], "one frame means a plain read"
