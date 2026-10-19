"""The classifier: frames of one camera in, a per-minute observation out.

A *burst* of frames becomes a count, and `vision/rules.decide` turns the count into a state.
It sits behind `pipeline.Pipeline`, so the worker, the pages and the scripts share one path.

**What one observation is.** The detector runs on every frame of the burst (boxes only; the
counting rule reads feet, not masks). `counting.count_inside` places each detection inside or
outside the camera's boundary, `counting.persist` takes the median count over the burst, the
motion cue between consecutive burst frames is averaged, and `decide` reads the result. If the
model produces masks, one more pass runs on the frame the verdict rests on, for the overlay
alone - masks change what a reader sees, never what is counted (`vision/detector.py`).

**Two motion cues, kept apart.** The burst cue - frames about a second apart - is what the
rule's thresholds are fitted at. The minute cue - this burst's last frame against the
previous minute's - is measured at a different gap again, and is recorded for the conditions
record without being read by any row. A threshold fitted at one gap is wrong at another, and
mixing the two in one field would invite that mistake.

**Per-person movement** (`vision/movement.py`) and the **brightness** of the pitch are measured
here too and handed to the rule: who moved between burst frames, and whether the picture is
too dark for an empty count to be believed.

**A burst stops early** once two frames agree on fewer people than a game needs: the count is
a median, so the remaining frames cannot change the state. Most minutes are an empty pitch,
so on a small machine this saves about a third of the detector's work.

**`__call__` keeps the old contract, with its limits stated.** `worker.Classifier` returns a
class and a confidence and cannot say "declined". Called on a still with no boundary this
counts the whole frame and says so in the verdict's trace, as the interactive pages do; called
when the detector cannot load it raises, because the one thing a legacy caller must not receive
from a broken detector is a plausible class.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import mean

import numpy as np

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision.counting import PitchCount, count_inside, persist
from pitch_occupancy.vision.detector import Detection, Detector
from pitch_occupancy.vision.motion import motion_cue
from pitch_occupancy.vision.rules import FrameVerdict, MinuteState, RuleConfig, decide

__all__ = ["CameraObservation", "DetectorFirstClassifier"]


@dataclass(frozen=True, slots=True)
class CameraObservation:
    """One camera, one minute: the verdict and what it rests on."""

    camera_id: str
    verdict: FrameVerdict
    #: How many burst frames were read, and which of them the verdict rests on - the frame
    #: whose count equals the persisted count, which is the one to show and to save.
    frames: int
    evidence_index: int = 0
    #: Detections for the overlay: masks when the model has them, else the counted boxes.
    instances: tuple[Detection, ...] = ()
    motion_burst: float | None = None
    motion_minute: float | None = None

    @property
    def decided(self) -> bool:
        return self.verdict.decided


def _settled(counts: list[PitchCount], play_min: int, total: int) -> bool:
    """Whether the median person count of the burst is already fixed below ``play_min``."""
    if len(counts) < 2 or len(counts) >= total:
        return False
    seen = [c.people_inside for c in counts]
    return seen[-1] < play_min and seen.count(seen[-1]) > total // 2


def brightness(frame: np.ndarray, polygon: list[list[float]] | None) -> float:
    """Mean grey level inside the boundary (or the whole frame without one), 0-255."""
    import cv2

    from pitch_occupancy.vision.counting import polygon_mask

    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask = polygon_mask(polygon, grey.shape[:2])
    if mask is not None and mask.any():
        return float(grey[mask.astype(bool)].mean())
    return float(grey.mean())


class DetectorFirstClassifier:
    """A registered detector and a rule config, behind `worker.Classifier` and more."""

    def __init__(self, detector: Detector, cfg: RuleConfig) -> None:
        self.detector = detector
        self.cfg = cfg

    # --- the burst ------------------------------------------------------------------

    def observe(
        self,
        camera_id: str,
        frames: list[np.ndarray],
        polygon: list[list[float]] | None,
        *,
        previous: np.ndarray | None = None,
        require_boundary: bool = True,
    ) -> CameraObservation:
        """Count the burst inside the boundary and decide. Never raises on a detector fault."""
        frames = [f for f in frames if f is not None]
        if not frames:
            raise ValueError("an observation needs at least one frame")
        cfg = self.cfg
        shape = frames[0].shape[:2]
        min_height = cfg.min_height_at(shape[0])
        floor = min(cfg.person_conf, cfg.ball_conf)

        counts: list[PitchCount] = []
        for frame in frames:
            # Not with `require_ball`: there a ball seen in the last frame can still decide.
            if not cfg.require_ball and _settled(counts, cfg.play_min, len(frames)):
                # The burst's count is a median, and two frames that agree on fewer people
                # than a game needs fix it whatever the rest say. On a small CPU most minutes
                # are an empty pitch, so this skips a third of the detector's work.
                break
            found = self.detector.detect(frame, confidence=floor, imgsz=cfg.imgsz,
                                         tiles=cfg.tiles)
            if found is None:
                verdict = decide(None, motion=None, cfg=cfg)
                return CameraObservation(camera_id, replace(verdict, polygon=polygon),
                                         frames=len(frames))
            counts.append(count_inside(
                found, polygon, shape, person_conf=cfg.person_conf, ball_conf=cfg.ball_conf,
                min_height_at=min_height,
                frame_bgr=frame if cfg.hi_vis_min_fraction is not None else None,
                hi_vis_min_fraction=cfg.hi_vis_min_fraction,
                edge_margin=cfg.edge_margin,
            ))

        count = persist(counts)
        evidence_index = next(
            (i for i, c in enumerate(counts) if c.people_inside == count.people_inside), 0)
        motion_burst = (mean(motion_cue(a, b, polygon)
                             for a, b in zip(frames, frames[1:], strict=False))
                        if len(frames) >= 2 else None)
        motion_minute = motion_cue(previous, frames[-1], polygon) if previous is not None else None

        verdict = decide(count, motion=motion_burst, cfg=cfg, require_boundary=require_boundary,
                         brightness=brightness(frames[evidence_index], polygon))
        verdict = replace(verdict, polygon=polygon)

        # The overlay draws the evidence frame, so its instances are that frame's own - the
        # persisted count may carry a ball from a different frame of the burst.
        evidence = counts[evidence_index]
        instances: tuple[Detection, ...] = evidence.people + evidence.balls
        if self.detector.spec.segment and verdict.decided:
            with_masks = self.detector.detect(frames[evidence_index], confidence=floor,
                                              imgsz=cfg.imgsz, tiles=cfg.tiles, masks=True)
            if with_masks:
                masked = count_inside(with_masks, polygon, shape, person_conf=cfg.person_conf,
                                      ball_conf=cfg.ball_conf, min_height_at=min_height,
                                      edge_margin=cfg.edge_margin)
                instances = masked.people + masked.balls
        return CameraObservation(camera_id, verdict, frames=len(frames),
                                 evidence_index=evidence_index, instances=instances,
                                 motion_burst=motion_burst, motion_minute=motion_minute)

    # --- the legacy contract --------------------------------------------------------

    def __call__(self, image_bgr: np.ndarray, *,
                 polygon: list[list[float]] | None = None) -> tuple[Class3, float]:
        observation = self.observe("still", [image_bgr], polygon, require_boundary=False)
        verdict = observation.verdict
        if verdict.state is MinuteState.UNCERTAIN:
            raise RuntimeError("; ".join(verdict.trace) or "the detector declined")
        return verdict.as_pair()
