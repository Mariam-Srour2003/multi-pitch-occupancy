"""What stands inside the boundary, from a list of detections (A36, WP9-T3).

Pure geometry, no detector import, so every rule that reads a count can be tested with
detections written by hand. `vision/people.detect_inside` did this and ran the detector in
the same breath; this is that arithmetic on its own, extended with what the decision table
(`thesis/preregistration.md` A36) reads that the gate did not: vehicles, hi-vis, how spread
out the people are, and a count persisted over a burst.

**A person is placed by the foot of their box, a ball by its centre** (`vision/people.py`,
A16). Someone standing at the touchline has their centre over the pitch and their feet
outside it, and it is the feet that say where they stand; a ball spends much of its time in
the air and has no feet to stand on.

**`raw_inside` against `people_inside`.** The minimum-height filter (fitted on camera A in
WP9-T5) drops a detection too small to be a person at that depth of the frame - the goalpost,
the bag, the shadow that 11% of camera B's recorded empty frames carry (A16). What it dropped
is kept as a number, because the EMPTY confidence in row 3 of the table is lowered by it: an
empty pitch on which nothing looked like a person is a stronger EMPTY than one on which
something did.

**Persistence is over counts, not identities.** Nothing here tracks a person from frame to
frame; the persisted count over a burst is the *median* of the per-frame counts, which for
three frames is the value at least two of them reached or exceeded. A detection that appears
in one frame of three does not survive it; a real person who was missed in one frame of three
does. That is the whole of what "a person must persist in two of three frames" means in
code, and it is stated here so nobody reads a tracker into it.

**Hi-vis is a pixel fraction and a best effort.** Row 5 of the table names it because the
labelling protocol's own maintenance cue is workwear plus a tool (§2.5), and the corpus has
zero recorded maintenance frames to fit anything better on. The fraction of a person's box
that is saturated yellow-green or orange is what the rule reads; the threshold lives with
the other rule parameters and is flagged as unevaluated wherever it appears.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import median

import numpy as np

from pitch_occupancy.vision.detector import PERSON, SPORTS_BALL, VEHICLES, Detection

__all__ = [
    "PitchCount", "polygon_mask", "inside", "hi_vis_fraction", "spread_of",
    "count_inside", "persist",
]


@dataclass(frozen=True, slots=True)
class PitchCount:
    """What one pass found inside the boundary.

    ``people_inside`` is after the minimum-height filter; ``raw_inside`` before it.
    ``ball_seen`` is True when a ball was seen and False when the frame was checked and none
    was - which is not "there is no ball" (`vision/people.py`, A17). ``spread`` is the mean
    distance between the feet of the counted people over the boundary's diagonal, None with
    fewer than two of them.
    """

    people_inside: int
    people_total: int
    raw_inside: int
    ball_seen: bool
    ball_confidence: float = 0.0
    ball_area_px: int = 0
    vehicles_inside: int = 0
    hi_vis_people: int = 0
    spread: float | None = None
    #: The counted people and the balls inside, for the overlay and the evidence record.
    people: tuple[Detection, ...] = ()
    balls: tuple[Detection, ...] = ()
    #: Whether a boundary was applied at all. Without one everything counts, which is the
    #: pre-boundary behaviour, and the caller reports it rather than pretends otherwise.
    bounded: bool = True


def polygon_mask(polygon: Sequence[Sequence[float]] | None,
                 shape: tuple[int, int]) -> np.ndarray | None:
    """The boundary rasterised at ``shape`` ``(height, width)``, or None for no boundary."""
    if not polygon:
        return None
    import cv2

    height, width = shape
    mask = np.zeros((height, width), np.uint8)
    pts = np.array([[int(x * width), int(y * height)] for x, y in polygon], np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def inside(mask: np.ndarray | None, point: tuple[int, int]) -> bool:
    """Whether ``point`` ``(x, y)`` lies inside the mask. No mask means everything is inside.

    **A point on the frame's edge is read at the last row or column, not discarded.** A
    person close enough to the camera for the frame to cut their feet off has
    ``foot_y == frame_height`` - one past the last row - because `Detection.foot` is the
    bottom of a box the frame itself truncated. This used to test ``y < height`` and answer
    False, which put that person outside *every* polygon including one covering the whole
    frame: counting with no boundary found them and counting with a boundary found none.

    Reported from use on 2026-09-20 - two people standing in the foreground of a clip read as
    C1_EMPTY at zero people once a boundary was drawn. Clamping is the honest reading: the
    frame stops, the pitch does not, and the lowest row the camera can see is where a
    truncated box stands. What the clamp cannot do is move anyone across a boundary, because
    it only ever moves a point that was already off the edge onto the edge, and a polygon
    that does not reach the edge still excludes it (`tests/test_detector.py`).
    """
    if mask is None:
        return True
    x, y = point
    height, width = mask.shape[:2]
    if not (0 <= y <= height and 0 <= x <= width):
        return False
    return bool(mask[min(y, height - 1), min(x, width - 1)])


def hi_vis_fraction(frame_bgr: np.ndarray, det: Detection) -> float:
    """The share of a detection's pixels that are saturated hi-vis colours.

    Yellow-green (hue 20-45 in OpenCV's 0-180 scale) or orange (5-20), saturated and bright.
    The mask is used when the detection has one, else the box. A referee's bib will score
    here too; the rule reads this only for a small group with no vehicle, and says so.
    """
    import cv2

    x1, y1, x2, y2 = det.box
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    bright = (s > 150) & (v > 150)
    hue_ok = ((h >= 20) & (h <= 45)) | ((h >= 5) & (h < 20))
    hit = bright & hue_ok
    if det.mask is not None:
        region = det.mask[y1:y2, x1:x2]
        if region.shape == hit.shape and region.any():
            return float(hit[region].mean())
    return float(hit.mean())


def spread_of(people: Sequence[Detection], polygon: Sequence[Sequence[float]] | None,
              shape: tuple[int, int]) -> float | None:
    """Mean pairwise foot distance over the boundary's bounding-box diagonal, or None.

    Relative to the boundary rather than the frame, so a camera that sees half a pitch and
    one that sees all of it are read on the same scale.
    """
    if len(people) < 2:
        return None
    height, width = shape
    if polygon:
        xs = [x * width for x, _ in polygon]
        ys = [y * height for _, y in polygon]
        diagonal = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    else:
        diagonal = float(np.hypot(width, height))
    if diagonal <= 0:
        return None
    feet = [d.foot for d in people]
    distances = [float(np.hypot(ax - bx, ay - by)) for (ax, ay), (bx, by) in combinations(feet, 2)]
    return float(np.mean(distances)) / diagonal


def count_inside(
    detections: Sequence[Detection],
    polygon: Sequence[Sequence[float]] | None,
    shape: tuple[int, int],
    *,
    person_conf: float = 0.25,
    ball_conf: float = 0.10,
    min_height_at: Callable[[int], float] | None = None,
    frame_bgr: np.ndarray | None = None,
    hi_vis_min_fraction: float | None = None,
) -> PitchCount:
    """Count people (by foot), the ball (by centre) and vehicles (by foot) inside ``polygon``.

    ``min_height_at(foot_y) -> pixels`` is the smallest box height a person can have with
    their feet at that row of the frame; detections shorter than it are counted in
    ``raw_inside`` and dropped from ``people_inside``. ``hi_vis_min_fraction`` turns on the
    hi-vis count and needs ``frame_bgr``; both default off, since the measurement that would
    set them does not exist yet (WP9-T5).
    """
    mask = polygon_mask(polygon, shape)
    people: list[Detection] = []
    balls: list[Detection] = []
    raw_inside = 0
    people_total = 0
    vehicles = 0
    hi_vis = 0
    for det in detections:
        if det.cls == PERSON:
            if det.conf < person_conf:
                continue
            people_total += 1
            if not inside(mask, det.foot):
                continue
            raw_inside += 1
            if min_height_at is not None and det.height < min_height_at(det.foot[1]):
                continue
            people.append(det)
            if (hi_vis_min_fraction is not None and frame_bgr is not None
                    and hi_vis_fraction(frame_bgr, det) >= hi_vis_min_fraction):
                hi_vis += 1
        elif det.cls == SPORTS_BALL:
            if det.conf >= ball_conf and inside(mask, det.centre):
                balls.append(det)
        elif det.cls in VEHICLES:
            if det.conf >= person_conf and inside(mask, det.foot):
                vehicles += 1
    best_ball = max(balls, key=lambda d: d.conf, default=None)
    return PitchCount(
        people_inside=len(people),
        people_total=people_total,
        raw_inside=raw_inside,
        ball_seen=best_ball is not None,
        ball_confidence=best_ball.conf if best_ball else 0.0,
        ball_area_px=best_ball.area if best_ball else 0,
        vehicles_inside=vehicles,
        hi_vis_people=hi_vis,
        spread=spread_of(people, polygon, shape),
        people=tuple(people),
        balls=tuple(balls),
        bounded=mask is not None,
    )


def persist(counts: Sequence[PitchCount]) -> PitchCount:
    """One count for a burst: the median person count, any ball, the most people seen.

    See the module docstring for what this is and is not. The people and balls carried are
    those of the frame whose count equals the median (the first such), so the overlay shows a
    frame the verdict actually rests on.
    """
    if not counts:
        raise ValueError("a burst needs at least one frame")
    if len(counts) == 1:
        return counts[0]
    persisted = int(median(c.people_inside for c in counts))
    representative = next((c for c in counts if c.people_inside == persisted), counts[0])
    balls = [c for c in counts if c.ball_seen]
    best = max(balls, key=lambda c: c.ball_confidence, default=None)
    return PitchCount(
        people_inside=persisted,
        people_total=max(c.people_total for c in counts),
        raw_inside=max(c.raw_inside for c in counts),
        ball_seen=best is not None,
        ball_confidence=best.ball_confidence if best else 0.0,
        ball_area_px=best.ball_area_px if best else 0,
        vehicles_inside=int(median(c.vehicles_inside for c in counts)),
        hi_vis_people=int(median(c.hi_vis_people for c in counts)),
        spread=representative.spread,
        people=representative.people,
        balls=best.balls if best else (),
        bounded=all(c.bounded for c in counts),
    )
