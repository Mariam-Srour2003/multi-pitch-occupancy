"""Measure a pitch boundary from footage, instead of drawing one by hand (WP3-T1, A36).

The routine `scripts/derive_roi.py` has run over the corpus since A14, moved into the package
so the paths that need a boundary at run time can call it without `sys.path` surgery:
`api/clip_review.py`, `scripts/run_slot_on_video.py` and `experiments/clock_rule_on_video.py`
each carried their own copy of "insert the scripts folder, import two functions", and the
worker - the one place a missing boundary costs a night's verdicts - had no copy at all. The
script keeps its CLI and the two measured-and-rejected variants (`turf_mask_flat`,
`turf_mask_grown`, A19); this module holds only what is deployed.

The pitch is the large connected green region, and it is the only large green region in any
of these frames:

1. **Median over many frames**, which removes people - they move, the pitch does not.
2. **Excess-green** ``2G - R - B``, thresholded with Otsu. Hue-based masks fail at night,
   when floodlit artificial turf is closer to grey than to green; excess-green survives it
   because the *relative* channel order holds even when saturation collapses.
3. **Largest connected component**, closed and hole-filled, so a player-shaped gap or a line
   marking does not cut the pitch in two.
4. **Convex hull**, simplified to a handful of points.

The hull is deliberate and it is a limitation worth stating: a pitch seen from the corner is
convex, but a hull cannot exclude a dugout or a neighbouring pitch that lies *inside* the
outline's span. It is a floor on what the boundary achieves, not a ceiling, and it is still
strictly better than the whole frame - A19 measured the whole frame at 0.74 false-play on the
unseen clip against 0.38 with a derived outline.

**Frames are spread over the whole recording, not taken from its start.** The previous
callers pooled every fortieth frame until forty were held, which on a ten-second clip is
eight frames and on an hour's recording is the first fifty-three seconds - a median over
one passage of play, in which the players do not move out of the way. `derive_from_video`
seeks to forty positions spread across the file instead; a stream that reports no frame
count falls back to reading as it comes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.vision import roi

__all__ = [
    "MAX_FRAMES", "WORK", "median_of_frames", "median_of_paths", "largest", "exg", "clean",
    "turf_mask", "polygon_from_mask", "iou", "derive_from_frames", "derive_from_video",
]

#: How many frames a median is taken over. Forty is what every derived boundary in
#: `configs/roi_derived.json` was measured with.
MAX_FRAMES = 40

#: The working size the median and the mask are computed at. A boundary is a handful of
#: points in frame fractions, and 320x180 is plenty to place them; at full resolution the
#: median of forty 1080p frames is a quarter of a gigabyte for nothing.
WORK = (320, 180)


def median_of_frames(frames: Iterable[np.ndarray | None]) -> np.ndarray | None:
    """The per-pixel median of up to :data:`MAX_FRAMES` frames at :data:`WORK`, or None."""
    pool = []
    for frame in frames:
        if frame is None:
            continue
        pool.append(cv2.resize(np.asarray(frame), WORK))
        if len(pool) >= MAX_FRAMES:
            break
    if not pool:
        return None
    return np.median(np.stack(pool), axis=0).astype(np.uint8)


def median_of_paths(paths: Sequence[Path]) -> np.ndarray | None:
    """:func:`median_of_frames` over image files; unreadable files are skipped."""
    return median_of_frames(cv2.imread(str(p)) for p in paths[:MAX_FRAMES])


def largest(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected region of a binary mask."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == biggest, 255, 0).astype(np.uint8)


def exg(bgr: np.ndarray) -> np.ndarray:
    """Excess-green ``2G - R - B``, normalised and smoothed.

    Excess-green rather than a hue mask, because floodlit artificial turf at night is closer
    to grey than to green and a hue threshold loses it - the *relative* channel order survives
    when saturation does not.
    """
    bgr = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX)
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    out = np.clip(2 * g - r - b, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(out, (5, 5), 0)


def clean(mask: np.ndarray) -> np.ndarray:
    """Close a player-shaped gap, drop specks, keep the largest region."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return largest(mask)


def turf_mask(bgr: np.ndarray) -> np.ndarray:
    """The pitch as the largest connected excess-green region.

    A luminance fallback was written here for scenes too dark for excess-green, on the
    strength of two boundaries that looked wrong in a rendered check. Measured afterwards, it
    fired for **none of the 70 cameras**, and the boundaries in question turned out to be
    correct - the dome's pitch really does occupy only the lower band of its frame. The
    branch is gone rather than kept as an untested path that never runs, which is the defect
    this project keeps finding in its own guards.

    `normalize` stays: it changed four cameras by an IoU of 0.97-0.98, which is small but is
    a real effect on contrast-poor medians rather than a hypothetical one.
    """
    _, mask = cv2.threshold(exg(bgr), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clean(mask)


def polygon_from_mask(mask: np.ndarray, max_points: int = 8, *,
                      hull: bool = True) -> roi.Polygon | None:
    """Simplify the largest region of ``mask`` to a polygon of at most ``max_points``.

    ``hull=False`` follows the region's own outline instead of its convex hull, which is
    tighter wherever the pitch is seen at an angle - a hull spans from the far corner of the
    pitch to the near one and swallows whatever lies between, which at venue_01 camera B is
    the car park behind the goal. It is offered rather than assumed because a ragged mask
    makes a ragged polygon, and `roi.validate` rejects a self-intersecting one; see A23 for
    what it is actually worth.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    outline = max(contours, key=cv2.contourArea)
    if hull:
        outline = cv2.convexHull(outline)
    perimeter = cv2.arcLength(outline, True)
    approx = outline
    for eps in np.linspace(0.005, 0.08, 40):
        approx = cv2.approxPolyDP(outline, eps * perimeter, True)
        if len(approx) <= max_points:
            break
    height, width = mask.shape
    points = [[float(x) / width, float(y) / height] for [[x, y]] in approx]
    if len(points) < 3:
        return None
    try:
        return roi.validate(points)
    except ValueError:
        return None


def iou(poly_a: roi.Polygon, poly_b: roi.Polygon, size: tuple[int, int] = (360, 640)) -> float:
    """Intersection over union of two boundaries, rasterised at ``size`` (rows, cols)."""

    def fill(polygon: roi.Polygon) -> np.ndarray:
        mask = np.zeros(size, np.uint8)
        pts = np.array([[int(x * size[1]), int(y * size[0])] for x, y in polygon], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask > 0

    a, b = fill(poly_a), fill(poly_b)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def derive_from_frames(frames: Iterable[np.ndarray | None], *,
                       hull: bool = True) -> roi.Polygon | None:
    """A boundary measured from a set of frames of one camera, or None if no turf was found."""
    median = median_of_frames(frames)
    if median is None:
        return None
    return polygon_from_mask(turf_mask(median), hull=hull)


def derive_from_video(path: Path | str, *, n_frames: int = MAX_FRAMES,
                      hull: bool = True) -> roi.Polygon | None:
    """A boundary measured from a recording or a stream, or None.

    Frames are taken at ``n_frames`` positions spread over the whole file. A stream, or a
    file whose container reports no frame count, is read as it comes with every fortieth
    frame kept, which is the behaviour every previous caller had.

    Never raises: a boundary is an improvement, not a precondition, and the caller says
    which it got.
    """
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None
        pool: list[np.ndarray] = []
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                for index in np.linspace(0, total - 1, n_frames).astype(int):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, frame = capture.read()
                    if ok:
                        pool.append(frame)
            else:
                i = 0
                while len(pool) < n_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if i % 40 == 0:
                        pool.append(frame)
                    i += 1
        finally:
            capture.release()
        return derive_from_frames(pool, hull=hull)
    except Exception:  # noqa: BLE001 - see the docstring
        return None
