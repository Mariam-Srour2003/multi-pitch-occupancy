"""Derive a pitch boundary per camera from the footage, instead of drawing one by hand.

Two problems this solves at once.

**The stored outlines belong to cameras that do not exist in the corpus.** `configs/roi.json`
holds `cam` and `cam2`; the manifest has 70 cameras and none is called that, so `roi.get`
returns None for every frame and the boundary never applies to any of them.

**Nine venues have no outline at all and never will by hand.** They are 10-second clips from
cameras nobody here controls. A boundary that cannot be drawn for most of the corpus cannot
be part of the method.

So the outline is measured. The pitch is the large connected green region, and it is the only
large green region in any of these frames:

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
strictly better than the whole frame.

    uv run python scripts/derive_roi.py --out configs/roi_derived.json
    uv run python scripts/derive_roi.py --identify      # which stored outline is which camera
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.vision import roi

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
MAX_FRAMES = 40
WORK = (320, 180)


def _median(paths: list[Path]) -> np.ndarray | None:
    imgs = []
    for p in paths[:MAX_FRAMES]:
        im = cv2.imread(str(p))
        if im is not None:
            imgs.append(cv2.resize(im, WORK))
    if not imgs:
        return None
    return np.median(np.stack(imgs), axis=0).astype(np.uint8)


def _largest(m: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == biggest, 255, 0).astype(np.uint8)


def _exg(bgr: np.ndarray) -> np.ndarray:
    """Excess-green ``2G - R - B``, normalised and smoothed.

    Excess-green rather than a hue mask, because floodlit artificial turf at night is closer
    to grey than to green and a hue threshold loses it - the *relative* channel order survives
    when saturation does not.
    """
    bgr = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX)
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    exg = np.clip(2 * g - r - b, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(exg, (5, 5), 0)


def _clean(m: np.ndarray) -> np.ndarray:
    """Close a player-shaped gap, drop specks, keep the largest region."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return _largest(m)


def flat_field(exg: np.ndarray) -> np.ndarray:
    """Divide out the smooth illumination surface, leaving the sharp edges.

    A floodlit pitch is brighter and greener near the camera than at the far end, and Otsu
    applies **one** threshold to the whole frame - so it splits the pitch rather than
    separating pitch from not-pitch. On the unseen clip that cost the far third, up to and
    including the goal: coverage 56%, and a person standing there was not counted.

    Dividing by a heavily blurred copy of the image is the standard correction for exactly
    that. The blur (sigma = width/6) is far wider than any pitch feature and far narrower than
    the frame, so it follows the lighting and not the turf. There is no threshold to tune here
    and no constant fitted to a venue.
    """
    background = cv2.GaussianBlur(exg.astype(np.float32), (0, 0), sigmaX=exg.shape[1] / 6)
    corrected = exg.astype(np.float32) / np.maximum(background, 1e-3) * float(background.mean())
    return np.clip(corrected, 0, 255).astype(np.uint8)


def turf_mask_flat(bgr: np.ndarray) -> np.ndarray:
    """:func:`turf_mask` with the lighting gradient removed before thresholding.

    **Measured and rejected (A19).** It does what it was meant to - it recovers the far third
    of the unseen clip's pitch, 56% coverage to 86% - and it is unusable, because it destroys
    the mask on frames that *have* no gradient. Dividing a fairly uniform image by a blurred
    copy of itself leaves near-constant noise, and Otsu then splits the noise: venue_01
    camera A, where the six recorded C3 frames live, collapses from 49% coverage to **3%**.
    Kept as the measured alternative that `roi_flat_field.py` compares against, not as a path
    anything calls.
    """
    _, m = cv2.threshold(flat_field(_exg(bgr)), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _clean(m)


#: How far below Otsu's threshold a pixel may sit and still join the pitch, as a fraction of
#: the threshold. Hysteresis needs two levels and this is the second; 0.6 is the value the
#: table in `roi_flat_field.py` was measured at. It is a floor, not a fit: the growth is
#: bounded by connectivity, so a low value costs nothing where there is nothing adjacent to
#: grow into.
GROW_FRACTION = 0.6


def turf_mask_grown(bgr: np.ndarray, fraction: float = GROW_FRACTION) -> np.ndarray:
    """Otsu, then grow into whatever is connected to it and nearly as green (A19).

    The defect this addresses is real: one global threshold on a floodlit pitch splits the
    pitch, because the far end is dimmer than the near end. `turf_mask_flat` addressed it by
    removing the gradient, and destroyed frames that had none.

    Hysteresis is the version that degrades gracefully. Otsu's threshold still decides what is
    *certainly* pitch; a second, lower threshold decides what may *join* it, and only regions
    touching the certain ones are kept. Where the far end is dim but connected it comes back;
    where there is nothing adjacent and nearly-green, the result is exactly
    :func:`turf_mask`. That is the property `turf_mask_flat` lacked.

    It is the same rule Canny uses on edges, applied to a region instead.

    **Measured and rejected (A19), and not for the reason `turf_mask_flat` was.** It behaves:
    no camera collapses, the minimum coverage across 69 cameras rises from 26% to 43%, and on
    923 labelled frames it halves the number of cross-venue play frames sitting in the 1-4
    person band. End to end on the unseen clip it is worse - the area it recovers reaches the
    barrier where three people stand watching, which turns two empty minutes into C3 and
    promotes the minute with one walker to a match. The boundary's value is in what it
    excludes, and a looser threshold recovers pitch and touchline at the same rate.
    """
    exg = _exg(bgr)
    high, strong = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    weak = (exg >= high * fraction).astype(np.uint8) * 255

    # Keep only the weak components that touch a strong one. `connectedComponents` on the
    # weak mask, then the set of labels that overlap the strong mask.
    n, lab = cv2.connectedComponents(weak, 8)
    if n <= 1:
        return _clean(strong)
    keep = set(np.unique(lab[strong > 0])) - {0}
    grown = np.isin(lab, list(keep)).astype(np.uint8) * 255
    return _clean(grown)


def turf_mask(bgr: np.ndarray) -> np.ndarray:
    """The pitch as the largest connected excess-green region.

    Excess-green rather than a hue mask, because floodlit artificial turf at night is closer
    to grey than to green and a hue threshold loses it - the *relative* channel order
    survives when saturation does not.

    A luminance fallback was written here for scenes too dark for excess-green, on the
    strength of two boundaries that looked wrong in a rendered check. Measured afterwards, it
    fired for **none of the 70 cameras**, and the boundaries in question turned out to be
    correct - the dome's pitch really does occupy only the lower band of its frame. The
    branch is gone rather than kept as an untested path that never runs, which is the defect
    this project keeps finding in its own guards.

    `normalize` stays: it changed four cameras by an IoU of 0.97-0.98, which is small but is
    a real effect on contrast-poor medians rather than a hypothetical one.
    """
    _, m = cv2.threshold(_exg(bgr), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _clean(m)


def polygon_from_mask(mask: np.ndarray, max_points: int = 8) -> list[list[float]] | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    approx = hull
    for eps in np.linspace(0.005, 0.08, 40):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) <= max_points:
            break
    h, w = mask.shape
    pts = [[float(x) / w, float(y) / h] for [[x, y]] in approx]
    if len(pts) < 3:
        return None
    try:
        return roi.validate(pts)
    except ValueError:
        return None


def _iou(poly_a, poly_b, size=(360, 640)) -> float:
    def fill(p):
        m = np.zeros(size, np.uint8)
        pts = np.array([[int(x * size[1]), int(y * size[0])] for x, y in p], np.int32)
        cv2.fillPoly(m, [pts], 255)
        return m > 0

    a, b = fill(poly_a), fill(poly_b)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "roi_derived.json")
    ap.add_argument("--identify", action="store_true",
                    help="match the stored cam/cam2 outlines to real cameras and stop")
    args = ap.parse_args()

    # Generated frames included. Their `camera` is `synthetic_<batch>`, so they have no
    # outline of their own, and without one they would be the only frames in a cache pooled
    # over the whole image while every recorded frame was pooled inside a boundary. Mixing
    # two pooling conventions inside one training set is the skew this work exists to remove,
    # arriving from the other side. The boundary is measured from the generated frames
    # themselves rather than borrowed from the camera that conditioned them, because the
    # generator does not hold the framing exactly.
    rows = read_manifest(DATASET / "manifest.csv")
    by_cam: dict[str, list[Path]] = defaultdict(list)
    for r in rows:
        by_cam[r.camera].append(DATASET / r.file)

    derived: dict[str, list[list[float]]] = {}
    for cam, paths in sorted(by_cam.items()):
        med = _median(paths)
        if med is None:
            continue
        poly = polygon_from_mask(turf_mask(med))
        if poly is None:
            print(f"  {cam:<34} no boundary found")
            continue
        derived[cam] = poly
        print(f"  {cam:<34} {len(poly)} pts, covers {roi.coverage(poly):.1%} of frame")

    print(f"\nderived {len(derived)} / {len(by_cam)} cameras")

    if args.identify:
        stored = roi.load_all()
        print("\nmatching stored outlines against derived ones (IoU):")
        for name, poly in stored.items():
            scores = sorted(((_iou(poly, d), c) for c, d in derived.items()), reverse=True)
            print(f"\n  {name}:")
            for s, c in scores[:3]:
                print(f"    {s:.3f}  {c}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": [
            "Pitch boundaries derived from the footage by scripts/derive_roi.py, not drawn.",
            "One per camera, [x, y] as fractions of the frame. Convex hull of the largest",
            "green region in a per-camera median frame - see the script for what that does",
            "and does not achieve.",
        ],
        **derived,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
