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


def turf_mask(bgr: np.ndarray) -> np.ndarray:
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    exg = np.clip(2 * g - r - b, 0, 255).astype(np.uint8)
    exg = cv2.GaussianBlur(exg, (5, 5), 0)
    _, m = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == biggest, 255, 0).astype(np.uint8)


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

    rows = [r for r in read_manifest(DATASET / "manifest.csv") if r.source != "synthetic"]
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
