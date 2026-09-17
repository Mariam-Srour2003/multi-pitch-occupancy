"""A boundary from the pitch's line markings, attempted and not landed (A24).

A19 and A23 bracket the colour-threshold boundary from both sides: loosening it admits
spectators, tightening it drops players, and every remaining error is a colour error - a tree
read as turf, a dim far end read as not-turf, a car park bridged by a convex hull. Both entries
end with the same recommendation, that the boundary should come from the pitch's **line
markings**, which bound the playing surface exactly and are white on green under any lighting
these cameras see.

Recommending it three times without trying it is not a finding. This is the attempt.

**Three stages, and the third is where it fails.**

1. **Markings are detectable.** A white top-hat - brighter than a neighbourhood wider than the
   line - with a low-saturation test, on the per-camera median. On venue_01 camera B this finds
   the halfway line, the penalty box and the far touchline clearly.
2. **And so is everything else bright and thin.** Fences, netting, window frames, the edges of
   buildings. Unrestricted, the hull of "marking" pixels covers **89%** of the frame. Gating on
   the turf mask fixes that and inherits its defect: `turf_mask` calls the trees beyond the
   fence turf, so tree highlights come through as markings.
3. **Requiring straightness removes the contamination and most of the markings with it.**
   `HoughLinesP` at a minimum length of 12% of the frame width finds **7** segments at
   venue_01 camera B, and their hull covers **14%** of the frame against the current boundary's
   55%. The near half of the pitch contributes nothing.

**Why stage 3 fails is specific and is the useful part.** A single top-hat kernel matches a
line of one width. Under perspective a touchline is several pixels across at the near edge of
the frame and sub-pixel at the far edge, so one kernel can be right for one band of the image
and is wrong above and below it. At camera B the kernel suits the far half, which is why the
detected segments are all in the far half. Lowering the Otsu threshold to catch the rest brings
the fence back.

**What would work, stated so the next attempt starts further along.** Either a kernel whose
width scales with image row - perspective is a smooth function of height for a fixed camera,
and the median frame is stable enough to estimate it - or, better, fitting a homography from a
pitch template to the detected segments, which uses the fact that the markings are not just
lines but *a known arrangement of lines*. The second is the standard approach in the sports
analytics literature and needs the full outline of a pitch in frame, which most of these
cameras do not have.

**Nothing here is wired in.** The failure is reported because "use line markings" is otherwise
an untested recommendation appearing in three log entries, and an untested recommendation that
has been tried and found hard is worth more than one that has not.

    uv run python experiments/line_marking_boundary.py --camera slot_20260711_1000_camB
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.vision import roi

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
WORK = (960, 540)

sys.path.insert(0, str(ROOT / "scripts"))

#: Top-hat kernel. A marking must be brighter than a neighbourhood wider than itself, so this
#: is "how wide is a line" - and the finding above is that there is no one answer under
#: perspective.
TOPHAT_WIDTH = 7

#: A marking is white. Above this saturation it is something else that happens to be bright.
MAX_SATURATION = 90

#: A marking is long. As a fraction of frame width.
MIN_LINE_FRACTION = 0.12


def median_frame(paths: list[Path], limit: int = 40) -> np.ndarray | None:
    imgs = []
    for p in paths[:limit]:
        im = cv2.imread(str(p))
        if im is not None:
            imgs.append(cv2.resize(im, WORK))
    if not imgs:
        return None
    return np.median(np.stack(imgs), axis=0).astype(np.uint8)


def marking_mask(bgr: np.ndarray, *, on_turf: bool) -> np.ndarray:
    """Thin bright low-saturation ridges, optionally restricted to the turf region."""
    from derive_roi import turf_mask

    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_WIDTH, TOPHAT_WIDTH))
    top = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[hsv[:, :, 1] > MAX_SATURATION] = 0
    if on_turf:
        small = turf_mask(cv2.resize(bgr, (320, 180)))
        turf = cv2.resize(small, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        mask[turf == 0] = 0
    return mask


def hull_of(points, shape) -> list[list[float]] | None:
    if len(points) < 3:
        return None
    hull = cv2.convexHull(np.asarray(points, np.int32))
    approx = cv2.approxPolyDP(hull, 0.015 * cv2.arcLength(hull, True), True)
    h, w = shape[:2]
    pts = [[float(x) / w, float(y) / h] for [[x, y]] in approx]
    try:
        return roi.validate(pts) if len(pts) >= 3 else None
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", action="append", default=None)
    ap.add_argument("--limit-cameras", type=int, default=12)
    args = ap.parse_args()

    rows = [r for r in read_manifest(DATASET / "manifest.csv") if r.source != "synthetic"]
    by_cam: dict[str, list[Path]] = {}
    for r in rows:
        by_cam.setdefault(r.camera, []).append(DATASET / r.file)
    cameras = args.camera or sorted(by_cam)[: args.limit_cameras]

    print(f"{len(cameras)} cameras | top-hat {TOPHAT_WIDTH}px, saturation <= {MAX_SATURATION}, "
          f"segments >= {MIN_LINE_FRACTION:.0%} of frame width\n")
    print(f"{'camera':<30}{'current':>10}{'raw marks':>12}{'on turf':>10}"
          f"{'straight':>10}{'segments':>10}")

    records = []
    for cam in cameras:
        med = median_frame(by_cam.get(cam, []))
        if med is None:
            continue
        current = roi.coverage(roi.get(cam)) if roi.get(cam) else float("nan")

        raw = marking_mask(med, on_turf=False)
        ys, xs = np.nonzero(raw)
        p_raw = hull_of(list(zip(xs.tolist(), ys.tolist(), strict=True)), med.shape)

        gated = marking_mask(med, on_turf=True)
        ys, xs = np.nonzero(gated)
        p_turf = hull_of(list(zip(xs.tolist(), ys.tolist(), strict=True)), med.shape)

        segs = cv2.HoughLinesP(gated, 1, np.pi / 360, threshold=60,
                               minLineLength=int(MIN_LINE_FRACTION * med.shape[1]),
                               maxLineGap=12)
        n_seg = 0 if segs is None else len(segs)
        ends = []
        if n_seg:
            for x1, y1, x2, y2 in segs.reshape(-1, 4):
                ends += [[int(x1), int(y1)], [int(x2), int(y2)]]
        p_line = hull_of(ends, med.shape)

        def show(p):
            return f"{roi.coverage(p):.0%}" if p else "-"

        print(f"{cam:<30}{current:>9.0%}{show(p_raw):>12}{show(p_turf):>10}"
              f"{show(p_line):>10}{n_seg:>10}")
        records.append({
            "camera": cam, "current": round(current, 4),
            "raw_marks": round(roi.coverage(p_raw), 4) if p_raw else "",
            "on_turf": round(roi.coverage(p_turf), 4) if p_turf else "",
            "straight_only": round(roi.coverage(p_line), 4) if p_line else "",
            "segments": n_seg,
        })

    ok = [r for r in records if r["straight_only"] != ""]
    if ok:
        near = [r for r in ok if abs(r["straight_only"] - r["current"]) < 0.10]
        print(f"\n{len(ok)} of {len(records)} cameras produced a polygon from straight "
              f"segments at all;\n{len(near)} of those land within 10 percentage points of "
              f"the current boundary.")
        print("A boundary that only works where the current one already does is not a "
              "replacement.")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "line_marking_boundary.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
