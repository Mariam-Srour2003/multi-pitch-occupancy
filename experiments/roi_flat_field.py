"""The derived boundary clips the far end of a pitch, and what fixing it costs (A19).

Rendered with the detections drawn on, the boundary for the unseen clip keeps **56%** of the
frame and stops a third of the way up the pitch: the far third, up to and including the goal,
is outside. A person standing there is not counted, which is why minute 12 of that clip - a
person plainly visible - reads EMPTY. The same thing shows in the corpus: of the six recorded
C3 frames, three detect people in the frame and none inside the boundary.

**The cause is a global threshold on an image with a lighting gradient.** `turf_mask`
thresholds excess-green with Otsu, one value for the whole frame. Under floodlights the far
end of a pitch is dimmer and hazier than the near end, so a single threshold splits the pitch
itself rather than separating pitch from not-pitch.

**Two candidate fixes, and the obvious one is wrong.** Flat-field correction - divide the
excess-green image by a heavily blurred copy of itself - removes the gradient and recovers the
clip's far third. It also destroys frames that have no gradient: a near-uniform image divided
by a blur of itself is noise, and Otsu splits the noise. Hysteresis is the version that
degrades gracefully: Otsu still decides what is certainly pitch, a lower threshold decides
what may join it, and only regions touching the certain ones survive. Where there is nothing
adjacent to grow into it reduces exactly to the current mask.

**A wider boundary is not automatically a better one**, and that is what this measures. The
boundary exists to keep spectators, staff and the neighbouring pitch out of the count. Every
pixel it gains is a chance to let one in. So both directions are measured on frames that carry
labels:

- **venue_01 camera B, 243 recorded EMPTY frames** - the safety side. A boundary that admits
  off-pitch people shows up here as frames that stop counting zero, and would break the person
  gate directly.
- **venue_01 camera B, 278 recorded ACTIVE_PLAY frames**, and **396 clip-venue frames**, all
  genuine play - the recall side. A boundary that reaches the far end should find more of the
  players who were always there.
- **the 6 recorded C3 frames**, where the current boundary finds nobody.

The detector runs once per frame and both boundaries are applied to the same boxes, so the
comparison is exact and costs one pass rather than two.

    uv run python experiments/roi_flat_field.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.explain import detect_objects
from pitch_occupancy.vision.people import BALL_CONFIDENCE, DETECT_CONFIDENCE, DETECT_IMGSZ

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
EMPTY, PLAY, C3 = "C1_EMPTY", "C2_ACTIVE_PLAY", "C3_MAINTENANCE_NON_SPORTING"

sys.path.insert(0, str(ROOT / "scripts"))


def _inside(boxes, polygon, shape) -> int:
    if polygon is None:
        return len(boxes)
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array([[int(x * w), int(y * h)] for x, y in polygon], np.int32)], 1)
    n = 0
    for x1, _y1, x2, y2 in boxes:
        fx, fy = (x1 + x2) // 2, y2
        if 0 <= fy < h and 0 <= fx < w and mask[fy, fx]:
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="cap frames per venue, 0 = all")
    args = ap.parse_args()

    from derive_roi import (
        _median,
        polygon_from_mask,
        turf_mask,
        turf_mask_flat,
        turf_mask_grown,
    )

    ARMS = {"current": turf_mask, "flat-field": turf_mask_flat, "grown": turf_mask_grown}

    rows = [r for r in read_manifest(DATASET / "manifest.csv") if r.source != "synthetic"]

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    frames = [r for r in rows if r.venue == "venue_01" and cam(r) == "camera_B"]
    frames += [r for r in rows if r.venue.startswith("clipvenue_")]
    frames += [r for r in rows if r.class3 == C3 and r not in frames]
    if args.limit:
        by: dict[str, list] = {}
        for r in frames:
            by.setdefault(r.venue, []).append(r)
        frames = [r for v in by.values() for r in v[: args.limit]]

    # One boundary pair per camera, both derived from the same median, so the only difference
    # between the arms is the threshold.
    cameras = sorted({r.camera for r in frames})
    paths: dict[str, list[Path]] = {}
    for r in frames:
        paths.setdefault(r.camera, []).append(DATASET / r.file)
    polys: dict[str, dict[str, list]] = {arm: {} for arm in ARMS}
    print(f"deriving {len(ARMS)} boundaries for {len(cameras)} cameras ...")
    for c in cameras:
        med = _median(paths[c])
        if med is None:
            continue
        for arm, fn in ARMS.items():
            polys[arm][c] = polygon_from_mask(fn(med))

    from pitch_occupancy.vision import roi

    # A boundary that collapses is the failure mode that matters, and a mean hides it: one
    # camera at 3% of the frame counts for as little as one at 50%. The minimum is reported
    # beside the median for that reason.
    print(f"\n{'arm':<12}{'mean coverage':>15}{'median':>9}{'min':>8}{'under 20%':>11}")
    base = None
    for arm in ARMS:
        cov = np.array([roi.coverage(polys[arm][c]) for c in cameras
                        if polys[arm].get(c)])
        if base is None:
            base = cov
        print(f"{arm:<12}{cov.mean():>15.1%}{np.median(cov):>9.1%}{cov.min():>8.1%}"
              f"{(cov < 0.20).sum():>11}")

    records = []
    for n, r in enumerate(frames, 1):
        img = cv2.imread(str(DATASET / r.file))
        if img is None:
            continue
        found = detect_objects(img, confidence=min(DETECT_CONFIDENCE, BALL_CONFIDENCE),
                               imgsz=DETECT_IMGSZ, classes=(0, 32))
        if found is None:
            continue
        boxes = [box for cls, box, conf in found if cls == 0 and conf >= DETECT_CONFIDENCE]
        rec = {"file": r.file, "venue": r.venue, "class3": r.class3,
               "detected": len(boxes)}
        for arm in ARMS:
            rec[arm] = _inside(boxes, polys[arm].get(r.camera), img.shape)
        records.append(rec)
        if n % 50 == 0:
            print(f"  {n}/{len(frames)}", end="\r", flush=True)
    print(" " * 30, end="\r")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "roi_flat_field.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    def report(label: str, sub: list[dict], *, safety: bool) -> None:
        if not sub:
            return
        cells = []
        for arm in ARMS:
            v = np.array([d[arm] for d in sub])
            # On EMPTY frames the question is how often the boundary still finds nobody,
            # because that is the verdict the person gate turns into EMPTY. On play frames it
            # is how many of the players who were always there it now reaches.
            cells.append(f"{(v == 0).mean():>11.1%}" if safety
                         else f"{np.median(v):>11.1f}")
        print(f"{label:<34}{len(sub):>5}" + "".join(cells))

    v01 = [d for d in records if d["venue"] == "venue_01"]
    clips = [d for d in records if d["venue"].startswith("clipvenue_")]

    print("\nsafety - how often the boundary finds nobody "
          "(higher is safer on EMPTY, and these frames are empty):")
    print(f"{'frames':<34}{'n':>5}" + "".join(f"{a:>11}" for a in ARMS))
    report("venue_01 camera B, EMPTY", [d for d in v01 if d["class3"] == EMPTY], safety=True)

    print("\nrecall - median people found inside the boundary "
          "(these frames all have players on them):")
    print(f"{'frames':<34}{'n':>5}" + "".join(f"{a:>11}" for a in ARMS))
    report("venue_01 camera B, ACTIVE_PLAY", [d for d in v01 if d["class3"] == PLAY],
           safety=False)
    report("nine clip venues, ACTIVE_PLAY", clips, safety=False)

    c3 = [d for d in records if d["class3"] == C3]
    if c3:
        print(f"\nthe {len(c3)} recorded C3 frames, one at a time:")
        print(f"{'file':<40}{'detected':>10}" + "".join(f"{a:>11}" for a in ARMS))
        for d in c3:
            print(f"{Path(d['file']).name:<40}{d['detected']:>10}"
                  + "".join(f"{d[a]:>11}" for a in ARMS))

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
