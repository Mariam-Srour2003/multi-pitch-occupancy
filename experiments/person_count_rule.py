"""Count people inside the boundary and decide from that, instead of from the probe.

The proposal: nobody on the pitch is EMPTY; a few people are present-but-not-playing; a
crowd is a match. It is the rule a person would write, and until now the reason not to try it
was `synthetic_data_protocol.md` §3a - `redact_people` returned **zero detections on a frame
with four visible people**, so a "no people means empty" rule would have called everything
empty.

That measurement was on venue_01 frames where the people are in the dugout, small and
partly occluded. On the unseen clip the same detector finds the walker and reports **zero on
all thirteen empty minutes**. So the question is open again and worth measuring properly:
how well does a person count actually separate the classes on *recorded* frames with labels?

Two things this does that the earlier measurement did not:

- **Counts only inside the pitch boundary**, using the box's foot point rather than its
  centre, since a person is standing where their feet are. Spectators behind a fence are not
  people on the pitch, which is the whole reason the boundary exists.
- **Reports the count distribution per class**, not just an accuracy. A rule needs a
  threshold, and a threshold needs to be read off a distribution rather than guessed.

    uv run python experiments/person_count_rule.py --backbone dinov2
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision import roi

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
EMPTY, PLAY, C3 = "C1_EMPTY", "C2_ACTIVE_PLAY", "C3_MAINTENANCE_NON_SPORTING"


def count_inside(model, frame, polygon, *, imgsz: int, conf: float) -> tuple[int, int]:
    """(total detected, detected standing inside the boundary)."""
    res = model.predict(frame, verbose=False, conf=conf, classes=[0], imgsz=imgsz)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    if polygon is None:
        return len(boxes), len(boxes)
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array([[int(x * w), int(y * h)] for x, y in polygon], np.int32)], 1)
    inside = 0
    for b in boxes:
        fx, fy = int((b[0] + b[2]) / 2), int(b[3])  # feet
        if 0 <= fy < h and 0 <= fx < w and mask[fy, fx]:
            inside += 1
    return len(boxes), inside


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="cap frames per class, 0 = all")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    rows = [r for r in read_manifest(DATASET / "manifest.csv") if r.source != "synthetic"]

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    test = [r for r in rows if r.venue == "venue_01" and cam(r) == "camera_B"]
    if args.limit:
        by: dict[str, list] = {}
        for r in test:
            by.setdefault(r.class3, []).append(r)
        test = [r for v in by.values() for r in v[: args.limit]]

    print(f"{len(test)} recorded frames from venue_01 camera B "
          f"({Counter(r.class3 for r in test)})")
    print(f"detector yolov8n, imgsz={args.imgsz}, conf={args.conf}, counted inside the "
          f"camera's boundary\n")

    counts: dict[str, list[int]] = {}
    records = []
    for n, r in enumerate(test, 1):
        frame = cv2.imread(str(DATASET / r.file))
        if frame is None:
            continue
        total, inside = count_inside(model, frame, roi.get(r.camera),
                                     imgsz=args.imgsz, conf=args.conf)
        counts.setdefault(r.class3, []).append(inside)
        records.append({"file": r.file, "class3": r.class3, "detected": total,
                        "inside": inside})
        if n % 100 == 0:
            print(f"  {n}/{len(test)}", end="\r", flush=True)
    print()

    print(f"{'class':<32}{'n':>6}{'median':>8}{'mean':>8}{'zero':>8}{'1-4':>7}{'>=5':>7}")
    for cls in (EMPTY, PLAY, C3):
        v = np.array(counts.get(cls, []))
        if not len(v):
            continue
        print(f"{cls:<32}{len(v):>6}{np.median(v):>8.1f}{v.mean():>8.2f}"
              f"{(v == 0).mean():>8.0%}{((v >= 1) & (v <= 4)).mean():>7.0%}"
              f"{(v >= 5).mean():>7.0%}")

    # The rule, read off those distributions rather than guessed.
    e = np.array(counts.get(EMPTY, []))
    p = np.array(counts.get(PLAY, []))
    if len(e) and len(p):
        print("\nthe rule 'zero inside the boundary means EMPTY':")
        print(f"  correct on EMPTY frames      {(e == 0).mean():.3f}  ({(e == 0).sum()}/{len(e)})")
        print(f"  wrongly EMPTY on PLAY frames {(p == 0).mean():.3f}  ({(p == 0).sum()}/{len(p)})")
        best, best_bal = None, -2.0
        for thr in range(1, 12):
            rec = float((p >= thr).mean())
            fp = float((e >= thr).mean())
            if rec - fp > best_bal:
                best, best_bal = thr, rec - fp
        rec = float((p >= best).mean())
        fp = float((e >= best).mean())
        print(f"\nbest 'PLAY if >= k people': k={best}  "
              f"recall {rec:.3f}, false-play {fp:.3f}, balanced {rec - fp:+.3f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "person_count_rule.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "class3", "detected", "inside"])
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
