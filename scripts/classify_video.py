"""Run the deployed classifier over a video, with a boundary, and report what it does.

Built to answer a specific complaint: a frame of an empty pitch came back `C2_ACTIVE_PLAY`
at confidence 1.000, with zero people detected and nothing moving.

Reports, per sampled frame:

- the verdict and its confidence
- the **motion** cue against the previous sample, so "nothing moved" is a number
- how much of the frame the boundary keeps

and a strip of thumbnails so the verdicts can be read against the pictures rather than
trusted.

    uv run python scripts/classify_video.py VIDEO --polygon cam2 --every 5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.vision import roi

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--polygon", default="cam2",
                    help="key in configs/roi.json, or 'none'")
    ap.add_argument("--every", type=float, default=5.0, help="seconds between samples")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=RESULTS / "video_verdicts.csv")
    ap.add_argument("--strip", type=Path, default=None)
    args = ap.parse_args()

    polygon = None
    if args.polygon == "derive":
        # The outline measured from this video's own frames. A boundary belongs to a camera,
        # and a stored one drawn for a different camera is not a boundary for this footage -
        # it is an arbitrary polygon that happens to be the right shape for somewhere else.
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        from derive_roi import polygon_from_mask, turf_mask

        cap0 = cv2.VideoCapture(str(args.video))
        grabbed, k = [], 0
        while len(grabbed) < 40:
            ok, fr = cap0.read()
            if not ok:
                break
            if k % 40 == 0:
                grabbed.append(cv2.resize(fr, (320, 180)))
            k += 1
        cap0.release()
        med = np.median(np.stack(grabbed), axis=0).astype(np.uint8)
        polygon = polygon_from_mask(turf_mask(med))
        if polygon is None:
            raise SystemExit("could not derive a boundary from this video")
    elif args.polygon != "none":
        polygon = roi.get(args.polygon)
        if polygon is None:
            raise SystemExit(f"no boundary named {args.polygon!r} in configs/roi.json; "
                             f"have {sorted(roi.load_all())}")
    print(f"boundary {args.polygon}: "
          + (f"{len(polygon)} points, keeps {roi.coverage(polygon):.1%} of the frame"
             if polygon else "none - the whole frame reaches the model"))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(args.every * fps)))
    frames, times = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(frame)
            times.append(i / fps)
        i += 1
    cap.release()
    print(f"{i} frames at {fps:.1f} fps -> {len(frames)} sampled every {args.every:g}s")

    from pitch_occupancy.vision.classifier import load_classifier

    clf = load_classifier(args.model)
    verdicts = clf.classify_batch(frames, polygon=polygon)

    # Motion against the previous sample, the same cue `motion_cue_probe` measured.
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
             for f in frames]
    motion = [float("nan")] + [float(np.abs(b - a).mean())
                               for a, b in zip(small, small[1:])]

    rows = []
    print(f"\n{'t':>8}{'verdict':>26}{'conf':>8}{'motion':>9}")
    for t, (cls, conf), m in zip(times, verdicts, motion):
        name = cls.name if hasattr(cls, "name") else str(cls)
        print(f"{t:>8.1f}{name:>26}{conf:>8.3f}"
              + (f"{m:>9.3f}" if m == m else f"{'-':>9}"))
        rows.append({"t_s": round(t, 1), "verdict": name, "confidence": round(conf, 4),
                     "motion": round(m, 4) if m == m else "",
                     "polygon": args.polygon})

    from collections import Counter
    tally = Counter(r["verdict"] for r in rows)
    print("\n" + "  ".join(f"{k}={v}" for k, v in tally.most_common()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    if args.strip:
        cols = 6
        tw, th = 300, 170
        n = len(frames)
        sheet = np.zeros((((n - 1) // cols + 1) * (th + 20), cols * tw, 3), np.uint8)
        for k, (f, (cls, conf), t) in enumerate(zip(frames, verdicts, times)):
            im = cv2.resize(f, (tw, th))
            if polygon:
                pts = np.array([[int(x * tw), int(y * th)] for x, y in polygon], np.int32)
                cv2.polylines(im, [pts], True, (0, 255, 255), 1)
            name = cls.name if hasattr(cls, "name") else str(cls)
            x, y = (k % cols) * tw, (k // cols) * (th + 20)
            sheet[y + 20:y + 20 + th, x:x + tw] = im
            colour = (0, 0, 255) if "ACTIVE" in name else (0, 255, 0)
            cv2.putText(sheet, f"{t:.0f}s {name[:18]} {conf:.2f}", (x + 3, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)
        cv2.imwrite(str(args.strip), sheet)
        print(f"wrote {args.strip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
