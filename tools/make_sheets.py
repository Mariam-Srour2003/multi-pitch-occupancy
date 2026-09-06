"""Build contact sheets (thumbnail grids) of unlabeled frames for fast visual review.

Usage: python tools/make_sheets.py <camera_folder_name> [--out DIR]
Each sheet: 8 cols x 6 rows = 48 thumbs (320x180), captioned with frame index + t-seconds.
Also writes <out>/<cam>_index.csv mapping index -> filename.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNLABELED = ROOT / "data" / "dataset" / "unlabeled"

COLS, ROWS = 8, 6
TW, TH = 320, 180
CAP = 22  # caption strip height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cam_dir = UNLABELED / args.cam
    frames = sorted(cam_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {cam_dir}")
    out_dir = Path(args.out) if args.out else ROOT / "results" / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / f"{args.cam}_index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "file"])
        for i, p in enumerate(frames):
            w.writerow([i, p.name])

    per = COLS * ROWS
    n_sheets = (len(frames) + per - 1) // per
    for s in range(n_sheets):
        sheet = np.full((ROWS * (TH + CAP), COLS * TW, 3), 30, np.uint8)
        for j in range(per):
            gi = s * per + j
            if gi >= len(frames):
                break
            img = cv2.imread(str(frames[gi]))
            if img is None:
                continue
            thumb = cv2.resize(img, (TW, TH))
            r, c = divmod(j, COLS)
            y = r * (TH + CAP)
            sheet[y:y + TH, c * TW:(c + 1) * TW] = thumb
            t = frames[gi].stem.split("_t")[-1]
            cv2.putText(sheet, f"#{gi} t{t}", (c * TW + 4, y + TH + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        out = out_dir / f"{args.cam}_sheet{s:02d}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(out)


if __name__ == "__main__":
    main()
