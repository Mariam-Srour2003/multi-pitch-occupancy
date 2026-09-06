"""Keyboard-driven frame labeling tool.

Shows each unlabeled frame in a window; press a key to file it into a class folder.
Progress is appended to data/dataset/labels.csv so you can stop and resume anytime.

Keys:
    1  -> 1_empty
    2  -> 2_playing            (match, warmup, drills, academy training)
    3  -> 3_people_not_playing (people inside ROI but idle / strolling / photos)
    4  -> 4_maintenance        (hi-vis staff, brooms, mowers, line painting)
    s  -> skip (leave in unlabeled, shown again next run)
    u  -> undo last label
    d  -> delete frame (blurry / broken)
    q  -> quit

Usage:
    python tools/label_tool.py [--shuffle]
"""
import argparse
import csv
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset"
UNLABELED = DATASET / "unlabeled"
LABELS_CSV = DATASET / "labels.csv"

CLASSES = {
    ord("1"): "1_empty",
    ord("2"): "2_playing",
    ord("3"): "3_people_not_playing",
    ord("4"): "4_maintenance",
}

HUD = "1=empty  2=playing  3=people-not-playing  4=maintenance  s=skip  u=undo  d=delete  q=quit"


def append_label(path_rel: str, label: str) -> None:
    new = not LABELS_CSV.exists()
    with LABELS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "label", "labeled_at"])
        w.writerow([path_rel, label, datetime.now().isoformat(timespec="seconds")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle", action="store_true", help="label in random order")
    args = ap.parse_args()

    frames = sorted(UNLABELED.rglob("*.jpg"))
    if args.shuffle:
        random.shuffle(frames)
    if not frames:
        print("No unlabeled frames. Run tools/extract_frames.py first.")
        return

    print(f"{len(frames)} frames to label.\n{HUD}")
    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("label", 1600, 900)

    history: list[tuple[Path, Path]] = []  # (new_path, original_path) for undo
    done = 0
    i = 0
    while i < len(frames):
        src = frames[i]
        if not src.exists():
            i += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            i += 1
            continue

        disp = img.copy()
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(disp, f"[{done} done / {len(frames) - i} left] {src.parent.name}/{src.name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(disp, (0, disp.shape[0] - 44), (disp.shape[1], disp.shape[0]), (0, 0, 0), -1)
        cv2.putText(disp, HUD, (10, disp.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("label", disp)

        k = cv2.waitKey(0) & 0xFF
        if k == ord("q") or k == 27:
            break
        elif k == ord("s"):
            i += 1
        elif k == ord("d"):
            src.unlink()
            i += 1
        elif k == ord("u") and history:
            moved_to, original = history.pop()
            shutil.move(str(moved_to), str(original))
            done -= 1
            # step back to re-show it
            try:
                i = frames.index(original)
            except ValueError:
                frames.insert(i, original)
        elif k in CLASSES:
            cls = CLASSES[k]
            dst_dir = DATASET / cls / src.parent.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            shutil.move(str(src), str(dst))
            append_label(str(dst.relative_to(DATASET)), cls)
            history.append((dst, src))
            done += 1
            i += 1

    cv2.destroyAllWindows()
    remaining = len(list(UNLABELED.rglob("*.jpg")))
    print(f"Labeled {done} this session. {remaining} still unlabeled.")


if __name__ == "__main__":
    main()
