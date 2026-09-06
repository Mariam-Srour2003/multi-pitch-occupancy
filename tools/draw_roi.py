"""Draw the ROI polygon (the pitch area) for each camera view.

For every camera folder in data/dataset/unlabeled/ (or an explicit image via --image),
shows a reference frame; left-click to add polygon points around the pitch,
right-click to remove the last point.

Keys:
    ENTER -> save polygon for this camera and go to the next
    r     -> reset points
    v     -> preview mask
    q     -> quit without saving current camera

Polygons are stored in config/cameras.json as normalized [0-1] coordinates so they
survive any resolution change:
    {"slot_20260711_1000_camA": {"roi": [[x, y], ...]}, ...}

Usage:
    python tools/draw_roi.py            # iterate over all camera folders
    python tools/draw_roi.py --image path/to/frame.jpg --name camA
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNLABELED = ROOT / "data" / "dataset" / "unlabeled"
CONFIG = ROOT / "config" / "cameras.json"

points: list[tuple[int, int]] = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        points.pop()


def edit_polygon(img, name: str, existing: list | None) -> list | None:
    """Returns normalized polygon or None if skipped."""
    global points
    h, w = img.shape[:2]
    points = [(int(px * w), int(py * h)) for px, py in existing] if existing else []

    win = f"ROI: {name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = img.copy()
        if len(points) >= 2:
            cv2.polylines(disp, [np.array(points)], False, (0, 255, 255), 2)
        for p in points:
            cv2.circle(disp, p, 5, (0, 0, 255), -1)
        cv2.rectangle(disp, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(disp, f"{name} | L-click add, R-click undo, ENTER save, r reset, v preview, q skip",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10):  # ENTER
            if len(points) < 3:
                print("  need at least 3 points")
                continue
            cv2.destroyWindow(win)
            return [[round(x / w, 4), round(y / h, 4)] for x, y in points]
        elif k == ord("r"):
            points = []
        elif k == ord("v") and len(points) >= 3:
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.array(points)], 255)
            preview = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow(win, preview)
            cv2.waitKey(0)
        elif k == ord("q") or k == 27:
            cv2.destroyWindow(win)
            return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="single image to draw on")
    ap.add_argument("--name", help="camera name for --image mode")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}

    tasks: list[tuple[str, Path]] = []
    if args.image:
        tasks.append((args.name or Path(args.image).stem, Path(args.image)))
    else:
        for cam_dir in sorted(p for p in UNLABELED.iterdir() if p.is_dir()):
            frames = sorted(cam_dir.glob("*.jpg"))
            if frames:
                tasks.append((cam_dir.name, frames[len(frames) // 2]))

    if not tasks:
        print("No camera folders found. Run tools/extract_frames.py first.")
        return

    for name, frame_path in tasks:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        existing = cfg.get(name, {}).get("roi")
        print(f"Editing ROI for {name} ({'existing' if existing else 'new'})")
        poly = edit_polygon(img, name, existing)
        if poly:
            cfg.setdefault(name, {})["roi"] = poly
            CONFIG.parent.mkdir(exist_ok=True)
            CONFIG.write_text(json.dumps(cfg, indent=2))
            print(f"  saved ({len(poly)} points) -> {CONFIG}")

    print("Done.")


if __name__ == "__main__":
    main()
