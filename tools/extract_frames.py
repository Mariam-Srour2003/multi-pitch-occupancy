"""Extract frames from the StatBox replay videos for dataset building.

- Samples 1 frame every N seconds (default 15) from each video in data/.
- Additionally mines "high-motion" frames via MOG2 background subtraction so the
  PLAYING class gets diverse poses (--motion, on by default).
- Output: data/dataset/unlabeled/<camera_tag>/<camera_tag>_tXXXXXX.jpg
  where XXXXXX is the timestamp in seconds within the video (zero-padded).

Usage:
    python tools/extract_frames.py [--interval 15] [--no-motion] [--quality 90]
"""
import argparse
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "dataset" / "unlabeled"


def camera_tag(video_path: Path) -> str:
    """StatBox_Replay_<venue>_2026-07-11_10-00 (1).mp4 -> slot_20260711_1000_camB"""
    name = video_path.stem
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})", name)
    slot = f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}{m.group(5)}" if m else "unknown"
    cam = "camB" if "(1)" in name else "camA"
    return f"slot_{slot}_{cam}"


def extract(video_path: Path, interval_s: float, use_motion: bool, quality: int) -> tuple[int, int]:
    tag = camera_tag(video_path)
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  !! cannot open {video_path.name}", flush=True)
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps * interval_s)))

    mog = cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=False) if use_motion else None
    # do not save motion frames closer than this to any already-saved frame
    min_gap = int(round(fps * 5))
    last_saved = -min_gap
    n_regular = n_motion = 0

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        save_reason = None
        if idx % step == 0:
            save_reason = "regular"
        elif mog is not None and idx % int(fps) == 0:  # motion check once per second
            small = cv2.resize(frame, (480, 270))
            mask = mog.apply(small)
            motion_ratio = cv2.countNonZero(mask) / mask.size
            # sustained noticeable motion and not too close to the previous save
            if motion_ratio > 0.02 and (idx - last_saved) >= min_gap:
                save_reason = "motion"

        if save_reason:
            t = int(idx / fps)
            suffix = "" if save_reason == "regular" else "_m"
            out = out_dir / f"{tag}_t{t:06d}{suffix}.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            last_saved = idx
            if save_reason == "regular":
                n_regular += 1
            else:
                n_motion += 1

        idx += 1
        if idx % 20000 == 0:
            print(f"  {video_path.name}: {idx}/{total} frames scanned "
                  f"({n_regular} regular + {n_motion} motion saved)", flush=True)

    cap.release()
    print(f"  DONE {video_path.name}: {n_regular} regular + {n_motion} motion frames -> {out_dir}", flush=True)
    return n_regular, n_motion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=15.0, help="seconds between regular samples")
    ap.add_argument("--no-motion", action="store_true", help="disable motion-mined extra frames")
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()

    videos = sorted(DATA.glob("*.mp4"))
    if not videos:
        sys.exit(f"no .mp4 files found in {DATA}")

    print(f"Extracting from {len(videos)} videos (interval={args.interval}s, "
          f"motion={'off' if args.no_motion else 'on'})", flush=True)
    tot_r = tot_m = 0
    for v in videos:
        r, m = extract(v, args.interval, not args.no_motion, args.quality)
        tot_r += r
        tot_m += m
    print(f"\nTOTAL: {tot_r} regular + {tot_m} motion frames in {OUT}", flush=True)


if __name__ == "__main__":
    main()
