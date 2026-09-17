"""Median empty pitches from the whole clip, not from six sampled frames (A29).

`make_median_empties.py` medians the frames already in `data/processed/`, which is **six per
clip** - the sampling rate the dataset was built at. Six frames of a ten-second highlight is a
thin stack: a player who is slow, or who happens to be near where another player was, survives
the median. That is why three checks were needed and why only **5 frames from one venue**
survived them (A28).

The source clips hold **250-350 frames each**. A median over all of them is the same operation
with fifty times the evidence, and a player would have to stand still for ten seconds to
survive it. Nothing else changes: same subtraction, same real pixels, same exposure.

This reads `configs/clip_venues.csv` and the raw clips, and applies the same three checks
`make_median_empties.py` uses, including the person detector that asks directly whether the
people went.

**The locked final test venues are excluded by name**, as everywhere else. A median built from
a test venue's pixels is training data derived from the test set.

    uv run python scripts/median_empties_from_clips.py --clips DIR
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.splits import load_final_venues

ROOT = Path(__file__).resolve().parents[1]
VENUE_MAP = ROOT / "configs" / "clip_venues.csv"

#: The same two pixel checks `make_median_empties.py` uses, for the same reasons. They are
#: looser here only in that a 300-frame stack makes both easier to pass honestly.
MAX_RESIDUAL = 3.0
MIN_SPREAD = 3.5

#: Cap on frames read per clip. A ten-second clip at 25 fps is 250; reading every frame of a
#: 14-second one costs nothing and the median is over whatever arrives.
MAX_FRAMES = 400

#: Work size for the median itself. Full resolution would be 300 x 1080p frames in memory per
#: clip; the median is computed at this size and the output is written at it.
WORK = (960, 540)


def _small(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)


def load_map(path: Path) -> dict[str, tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["file"]: (r["venue"], r["venue_code"]) for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=Path, required=True, help="directory of source .mp4 clips")
    ap.add_argument("--out", type=Path, default=ROOT / "data/interim/median_empties_full")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = ap.parse_args()

    venues = load_map(VENUE_MAP)
    locked = load_final_venues()
    print(f"excluded the locked final test set: {', '.join(sorted(locked))}")

    args.out.mkdir(parents=True, exist_ok=True)
    made, refused = [], []
    per_venue: dict[str, int] = defaultdict(int)

    from pitch_occupancy.vision import roi
    from pitch_occupancy.vision.people import detect_inside

    clips = sorted(p for p in args.clips.glob("*.mp4") if p.name in venues)
    print(f"{len(clips)} clips with a venue assignment\n")
    for n, path in enumerate(clips, 1):
        venue, code = venues[path.name]
        if venue in locked:
            refused.append((venue, path.name, "locked final test venue"))
            continue
        cap = cv2.VideoCapture(str(path))
        frames = []
        while len(frames) < args.max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(fr, WORK))
        cap.release()
        if len(frames) < 30:
            refused.append((venue, path.name, f"only {len(frames)} frames readable"))
            continue

        med = np.median(np.stack(frames), axis=0).astype(np.uint8)
        ms = _small(med)
        diffs = [float(np.abs(_small(f) - ms).mean()) for f in frames]
        residual, spread = min(diffs), max(diffs)
        if residual > MAX_RESIDUAL:
            refused.append((venue, path.name, f"residual {residual:.2f}"))
            continue
        if spread < MIN_SPREAD:
            refused.append((venue, path.name, f"spread {spread:.2f} - nothing moved"))
            continue

        # The clip id is the camera tag the manifest uses, so the derived boundary is the one
        # this venue's frames were evaluated under.
        camera = f"{code}_{path.stem.rsplit('-', 1)[-1]}"
        counted = detect_inside(med, roi.get(camera))
        if counted is not None and counted.people:
            refused.append((venue, path.name,
                            f"detector finds {counted.people} inside the boundary"))
            continue

        name = f"medfull_{camera}.jpg"
        cv2.imwrite(str(args.out / name), med, [cv2.IMWRITE_JPEG_QUALITY, 92])
        made.append({"file": name, "venue": venue, "camera": camera,
                     "n_frames": len(frames), "residual": round(residual, 3),
                     "spread": round(spread, 3), "lighting": "night"})
        per_venue[venue] += 1
        if n % 10 == 0:
            print(f"  {n}/{len(clips)}", end="\r", flush=True)
    print(" " * 30, end="\r")

    with (args.out / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "venue", "camera", "n_frames",
                                           "residual", "spread", "lighting"])
        w.writeheader()
        w.writerows(made)

    print(f"made {len(made)} median frames from {len(clips)} clips")
    for v, k in sorted(per_venue.items()):
        print(f"  {v:<32}{k:>4}")
    print(f"\nrefused {len(refused)}:")
    why: dict[str, int] = defaultdict(int)
    for _v, _f, reason in refused:
        why[reason.split(" ")[0]] += 1
    for reason, k in sorted(why.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<20}{k:>4}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
