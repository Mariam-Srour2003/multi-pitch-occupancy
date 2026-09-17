"""Manufacture EMPTY frames by temporal median, from real pixels and no generative model.

The finding that prompted this: generated EMPTY frames repaired the false-play collapse and
generated C3 frames did nothing, because the first were made by **subtraction** - people
removed from a real frame, leaving real pitch - and the second by addition, which teaches the
probe what a drawn person looks like.

A temporal median is subtraction without a model. Players move; the pitch does not. Take
several frames of one camera and the per-pixel median is the pitch with the people gone, in
the camera's own pixels, at the camera's own exposure. It is the same operation
`derive_roi.py` already relies on to find the turf.

**This is not a substitute for a recorded empty pitch and must not be reported as one.** What
it produces is *a real venue with nobody on it*, which is exactly the class the corpus lacks -
but every one is an average of frames that had people in them, so a median over too few, or
over frames where someone stood still, leaves a smear where a person was. That is why the
output is checked rather than trusted: `--min-frames` refuses a thin stack, the residual and
spread checks below reject a median that resembles no frame or that nothing moved in, and a
person detector then asks the question those two only approximate.

    uv run python scripts/make_median_empties.py --out data/interim/median_empties
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import load_final_venues

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"

#: A median over fewer frames than this is not a median, it is one of the frames. Six is the
#: length of a clip-venue clip, so this admits them while refusing anything thinner.
MIN_FRAMES = 6

#: Two checks, because one of them was the wrong question.
#:
#: `MAX_RESIDUAL` compares the median to the *closest* of its own members: a median that
#: resembles no frame at all is a smear. That check alone passed a frame from
#: `clipvenue_e_pink_boards` with two people plainly standing in it - because when nobody
#: moves, the median reproduces the crowd and is extremely close to every member. It was
#: measuring "does this look like a frame", not "did the people go".
#:
#: `MIN_SPREAD` asks the question that was missing: the *furthest* member must differ from the
#: median by at least this much, which is only true if something moved enough to be averaged
#: out. A stack of six frames of people standing still fails it, as it should.
MAX_RESIDUAL = 3.0
MIN_SPREAD = 3.5

#: And a third check, because the first two are proxies and this is the question (A28).
#:
#: Both thresholds above reason about pixel differences and infer from them whether the people
#: went. A detector answers it directly. Run on the 11 frames the two thresholds passed, it
#: found people inside the boundary on **four** of them - two people on two of the
#: `clipvenue_a` medians, one on a third, and **eight** on the `clipvenue_e` median, which is
#: the same venue whose survivors prompted `MIN_SPREAD` in the first place. Tightening a
#: threshold was never going to fix that; it was the wrong instrument.
#:
#: **What this costs.** A frame selected by the person detector cannot afterwards be used to
#: evaluate the person detector, or the gate built on it. It can evaluate a probe, and that is
#: what `median_empty_night.py` does with the output.
REJECT_IF_ANYONE_INSIDE = True


def _small(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/interim/median_empties")
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    args = ap.parse_args()

    # The locked final test set is never a source. A median built from its frames is training
    # data derived from the test set's own pixels - the same leak `scripts/derive_roi.py`
    # refuses by name, and it was not refused here until this script had already produced two
    # frames from `clipvenue_b_floodlit_track`.
    locked = load_final_venues()
    rows = [r for r in read_manifest(DATASET / "manifest.csv")
            if r.source != "synthetic" and r.venue not in locked]
    print(f"excluded the locked final test set: {', '.join(sorted(locked))}")
    # Grouped by camera *and* slot: a camera's appearance changes between a morning and a
    # floodlit night, and a median across both would be an average of two lightings that the
    # camera never produced.
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        groups[(r.venue, r.camera)].append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    made, refused = [], []
    for (venue, camera), members in sorted(groups.items()):
        if len(members) < args.min_frames:
            refused.append((venue, camera, f"only {len(members)} frames"))
            continue
        imgs = []
        for r in members[:60]:
            im = cv2.imread(str(DATASET / r.file))
            if im is not None:
                imgs.append(im)
        if len(imgs) < args.min_frames:
            refused.append((venue, camera, f"only {len(imgs)} readable"))
            continue
        h, w = imgs[0].shape[:2]
        stack = np.stack([cv2.resize(im, (w, h)) for im in imgs])
        med = np.median(stack, axis=0).astype(np.uint8)

        # The check: against the member closest to the median, not the mean of all of them.
        # A stack where everyone moved has one frame that already looks like the empty pitch;
        # a stack where someone stood still does not.
        ms = _small(med)
        diffs = [float(np.abs(_small(im) - ms).mean()) for im in imgs]
        residual, spread = min(diffs), max(diffs)
        if residual > MAX_RESIDUAL:
            refused.append((venue, camera, f"residual {residual:.2f} - median resembles no frame"))
            continue
        if spread < MIN_SPREAD:
            refused.append((venue, camera, f"spread {spread:.2f} - nothing moved, people survive"))
            continue

        # The direct question, after two proxies for it.
        if REJECT_IF_ANYONE_INSIDE:
            from pitch_occupancy.vision import roi
            from pitch_occupancy.vision.people import detect_inside

            counted = detect_inside(med, roi.get(camera))
            if counted is not None and counted.people:
                refused.append((venue, camera,
                                f"detector finds {counted.people} inside the boundary"))
                continue

        name = f"medempty_{camera}.jpg"
        cv2.imwrite(str(args.out / name), med, [cv2.IMWRITE_JPEG_QUALITY, 92])
        made.append({"file": name, "venue": venue, "camera": camera,
                     "n_frames": len(imgs), "residual": round(residual, 3),
                     "spread": round(spread, 3),
                     "lighting": members[0].lighting})

    with (args.out / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["file", "venue", "camera", "n_frames",
                                            "residual", "spread", "lighting"])
        w_.writeheader()
        w_.writerows(made)

    print(f"made {len(made)} median frames from {len(groups)} cameras")
    per_venue = defaultdict(int)
    for m in made:
        per_venue[m["venue"]] += 1
    for v, n in sorted(per_venue.items()):
        print(f"  {v:<30}{n:>4}")
    print(f"\nrefused {len(refused)}:")
    for v, c, why in refused[:8]:
        print(f"  {v:<30}{c:<26}{why}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
