"""Group frames into scenes, so a count of frames stops being read as a count of evidence.

1,692 recorded frames carry **179 distinct scenes**. The EMPTY class is 494 frames and **5**
scenes. Training on all of them tells the fit that those five backgrounds *are* what an empty
pitch looks like, with the confidence a hundred observations would justify - and refitting on
one frame per scene takes false-play on unseen footage from 0.31 to 0.00.

So scenes have to be addressable. This writes `data/processed/scene_ids.csv`, one row per
frame, and `splits.distinct_rows` reads it.

**A sidecar rather than a manifest column, deliberately.** `build_manifest` regenerates
`manifest.csv` from the filenames on disk, and the 189 generated rows were written into it by
`ingest_synthetic.py` - a regeneration would drop them. Adding a field that only a
regeneration can populate would make the two files disagree the first time anyone ran it.

Frames are grouped **within (venue, class)**, never across. Two venues that happen to hash
alike are still two venues, and collapsing them would discard the only cross-venue variation
the corpus has. Two classes that hash alike are a labelling problem, not a duplicate.

    uv run python scripts/assign_scene_ids.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from pitch_occupancy.data.manifest import read_manifest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"

#: Hamming distance between dHashes below which two frames are the same scene. 5 of 64 bits.
#: `dedup.py` uses the same figure for the effective-sample audit, and the two numbers must
#: agree or "distinct scenes" means one thing in the training set and another in the report.
THRESHOLD = 5


def dhash(path: Path, size: int = 8) -> int:
    im = np.asarray(Image.open(path).convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (im[:, 1:] > im[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATASET / "scene_ids.csv")
    args = ap.parse_args()

    rows = read_manifest(DATASET / "manifest.csv")
    print(f"hashing {len(rows)} frames ...")
    h = {r.file: dhash(DATASET / r.file) for r in rows}

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        groups[(r.venue, r.class3)].append(r)

    out: list[dict[str, object]] = []
    n_scenes = 0
    for (venue, cls), members in sorted(groups.items()):
        # Greedy single-link over the hash: the first frame opens a scene, each later frame
        # joins the first scene it is close to, or opens its own. Order-dependent by nature,
        # so the manifest order is used and recorded rather than a shuffle.
        reps: list[tuple[int, str]] = []
        for r in members:
            code = h[r.file]
            joined = None
            for rep_code, scene in reps:
                if bin(rep_code ^ code).count("1") <= THRESHOLD:
                    joined = scene
                    break
            if joined is None:
                n_scenes += 1
                joined = f"s{n_scenes:04d}"
                reps.append((code, joined))
            out.append({"file": r.file, "scene_id": joined, "venue": venue, "class3": cls})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "scene_id", "venue", "class3"])
        w.writeheader()
        w.writerows(out)

    per_class: dict[str, set] = defaultdict(set)
    for row in out:
        per_class[str(row["class3"])].add(row["scene_id"])
    print(f"\n{len(rows)} frames -> {n_scenes} scenes ({n_scenes / len(rows):.0%})")
    for cls, scenes in sorted(per_class.items()):
        n = sum(1 for row in out if row["class3"] == cls)
        print(f"  {cls:<32}{n:>6} frames{len(scenes):>7} scenes")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
