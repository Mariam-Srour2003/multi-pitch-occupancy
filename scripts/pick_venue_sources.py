"""Pick one source frame per development clip venue, for conditioning generated frames.

Two rules, and the first is the one that matters:

1. **Never a locked venue.** ``clipvenue_b_floodlit_track`` and ``clipvenue_c_teal_boards``
   are the final test set. Deriving training data from a frame of theirs would put the
   final test set's own pixels into training - the most damaging leak available here, and
   an invisible one, because the resulting frames would look like any other.
2. **Fewest people.** Every clip venue is active play, so a source frame has people in it.
   Picking the emptiest frame minimises both the redaction work and what leaves the project.

Prints the choice per venue and, with ``--stage``, copies them out for upload.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKED = {"clipvenue_b_floodlit_track", "clipvenue_c_teal_boards"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=Path, default=None, help="copy the chosen frames here")
    args = ap.parse_args()

    counts = {
        r["file"]: int(r["n_person"])
        for r in csv.DictReader(
            (ROOT / "data/interim/clip_person_counts.csv").open(encoding="utf-8")
        )
    }
    frames = list(
        csv.DictReader((ROOT / "data/interim/clip_frames.csv").open(encoding="utf-8"))
    )

    best: dict[str, tuple[int, str, str]] = {}
    for r in frames:
        venue = r["venue"]
        if venue in LOCKED:
            continue
        n = counts.get(r["file"], 999)
        if venue not in best or n < best[venue][0]:
            best[venue] = (n, r["file"], r["lighting"])

    base = ROOT / "data/interim/frames" / "clips"
    for venue, (n, f, lighting) in sorted(best.items()):
        src = base / f
        print(f"{venue:30s} people={n:3d}  {lighting:6s}  {f}  exists={src.exists()}")
        if args.stage and src.exists():
            args.stage.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, args.stage / f"SRC_{venue}.jpg")

    print(f"\nskipped (locked final test set): {', '.join(sorted(LOCKED))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
