"""Move `3_people_not_playing` and `4_maintenance` into one folder (2026-09-21).

The prediction path stopped distinguishing them in A40 because the corpus could not measure
the split - 6 real frames in one folder, 0 in the other. This finishes the job on disk, so
the folders a frame can be filed in are the classes the system reports, and `data/taxonomy`
stops carrying a four-to-three mapping that was one-to-one in every way that mattered.

**What moves and what does not.** Frame files move. Every sidecar that addresses a frame *by
path* is rewritten to match, because a path that points at nothing is not a record of
anything - `labels.csv`, `scene_ids.csv`, `results/hand_counts.csv` and the frozen-backbone
caches under `data/cache/`, which key their feature rows by filename and would silently drop
175 frames otherwise. The `label` column of `labels.csv` keeps the value the labeller
actually chose; `manifest.build` parses both names onto the same class, so an old value is
history rather than a mismatch.

**Nothing under `results/` that reports a measurement is touched.** Those CSVs record what
was true when they were written, and editing them to match today's vocabulary is how a
record stops being one. `hand_counts.csv` is the exception and is not a measurement: it is an
index of frames to be counted by hand, and an index of paths that do not exist is useless.

Idempotent: a second run finds nothing to do. Dry by default.

    uv run python scripts/collapse_label_folders.py --apply
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"

OLD = ("3_people_not_playing", "4_maintenance")
NEW = "3_maintenance_non_sporting"

#: Files that address frames by path and must follow them. Everything else under `results/`
#: is a measurement and is deliberately left as written.
SIDECARS = (
    DATASET / "labels.csv",
    DATASET / "scene_ids.csv",
    ROOT / "results" / "hand_counts.csv",
)


def rewrite_paths(text: str) -> tuple[str, int]:
    hits = sum(text.count(f"{old}/") for old in OLD)
    for old in OLD:
        text = text.replace(f"{old}/", f"{NEW}/")
    return text, hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write; otherwise report only")
    args = ap.parse_args()
    apply = args.apply
    dest = DATASET / NEW

    # --- the frames -------------------------------------------------------------------
    moved = clashes = 0
    for old in OLD:
        src = DATASET / old
        if not src.is_dir():
            continue
        existing = {p.name for p in dest.iterdir()} if dest.is_dir() else set()
        for frame in sorted(src.iterdir()):
            if frame.name in existing:
                print(f"  CLASH {old}/{frame.name} already in {NEW}/ - left in place")
                clashes += 1
                continue
            moved += 1
            if apply:
                dest.mkdir(exist_ok=True)
                shutil.move(str(frame), str(dest / frame.name))
        if apply and not clashes and src.is_dir() and not any(src.iterdir()):
            src.rmdir()
    print(f"frames: {moved} to move, {clashes} name clash(es)")

    # --- sidecars that address frames by path -------------------------------------------
    for path in SIDECARS:
        if not path.exists():
            print(f"  {path.relative_to(ROOT)}: absent")
            continue
        text = path.read_text(encoding="utf-8")
        new_text, hits = rewrite_paths(text)
        print(f"  {path.relative_to(ROOT)}: {hits} path(s)")
        if apply and hits:
            path.write_text(new_text, encoding="utf-8")

    # --- the feature caches, which key their rows by filename ---------------------------
    # Only the `files` array is touched and only its strings; the feature rows keep their
    # order, so row i still belongs to files[i]. Recomputing them instead would cost hours
    # of DINOv2 on 1,892 frames to arrive at identical numbers.
    for npz in sorted(CACHE.glob("*.npz")):
        with np.load(npz, allow_pickle=True) as z:
            keys = list(z.keys())
            if "files" not in keys:
                continue
            data = {k: z[k] for k in keys}
        files = np.array([str(f) for f in data["files"]])
        hits = int(sum(any(str(f).startswith(f"{old}/") for old in OLD) for f in files))
        if not hits:
            continue
        print(f"  {npz.relative_to(ROOT)}: {hits} of {len(files)} entries")
        if apply:
            data["files"] = np.array([rewrite_paths(str(f))[0] for f in files])
            np.savez(npz, **data)

    print("\napplied" if apply else "\ndry run - pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
