"""Ingest generated frames into the corpus (A13).

Generated frames are a **training-side-only** augmentation. This script does three things
the eye cannot be trusted to do consistently across 83 files:

1. copies them in under a name that can never be mistaken for a recording,
2. writes manifest rows carrying ``source=synthetic`` - the value ``splits.check_split``
   refuses on a test side - and the venue of the frame that conditioned them, never a new one,
3. records, per frame, the defects found by inspection, in the ``quality`` column, so a
   frame with a burned-in CCTV timestamp cannot quietly become training data.

The timestamp defect is the one that matters. Real frames in this corpus carry no overlay,
so "has a timestamp" would predict "maintenance" perfectly, and a frozen backbone would find
that before it found anything about grass. It is the exact failure this thesis is about.

Usage::

    python scripts/ingest_synthetic.py --src "<folder of images>" --batch gemini01 \
        [--defects scripts/synthetic_defects_gemini01.csv] [--apply]

Dry run by default: it prints what it would write and changes nothing.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MANIFEST = PROCESSED / "manifest.csv"
LABELS = PROCESSED / "labels.csv"

#: Every recorded frame in this corpus is 1920x1080. Generated frames arrive at whatever the
#: image model felt like (1356-1408 x 768 in batch gemini01, varying *within* the batch), and
#: a resolution that correlates with a class is a shortcut feature like any other. Normalising
#: is an alteration, so it is recorded here rather than done silently.
TARGET_SIZE = (1920, 1080)

def classify_lighting(_path: Path) -> str:
    """Always ``"unknown"``, and that is the finding rather than a shortcut.

    Three discriminators were calibrated against the 1,692 labelled frames before this was
    written - mean brightness, median brightness of the sky strip, and the fraction of
    near-black pixels. The best scored **0.76 balanced accuracy**, and two of the three
    *reverse sign* on real data: day frames here hold more deep shadow than floodlit night
    frames (dark-pixel fraction 0.035 vs 0.0001), because the night recording is evenly lit
    and the day one is full of hard shadow under the structures.

    `MAINTENANCE x night` is the empty cell this batch is meant to fill, so a guessed
    lighting value would not be a small inaccuracy - it would be the answer to the question.
    These frames need a human pass; until then the column says so.
    """
    return "unknown"


def load_defects(path: Path | None) -> dict[int, str]:
    """``{index: "defect;defect"}`` from a CSV of ``idx,defects``.

    Not passing ``--defects`` means no defects. **Passing one that does not exist is an
    error**, where it used to mean the same thing as not passing one at all - so a mistyped
    path, or a value that was never a path (``--defects "rain_heavy"`` reads as a filename
    once argparse casts it), recorded every frame as clean and said nothing. Defect flags are
    the only record that a generated frame carries a known problem; losing them silently is
    the one failure this column cannot afford.
    """
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(
            f"--defects {path} does not exist. It takes a CSV of `idx,defects`, not a "
            f"defect string. Write the file, or omit the flag to record no defects."
        )
    out: dict[int, str] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[int(row["idx"])] = row["defects"].strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="folder of generated images")
    ap.add_argument("--batch", required=True, help="batch id, e.g. gemini01")
    ap.add_argument("--class4", default="4_maintenance")
    ap.add_argument("--class3", default="C3_MAINTENANCE_NON_SPORTING")
    ap.add_argument("--venue", default="venue_01",
                    help="venue of the CONDITIONING frame. A13 forbids inventing a new one.")
    ap.add_argument("--defects", type=Path, default=None)
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry run")
    args = ap.parse_args()

    files = sorted(p for p in args.src.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        print(f"no images in {args.src}")
        return 1

    defects = load_defects(args.defects)
    dest_dir = PROCESSED / args.class4
    rows, clean, flagged = [], 0, 0

    for i, src in enumerate(files, 1):
        name = f"syn_{args.batch}_{i:03d}.jpg"
        lighting = classify_lighting(src)
        defect = defects.get(i, "")
        quality = f"synthetic:{defect}" if defect else "synthetic:ok"
        if defect:
            flagged += 1
        else:
            clean += 1
        rows.append({
            "file": f"{args.class4}/{name}",
            "class4": args.class4,
            "class3": args.class3,
            "venue": args.venue,
            "camera": f"synthetic_{args.batch}",
            "slot_date": "",
            "slot_time": "",
            # A group key of its own, so a grouped split can never put a generated frame
            # and its own conditioning frame on opposite sides.
            "slot_id": f"synthetic_{args.batch}",
            "t_s": i,
            "source": "synthetic",
            "labeled_by": "synthetic",
            "lighting": lighting,
            "quality": quality,
            "split_role": "train",
        })
        if args.apply:
            dest_dir.mkdir(parents=True, exist_ok=True)
            Image.open(src).convert("RGB").resize(TARGET_SIZE).save(
                dest_dir / name, quality=92)

    print(f"{len(rows)} frame(s): {clean} clean, {flagged} carrying a recorded defect")
    for d in sorted({r["quality"] for r in rows}):
        print(f"  {d}: {sum(1 for r in rows if r['quality'] == d)}")
    print("  lighting: unknown for all - see classify_lighting.__doc__; "
          "MAINTENANCE x night stays unfilled until a human pass")

    if not args.apply:
        print("\ndry run - nothing written. Re-run with --apply.")
        return 0

    with MANIFEST.open(encoding="utf-8") as fh:
        fieldnames = next(csv.reader(fh))
    with MANIFEST.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writerows(rows)
    with LABELS.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for r in rows:
            w.writerow([r["file"], r["class4"], "2026-09-13T00:00:00"])
    print(f"\nwrote {len(rows)} rows to {MANIFEST.name} and {LABELS.name}")
    print(f"copied images to {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
