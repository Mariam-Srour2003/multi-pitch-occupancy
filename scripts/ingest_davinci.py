"""Ingest the operator's DaVinci exports into the labelled dataset (2026-09-21).

Fifteen short exports - thirteen clips and two stills - labelled by their filenames and
assigned to venues in `configs/davinci_venues.csv`. They matter because **C3 has almost no
real data**: before this the class held 6 recorded frames and 158 generated ones, so every
number about it was about pictures a model drew.

**Venue is read from the config, not invented from the folder.** Thirteen of the fifteen are
venues the corpus already has - `playing day 3` is `clipvenue_g_netting`, `playing day` is
`clipvenue_b_floodlit_track`, six of them are `venue_01`. Filing them under a new name would
put the same camera on both sides of a leave-one-venue-out fold, which is the failure this
project exists to avoid. The config records the evidence for every identification.

**Frames are sampled by difference, not by clock.** These are four-second exports at 24 fps;
sampling evenly would give six near-identical frames of one moment and count them as six
observations. Each candidate is kept only if its dHash is at least ``--min-distance`` from
every frame already kept from that clip, so a clip of people standing still yields one or two
frames and a clip with a game in it yields more. What a clip actually contributed is printed
and written to the sidecar.

**One clip per class is held out and never extracted.** `split_role=test` in the config marks
them; their frames are not written at all, so they cannot reach a training set by accident.
They stay in `data/raw/` as whole videos, which is what a holdout should be.

    uv run python scripts/ingest_davinci.py --apply
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import cv2

from pitch_occupancy.data.dedup import dhash, hamming
from pitch_occupancy.data.splits import load_final_venues
from pitch_occupancy.data.taxonomy import Class3, Label

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "davinci_2026-09-21"
CONFIG = ROOT / "configs" / "davinci_venues.csv"
SOURCES = (Path(r"C:\Users\maria\Downloads\davinci\davinci"),
           Path(r"C:\Users\maria\Downloads\Davinci new\Davinci new"))

#: Venues held back for the final evaluation, read from the same place every
#: split reads it. Frames are never extracted into them.
LOCKED: frozenset[str] = frozenset()

FOLDER = {Class3.EMPTY: Label.EMPTY, Class3.ACTIVE_PLAY: Label.PLAYING,
          Class3.MAINTENANCE_NON_SPORTING: Label.MAINTENANCE_NON_SPORTING}

#: Short venue codes for the frame names, matching `clip_<code>_<id>_t<ms>.jpg` in
#: `data.extract` so the existing manifest parser reads these frames with no change.
CODE = {"venue_01": "dv01", "clipvenue_a_blue_barrier": "dvca",
        "clipvenue_b_floodlit_track": "dvcb", "clipvenue_f_outdoor_bldg": "dvcf",
        "clipvenue_g_netting": "dvcg", "clipvenue_h_teal_pitch": "dvch",
        "davinci_j_maint_outdoor": "dvdj", "davinci_l_city_pitch": "dvdl"}


def read_config() -> list[dict]:
    lines = [ln for ln in CONFIG.read_text(encoding="utf-8").splitlines()
             if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def find_source(name: str) -> Path | None:
    for folder in SOURCES:
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def sample(path: Path, *, min_distance: int, cap_frames: int) -> list[tuple[int, "cv2.Mat"]]:
    """Frames from one clip that differ from each other, newest-first by position."""
    if path.suffix.lower() != ".mp4":
        image = cv2.imread(str(path))
        return [] if image is None else [(0, image)]
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    kept: list[tuple[int, "cv2.Mat"]] = []
    hashes: list[int] = []
    step = max(1, int(fps / 4))          # look four times a second
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        h = dhash(frame)
        if all(hamming(h, seen) >= min_distance for seen in hashes):
            hashes.append(h)
            kept.append((int(i * 1000 / fps), frame))
        if len(kept) >= cap_frames:
            break
    cap.release()
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-distance", type=int, default=6,
                    help="dHash bits a frame must differ by from every frame already kept "
                         "from the same clip (default 6, one above the scene threshold of 5)")
    ap.add_argument("--max-per-clip", type=int, default=8)
    args = ap.parse_args()

    rows = read_config()
    global LOCKED
    LOCKED = load_final_venues()
    sidecar_rows: list[dict] = []
    per_class: Counter = Counter()
    missing = []
    print(f"{'clip':<30}{'venue':<28}{'class':<6}{'role':<6}{'kept':>5}")
    for row in rows:
        src = find_source(row["clip"])
        if src is None:
            missing.append(row["clip"])
            continue
        cls = Class3(row["class3"])
        folder = FOLDER[cls]
        if row["venue"] in LOCKED:
            # The locked venues are opened once, at the end, on footage nobody has looked
            # at. `playing day.mp4` is clipvenue_b_floodlit_track, and seven of its frames
            # went into the labelled set on 2026-09-21 before this guard existed - after I
            # had rendered the clip, gridded it and compared it frame by frame against that
            # venue to decide it *was* that venue. Inspected footage cannot be a held-out
            # test set, and `development_rows` dropping it from training does not undo that.
            # The source video stays in data/raw/; nothing is extracted from it.
            print(f"{row['clip'][:29]:<30}{row['venue']:<28}{cls.value[:2]:<6}"
                  f"{'LOCKED':<6}{'-':>5}  a final-test venue, not extracted")
            continue
        if args.apply:
            RAW.mkdir(parents=True, exist_ok=True)
            if not (RAW / src.name).exists():
                shutil.copy2(src, RAW / src.name)
        if row["split_role"] == "test":
            print(f"{row['clip'][:29]:<30}{row['venue']:<28}{cls.value[:2]:<6}"
                  f"{'TEST':<6}{'-':>5}  held out, not extracted")
            continue
        frames = sample(src, min_distance=args.min_distance, cap_frames=args.max_per_clip)
        code = CODE[row["venue"]]
        # md5, not hash(): Python randomises string hashing per process, so `hash()` here
        # would give every re-run a different filename for the same frame and quietly
        # duplicate the whole batch on disk.
        clip_id = f"{int(hashlib.md5(row['clip'].encode()).hexdigest()[:12], 16) % 10**10:010d}"
        for t_ms, image in frames:
            name = f"clip_{code}_{clip_id}_t{t_ms:06d}.jpg"
            sidecar_rows.append({
                "file": name, "venue": row["venue"], "venue_code": code,
                "clip_id": clip_id, "t_ms": t_ms,
                "brightness": round(float(image.mean()), 2),
                # From the operator's own filename. `night` is the default because every
                # ambiguous export here is floodlit; `configs/davinci_venues.csv` carries
                # the clip name, so a wrong call is visible rather than buried.
                "lighting": "day" if "day" in row["clip"].lower() else "night",
                "source_clip": row["clip"]})
            if args.apply:
                (DATASET / folder.value).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(DATASET / folder.value / name), image,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
        per_class[cls] += len(frames)
        print(f"{row['clip'][:29]:<30}{row['venue']:<28}{cls.value[:2]:<6}"
              f"{'dev':<6}{len(frames):>5}")

    if missing:
        print(f"\n{len(missing)} clip(s) not found: {missing}")
    print(f"\n{sum(per_class.values())} frame(s) from {len(sidecar_rows) and len(rows)} clips")
    for cls, n in sorted(per_class.items()):
        print(f"  {cls.value:<30}{n:>5}")
    if args.apply:
        # Appended to the sidecar `data.extract` writes and `manifest` reads, because a
        # second sidecar would be a second place for a venue to be recorded and a second
        # chance for the two to disagree. Keyed by filename, so a re-run replaces.
        from pitch_occupancy.data.extract import CLIP_SIDECAR

        fields = ["file", "venue", "venue_code", "clip_id", "t_ms", "brightness", "lighting"]
        existing: dict[str, dict] = {}
        if CLIP_SIDECAR.exists():
            with CLIP_SIDECAR.open(newline="", encoding="utf-8") as fh:
                existing = {r["file"]: r for r in csv.DictReader(fh)}
        for r in sidecar_rows:
            existing[r["file"]] = {k: r[k] for k in fields}
        CLIP_SIDECAR.parent.mkdir(parents=True, exist_ok=True)
        with CLIP_SIDECAR.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(existing.values())
        provenance = RAW / "provenance.csv"
        with provenance.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sidecar_rows[0]))
            w.writeheader()
            w.writerows(sidecar_rows)
        print(f"\n{len(sidecar_rows)} row(s) into {CLIP_SIDECAR}")
        print(f"sources copied to {RAW}, provenance in {provenance.name}")
        print("next: `uv run pitch manifest`, then scripts/assign_scene_ids.py")
    else:
        print("\ndry run - pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
