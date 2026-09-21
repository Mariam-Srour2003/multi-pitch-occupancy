"""Rebuild the manifest rows for generated frames (2026-09-21 recovery).

A rebuild of `manifest.csv` during the folder collapse dropped all 200 generated frames:
`build_manifest` cannot parse `syn_<batch>_<n>.jpg`, reported each as a problem and wrote the
file without them. This puts them back from what survived, and says plainly which field could
not be recovered.

Recovered exactly, because each is a function of something still on disk:

    file        the frame itself
    label       the folder it sits in
    class3      taxonomy.to_class3 of that folder
    camera      `synthetic_<batch>`, from the filename
    slot_id     the same - a group of its own, so a split cannot put a generated frame and
                its conditioning frame on opposite sides (ingest_synthetic.__doc__)
    t_s         the index in the filename
    source      "synthetic"; labeled_by "synthetic"; split_role "train"
    lighting    "unknown" - `ingest_synthetic.classify_lighting` returns that for every
                frame by design, so there is nothing to recover
    venue       `scene_ids.csv`, which was written after ingestion and still has all 189
                of the frames it covers

**`quality` is the one field that is genuinely lost, for 106 of the 200 frames.** It recorded
the per-frame defect triage. `scripts/synthetic_defects_gemini01.csv` survives for the
83-frame `gemini01` batch and is read here; four weather variants name their defect in the
filename (`cvf_e_fog`, `cvf_e_rainH`, `cvf_e_rainL`). For the rest this writes
``synthetic:unrecorded`` rather than ``synthetic:ok``, because the two are different claims
and only one of them is true. `results/coverage.md` records what the distribution was before
the loss: 154 plain `ok` against 35 carrying a defect, of which ten were `frozen_timestamp`
flags on EMPTY frames that cannot now be attributed to individual files.

    uv run python scripts/rebuild_synthetic_rows.py --apply
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from pitch_occupancy.data.manifest import ManifestRow, read_manifest, write_manifest
from pitch_occupancy.data.taxonomy import Label, to_class3

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
SYN_RE = re.compile(r"^syn_(?P<batch>.+)_(?P<idx>\d+)\.jpg$")

#: Defect names the filename itself carries, for batches generated as weather variants.
FROM_NAME = {"cvf_e_fog": "ok;frozen_timestamp;fog",
             "cvf_e_rainH": "ok;frozen_timestamp;rain_heavy",
             "cvf_e_rainL": "ok;frozen_timestamp;rain_light",
             "cvf_e_dry": "ok;frozen_timestamp"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    manifest = DATASET / "manifest.csv"
    rows = read_manifest(manifest)
    have = {r.file for r in rows}
    venues = {r["file"]: r["venue"]
              for r in csv.DictReader((DATASET / "scene_ids.csv").open(encoding="utf-8"))}
    defects = {int(r["idx"]): r["defects"] for r in csv.DictReader(
        (ROOT / "scripts" / "synthetic_defects_gemini01.csv").open(encoding="utf-8"))}

    added, no_venue, unrecorded = [], [], 0
    for folder in sorted(DATASET.glob("[0-9]_*")):
        label = Label.parse(folder.name)
        for frame in sorted(folder.glob("syn_*.jpg")):
            rel = f"{folder.name}/{frame.name}"
            if rel in have:
                continue
            m = SYN_RE.match(frame.name)
            if m is None:
                continue
            batch, idx = m["batch"], int(m["idx"])
            venue = venues.get(rel)
            if venue is None:
                no_venue.append(rel)
                continue
            if batch == "gemini01":
                quality = "synthetic:" + (defects.get(idx) or "ok")
            elif batch in FROM_NAME:
                quality = "synthetic:" + FROM_NAME[batch]
            else:
                quality = "synthetic:unrecorded"
                unrecorded += 1
            added.append(ManifestRow(
                file=rel, label=label.value, class3=to_class3(label).value, venue=venue,
                camera=f"synthetic_{batch}", slot_date="", slot_time="",
                slot_id=f"synthetic_{batch}", t_s=idx, source="synthetic",
                labeled_by="synthetic", lighting="unknown", quality=quality,
                split_role="train"))

    print(f"{len(rows)} rows in the manifest, {len(added)} generated rows to restore")
    print(f"  quality recovered for {len(added) - unrecorded}, "
          f"unrecorded for {unrecorded}")
    if no_venue:
        print(f"  {len(no_venue)} frame(s) have no scene_ids entry and no venue - NOT "
              f"restored, they would need one invented: {no_venue[:4]}")
    if args.apply and added:
        write_manifest(rows + added, manifest)
        print(f"\nwrote {len(rows) + len(added)} rows to {manifest}")
    elif not args.apply:
        print("\ndry run - pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
