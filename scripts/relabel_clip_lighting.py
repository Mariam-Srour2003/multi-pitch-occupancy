"""Correct the `lighting` column for the 396 recorded clip frames (A25, WP3-T5).

`data/extract.py` assigns lighting by **mean frame brightness**: below 80 is night, above is
day, "calibrated against the known day/night recordings in raw/venue_01". That calibration does
not survive leaving venue_01. A floodlit five-a-side pitch fills most of its frame with
intensely lit turf, so its mean brightness is *higher* than an overcast afternoon at venue_01 -
and the rule files night football as daylight. `rq_matrix.md` already records one instance,
`f_outdoor_bldg`, and parks the rest on this task.

**The audit, one venue at a time, on a rendered frame from the middle of the clip set.** The
evidence is stated per venue so the judgement can be disagreed with rather than trusted:

| venue | was | is | what the frame shows |
|---|---|---|---|
| a_blue_barrier | night | night | dark sky above the barrier; already correct |
| b_floodlit_track | **day** | night | black sky, floodlight fixtures visibly lit |
| c_teal_boards | **day** | night | burned-in timestamp reads `08-23-2026 Sun 21:02` |
| d_indoor_dome | **day** | night | air-dome interior, ceiling lights on, no daylight |
| e_pink_boards | **day** | night | indoor hall, ceiling lights, no sky in frame |
| f_outdoor_bldg | **day** | night | black sky, floodlight flare; the known instance |
| g_netting | **mixed** | night | dark sky through the netting, floodlights above |
| h_teal_pitch | **day** | night | dark sky beyond the cage, floodlights along the top |
| i_outdoor_trees | **mixed** | night | night sky behind the trees |

**Not one of the nine clip venues shows daylight.** Every recorded clip frame in this corpus is
under artificial light, and the corpus's only daylight is venue_01's.

**What `night` is being made to mean, and the limitation that carries.** The field is binary,
and these venues are two different things: floodlit outdoor pitches and indoor halls. Both are
"artificial light, no daylight", which is the distinction the thesis's adverse-light claim
rests on, so both become `night`. That floodlit-outdoor and indoor-artificial are not
distinguished is a real limitation of the field and is not repaired here.

**Generated frames are not touched.** They carry these venue names and `lighting = unknown`,
and the audit is of recorded footage; what light a generated image depicts is the generator's
business and `unknown` is the honest value for it.

**Why the sidecar and the manifest are edited in place rather than rebuilt.** `build_manifest`
regenerates from filenames on disk and would drop the 189 generated rows, which is the failure
A15 records. This touches one column of the rows it names and nothing else, and `--check`
reports what it would do without doing it.

    uv run python scripts/relabel_clip_lighting.py --check
    uv run python scripts/relabel_clip_lighting.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
SIDECAR = ROOT / "data" / "interim" / "clip_frames.csv"

#: The audit. One judgement per venue, with the evidence in the table above. Every clip venue
#: is artificial light; the mapping is uniform because the finding is uniform, and it is
#: written out per venue anyway so that a disagreement can be about one venue.
AUDIT: dict[str, str] = {
    "clipvenue_a_blue_barrier": "night",
    "clipvenue_b_floodlit_track": "night",
    "clipvenue_c_teal_boards": "night",
    "clipvenue_d_indoor_dome": "night",
    "clipvenue_e_pink_boards": "night",
    "clipvenue_f_outdoor_bldg": "night",
    "clipvenue_g_netting": "night",
    "clipvenue_h_teal_pitch": "night",
    "clipvenue_i_outdoor_trees": "night",
}


def _rewrite(path: Path, venue_of, keep, *, apply: bool) -> Counter:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "lighting" not in fields:
        raise SystemExit(f"{path} has no lighting column")

    changed: Counter = Counter()
    for row in rows:
        venue = venue_of(row)
        want = AUDIT.get(venue)
        if want is None or not keep(row) or row["lighting"] == want:
            continue
        changed[f"{venue}: {row['lighting']} -> {want}"] += 1
        if apply:
            row["lighting"] = want

    if apply and changed:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report and change nothing")
    args = ap.parse_args()
    apply = not args.check

    total: Counter = Counter()
    # The manifest holds generated frames at these same venue names, and the audit does not
    # cover them: a generated frame's lighting is whatever the generator produced, and
    # `unknown` is the honest value for it. Only recorded clip rows are touched.
    for path, venue_of, keep in (
        (SIDECAR, lambda r: r.get("venue", ""), lambda _r: True),
        (MANIFEST, lambda r: r.get("venue", ""), lambda r: r.get("source") == "clip"),
    ):
        if not path.exists():
            print(f"  {path} does not exist, skipped")
            continue
        changed = _rewrite(path, venue_of, keep, apply=apply)
        total.update(changed)
        print(f"{'would change' if args.check else 'changed'} {sum(changed.values()):>4} "
              f"rows in {path.relative_to(ROOT)}")

    print()
    for label, n in sorted(total.items()):
        print(f"  {label:<48}{n:>5}")
    if not total:
        print("  nothing to change - the labels already match the audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
