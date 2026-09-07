"""How much of the dataset is actually distinct, and what does a split leak? (WP2-T4)

1,692 labelled frames is the headline number. Frames sampled every fifteen seconds from a
fixed camera on a mostly-static pitch are not 1,692 independent observations, and that
matters in three places this project has already met:

* **H1** found a 16-bin colour histogram beating a deep probe under a random split. If
  near-identical frames sit on both sides, that is memorisation scoring as generalisation.
* **C3** could not be evaluated at all - six frames, four of them seconds apart.
* **Label-efficiency** curves are drawn against "number of labels", which means something
  different if many of those labels are copies.

Duplication is reported **pairwise**. The first version of this script grouped frames by
single-link chaining and announced that 96.9% of the dataset was redundant, with "52 distinct
scenes". Its own output refuted it: the largest group held 512 frames whose maximum internal
distance was **19** against a threshold of 6, and contained both EMPTY and ACTIVE_PLAY
frames. A fixed camera drifts slowly, so every frame is near its neighbour in time and the
whole slot chains into one blob. An empty pitch and a match in progress are not duplicates of
each other, and the number was an artifact of the linkage, not a property of the data.

    uv run python experiments/near_duplicate_audit.py
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict

import cv2

from pitch_occupancy.config import settings
from pitch_occupancy.data.dedup import (
    DEFAULT_THRESHOLD,
    dhash,
    duplicate_rate,
    find_near_duplicates,
    hamming,
    near_duplicate_pairs,
)

OUT = settings.results_dir / "near_duplicates.csv"


def report_separation(hashes: dict[str, int], meta: dict[str, dict]) -> None:
    """Show the threshold sits in a real gap rather than being asserted.

    If frames from unrelated slots routinely fall under it, the threshold is merging
    different scenes and every number below it is meaningless.
    """
    random.seed(0)
    by_slot: dict[str, list[str]] = defaultdict(list)
    for f in hashes:
        by_slot[meta[f]["slot_id"]].append(f)

    same, different = [], []
    for members in by_slot.values():
        if len(members) < 2:
            continue
        for _ in range(min(200, len(members) * 2)):
            a, b = random.sample(members, 2)
            same.append(hamming(hashes[a], hashes[b]))
    files = list(hashes)
    for _ in range(2000):
        a, b = random.sample(files, 2)
        if meta[a]["slot_id"] != meta[b]["slot_id"]:
            different.append(hamming(hashes[a], hashes[b]))

    def pct(values, p):
        return sorted(values)[int(len(values) * p / 100)] if values else float("nan")

    print(f"  within one slot ({len(same):5} pairs): "
          f"p5={pct(same, 5):3} median={pct(same, 50):3} p95={pct(same, 95):3}")
    print(f"  across slots    ({len(different):5} pairs): "
          f"p5={pct(different, 5):3} median={pct(different, 50):3} p95={pct(different, 95):3}")
    below = sum(1 for d in different if d <= DEFAULT_THRESHOLD) / max(len(different), 1)
    print(f"  frames from different slots under the threshold: {below:.2%} "
          f"(a large value would mean the threshold merges unrelated scenes)")


def report_leakage(pairs, meta) -> None:
    """The number H1 was really about: how many near-duplicate pairs a split separates.

    A pair straddling the train/test boundary is one frame being scored against a copy of
    itself. This compares the deliberately leaky random split against the grouped split that
    replaced it, on exactly the same frames.
    """
    from pitch_occupancy.data.manifest import read_manifest
    from pitch_occupancy.data.splits import development_rows, grouped_split, random_split

    rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    known = {r.file for r in rows}
    usable = [(a, b) for a, b, _ in pairs if a in known and b in known]
    if not usable:
        print("\n  no near-duplicate pairs among the development rows")
        return
    print(f"\n--- how many of the {len(usable)} pairs does each split separate? ---")
    for name, split in (("random split", random_split(rows)),
                        ("grouped split", grouped_split(rows))):
        train = {r.file for r in split.train}
        test = {r.file for r in split.test}
        straddling = sum(
            1 for a, b in usable
            if (a in train and b in test) or (a in test and b in train)
        )
        print(f"  {name:16} {straddling:6} pairs straddle the boundary "
              f"({straddling / len(usable):.1%})")
    print("  Every straddling pair is a frame scored against a copy of itself.")


def main() -> None:
    manifest = settings.dataset_dir / "manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    hashes: dict[str, int] = {}
    meta: dict[str, dict] = {}
    for row in rows:
        image = cv2.imread(str(settings.dataset_dir / row["file"]))
        if image is None:
            continue
        hashes[row["file"]] = dhash(image)
        meta[row["file"]] = row
    print(f"hashed {len(hashes)} frames")

    print(f"\n--- is the threshold justified by this data? (default {DEFAULT_THRESHOLD}) ---")
    report_separation(hashes, meta)

    pairs = near_duplicate_pairs(hashes)
    twinned = {f for a, b, _ in pairs for f in (a, b)}
    print(f"\n--- duplication at threshold {DEFAULT_THRESHOLD}, measured pairwise ---")
    print(f"{len(pairs)} directly similar pairs")
    print(f"{len(twinned)} of {len(hashes)} frames have at least one near-duplicate "
          f"({len(twinned) / len(hashes):.1%})")

    groups = find_near_duplicates(hashes)
    if groups:
        biggest = groups[0]
        classes = sorted({meta[m]["class3"] for m in biggest.members})
        print("\nwhy this is pairwise and not grouped:")
        print(f"  single-link chaining puts {len(biggest.members)} frames in one group whose")
        print(f"  maximum internal distance is {biggest.max_distance}, against a threshold "
              f"of {DEFAULT_THRESHOLD},")
        print(f"  spanning classes {classes} - not a duplicate count.")

    print("\nper class (frames with a near-duplicate):")
    for cls in sorted({r["class3"] for r in meta.values()}):
        subset = {f: h for f, h in hashes.items() if meta[f]["class3"] == cls}
        print(f"  {cls:30} {len(subset):5} frames  {duplicate_rate(subset):6.1%}")

    print("\nper venue:")
    for venue in sorted({r["venue"] for r in meta.values()}):
        subset = {f: h for f, h in hashes.items() if meta[f]["venue"] == venue}
        print(f"  {venue:30} {len(subset):5} frames  {duplicate_rate(subset):6.1%}")

    cross = [(a, b, d) for a, b, d in pairs if meta[a]["class3"] != meta[b]["class3"]]
    print(f"\nnear-identical pairs carrying different labels: {len(cross)}")
    for a, b, d in sorted(cross, key=lambda x: x[2])[:5]:
        print(f"  d={d}  {meta[a]['class3']:28} {a}")
        print(f"        {meta[b]['class3']:28} {b}")

    report_leakage(pairs, meta)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file_a", "file_b", "distance", "class_a", "class_b",
                    "slot_a", "slot_b", "same_slot"])
        for a, b, d in pairs:
            w.writerow([a, b, d, meta[a]["class3"], meta[b]["class3"],
                        meta[a]["slot_id"], meta[b]["slot_id"],
                        meta[a]["slot_id"] == meta[b]["slot_id"]])
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
