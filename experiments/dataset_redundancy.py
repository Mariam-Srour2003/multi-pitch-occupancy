"""How much of the dataset is a second copy of something already in it? (2026-09-21)

A frame count is not an evidence count. These frames were sampled every fifteen seconds from
fixed cameras watching mostly-static pitches, and from clips a few seconds long, so many of
them differ by cloud movement and nothing else. This reports the gap between the two numbers
for every venue, class and source, so "1,720 recorded frames" stops being quotable without
the figure that matters beside it.

**Nothing is deleted.** `data/dedup.py` and `splits.distinct_rows` already exist and the
protocol for using them is settled: pruning to one frame per scene is a *training-side* tool
and a test set pruned the same way would change what its number means. Deleting the copies
would also destroy the only EMPTY footage this project has - 525 frames of five scenes is a
thin dataset, but 5 frames is not a dataset at all. So this measures, names the worst
offenders, and leaves the decision where it belongs.

The scene threshold is `dedup.THRESHOLD`, 5 bits of a 64-bit dHash, the same number
`scripts/assign_scene_ids.py` groups by - so "distinct" means one thing everywhere.

    uv run python experiments/dataset_redundancy.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import scene_ids
from pitch_occupancy.evaluation.experiment_log import record

RESULTS = settings.results_dir


def summarise(rows, scenes: dict[str, str], key) -> list[dict]:
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    out = []
    for name, group in sorted(groups.items()):
        distinct = {scenes.get(r.file) for r in group} - {None}
        n_scenes = len(distinct) or len(group)
        out.append({
            "group": name,
            "frames": len(group),
            "scenes": n_scenes,
            "frames_per_scene": round(len(group) / max(n_scenes, 1), 1),
            "redundancy": round(1 - n_scenes / max(len(group), 1), 3),
        })
    return out


def table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'':<34}{'frames':>8}{'scenes':>8}{'per scene':>11}{'redundant':>11}")
    for r in sorted(rows, key=lambda r: -r["frames"]):
        print(f"{r['group'][:32]:<34}{r['frames']:>8}{r['scenes']:>8}"
              f"{r['frames_per_scene']:>11}{r['redundancy']:>11.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-synthetic", action="store_true")
    args = ap.parse_args()

    everything = read_manifest(settings.dataset_dir / "manifest.csv")
    rows = everything if args.include_synthetic else [r for r in everything
                                                     if r.source != "synthetic"]
    scenes = scene_ids()
    distinct = {scenes.get(r.file) for r in rows} - {None}
    print(f"{len(rows)} frames, {len(distinct)} distinct scenes "
          f"({len(distinct) / max(len(rows), 1):.0%} of the frame count)")
    if not args.include_synthetic:
        print(f"(recorded frames only; {len(everything) - len(rows)} generated frames excluded)")

    by_class = summarise(rows, scenes, lambda r: r.class3)
    by_venue = summarise(rows, scenes, lambda r: r.venue)
    by_source = summarise(rows, scenes, lambda r: r.source)
    by_camera = summarise(rows, scenes, lambda r: f"{r.venue}/{r.camera}")
    table("by class", by_class)
    table("by venue", by_venue)
    table("by source", by_source)

    # The single number worth quoting: how concentrated the corpus is.
    sizes = Counter()
    for row in rows:
        sizes[scenes.get(row.file, row.file)] += 1
    biggest = sizes.most_common(6)
    top = sum(n for _, n in biggest[:4])
    print(f"\nthe four largest scenes hold {top} frames, "
          f"{top / max(len(rows), 1):.0%} of the recorded corpus:")
    scene_of = {}
    for row in rows:
        scene_of.setdefault(scenes.get(row.file), row)
    for scene, n in biggest:
        example = scene_of.get(scene)
        where = f"{example.venue} {example.class3[:2]} {example.lighting}" if example else "?"
        print(f"  {str(scene):<8}{n:>6}  {where}")
    singles = sum(1 for _, n in sizes.items() if n == 1)
    print(f"\n{singles} scene(s) are a single frame; "
          f"{sum(1 for _, n in sizes.items() if n >= 20)} hold twenty or more")

    print("\nthe ten most repeated cameras:")
    table("", sorted(by_camera, key=lambda r: -r["frames"])[:10])

    # --- the per-video concentration, which is the bias a fit would actually learn -------
    from pitch_occupancy.data.splits import balanced_rows, distinct_rows

    per_video = Counter(r.camera for r in rows)
    top4 = sum(n for _, n in per_video.most_common(4))
    print(f"\n{len(per_video)} source videos. The four largest are {top4} frames, "
          f"{top4 / max(len(rows), 1):.0%} of every recorded frame:")
    for cam, n in per_video.most_common(6):
        example = next(r for r in rows if r.camera == cam)
        print(f"  {n:>6}  {n / len(rows):>6.1%}  {cam[:40]:<42}"
              f"{example.venue} {example.class3[:2]}")
    median = sorted(per_video.values())[len(per_video) // 2]
    print(f"  the median source video contributes {median}")

    print("\nwhat a per-video cap costs and buys (splits.balanced_rows):")
    print(f"{'arm':<24}{'frames':>8}{'videos':>8}{'top 4':>8}   by class")
    arms = [("all recorded", rows), ("distinct scenes", distinct_rows(rows))]
    arms += [(f"cap {cap}/video", balanced_rows(rows, per_video=cap))
             for cap in (6, 12, 20, 40)]
    caps = []
    for name, subset in arms:
        counts = Counter(r.camera for r in subset)
        share = sum(n for _, n in counts.most_common(4)) / max(len(subset), 1)
        by_cls = Counter(r.class3 for r in subset)
        print(f"{name:<24}{len(subset):>8}{len(counts):>8}{share:>8.0%}   "
              + "  ".join(f"{k[:2]}={v}" for k, v in sorted(by_cls.items())))
        caps.append({"axis": "cap", "group": name, "frames": len(subset),
                     "scenes": len(counts), "frames_per_scene": round(share * 100, 1),
                     "redundancy": round(share, 3)})
    print("  `frames_per_scene` on the cap rows is the top-four share as a percentage, and")
    print("  `scenes` is the number of source videos - the columns carry the analogous thing.")
    print("\nThere is no free value. The four videos carrying the bias are also the only")
    print("EMPTY footage there is, so a hard cap trades one problem for the other. This is a")
    print("training-side knob and the test side is never capped.")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "dataset_redundancy.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["axis", "group", "frames", "scenes",
                                                "frames_per_scene", "redundancy"])
        writer.writeheader()
        for axis, group in (("class", by_class), ("venue", by_venue),
                            ("source", by_source), ("camera", by_camera)):
            for r in group:
                writer.writerow({"axis": axis, **r})
        writer.writerows(caps)
    print(f"\nwrote {path}")
    print("nothing was deleted - see the module docstring for why")

    worst = max(by_class, key=lambda r: r["redundancy"])
    record(
        "dataset redundancy: frames against distinct scenes",
        "uv run python experiments/dataset_redundancy.py",
        "dataset_redundancy.csv",
        f"{len(rows)} recorded frames carry {len(distinct)} distinct scenes "
        f"({len(distinct) / max(len(rows), 1):.0%}); the four largest scenes hold "
        f"{top / max(len(rows), 1):.0%} of the corpus; worst class {worst['group']} at "
        f"{worst['frames_per_scene']} frames per scene; nothing deleted - pruning is a "
        f"training-side tool (splits.distinct_rows) and the copies are the only EMPTY "
        f"footage there is",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
