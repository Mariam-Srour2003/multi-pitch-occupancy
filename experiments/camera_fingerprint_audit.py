"""WP2-T10 - identify a camera view from its pixels, and check the venue labels with it.

Three questions, in order of how much rests on them:

1. **Is the `venue_01` camera labelling right?** SUPER_PLAN gotcha 2.7 says the `(1).mp4`
   suffix means a different physical camera on the two recording days. That was written from
   memory. Here it is either measured or withdrawn.
2. **Can a view be assigned to a known venue from pixels alone?** The operational question:
   new footage arrives, which facility is it? Scored as leave-one-view-out 1-NN against the
   hand grouping.
3. **Does the hand grouping survive an unsupervised check?** The visual venue audit
   (EXPERIMENT_LOG, 2026-09-06) raised two groups from `confidence=low` by eye. Clustering
   the fingerprints is the same audit done by measurement.

    uv run python experiments/camera_fingerprint_audit.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import adjusted_rand_score, homogeneity_score

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.vision.fingerprint import (
    ViewFingerprint,
    cluster_views,
    distance,
    fingerprint_view,
    mutual_nearest_pairs,
    suggest_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CLIP_FRAMES = ROOT / "data" / "interim" / "frames" / "clips"
CLIP_VENUES = ROOT / "configs" / "clip_venues.csv"
RESULTS = ROOT / "results"

#: Frames per view fed to the median. The clips only have six; venue_01's camera-slots have
#: hundreds and are subsampled, because a median over 25 well-spread frames already has no
#: moving object left in it and 500 would only cost memory.
MAX_FRAMES = 25

#: The two groups the visual audit had to adjudicate.
AUDITED = ("clipvenue_g_netting", "clipvenue_h_teal_pitch")

BLOCKS = ("descriptor", "structure", "chroma")


def clip_venue_map() -> dict[str, str]:
    """`ca_1788518168281` -> `clipvenue_a_blue_barrier`, from the hand-made grouping."""
    out: dict[str, str] = {}
    for row in csv.DictReader(CLIP_VENUES.open(encoding="utf-8")):
        clip_id = row["file"].replace("statbox-watermarked-", "").replace(".mp4", "")
        out[f"{row['venue_code']}_{clip_id}"] = row["venue"]
    return out


def load(paths: list[Path]) -> list[np.ndarray]:
    step = max(1, len(paths) // MAX_FRAMES)
    frames = [cv2.imread(str(p)) for p in paths[::step][:MAX_FRAMES]]
    return [f for f in frames if f is not None]


def collect_views() -> list[ViewFingerprint]:
    """One fingerprint per camera view: 66 clip cameras plus venue_01's four camera-slots."""
    venues = clip_venue_map()
    by_view: dict[str, list[Path]] = defaultdict(list)

    for path in sorted(CLIP_FRAMES.glob("clip_*.jpg")):
        _, code, clip_id, _ = path.name.split("_", 3)
        by_view[f"{code}_{clip_id}"].append(path)

    for row in read_manifest(DATASET / "manifest.csv"):
        if row.venue == "venue_01":
            by_view[row.camera].append(DATASET / row.file)
            venues.setdefault(row.camera, "venue_01")

    prints: list[ViewFingerprint] = []
    for view, paths in sorted(by_view.items()):
        frames = load(sorted(paths))
        if not frames:
            print(f"  (skipping {view}: no readable frames)")
            continue
        prints.append(fingerprint_view(view, frames, venue=venues.get(view)))
    return prints


def report_camera_identity(prints: list[ViewFingerprint]) -> list[dict]:
    """Question 1: does the `(1).mp4` suffix mean the same camera on both days?"""
    v1 = [f for f in prints if f.venue == "venue_01"]
    print("=== venue_01 camera identity (gotcha 2.7) ===")
    if len(v1) < 4:
        print("  fewer than four camera-slots found; skipping\n")
        return []

    for a in v1:
        d, nearest = min(
            (distance(a.descriptor, b.descriptor), b.camera) for b in v1 if b is not a
        )
        same_suffix = nearest.endswith(a.camera[-1])
        print(
            f"  {a.camera:<28} nearest {nearest:<28} d={d:.4f}  "
            f"{'same suffix' if same_suffix else 'OPPOSITE suffix'}"
        )

    pairs = mutual_nearest_pairs(v1)
    print("\n  mutual nearest pairs (what a single noisy distance cannot fake):")
    rows: list[dict] = []
    for a, b, d in pairs:
        cross_day = a.split("_cam")[0] != b.split("_cam")[0]
        opposite = a[-1] != b[-1]
        flag = "  [different days, OPPOSITE suffixes]" if cross_day and opposite else ""
        print(f"    {a} <-> {b}  d={d:.4f}{flag}")
        rows.append(
            {
                "question": "camera_identity",
                "view": a,
                "other": b,
                "dist": round(d, 4),
                "note": "mutual-NN cross-day opposite suffix" if flag else "mutual-NN",
            }
        )

    swapped = len(pairs) == 2 and all(a[-1] != b[-1] for a, b, _ in pairs)
    verdict = (
        "the suffix swaps between the two days - gotcha 2.7 is MEASURED, not remembered"
        if swapped
        else "no consistent swap found; gotcha 2.7 is not reproduced here"
    )
    print(f"\n  VERDICT: {verdict}.\n")
    rows.append(
        {"question": "camera_identity", "view": "", "other": "", "dist": "", "note": verdict}
    )
    return rows


def report_assignment(prints: list[ViewFingerprint], block: str) -> tuple[int, int, list[dict]]:
    """Question 2: leave-one-view-out 1-NN venue assignment."""
    labelled = [f for f in prints if f.venue]
    rows, hits = [], 0
    for a in labelled:
        d, nearest = min(
            (distance(getattr(a, block), getattr(b, block)), b) for b in labelled if b is not a
        )
        if nearest.venue == a.venue:
            hits += 1
        else:
            rows.append(
                {
                    "question": f"assignment_{block}",
                    "view": a.camera,
                    "other": nearest.camera,
                    "dist": round(d, 4),
                    "note": f"MISS: {a.venue} -> {nearest.venue}",
                }
            )
    return hits, len(labelled), rows


def report_grouping(prints: list[ViewFingerprint]) -> list[dict]:
    """Question 3: does clustering agree with the hand grouping?

    **Two points on a trade-off, not one score.** This module identifies a camera *view*; a
    venue label names a *facility*, and a facility with two pitches has two views that
    genuinely do not look alike - `clipvenue_g_netting` is exactly that case (the visual audit
    identified it by the pitch-4 camera seeing pitch 5's number board) and `venue_01` has two
    cameras on opposite halves. So a venue splitting into its cameras is correct behaviour,
    which ARI punishes; while purity on its own is gamed by splitting everything into
    singletons - maximising it lands on 30 clusters for 70 views and says nothing. Both were
    tried, in that order, before this.

    So both ends of the curve get reported: the coarsest threshold at which no cluster yet
    mixes two facilities, and the threshold of best agreement with the hand labels. The gap
    between them is the finding.
    """
    labelled = [f for f in prints if f.venue]
    truth = [f.venue for f in labelled]
    sep = suggest_threshold(labelled)

    curve: list[dict] = []
    for t in np.arange(0.05, 0.65, 0.01):
        labels = cluster_views(labelled, threshold=float(t))
        assigned = [labels[f.camera] for f in labelled]
        per_cluster: dict[int, set[str]] = defaultdict(set)
        for f in labelled:
            per_cluster[labels[f.camera]].add(f.venue)
        curve.append(
            {
                "threshold": float(t),
                "clusters": len(set(assigned)),
                "mixed": sum(1 for v in per_cluster.values() if len(v) > 1),
                "homogeneity": homogeneity_score(truth, assigned),
                "ari": adjusted_rand_score(truth, assigned),
                "labels": labels,
            }
        )

    pure = [c for c in curve if c["mixed"] == 0]
    coarsest_pure = max(pure, key=lambda c: c["threshold"]) if pure else None
    best = max(curve, key=lambda c: c["ari"])

    print("=== venue grouping vs the hand labels ===")
    print(
        f"  separation: same-venue mean {sep.same_mean:.4f} (max {sep.same_max:.4f}) | "
        f"different mean {sep.different_mean:.4f} (min {sep.different_min:.4f})"
    )
    overlap_note = (
        "OVERLAP - thresholding alone cannot settle venue identity"
        if sep.overlaps
        else "are separated"
    )
    print(f"  AUC {sep.auc:.4f}  |  distributions {overlap_note}")
    if coarsest_pure:
        print(
            f"  coarsest grouping with no facility mixed: threshold "
            f"{coarsest_pure['threshold']:.2f} -> {coarsest_pure['clusters']} clusters for "
            f"{len(labelled)} views (ARI {coarsest_pure['ari']:.3f})"
        )
    print(
        f"  best agreement with the hand labels: threshold {best['threshold']:.2f} -> ARI "
        f"{best['ari']:.3f}, {best['clusters']} clusters vs {len(set(truth))} facilities, "
        f"{best['mixed']} of them mixing facilities\n"
    )

    labels = best["labels"]
    per_venue: dict[str, set[int]] = defaultdict(set)
    per_cluster = defaultdict(set)
    for f in labelled:
        per_venue[f.venue].add(labels[f.camera])
        per_cluster[labels[f.camera]].add(f.venue)

    print("  at best agreement:")
    for venue in sorted(per_venue):
        marker = "  <- audited" if venue in AUDITED else ""
        n = len(per_venue[venue])
        note = "" if n == 1 else f"  (split into {n})"
        print(f"    {venue:<34} clusters {sorted(per_venue[venue])}{note}{marker}")

    merged = {c: v for c, v in per_cluster.items() if len(v) > 1}
    print(
        f"\n  clusters mixing facilities - the error that would break a leave-one-venue-out "
        f"split: {len(merged)}"
    )
    for cluster, venues_in in sorted(merged.items()):
        print(f"    cluster {cluster}: {', '.join(sorted(venues_in))}")

    g, h = AUDITED
    print("\n  the claims the visual audit made:")
    if g in per_venue and h in per_venue:
        shared = per_venue[g] & per_venue[h]
        state = (
            "CONFIRMED - no shared cluster"
            if not shared
            else f"CONTRADICTED - share {sorted(shared)}"
        )
        print(f"    {g} and {h} are different facilities: {state}")
    if g in per_venue:
        n = len(per_venue[g])
        print(
            f"    {g} is one facility: not testable this way - it splits into {n} view"
            f"{'s' if n != 1 else ''}, which is what a multi-pitch facility should do. The "
            f"audit established it from a pitch number visible across the fence, and a view "
            f"fingerprint cannot read a number board."
        )
    print()

    rows: list[dict] = [
        {
            "question": "grouping",
            "view": f.camera,
            "other": "",
            "dist": "",
            "note": f"venue={f.venue} cluster={labels[f.camera]}",
        }
        for f in labelled
    ]
    rows += [
        {
            "question": "grouping_curve",
            "view": "",
            "other": "",
            "dist": round(c["threshold"], 2),
            "note": (
                f"clusters={c['clusters']}; mixed={c['mixed']}; "
                f"homogeneity={c['homogeneity']:.3f}; ari={c['ari']:.3f}"
            ),
        }
        for c in curve
    ]
    pure_note = f"{coarsest_pure['threshold']:.2f}" if coarsest_pure else "none"
    rows.append(
        {
            "question": "grouping_summary",
            "view": "",
            "other": "",
            "dist": round(best["threshold"], 2),
            "note": (
                f"best_ari={best['ari']:.3f}; AUC={sep.auc:.4f}; overlaps={sep.overlaps}; "
                f"clusters={best['clusters']}; hand_venues={len(set(truth))}; "
                f"mixed={best['mixed']}; coarsest_pure_threshold={pure_note}"
            ),
        }
    )
    return rows


def main() -> None:
    prints = collect_views()
    n_venues = len({f.venue for f in prints if f.venue})
    print(f"{len(prints)} camera views | {n_venues} hand-labelled venues\n")

    rows: list[dict] = []
    rows += report_camera_identity(prints)

    print("=== venue assignment from pixels (leave-one-view-out 1-NN) ===")
    labelled_prints = [f for f in prints if f.venue]
    for block in BLOCKS:
        hits, total, miss_rows = report_assignment(prints, block)
        # The per-block AUC is the whole basis for keeping both descriptor blocks
        # ("structure identifies, chromaticity separates"), and it used to be computed
        # nowhere: `suggest_threshold` was called once on the combined descriptor, so only
        # that single AUC reached the CSV while the write-up quoted three. A claim argued
        # from a number the artefact does not contain is not reproducible, whatever the
        # number turns out to be - and `suggest_threshold` already takes `block`.
        block_sep = suggest_threshold(labelled_prints, block=block)
        print(
            f"  {block:<11} {hits}/{total} = {hits / total:.3f}"
            f"   AUC {block_sep.auc:.4f}"
        )
        for r in miss_rows:
            print(f"      {r['view']:<28} {r['note']}  d={r['dist']}")
        rows += miss_rows
        rows.append(
            {
                "question": f"assignment_{block}",
                "view": "",
                "other": "",
                "dist": f"{block_sep.auc:.4f}",
                "note": (
                    f"1NN venue accuracy {hits}/{total} = {hits / total:.4f}; "
                    f"separation AUC {block_sep.auc:.4f}; overlaps={block_sep.overlaps}"
                ),
            }
        )
    print()
    rows += report_grouping(prints)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "camera_fingerprint.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["question", "view", "other", "dist", "note"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")

    record(
        "WP2-T10 camera fingerprinting",
        "`python experiments/camera_fingerprint_audit.py`",
        f"`{out.name}`",
        f"{len(prints)} views, {n_venues} venues",
    )


if __name__ == "__main__":
    main()
