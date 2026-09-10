"""H3 sensitivity check - does the cross-venue result survive merging cg + ch?

Background (see EXPERIMENT_LOG 2026-09-06 "Venue-grouping audit"): the two originally
low-confidence venue groups were audited visually. `clipvenue_g_netting` is confirmed as
one facility (adjacent numbered pitches) and `clipvenue_h_teal_pitch` is plausibly one
venue; the audit also concluded they are *different* facilities from each other. This
script is the insurance policy for the residual doubt: if `g` and `h` were in fact one
facility, leave-one-venue-out would leak that facility's appearance into training whenever
only one of them is held out.

Design: identical to experiments/h3_cross_venue_recall.py (same seed, same models, same
recall metric, same fold exclusions) except that `clipvenue_g_netting` and
`clipvenue_h_teal_pitch` are relabelled to a single merged venue
`clipvenue_gh_merged_sensitivity` before the leave-one-group-out split. Under the merge,
holding the pair out removes BOTH from training - the pessimistic assumption.

Read the result as: if the models still meet (or miss) the 0.90 target the same way the
main H3 run did, the venue-grouping doubt cannot change the H3 conclusion, and the
worst-fold caveat can be quoted with confidence.

**This table reports play recall only, and inherits H3's control rather than repeating it.**
Every held-out venue here is 100% ACTIVE_PLAY, so recall on these folds is earned by
answering PLAY more often and a constant predictor scores 1.000 - the reason
`h3_with_false_play.csv` exists, and the reason the input ablation's best variant turned out
to have empty accuracy 0.000 (2026-09-11). Nothing in this script separates a model that
transfers from one that has shifted toward PLAY. It does not need to: the question it asks is
whether *merging two venue groups* changes what the main run concluded, which is a comparison
between folds rather than a claim about a model's sight. Quote its absolute numbers only
beside the control in `h3_with_false_play.csv`.

    uv run python experiments/h3_sensitivity_merged_venues.py
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import features_for, load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.stats import bootstrap_ci
from pitch_occupancy.vision.backbones import BACKBONES
from pitch_occupancy.vision.heads import ClockRule, LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
SEED = 42
PLAY = "C2_ACTIVE_PLAY"
TARGET_RECALL = 0.90

MERGE = {"clipvenue_g_netting", "clipvenue_h_teal_pitch"}
MERGED_NAME = "clipvenue_gh_merged_sensitivity"


def available_models(rows):
    yield "clock_rule", ClockRule(), np.zeros((len(rows), 1), np.float32)
    for key in sorted(BACKBONES):
        try:
            cached = load_cache(key, CACHE)
        except FileNotFoundError:
            print(f"  (skipping {key}: no feature cache)")
            continue
        X, kept = features_for(cached, rows)
        if len(kept) == len(rows):
            yield key, LinearProbe(key, seed=SEED), X
        else:
            print(f"  (skipping {key}: cache covers {len(kept)}/{len(rows)})")


def main() -> None:
    raw = development_rows(read_manifest(DATASET / "manifest.csv"))
    # relabel the two audited venues into one merged group (rows are frozen -> replace)
    rows = [
        replace(r, venue=MERGED_NAME) if r.venue in MERGE else r
        for r in raw
    ]
    n_merged = sum(1 for r in rows if r.venue == MERGED_NAME)
    pos = {r.file: i for i, r in enumerate(rows)}

    folds = [
        f for f in leave_one_group_out(rows, group_key="venue")
        if not f.name.endswith("venue_01")
    ]
    print(f"{len(rows)} development rows | {n_merged} rows in {MERGED_NAME} | "
          f"{len(folds)} held-out venue folds (one fewer than main H3)\n")

    records: list[dict] = []
    models = list(available_models(rows))

    for name, model, X in models:
        recalls: list[float] = []
        merged_fold_recall = float("nan")
        print(f"=== {name} ===")
        for fold in folds:
            venue = fold.name.split("__")[-1]
            tr = [pos[r.file] for r in fold.train]
            te = [pos[r.file] for r in fold.test]
            model.fit(X[tr], fold.train)
            pred = model.predict(X[te], fold.test)

            truth = [r.class3 for r in fold.test]
            play = [(p, t) for p, t in zip(pred, truth, strict=True) if t == PLAY]
            recall = sum(p == PLAY for p, _ in play) / len(play) if play else float("nan")
            recalls.append(recall)
            if venue == MERGED_NAME:
                merged_fold_recall = recall
            print(f"  {venue:<34} n={len(fold.test):>4}  play-recall {recall:.3f}")
            records.append(
                {
                    "model": name,
                    "held_out_venue": venue,
                    "n_train": len(fold.train),
                    "n_test": len(fold.test),
                    "play_recall": round(recall, 4),
                }
            )

        arr = np.array(recalls, dtype=float)
        ci = bootstrap_ci(arr, np.mean, resamples=5000, seed=SEED)
        meets = "MEETS" if ci.estimate >= TARGET_RECALL else "below"
        print(
            f"  --> mean {ci.estimate:.3f} [{ci.low:.3f}, {ci.high:.3f}]  "
            f"worst {arr.min():.3f}  merged-fold {merged_fold_recall:.3f}  "
            f"{meets} the {TARGET_RECALL:.2f} target\n"
        )
        records.append(
            {
                "model": name,
                "held_out_venue": "MEAN_ACROSS_FOLDS",
                "n_train": "",
                "n_test": "",
                "play_recall": round(float(ci.estimate), 4),
                "ci_low": round(ci.low, 4),
                "ci_high": round(ci.high, 4),
                "worst_fold": round(float(arr.min()), 4),
                "merged_fold": round(merged_fold_recall, 4),
                "meets_target": ci.estimate >= TARGET_RECALL,
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "h3_sensitivity_merged_venues.csv"
    cols = [
        "model", "held_out_venue", "n_train", "n_test", "play_recall",
        "ci_low", "ci_high", "worst_fold", "merged_fold", "meets_target",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"wrote {out}")

    record(
        "H3 sensitivity (cg+ch merged)",
        "`python experiments/h3_sensitivity_merged_venues.py`",
        f"seed {SEED}",
        f"`{out.name}`",
        f"{len(folds)} folds x {len(models)} models",
    )


if __name__ == "__main__":
    main()
