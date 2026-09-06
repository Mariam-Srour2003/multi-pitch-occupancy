"""H3 - does active-play detection transfer to venues never seen in training?

Pre-registered: a model trained on venue_01 plus a subset of clip venues reaches
ACTIVE_PLAY recall >= 0.90 on held-out venues.

**Recall on one class, not macro-F1, and that is deliberate.** None of the clip venues
contains an empty pitch, so every held-out venue fold has a test set that is 100%
ACTIVE_PLAY. Macro-F1 over a single present class is meaningless, and accuracy on a
single-class test set is just recall wearing a different name. Reporting recall openly is
the honest version of the same number.

This is the *only* generalisation claim the dataset supports. It says nothing about
whether the model can recognise an empty pitch at a new venue - there is no data for that,
and `thesis/preregistration.md` records it as unanswerable.

The `venue_01` fold is excluded: training on it would mean fitting on clip venues alone,
which contain no EMPTY frames at all, then testing on the only venue that has them.

    uv run python experiments/h3_cross_venue_recall.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import features_for, load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
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
    rows = development_rows(read_manifest(DATASET / "manifest.csv"))
    pos = {r.file: i for i, r in enumerate(rows)}

    folds = [
        f for f in leave_one_group_out(rows, group_key="venue")
        if not f.name.endswith("venue_01")
    ]
    print(f"{len(rows)} development rows | {len(folds)} held-out venue folds")
    print("(venue_01 fold excluded: training on it leaves no EMPTY frames at all)\n")

    records: list[dict] = []
    models = list(available_models(rows))

    for name, model, X in models:
        recalls: list[float] = []
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
            print(f"  {venue:<28} n={len(fold.test):>4}  play-recall {recall:.3f}")
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
            f"worst {arr.min():.3f}  {meets} the {TARGET_RECALL:.2f} target\n"
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
                "meets_target": ci.estimate >= TARGET_RECALL,
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "h3_cross_venue_recall.csv"
    cols = [
        "model", "held_out_venue", "n_train", "n_test", "play_recall",
        "ci_low", "ci_high", "worst_fold", "meets_target",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"wrote {out}")

    with (RESULTS / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n- {datetime.now(timezone.utc):%Y-%m-%d} | H3 | "
            f"`python experiments/h3_cross_venue_recall.py` | seed {SEED} | "
            f"`{out.name}` | {len(folds)} folds x {len(models)} models\n"
        )


if __name__ == "__main__":
    main()
