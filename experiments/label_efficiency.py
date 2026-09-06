"""How many labels does a new site actually need? (WP4-T2, RQ1/RQ2)

The operational question behind this is concrete: a new facility is onboarded, someone has
to label frames from it, and that labour is the real deployment cost. This traces macro-F1
against training-set size so the answer is a curve rather than a guess.

Design:

* training sizes {10, 25, 50, 100, 300, 1000, all}, 5 seeds each;
* the **grouped** split throughout - the leaky one would flatter every point equally and
  answer a question nobody asked;
* subsampling is **stratified where possible**, so a 10-frame budget is not spent entirely
  on the majority class;
* the clock rule is drawn as a horizontal line: it needs no labels at all, so any model
  below it at a given budget is not worth the labelling effort.

Cheap because it runs entirely on cached features - the backbones are never re-run.

    uv run python experiments/label_efficiency.py
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import features_for, load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, grouped_split
from pitch_occupancy.evaluation.metrics import evaluate
from pitch_occupancy.vision.backbones import BACKBONES
from pitch_occupancy.vision.heads import ClockRule, LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
SIZES = [10, 25, 50, 100, 300, 1000]
SEEDS = [0, 1, 2, 3, 4]


def stratified_sample(rows, n: int, rng: random.Random):
    """Take ``n`` rows, spreading across classes as evenly as the data allows."""
    if n >= len(rows):
        return list(rows)
    by_class: dict[str, list] = defaultdict(list)
    for r in rows:
        by_class[r.class3].append(r)
    for v in by_class.values():
        rng.shuffle(v)

    picked, cursor = [], 0
    classes = sorted(by_class)
    while len(picked) < n:
        added = False
        for c in classes:
            if cursor < len(by_class[c]) and len(picked) < n:
                picked.append(by_class[c][cursor])
                added = True
        if not added:
            break
        cursor += 1
    return picked


def main() -> None:
    rows = development_rows(read_manifest(DATASET / "manifest.csv"))
    split = grouped_split(rows, group_key="slot_id", seed=42)
    pos = {r.file: i for i, r in enumerate(rows)}
    te = [pos[r.file] for r in split.test]
    y_true = [r.class3 for r in split.test]
    print(f"train pool {len(split.train)}  test {len(split.test)}\n")

    records = []

    # zero-label reference
    clock = ClockRule().fit(np.zeros((len(split.train), 1)), split.train)
    ref = evaluate(y_true, clock.predict(np.zeros((len(split.test), 1)), split.test))
    print(f"clock_rule (0 labels): macroF1 {ref.macro_f1:.4f}  acc {ref.accuracy:.4f}\n")
    records.append(
        {"model": "clock_rule", "n_labels": 0, "seed": -1,
         "macro_f1": round(ref.macro_f1, 4), "accuracy": round(ref.accuracy, 4)}
    )

    for key in sorted(BACKBONES):
        try:
            cached = load_cache(key, CACHE)
        except FileNotFoundError:
            print(f"(skipping {key}: no cache)")
            continue
        X, kept = features_for(cached, rows)
        if len(kept) != len(rows):
            print(f"(skipping {key}: cache covers {len(kept)}/{len(rows)})")
            continue

        print(f"=== {key} ===")
        # clamp to the pool and dedupe: 1000 and "all" collapse to the same budget here
        budgets = sorted({min(s, len(split.train)) for s in [*SIZES, len(split.train)]})
        for n in budgets:
            scores = []
            for seed in SEEDS:
                sub = stratified_sample(split.train, n, random.Random(seed))
                if len({r.class3 for r in sub}) < 2:
                    continue  # a single-class training set cannot fit a classifier
                tr = [pos[r.file] for r in sub]
                model = LinearProbe(key, seed=seed).fit(X[tr], sub)
                rep = evaluate(y_true, model.predict(X[te], split.test))
                scores.append(rep.macro_f1)
                records.append(
                    {"model": key, "n_labels": len(sub), "seed": seed,
                     "macro_f1": round(rep.macro_f1, 4), "accuracy": round(rep.accuracy, 4)}
                )
                if n >= len(split.train):
                    break  # the full set is identical across seeds
            if scores:
                arr = np.array(scores)
                marker = " <- beats the 0-label rule" if arr.mean() > ref.macro_f1 else ""
                print(
                    f"  n={min(n, len(split.train)):>5}  macroF1 {arr.mean():.4f} "
                    f"+/- {arr.std():.4f}{marker}"
                )
        print()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "label_efficiency.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "n_labels", "seed", "macro_f1", "accuracy"])
        w.writeheader()
        w.writerows(records)
    print(f"wrote {out}")

    with (RESULTS / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n- {datetime.now(timezone.utc):%Y-%m-%d} | label efficiency | "
            f"`python experiments/label_efficiency.py` | `{out.name}` | "
            f"{len(SIZES) + 1} sizes x {len(SEEDS)} seeds\n"
        )


if __name__ == "__main__":
    main()
