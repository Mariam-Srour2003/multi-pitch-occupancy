"""H1 and H2 — split leakage, and whether trivial baselines already do the job.

Pre-registered in ``thesis/preregistration.md``:

* **H1** — a random split gives significantly higher macro-F1 than a grouped split,
  because frames sampled seconds apart land on both sides.
* **H2** — trivial baselines (majority class, a clock rule, mean intensity, a colour
  histogram) come within 2 macro-F1 points of the best frozen-backbone probe.

H2's prediction was registered *before* running this, on the strength of the clock rule
already scoring 98.4% on the labelled frames. Confirming it is the point: it would mean
the benchmark measures scene recognition rather than occupancy.

    uv run python experiments/h1_h2_baseline_floor.py

Writes ``results/h1_h2_baseline_floor.csv`` and appends to ``results/EXPERIMENT_LOG.md``.
Locked final-test venues are excluded throughout — every split here runs on development
rows only.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import features_for, load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import (
    Split,
    check_split,
    development_rows,
    grouped_split,
    random_split,
)
from pitch_occupancy.evaluation.metrics import evaluate
from pitch_occupancy.evaluation.stats import (
    bootstrap_metric_ci,
    holm_bonferroni,
    mcnemar,
)
from pitch_occupancy.vision.backbones import BACKBONES
from pitch_occupancy.vision.cheap_features import build_cheap_features
from pitch_occupancy.vision.heads import ClockRule, LinearProbe, MajorityClass

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
SEED = 42
RESAMPLES = 2000  # CI resamples; 10k for the final thesis numbers


def cheap_matrix(rows, kind: str) -> np.ndarray:
    """Cheap features, cached to disk so reruns are instant."""
    path = CACHE / f"cheap_{kind}.npz"
    files = [r.file for r in rows]
    if path.exists():
        z = np.load(path, allow_pickle=True)
        idx = {str(f): i for i, f in enumerate(z["files"])}
        if all(f in idx for f in files):
            return z["features"][[idx[f] for f in files]]
    X = build_cheap_features(files, DATASET, kind=kind)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, files=np.array(files, dtype=object), features=X)
    return X


def predictors(rows):
    """Every candidate, cheapest first, each paired with its feature matrix."""
    n = len(rows)
    yield "majority", MajorityClass(), np.zeros((n, 1), np.float32)
    yield "clock_rule", ClockRule(), np.zeros((n, 1), np.float32)
    for kind in ("intensity", "histogram"):
        yield f"cheap_{kind}", LinearProbe(f"cheap_{kind}", seed=SEED), cheap_matrix(rows, kind)
    for key in sorted(BACKBONES):
        try:
            cached = load_cache(key, CACHE)
        except FileNotFoundError:
            print(f"  (skipping {key}: no feature cache - run `uv run pitch cache`)")
            continue
        X, kept = features_for(cached, rows)
        if len(kept) != len(rows):
            print(f"  (warning: {key} cache covers {len(kept)}/{len(rows)} rows)")
            continue
        yield key, LinearProbe(key, seed=SEED), X


def run_split(split: Split, all_rows) -> list[dict]:
    """Fit every predictor on this split and score it."""
    pos = {r.file: i for i, r in enumerate(all_rows)}
    tr = [pos[r.file] for r in split.train]
    te = [pos[r.file] for r in split.test]
    y_true = [r.class3 for r in split.test]

    out = []
    for name, model, X in predictors(all_rows):
        t0 = time.perf_counter()
        model.fit(X[tr], split.train)
        y_pred = model.predict(X[te], split.test)
        fit_ms = (time.perf_counter() - t0) * 1000

        rep = evaluate(y_true, y_pred)
        ci = bootstrap_metric_ci(
            y_true, y_pred, lambda t, p: evaluate(t, p).macro_f1,
            resamples=RESAMPLES, seed=SEED,
        )
        out.append(
            {
                "split": split.name,
                "model": name,
                "n_train": len(split.train),
                "n_test": len(split.test),
                "accuracy": round(rep.accuracy, 4),
                "macro_f1": round(rep.macro_f1, 4),
                "macro_f1_lo": round(ci.low, 4),
                "macro_f1_hi": round(ci.high, 4),
                "balanced_acc": round(rep.balanced_accuracy, 4),
                "absent_classes": ";".join(rep.absent_classes),
                "fit_ms": round(fit_ms, 1),
                "_correct": np.array([p == t for p, t in zip(y_pred, y_true, strict=True)]),
            }
        )
    return out


def main() -> None:
    rows = development_rows(read_manifest(DATASET / "manifest.csv"))
    print(f"development rows: {len(rows)}  venues: {len({r.venue for r in rows})}\n")

    splits = [random_split(rows, seed=SEED), grouped_split(rows, group_key="slot_id", seed=SEED)]
    records: list[dict] = []

    for split in splits:
        print(f"=== {split.name} ===")
        print(f"  train {len(split.train)}  test {len(split.test)}")
        for w in check_split(split):
            print(f"  ! {w}")
        results = run_split(split, rows)
        records.extend(results)
        for r in sorted(results, key=lambda d: -d["macro_f1"]):
            print(
                f"  {r['model']:<16} acc {r['accuracy']:.4f}  "
                f"macroF1 {r['macro_f1']:.4f} [{r['macro_f1_lo']:.3f},{r['macro_f1_hi']:.3f}]"
            )
        print()

    # --- H2: every baseline against the best deep probe, on the honest split ---
    deep = set(BACKBONES)
    grouped = [r for r in records if r["split"].startswith("grouped")]
    print("=== H2: baselines vs the best deep probe (grouped split) ===")
    deep_rows = [r for r in grouped if r["model"] in deep]
    if not deep_rows:
        print("  no deep probe available yet - rerun after `uv run pitch cache`")
    else:
        champ = max(deep_rows, key=lambda r: r["macro_f1"])
        others = [r for r in grouped if r["model"] != champ["model"]]
        tests = [mcnemar(champ["_correct"], r["_correct"]) for r in others]
        adjusted, rejected = holm_bonferroni([t.p_value for t in tests])
        print(f"  champion: {champ['model']}  macroF1 {champ['macro_f1']:.4f}\n")
        for r, t, p_adj, rej in zip(others, tests, adjusted, rejected, strict=True):
            gap = champ["macro_f1"] - r["macro_f1"]
            print(
                f"  vs {r['model']:<16} dMacroF1 {gap:+.4f}  "
                f"p_holm {p_adj:.4g}  g {t.effect_size:.3f}  "
                f"{'differs' if rej else 'indistinguishable'}"
            )
            r["p_holm_vs_champion"] = round(p_adj, 6)
            r["differs_from_champion"] = rej

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "h1_h2_baseline_floor.csv"
    cols = [k for k in records[0] if not k.startswith("_")]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {out}")

    log = RESULTS / "EXPERIMENT_LOG.md"
    log.touch()
    with log.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n- {datetime.now(timezone.utc):%Y-%m-%d} · H1/H2 · "
            f"`python experiments/h1_h2_baseline_floor.py` · seed {SEED} · "
            f"`{out.name}` · {len(records)} rows over {len(splits)} splits\n"
        )


if __name__ == "__main__":
    main()
