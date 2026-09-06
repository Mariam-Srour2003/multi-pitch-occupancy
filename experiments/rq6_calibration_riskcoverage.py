"""RQ6 - how much human review buys a given reliability? (WP4-T5, WP4-T9)

The REVIEW band is the system's escape hatch: uncertain frames go to a person. That makes
confidence an operational quantity, not a diagnostic one - it decides how many slots a
facility has to inspect by hand. Two measurements follow:

1. **Calibration.** Is a 90%-confident prediction right 90% of the time? Reported as ECE,
   before and after temperature scaling fitted on a held-out slice of the training venues.
   Temperature only rescales logits, so accuracy is untouched by construction; that is
   asserted here rather than assumed.

2. **Risk-coverage.** Sweep the confidence threshold: answer the most confident fraction
   automatically, defer the rest. The output is the operating point a manager can act on -
   *"to reach 99% precision on automated verdicts, expect to review N% of slots."*

    uv run python experiments/rq6_calibration_riskcoverage.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import features_for, load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, grouped_split
from pitch_occupancy.evaluation.calibration import (
    coverage_for_target_accuracy,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    risk_coverage_curve,
)
from pitch_occupancy.vision.backbones import BACKBONES
from pitch_occupancy.vision.heads import LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
SEED = 42
TARGETS = (0.95, 0.99)


def main() -> None:
    rows = development_rows(read_manifest(DATASET / "manifest.csv"))
    split = grouped_split(rows, group_key="slot_id", seed=SEED)
    pos = {r.file: i for i, r in enumerate(rows)}

    # a calibration slice carved from TRAIN only - never the test side
    rng = np.random.default_rng(SEED)
    tr_rows = list(split.train)
    rng.shuffle(tr_rows)
    cut = max(1, len(tr_rows) // 5)
    cal_rows, fit_rows = tr_rows[:cut], tr_rows[cut:]
    print(f"fit {len(fit_rows)} | calibrate {len(cal_rows)} | test {len(split.test)}\n")

    curve_records, summary = [], []

    for key in sorted(BACKBONES):
        try:
            cached = load_cache(key, CACHE)
        except FileNotFoundError:
            print(f"(skipping {key}: no cache)")
            continue
        X, kept = features_for(cached, rows)
        if len(kept) != len(rows):
            continue

        probe = LinearProbe(key, seed=SEED).fit(X[[pos[r.file] for r in fit_rows]], fit_rows)
        classes = list(probe.classes_)

        def logits_for(subset):
            p = probe.predict_proba(X[[pos[r.file] for r in subset]])
            return np.log(np.clip(p, 1e-12, None))

        cal_logits = logits_for(cal_rows)
        cal_idx = np.array([classes.index(r.class3) for r in cal_rows])
        temperature = fit_temperature(cal_logits, cal_idx)

        te_logits = logits_for(split.test)
        truth = np.array([r.class3 for r in split.test])

        def evaluate_at(t: float):
            from pitch_occupancy.evaluation.calibration import apply_temperature

            p = apply_temperature(te_logits, t)
            pred = np.array([classes[i] for i in p.argmax(1)])
            return p.max(axis=1), pred == truth, pred

        conf_raw, ok_raw, pred_raw = evaluate_at(1.0)
        conf_cal, ok_cal, pred_cal = evaluate_at(temperature)

        assert (pred_raw == pred_cal).all(), "temperature must not change predictions"

        ece_raw = expected_calibration_error(conf_raw, ok_raw)
        ece_cal = expected_calibration_error(conf_cal, ok_cal)
        print(f"=== {key} ===")
        print(f"  accuracy {ok_raw.mean():.4f}   temperature {temperature:.2f}")
        print(f"  ECE {ece_raw:.4f} -> {ece_cal:.4f}  ({'better' if ece_cal < ece_raw else 'worse'})")

        for target in TARGETS:
            hit = coverage_for_target_accuracy(conf_cal, ok_cal, target=target)
            if hit is None:
                print(f"  {target:.0%} precision: unreachable at any coverage")
            else:
                cov, review = hit
                print(f"  {target:.0%} precision: automate {cov:.1%}, review {review:.1%}")

        bins = reliability_bins(conf_cal, ok_cal)
        worst = max(bins, key=lambda b: abs(b.gap))
        print(
            f"  worst bin [{worst.lo:.1f},{worst.hi:.1f}] n={worst.n}: "
            f"confidence {worst.mean_confidence:.3f} vs accuracy {worst.accuracy:.3f}\n"
        )

        for cov, acc, review in risk_coverage_curve(conf_cal, ok_cal, points=60):
            curve_records.append(
                {"model": key, "coverage": round(cov, 4),
                 "accuracy": round(acc, 4), "review_rate": round(review, 4)}
            )
        row = {
            "model": key, "accuracy": round(float(ok_raw.mean()), 4),
            "temperature": round(temperature, 3),
            "ece_raw": round(ece_raw, 4), "ece_calibrated": round(ece_cal, 4),
        }
        for target in TARGETS:
            hit = coverage_for_target_accuracy(conf_cal, ok_cal, target=target)
            row[f"coverage_at_{int(target * 100)}"] = round(hit[0], 4) if hit else ""
            row[f"review_at_{int(target * 100)}"] = round(hit[1], 4) if hit else ""
        summary.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "rq6_calibration.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with (RESULTS / "rq6_risk_coverage.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "coverage", "accuracy", "review_rate"])
        w.writeheader()
        w.writerows(curve_records)
    print("wrote results/rq6_calibration.csv and results/rq6_risk_coverage.csv")

    with (RESULTS / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n- {datetime.now(timezone.utc):%Y-%m-%d} | RQ6 | "
            f"`python experiments/rq6_calibration_riskcoverage.py` | seed {SEED} | "
            f"`rq6_calibration.csv`, `rq6_risk_coverage.csv`\n"
        )


if __name__ == "__main__":
    main()
