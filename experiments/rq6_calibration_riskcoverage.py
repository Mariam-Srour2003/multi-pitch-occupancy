"""RQ6 - how much human review buys a given reliability? (WP4-T5, WP4-T9)

The REVIEW band is the system's escape hatch: uncertain frames go to a person. That makes
confidence an operational quantity, not a diagnostic one - it decides how many slots a
facility has to inspect by hand. Two measurements follow:

1. **Calibration.** Is a 90%-confident prediction right 90% of the time? Reported as ECE,
   before and after temperature scaling fitted on a held-out slice of the training venues.
   Temperature only rescales logits, so accuracy is untouched by construction; that is
   asserted here rather than assumed.

2. **Risk-coverage, as a band rather than a line.** Sweep the confidence threshold: answer
   the most confident fraction automatically, defer the rest. The output is the operating
   point a manager can act on - *"to reach 99% precision on automated verdicts, expect to
   review N% of slots."*

   **"The most confident k" is only defined when the k-th and (k+1)-th confidences differ**,
   and on this data they very often do not: DINOv2's fitted temperature hits its grid
   boundary at 0.05, which sharpens the probabilities until **890 of 907** calibrated
   confidences are exactly 1.0. Sorting them puts equally confident frames in an order the
   model never expressed, and the published curve moved by up to 0.125 between runs for that
   reason alone. So each coverage now carries the best and worst accuracy that confidence
   permits, and the operating point is read off the **lower** bound - a promise that holds
   whichever way the ties fall. Where confidences are distinct the band is a line and
   nothing changes.

    uv run python experiments/rq6_calibration_riskcoverage.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import features_for, load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, grouped_split
from pitch_occupancy.evaluation.calibration import (
    confidence_ties,
    coverage_for_target_accuracy,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    risk_coverage_band,
)
from pitch_occupancy.evaluation.experiment_log import record
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
    reliability_records: list[dict] = []

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
        # Persisted so the reliability diagram can be drawn from a CSV rather than by
        # recomputing the probes inside the figure script - the rule every other figure here
        # follows, and the reason a figure cannot silently disagree with its source.
        for b in bins:
            reliability_records.append({
                "model": key, "bin_lo": round(b.lo, 4), "bin_hi": round(b.hi, 4),
                "n": b.n, "mean_confidence": round(b.mean_confidence, 4),
                "accuracy": round(b.accuracy, 4), "gap": round(b.gap, 4),
            })
        worst = max(bins, key=lambda b: abs(b.gap))
        print(
            f"  worst bin [{worst.lo:.1f},{worst.hi:.1f}] n={worst.n}: "
            f"confidence {worst.mean_confidence:.3f} vs accuracy {worst.accuracy:.3f}\n"
        )

        n_tied, largest_tie = confidence_ties(conf_cal)
        if n_tied:
            print(f"  ! {n_tied}/{len(conf_cal)} calibrated confidences are tied "
                  f"(largest group {largest_tie}) - the curve below is a band, not a line")
        for cov, acc_lo, acc_hi, review in risk_coverage_band(conf_cal, ok_cal, points=60):
            curve_records.append(
                {"model": key, "coverage": round(cov, 4),
                 "accuracy_worst": round(acc_lo, 4), "accuracy_best": round(acc_hi, 4),
                 "review_rate": round(review, 4),
                 "n_tied": n_tied, "largest_tie_group": largest_tie}
            )
        row = {
            "model": key, "accuracy": round(float(ok_raw.mean()), 4),
            "temperature": round(temperature, 3),
            "ece_raw": round(ece_raw, 4), "ece_calibrated": round(ece_cal, 4),
            "n_tied_confidences": confidence_ties(conf_cal)[0],
            "largest_tie_group": confidence_ties(conf_cal)[1],
            "n_test": int(conf_cal.size),
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
    with (RESULTS / "rq6_reliability.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(reliability_records[0]))
        w.writeheader()
        w.writerows(reliability_records)
    with (RESULTS / "rq6_risk_coverage.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "model", "coverage", "accuracy_worst", "accuracy_best", "review_rate",
            "n_tied", "largest_tie_group"])
        w.writeheader()
        w.writerows(curve_records)
    print("wrote results/rq6_calibration.csv and results/rq6_risk_coverage.csv")

    record(
        "RQ6",
        "`python experiments/rq6_calibration_riskcoverage.py`",
        f"seed {SEED}",
        "`rq6_calibration.csv`, `rq6_risk_coverage.csv`",
    )


if __name__ == "__main__":
    main()
