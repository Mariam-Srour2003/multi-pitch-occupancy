"""H4: is the compact backbone genuinely indistinguishable from ViT? (RQ2)

Pre-registered as: *ConvNeXtV2-Tiny + head is statistically indistinguishable from
ViT-Base + head on macro-F1 under grouped splitting, while being >= 2x faster per frame.*
Family: model comparisons, all pairs. Decision rule as written: *a null result confirms H4.*

**That decision rule is wrong and this script does not follow it.** "We failed to find a
difference" is absence of evidence; on 907 frames that are 99% one class it is close to
guaranteed whatever the models do. An equivalence claim needs the opposite shape of test:
an interval for the difference that lies entirely **inside** a margin declared in advance.
So the margin is declared here - 2 macro-F1 points, borrowed from H2's own threshold so it
is not chosen to fit the answer - and the verdict is read off the interval:

* interval entirely inside +/- margin  -> **equivalent** (H4 supported)
* interval entirely outside            -> **different** (H4 refuted)
* interval straddling the margin       -> **inconclusive**, the honest third answer that
  the pre-registered rule collapses into "confirmed"

Two further things this reports that the hypothesis assumes away:

**The metric is macro-F1, so the test is on macro-F1.** McNemar and paired-correctness
bootstraps answer a question about *accuracy*, and H2 was reported with a macro-F1 delta
beside an accuracy p-value that for one pair pointed the other way.
:func:`paired_bootstrap_metric_diff` resamples once per draw and scores both models on the
same frames.

**Effective sample size, per the standing rule.** The grouped test set is 907 frames but
**95** distinct scenes (`effective_sample_audit.csv`, on the split whose identity this file
records), so every comparison is run twice - on all frames and on one frame per scene - and
both are reported.

    uv run python experiments/h4_model_equivalence.py
"""

from __future__ import annotations

import csv
from collections import Counter
from itertools import combinations

import cv2
import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.dedup import DEFAULT_THRESHOLD, dhash, distinct_subset
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, grouped_split, split_identity
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.metrics import evaluate
from pitch_occupancy.evaluation.stats import (
    holm_bonferroni,
    mcnemar,
    paired_bootstrap_metric_diff,
)
from pitch_occupancy.vision.heads import LinearProbe

SEED = 42
RESAMPLES = 10_000
OUT = settings.results_dir / "h4_model_equivalence.csv"
PUBLISHED = settings.results_dir / "h1_h2_baseline_floor.csv"
LATENCY = settings.results_dir / "efficiency_latency.csv"

#: Declared before looking: 2 macro-F1 points, the same threshold H2 used for "competitive".
#: Reusing an existing number rather than inventing one is the point - a margin chosen after
#: seeing the interval is not a margin.
MARGIN = 0.02

#: The pre-registration says "4 models, all pairs" = 6 comparisons. Only three trained
#: probes are available on the full grouped split: OpenCLIP's image features cover 600 of
#: 1,578 frames, and its slice of this test set is 339 ACTIVE_PLAY against 6 EMPTY, which
#: cannot carry a macro-F1. The realised family is therefore 3 pairs, and Holm corrects over
#: 3 rather than 6 - declared here because a family that shrinks silently is a family that
#: was chosen after the fact.
CACHES = {"convnextv2": "convnextv2.npz", "dinov2": "dinov2.npz", "vit": "vit.npz"}

#: The pair H4 is actually about.
H4_PAIR = ("convnextv2", "vit")


def macro_f1(y_true, y_pred) -> float:
    return evaluate(list(y_true), list(y_pred)).macro_f1


def load_aligned(rows):
    """Rows every cache covers, with one aligned feature matrix per backbone."""
    loaded = {}
    for key, name in CACHES.items():
        d = np.load(settings.feature_cache_dir / name, allow_pickle=True)
        loaded[key] = ({str(f): i for i, f in enumerate(d["files"])}, d["features"])
    keep = [r for r in rows if all(r.file in idx for idx, _ in loaded.values())]
    if len(keep) != len(rows):
        print(f"  {len(rows) - len(keep)} row(s) missing from a cache - dropped, not zero-filled")
    return keep, {k: f[[i[r.file] for r in keep]] for k, (i, f) in loaded.items()}


def scene_indices(rows) -> list[int]:
    """One frame per distinct scene, by perceptual hash."""
    hashes = {}
    for i, row in enumerate(rows):
        image = cv2.imread(str(settings.dataset_dir / row.file))
        if image is not None:
            hashes[i] = dhash(image)
    return distinct_subset(hashes, threshold=DEFAULT_THRESHOLD)


def published_macro_f1() -> dict[str, float]:
    if not PUBLISHED.exists():
        return {}
    return {
        r["model"]: float(r["macro_f1"])
        for r in csv.DictReader(PUBLISHED.open(encoding="utf-8"))
        if r["split"].startswith("grouped")
    }


def latency_ratios() -> dict[str, tuple[float, float]]:
    """(single-frame median, 20-camera concurrent median) per backbone, in ms."""
    if not LATENCY.exists():
        return {}
    return {
        r["backbone"]: (float(r["single_median_ms"]), float(r["concurrent_median_ms"]))
        for r in csv.DictReader(LATENCY.open(encoding="utf-8"))
    }


def verdict(ci) -> str:
    """Equivalence read off the interval, not off a p-value."""
    if ci.low > -MARGIN and ci.high < MARGIN:
        return "equivalent"
    if ci.low > MARGIN or ci.high < -MARGIN:
        return "different"
    return "inconclusive"


def main() -> None:
    rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    rows, X = load_aligned(rows)
    split = grouped_split(rows, group_key="slot_id", seed=SEED)
    pos = {r.file: i for i, r in enumerate(rows)}
    tr = [pos[r.file] for r in split.train]
    te = [pos[r.file] for r in split.test]
    y_true = [r.class3 for r in split.test]

    identity = split_identity(split)
    # The seed is 42 here and 42 in `effective_sample_audit.py`, and the two scripts filter
    # to different feature caches - so a seed alone does not say they built the same split.
    # In September they did not (94 distinct scenes against 95, both correct); since the
    # reproducibility repair they do. Recording the identity is what makes that checkable,
    # and `tests/test_splits.py` compares the two files.
    print(f"grouped split: train {len(split.train)}  test {len(split.test)}  "
          f"identity {identity}")
    print(f"  test composition: {dict(Counter(y_true))}")

    preds: dict[str, list[str]] = {}
    for key in CACHES:
        probe = LinearProbe(key, seed=SEED).fit(X[key][tr], split.train)
        preds[key] = probe.predict(X[key][te], split.test)

    # Reproduce before extending - the rule the H3 harness follows.
    print("\n=== reproduction check against the published grouped-split macro-F1 ===")
    ref = published_macro_f1()
    ok = True
    for key in CACHES:
        got = macro_f1(y_true, preds[key])
        want = ref.get(key)
        agree = want is None or abs(got - want) < 5e-4
        ok &= agree
        shown = "-" if want is None else f"{want:.4f}"
        print(f"  {key:<14} {got:.4f}  published {shown}  {'ok' if agree else 'MISMATCH'}")
    if not ok:
        raise SystemExit("a macro-F1 does not reproduce the published CSV; fix that first")

    keep = scene_indices(list(split.test))
    print(f"\neffective sample: {len(keep)} distinct scenes of {len(split.test)} frames "
          f"({len(keep) / len(split.test):.1%})")
    print(f"  scene composition: {dict(Counter(y_true[i] for i in keep))}")

    pairs = list(combinations(CACHES, 2))
    records: list[dict] = []
    print(f"\n=== all {len(pairs)} pairs, margin +/-{MARGIN} macro-F1 ===")
    print(f"{'pair':<26}{'dMacroF1':>10}{'95% CI':>22}{'verdict':>14}{'McNemar p':>11}")

    for a, b in pairs:
        for label, idx in (("all frames", list(range(len(y_true)))), ("distinct scenes", keep)):
            yt = [y_true[i] for i in idx]
            pa = [preds[a][i] for i in idx]
            pb = [preds[b][i] for i in idx]
            ci = paired_bootstrap_metric_diff(
                yt, pa, pb, macro_f1, resamples=RESAMPLES, seed=SEED
            )
            mc = mcnemar(
                np.array([p == t for p, t in zip(pa, yt, strict=True)]),
                np.array([p == t for p, t in zip(pb, yt, strict=True)]),
            )
            v = verdict(ci)
            if label == "all frames":
                print(f"{a + ' vs ' + b:<26}{ci.estimate:+10.4f}"
                      f"{f'[{ci.low:+.4f}, {ci.high:+.4f}]':>22}{v:>14}{mc.p_value:11.3g}")
            records.append({
                "pair": f"{a}_vs_{b}", "sample": label, "n": len(idx),
                "split_identity": identity,
                "d_macro_f1": round(ci.estimate, 4),
                "ci_lo": round(ci.low, 4), "ci_hi": round(ci.high, 4),
                "margin": MARGIN, "equivalence_verdict": v,
                "mcnemar_p_accuracy": f"{mc.p_value:.3e}",
                "mcnemar_effect_g": round(mc.effect_size, 3),
                "n_discordant": mc.n_discordant,
            })

    # Holm within the declared family, on the accuracy-level tests only: the equivalence
    # verdicts are interval-based and a correction does not apply to them.
    for sample in ("all frames", "distinct scenes"):
        rs = [r for r in records if r["sample"] == sample]
        adj, rej = holm_bonferroni([float(r["mcnemar_p_accuracy"]) for r in rs])
        for r, p, k in zip(rs, adj, rej, strict=True):
            r["mcnemar_p_holm"] = f"{p:.3e}"
            r["mcnemar_differs"] = k

    lat = latency_ratios()
    print("\n=== H4's second clause: is ConvNeXtV2 >= 2x faster than ViT? ===")
    c, v_ = H4_PAIR
    if c in lat and v_ in lat:
        s_ratio = lat[v_][0] / lat[c][0]
        k_ratio = lat[v_][1] / lat[c][1]
        print(f"  single frame  : {lat[v_][0]:.1f} / {lat[c][0]:.1f} = {s_ratio:.2f}x"
              f"  {'meets 2x' if s_ratio >= 2 else 'BELOW 2x'}")
        print(f"  20 concurrent : {lat[v_][1]:.1f} / {lat[c][1]:.1f} = {k_ratio:.2f}x"
              f"  {'meets 2x' if k_ratio >= 2 else 'BELOW 2x'}   <- the condition the")
        print("                  pre-registration actually names")
        records.append({
            "pair": "latency_convnextv2_vs_vit", "sample": "single_frame_median",
            "split_identity": "",  # latency is measured, not split-derived
            "n": "", "d_macro_f1": "", "ci_lo": "", "ci_hi": "", "margin": "",
            "equivalence_verdict": f"{s_ratio:.2f}x", "mcnemar_p_accuracy": "",
            "mcnemar_effect_g": "", "n_discordant": "",
        })
        records.append({
            "pair": "latency_convnextv2_vs_vit", "sample": "concurrent20_median",
            "n": "", "d_macro_f1": "", "ci_lo": "", "ci_hi": "", "margin": "",
            "equivalence_verdict": f"{k_ratio:.2f}x", "mcnemar_p_accuracy": "",
            "mcnemar_effect_g": "", "n_discordant": "",
        })

    h4 = next(r for r in records
              if r["pair"] == f"{H4_PAIR[0]}_vs_{H4_PAIR[1]}" and r["sample"] == "all frames")
    print("\n=== H4 verdict ===")
    print(f"  accuracy half: {h4['equivalence_verdict']} "
          f"(d {h4['d_macro_f1']:+}, CI [{h4['ci_lo']:+}, {h4['ci_hi']:+}], margin +/-{MARGIN})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # `extrasaction="ignore"` below means a key absent from this list is dropped without a
    # word, which is how `split_identity` was added to every record and reached the file on
    # none of them. Anything worth recording has to be named here too.
    cols = ["pair", "sample", "split_identity", "n", "d_macro_f1", "ci_lo", "ci_hi", "margin",
            "equivalence_verdict", "mcnemar_p_accuracy", "mcnemar_p_holm",
            "mcnemar_differs", "mcnemar_effect_g", "n_discordant"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {OUT.name}")

    record(
        "H4 model equivalence",
        "`python experiments/h4_model_equivalence.py`",
        f"`{OUT.name}`",
        f"ConvNeXtV2 vs ViT: {h4['equivalence_verdict']} at margin {MARGIN}",
    )


if __name__ == "__main__":
    main()
