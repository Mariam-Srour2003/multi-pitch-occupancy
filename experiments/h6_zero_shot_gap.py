"""H6 - does zero-shot lag the trained probes? (WP4-T4b, RQ1)

The last of the six pre-registered hypotheses never to have been reported. As written:

> **H6 - Zero-shot lags trained probes.** OpenCLIP zero-shot is significantly worse than
> every trained probe on macro-F1, quantifying the cost of a no-label deployment at a new
> site. **Family:** zero-shot vs trained (4 comparisons).

It was blocked on two things, and both are now gone.

* **Coverage.** OpenCLIP's image cache held 600 of 1,578 frames, and that slice of the
  grouped test set was 339 ACTIVE_PLAY against 6 EMPTY - a macro-F1 resting on six frames.
  The cache now covers every development frame.
* **Selection bias.** The obvious zero-shot arm was `prompt_search.py`'s winner, which was
  chosen as the best of 375 prompt sets *on the folds it reports*. Amendment A6 says a
  searched configuration is **selected, not tested**, so using it here would have tested
  the search rather than the model. The arm is `vision.zeroshot.DECLARED_PROMPT_SET`
  instead: first descriptor per class, all five templates, fixed by position.

What is reported
----------------

**The pre-registered analysis, and then a robustness check that is labelled as one.** The
declared comparison is macro-F1 under the grouped split. WP4-T1 has since measured that
protocol's seed-to-seed spread at +-0.24, which is the same order as the differences being
tested, so the same comparison is repeated across five seeds. Seed 42 is the pre-registered
result; the other four say whether it is a property of the models or of one partition.

Each pair carries all three things the plan asks for and the earlier hypotheses lacked:

* a **paired bootstrap interval** on the macro-F1 difference, resampling once per draw so
  both models face the same frames - `mcnemar` works on per-frame *correctness*, which is
  accuracy, and H2 was once published with a macro-F1 delta beside an accuracy p-value that
  pointed the other way;
* **McNemar** with **Cohen's g** beside it, so "significant" is never read as "large";
* **Holm-Bonferroni** over the realised family, whose size is declared below rather than
  discovered afterwards.

**And it measures the search's optimism.** The declared prompt set and the search's winner
are scored on the winner's own protocol. The gap between them is what A6 calls the
optimistic bias of a selected configuration, as a number rather than a caveat.

    uv run python -m experiments.h6_zero_shot_gap

**Reproduce before extending.** The trained probes' grouped-split macro-F1 at seed 42 is
checked against `h1_h2_baseline_floor.csv` before any comparison is computed.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.dedup import DEFAULT_THRESHOLD, dhash, distinct_subset
from pitch_occupancy.data.feature_cache import features_for, load_cache
from pitch_occupancy.data.manifest import ManifestRow, read_manifest
from pitch_occupancy.data.splits import Split, development_rows, grouped_split
from pitch_occupancy.data.taxonomy import CLASS3_ORDER, Class3
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.metrics import evaluable_subset, evaluate
from pitch_occupancy.evaluation.stats import (
    holm_bonferroni,
    mcnemar,
    paired_bootstrap_metric_diff,
)
from pitch_occupancy.vision.backbones import BACKBONES
from pitch_occupancy.vision.heads import LinearProbe
from pitch_occupancy.vision.zeroshot import (
    DECLARED_PROMPT_SET,
    DESCRIPTORS,
    TEMPLATES,
    PromptSet,
    classify,
    encode_prompts,
)

SEED = 42
CACHE = settings.feature_cache_dir
RESULTS = settings.results_dir
OUT = RESULTS / "h6_zero_shot_gap.csv"
PUBLISHED = RESULTS / "h1_h2_baseline_floor.csv"
SEARCH_BEST = RESULTS / "prompt_search_best.json"
TOLERANCE = 5e-4
RESAMPLES = 2000

ZERO_SHOT = "clip_zeroshot"


def macro_f1(y_true, y_pred) -> float:
    return evaluate(y_true, y_pred).macro_f1


# --- the two arms ------------------------------------------------------------------


def zero_shot_labels(rows: list[ManifestRow], prompts: PromptSet) -> dict[str, str] | None:
    """One label per frame from a prompt set. No training, so no split is involved."""
    path = CACHE / "clip_image_features.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    index = {str(f): i for i, f in enumerate(z["files"])}
    missing = [r.file for r in rows if r.file not in index]
    if missing:
        print(f"  ({ZERO_SHOT}: {len(missing)} frame(s) not embedded - cannot run H6)")
        return None

    from experiments.prompt_search import make_text_encoder

    directions = encode_prompts(prompts, list(CLASS3_ORDER), make_text_encoder())
    X = z["features"][[index[r.file] for r in rows]]
    return dict(zip([r.file for r in rows], classify(X, directions, list(CLASS3_ORDER)),
                    strict=True))


def probe_predictions(split: Split, rows: list[ManifestRow]) -> dict[str, list[str]]:
    """Test-set predictions per trained backbone, plus the zero-shot arm's, aligned."""
    pos = {r.file: i for i, r in enumerate(rows)}
    tr = [pos[r.file] for r in split.train]
    te = [pos[r.file] for r in split.test]
    out: dict[str, list[str]] = {}
    for key in sorted(BACKBONES):
        try:
            cached = load_cache(key, CACHE)
        except FileNotFoundError:
            continue
        X, kept = features_for(cached, rows)
        if len(kept) != len(rows):
            continue
        probe = LinearProbe(key, seed=SEED).fit(X[tr], split.train)
        out[key] = probe.predict(X[te], split.test)
    return out


# --- the effective sample ------------------------------------------------------------


def distinct_scene_positions(rows: list[ManifestRow]) -> list[int]:
    """One position per distinct scene, at the project's default near-duplicate threshold.

    Every frame-level test in this project runs on footage sampled every fifteen seconds
    from fixed cameras, so the nominal n overstates the evidence. A comparison that does
    not survive this recount has not been shown.
    """
    import cv2

    hashes: dict[int, int] = {}
    for i, r in enumerate(rows):
        image = cv2.imread(str(settings.dataset_dir / r.file))
        if image is not None:
            hashes[i] = dhash(image)
    return distinct_subset(hashes, threshold=DEFAULT_THRESHOLD)


# --- the pre-registered comparison ----------------------------------------------------


def compare(split: Split, rows: list[ManifestRow], zs: dict[str, str],
            scene_positions: bool) -> tuple[list[dict], dict]:
    """Zero-shot against each trained probe on one split."""
    truth = [r.class3 for r in split.test]
    preds = probe_predictions(split, rows)
    if not preds:
        raise SystemExit("no trained probe available; run `uv run pitch cache`")
    preds[ZERO_SHOT] = [zs[r.file] for r in split.test]

    scores = {}
    for name, pred in preds.items():
        yt, yp, _ = evaluable_subset(truth, pred)
        scores[name] = (evaluate(yt, yp).macro_f1, evaluate(truth, pred).accuracy)

    rows_out, tests = [], []
    for name in sorted(preds):
        if name == ZERO_SHOT:
            continue
        # Direction: zero-shot minus trained. H6 predicts this is negative and significant.
        ci = paired_bootstrap_metric_diff(
            *_evaluable_triple(truth, preds[ZERO_SHOT], preds[name]),
            macro_f1, resamples=RESAMPLES, seed=SEED,
        )
        ok_zs = np.array([p == t for p, t in zip(preds[ZERO_SHOT], truth, strict=True)])
        ok_tr = np.array([p == t for p, t in zip(preds[name], truth, strict=True)])
        m = mcnemar(ok_zs, ok_tr)
        tests.append(m.p_value)
        rows_out.append({
            "split": split.name, "comparison": f"{ZERO_SHOT}_minus_{name}",
            "zero_shot_macro_f1": round(scores[ZERO_SHOT][0], 4),
            "trained_macro_f1": round(scores[name][0], 4),
            "delta_macro_f1": round(ci.estimate, 4),
            "ci_low": round(ci.low, 4), "ci_high": round(ci.high, 4),
            "zero_shot_accuracy": round(scores[ZERO_SHOT][1], 4),
            "trained_accuracy": round(scores[name][1], 4),
            "mcnemar_p": f"{m.p_value:.3e}", "cohens_g": round(m.effect_size, 3),
            "n_discordant": m.n_discordant, "n_test": len(truth),
        })

    adjusted, rejected = holm_bonferroni(tests)
    for row, p_adj, sig in zip(rows_out, adjusted, rejected, strict=True):
        row["p_holm"] = f"{p_adj:.3e}"
        row["significant"] = sig
        # H6 predicts zero-shot is *worse*. A significant difference in the other direction
        # refutes it rather than supporting it, so the verdict reads the sign as well.
        row["supports_h6"] = bool(sig and row["delta_macro_f1"] < 0)

    if scene_positions:
        picks = distinct_scene_positions(list(split.test))
        for row in rows_out:
            name = row["comparison"].split("_minus_")[1]
            a = np.array([p == t for p, t in zip(preds[ZERO_SHOT], truth, strict=True)])[picks]
            b = np.array([p == t for p, t in zip(preds[name], truth, strict=True)])[picks]
            row["n_distinct_scenes"] = len(picks)
            row["mcnemar_p_scenes"] = f"{mcnemar(a, b).p_value:.3e}"

    return rows_out, scores


def _evaluable_triple(truth, pred_a, pred_b):
    """Restrict all three vectors to the classes the macro average is defined over.

    `evaluable_subset` takes one prediction vector; a paired comparison needs both cut the
    same way or the two models are scored on different frames, which is exactly the pairing
    the bootstrap exists to preserve.
    """
    yt, _, dropped = evaluable_subset(truth, pred_a)
    if not dropped:
        return list(truth), list(pred_a), list(pred_b)
    keep = [i for i, t in enumerate(truth) if t not in dropped]
    return ([truth[i] for i in keep], [pred_a[i] for i in keep], [pred_b[i] for i in keep])


# --- is the verdict a property of the model or of one prompt? ---------------------------


def prompt_space_sensitivity(
    split: Split, rows: list[ManifestRow], best_trained: float
) -> dict | None:
    """Grouped-split macro-F1 for **every** prompt set in the declared space.

    A hypothesis about zero-shot should not turn on which descriptor happened to be listed
    first. Declaring one set in advance is the right way to get an unbiased *point*
    estimate, but it leaves the obvious objection - pick another and the verdict flips -
    unanswered. So the whole space is scored and the *distribution* reported.

    This is not a search and nothing is selected from it. The quantity of interest is what
    fraction of prompt sets beat the best trained probe: if it is nearly all of them, H6's
    refutation is a property of the model rather than of a lucky phrase.
    """
    path = CACHE / "clip_image_features.npz"
    if not path.exists():
        return None
    import itertools

    from experiments.prompt_search import make_text_encoder

    z = np.load(path, allow_pickle=True)
    index = {str(f): i for i, f in enumerate(z["files"])}
    test = list(split.test)
    X = z["features"][[index[r.file] for r in test]]
    truth = [r.class3 for r in test]

    classes = list(CLASS3_ORDER)
    encode = make_text_encoder()
    cache: dict[str, np.ndarray] = {}

    def encode_cached(phrases: list[str]) -> np.ndarray:
        missing = [p for p in phrases if p not in cache]
        if missing:
            for phrase, vec in zip(missing, encode(missing), strict=True):
                cache[phrase] = vec
        return np.stack([cache[p] for p in phrases])

    scores: list[float] = []
    combos = list(itertools.product(*(DESCRIPTORS[c] for c in classes)))
    for n_templates in (1, 3, 5):
        templates = tuple(TEMPLATES[:n_templates])
        for combo in combos:
            pset = PromptSet(dict(zip(classes, combo, strict=True)), templates)
            pred = classify(X, encode_prompts(pset, classes, encode_cached), classes)
            yt, yp, _ = evaluable_subset(truth, pred)
            scores.append(evaluate(yt, yp).macro_f1)

    arr = np.array(scores)
    declared_pred = classify(X, encode_prompts(DECLARED_PROMPT_SET, classes, encode_cached), classes)
    yt, yp, _ = evaluable_subset(truth, declared_pred)
    return {
        "n_prompt_sets": int(arr.size),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "worst": float(arr.min()),
        "best": float(arr.max()),
        "fraction_beating_best_trained": float(np.mean(arr > best_trained)),
        "best_trained": best_trained,
        "declared": float(evaluate(yt, yp).macro_f1),
        "declared_percentile": float((arr < evaluate(yt, yp).macro_f1).mean()),
    }


# --- the search's optimism -------------------------------------------------------------


def selection_bias(rows: list[ManifestRow]) -> dict | None:
    """How much better the *selected* prompt set looks than a pre-declared one.

    Scored on the prompt search's own protocol - cross-venue play recall with the false-play
    control - because that is the protocol the winner was selected under. The gap is not an
    unbiased estimate of the bias, but it is the concrete version of A6's caveat: it says
    how much of the winner's margin could be the search rather than the prompt.
    """
    if not SEARCH_BEST.exists():
        return None
    from experiments.prompt_search import image_features, make_text_encoder, score
    from pitch_occupancy.data.splits import leave_one_group_out

    best = json.loads(SEARCH_BEST.read_text(encoding="utf-8"))["best"]
    folds = [f for f in leave_one_group_out(rows, group_key="venue")
             if not f.name.endswith("venue_01")]
    classes = list(CLASS3_ORDER)
    encode = make_text_encoder()
    X = image_features(rows)

    selected = PromptSet(
        descriptors={Class3.EMPTY: best["desc_EMPTY"],
                     Class3.ACTIVE_PLAY: best["desc_ACTIVE_PLAY"],
                     Class3.MAINTENANCE_NON_SPORTING: best["desc_MAINTENANCE_NON_SPORTING"]},
        templates=tuple(DECLARED_PROMPT_SET.templates[:best["n_templates"]]),
    )
    out = {}
    for label, pset in (("declared", DECLARED_PROMPT_SET), ("selected", selected)):
        out[label] = score(rows, X, encode_prompts(pset, classes, encode), classes, folds)
    return out


# --- guard --------------------------------------------------------------------------


def check_reproduces(scores: dict[str, tuple[float, float]]) -> None:
    if not PUBLISHED.exists():
        print("  (h1_h2_baseline_floor.csv absent - nothing to check against)")
        return
    published = {
        r["model"]: float(r["macro_f1"])
        for r in csv.DictReader(PUBLISHED.open(encoding="utf-8"))
        if r["split"].startswith("grouped")
    }
    bad, checked = [], 0
    for name, (macro, _) in scores.items():
        if name not in published:
            continue
        checked += 1
        if abs(macro - published[name]) > TOLERANCE:
            bad.append(f"{name}: {macro:.4f} vs published {published[name]:.4f}")
    for line in bad:
        print(f"  MISMATCH {line}")
    if bad:
        raise SystemExit(f"{len(bad)} of {checked} published numbers did not reproduce.")
    print(f"  {checked} published numbers reproduce to {TOLERANCE:g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5, help="grouped splits to repeat over")
    ap.add_argument("--skip-scene-audit", action="store_true")
    ap.add_argument("--skip-selection-bias", action="store_true")
    ap.add_argument("--skip-prompt-space", action="store_true",
                    help="skip the sensitivity sweep over every declared prompt set")
    args = ap.parse_args()

    rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    print(f"development rows: {len(rows)}")
    print(f"declared prompt set: {DECLARED_PROMPT_SET.label()}\n")

    zs = zero_shot_labels(rows, DECLARED_PROMPT_SET)
    if zs is None:
        raise SystemExit("H6 needs OpenCLIP image features for every development frame")

    seeds = [SEED + i for i in range(args.seeds)]
    records: list[dict] = []
    for i, seed in enumerate(seeds):
        split = grouped_split(rows, group_key="slot_id", seed=seed)
        out, scores = compare(split, rows, zs, scene_positions=(not args.skip_scene_audit and i == 0))
        if seed == SEED:
            print("reproducing the published table before extending it")
            check_reproduces(scores)
            print()
        records.extend(out)

    # --- the pre-registered result ------------------------------------------------
    declared = [r for r in records if r["split"].endswith(f"seed{SEED}")]
    n_family = len(declared)
    print(f"=== H6, as pre-registered: grouped split, seed {SEED} ===")
    print(f"realised family: {n_family} comparison(s) (the hypothesis anticipated 4)")
    print(f"\n{'comparison':34}{'zero-shot':>10}{'trained':>9}{'delta':>9}"
          f"{'95% CI':>20}{'p (Holm)':>11}{'g':>7}{'H6?':>6}")
    for r in declared:
        print(f"{r['comparison']:34}{r['zero_shot_macro_f1']:>10.4f}{r['trained_macro_f1']:>9.4f}"
              f"{r['delta_macro_f1']:>+9.4f}"
              f"  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]{r['p_holm']:>11}"
              f"{r['cohens_g']:>7.3f}{'yes' if r['supports_h6'] else 'no':>6}")

    supporting = sum(r["supports_h6"] for r in declared)
    against = sum(r["significant"] and r["delta_macro_f1"] > 0 for r in declared)
    print("\n  H6 requires zero-shot to be significantly worse than *every* trained probe.")
    print(f"  It is significantly worse than {supporting} of {n_family}, "
          f"and significantly BETTER than {against}.")

    if not args.skip_scene_audit and any("n_distinct_scenes" in r for r in declared):
        scenes = next(r["n_distinct_scenes"] for r in declared if "n_distinct_scenes" in r)
        print(f"\n  effective sample: {scenes} distinct scenes in {declared[0]['n_test']} frames")
        for r in declared:
            if "mcnemar_p_scenes" in r:
                print(f"    {r['comparison']:34} p on distinct scenes {r['mcnemar_p_scenes']}")

    # --- is it a property of the models or of one split? ---------------------------
    print(f"\n=== the same comparison across {len(seeds)} grouped splits ===")
    print("  (WP4-T1 measured this protocol's seed spread at +-0.24 macro-F1, which is why)")
    print(f"\n{'comparison':34}{'mean delta':>12}{'sd':>8}{'splits favouring zero-shot':>28}")
    for name in sorted({r["comparison"] for r in records}):
        deltas = [r["delta_macro_f1"] for r in records if r["comparison"] == name]
        print(f"{name:34}{np.mean(deltas):>+12.4f}{np.std(deltas, ddof=1):>8.3f}"
              f"{sum(d > 0 for d in deltas):>20} of {len(deltas)}")

    # --- would another pre-declared prompt have said otherwise? --------------------
    space = None
    if not args.skip_prompt_space:
        first = grouped_split(rows, group_key="slot_id", seed=SEED)
        best_trained = max(r["trained_macro_f1"] for r in declared)
        space = prompt_space_sensitivity(first, rows, best_trained)
    if space:
        print("\n=== every prompt set in the declared space, not just the declared one ===")
        print(f"  {space['n_prompt_sets']} prompt sets scored on the same grouped split.")
        print("  Nothing is selected from this - the distribution is the point.")
        print(f"\n  best trained probe            {space['best_trained']:.4f}")
        print(f"  declared prompt set           {space['declared']:.4f}  "
              f"({space['declared_percentile']:.0%} of the space is below it)")
        print(f"  prompt space  worst / median / best   "
              f"{space['worst']:.4f} / {space['median']:.4f} / {space['best']:.4f}")
        print(f"  10th-90th percentile          {space['p10']:.4f} - {space['p90']:.4f}")
        print(f"\n  {space['fraction_beating_best_trained']:.1%} of prompt sets beat "
              f"the best trained probe.")
        records.append({
            "split": "PROMPT_SPACE", "comparison": "fraction_beating_best_trained",
            "zero_shot_macro_f1": round(space["median"], 4),
            "trained_macro_f1": round(space["best_trained"], 4),
            "delta_macro_f1": round(space["median"] - space["best_trained"], 4),
            "ci_low": round(space["p10"], 4), "ci_high": round(space["p90"], 4),
            "zero_shot_accuracy": round(space["declared"], 4),
            # The extremes, because the write-up quotes the span and the claims ledger
            # could not re-derive it from percentiles alone.
            "trained_accuracy": round(space["worst"], 4),
            "mcnemar_p": round(space["best"], 4),
            "cohens_g": round(space["best"] - space["worst"], 4),
            "n_discordant": "", "n_test": space["n_prompt_sets"], "p_holm": "",
            "significant": round(space["fraction_beating_best_trained"], 4),
            "supports_h6": "",
        })

    # --- what the search's selection was worth -------------------------------------
    bias = None if args.skip_selection_bias else selection_bias(rows)
    if bias:
        print("\n=== what the prompt search's selection was worth (A6, quantified) ===")
        print("  scored on the search's own protocol - the folds its winner was chosen on")
        print(f"\n{'prompt set':12}{'play recall':>13}{'false play':>12}{'balanced':>11}")
        for label in ("declared", "selected"):
            s = bias[label]
            print(f"{label:12}{s['play_recall']:>13.4f}{s['false_play']:>12.4f}"
                  f"{s['balanced']:>11.4f}")
        gap = bias["selected"]["balanced"] - bias["declared"]["balanced"]
        print(f"\n  The winner leads a pre-declared prompt set by {gap:+.4f} balanced score")
        print("  on the data it was selected from. That margin is the search's optimism as")
        print("  much as the prompt's quality, which is why H6 uses the declared set.")
        records.append({
            "split": "SELECTION_BIAS", "comparison": "selected_minus_declared",
            "zero_shot_macro_f1": "", "trained_macro_f1": "",
            "delta_macro_f1": round(gap, 4), "ci_low": "", "ci_high": "",
            "zero_shot_accuracy": round(bias["declared"]["balanced"], 4),
            "trained_accuracy": round(bias["selected"]["balanced"], 4),
            "mcnemar_p": "", "cohens_g": "", "n_discordant": "", "n_test": "",
            "p_holm": "", "significant": "", "supports_h6": "",
        })

    fields = list(dict.fromkeys(k for r in records for k in r))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nwrote {OUT.name}")

    # --- the verdict, from all three checks rather than the first ------------------
    #
    # Taking the declared set alone would read "refuted": zero-shot beats every probe,
    # significantly, on all five splits. The sensitivity sweep is what stops that being the
    # answer, and a summary that ignored it would be the more confident and the less true.
    print("\n=== verdict on H6 ===")
    print("  H6 predicts zero-shot is significantly *worse* than every trained probe.")
    print(f"  Declared prompt set: worse than {supporting} of {n_family}, "
          f"better than {against}. Direction is against the hypothesis.")
    confirmed = supporting == n_family
    fragile = space is not None and space["fraction_beating_best_trained"] < 0.5
    survives = any(
        float(r.get("mcnemar_p_scenes", 1.0)) <= 0.05 for r in declared
    ) if any("mcnemar_p_scenes" in r for r in declared) else None

    if confirmed:
        verdict = "confirmed"
    elif fragile:
        verdict = "inconclusive"
        print(f"\n  But it is not refuted either. The declared set scores above "
              f"{space['declared_percentile']:.0%} of the prompt space, and only "
              f"{space['fraction_beating_best_trained']:.1%} of prompt sets beat")
        print(f"  the best probe - the median one ({space['median']:.4f}) loses to it. The "
              f"answer depends on which prompt was declared, so it is not an answer.")
    else:
        verdict = "refuted"

    if survives is False:
        print("\n  And the frame-level significance does not survive the effective sample: "
              "on distinct scenes no comparison is significant.")
    if space:
        span = space["best"] - space["worst"]
        spread_trained = max(r["trained_macro_f1"] for r in declared) - min(
            r["trained_macro_f1"] for r in declared)
        print(f"\n  The finding worth keeping is the spread, not the ranking: prompt choice "
              f"moves macro-F1 by {span:.3f}")
        print(f"  ({space['worst']:.3f} to {space['best']:.3f}) while the three trained "
              f"backbones span {spread_trained:.3f}.")
        print("  On this benchmark the prompt matters more than the model.")

    print(f"\n  H6: {verdict.upper()}")
    record(
        "H6 zero-shot vs trained probes",
        "`python -m experiments.h6_zero_shot_gap`",
        f"`{OUT.name}`",
        f"declared prompt set; realised family {n_family}; H6 {verdict}"
        + (f"; prompt choice spans {space['worst']:.3f}-{space['best']:.3f} macro-F1"
           if space else ""),
    )


if __name__ == "__main__":
    main()
