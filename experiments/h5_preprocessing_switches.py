"""H5 - do the preprocessing switches contribute measurably? (WP3-T8, WP4-T4b, RQ1)

The last of the six pre-registered hypotheses to be reported. As written:

> **H5 - Preprocessing contributes measurably.** ROI masking improves macro-F1
> significantly under grouped splitting; CLAHE improves it on the night subset
> specifically. **Family:** preprocessing ablations (5 switches). **Decision rule:** each
> switch reported with delta and CI; adopted only if the gain is significant after
> correction.

It has two clauses and they are in different states, so it is reported clause by clause
rather than as one verdict.

The ROI clause cannot be run, and that is a fact about the data
-------------------------------------------------------------

**ROI masking has never been evaluated and cannot be**, because no ROI polygon exists:
`configs/cameras.json` was never produced, WP3-T1 needs a human to draw one per camera, and
`roi_mask` returns the frame untouched when the polygon is absent. All 88 search evaluations
carry `roi: False`, and `SWITCHES` deliberately excludes `roi` for exactly this reason - a
switch that cannot change the image must never enter a search, or the search records "ROI
masking does not help" from a transform that never ran.

So the clause is reported **unrunnable**, not refuted. The distinction matters: a null
result here would be manufactured.

The CLAHE clause can be run, and is
-----------------------------------

`clahe='on'` and `clahe='auto'` were both evaluated in round 1, differing from the baseline
in that switch alone, and their feature caches survive - so this is re-scoring, not
re-embedding.

**Stratified by lighting on venue_01 only.** The clause is about the night subset, and
`lighting` elsewhere is a brightness proxy that WP3-T5 found wrong for at least three clip
venues. Within venue_01 it is not a proxy: the two recording days are a daylight morning and
a floodlit night, confirmed by the camera fingerprint audit. The grouped split's test set is
venue_01 anyway, so the restriction costs nothing and removes the label error.

**The day column would be the control, and this corpus cannot supply it.** CLAHE equalises
contrast, so a gain confined to night is the hypothesis and a gain in both columns is
something else. But venue_01 has exactly two recording days - one daylight morning, one
floodlit night - and holding whole slots out fills the test side from *one* of them: across
five seeds four test sets are entirely night and the fifth entirely day. There is no
within-split contrast to draw, so the clause's "specifically" is untestable here, and the
two columns below come from different partitions and are reported as two observations rather
than as a comparison.

    uv run python -m experiments.h5_preprocessing_switches

**Reproduce before extending.** Each arm's cross-venue recall is checked against
`preprocess_search.json` before any new number is computed.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.dedup import DEFAULT_THRESHOLD, dhash, distinct_subset
from pitch_occupancy.data.manifest import ManifestRow, read_manifest
from pitch_occupancy.data.splits import Split, development_rows, grouped_split, leave_one_group_out
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.metrics import evaluable_subset, evaluate
from pitch_occupancy.evaluation.stats import (
    holm_bonferroni,
    mcnemar,
    paired_bootstrap_metric_diff,
)
from pitch_occupancy.vision.heads import LinearProbe
from pitch_occupancy.vision.preprocess import SWITCHES

SEED = 42
PLAY = "C2_ACTIVE_PLAY"
SEARCH_CACHE = settings.feature_cache_dir / "search"
SEARCH_JSON = settings.results_dir / "preprocess_search.json"
OUT = settings.results_dir / "h5_preprocessing_switches.csv"
CAMERAS = settings.root / "configs" / "cameras.json" if hasattr(settings, "root") else None
TOLERANCE = 5e-4
RESAMPLES = 2000

#: The clause's subject and its control. CLAHE equalises contrast, so a gain confined to
#: night is the hypothesis; a gain in both columns is something else.
SUBSET = {"night": "night", "day": "day"}


def macro_f1(y_true, y_pred) -> float:
    return evaluate(y_true, y_pred).macro_f1


def arms() -> dict[str, dict[str, str]]:
    """``{model: {clahe value: cache hash}}`` for the baseline and the two CLAHE arms.

    Read from the search's own record rather than hardcoded, so an arm that was never run
    is absent instead of silently pointing at the wrong cache.
    """
    blob = json.loads(SEARCH_JSON.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for ev in blob["evaluations"]:
        if ev["round"] > 1:
            continue
        value = ev["config"]["clahe"]
        if ev["round"] == 0:
            out.setdefault(ev["model"], {})["off"] = ev["hash"]
        elif value in ("on", "auto") and ev["label"] == f"clahe={value}":
            out.setdefault(ev["model"], {})[value] = ev["hash"]
    return {m: v for m, v in out.items() if len(v) >= 2}


def features(model: str, config_hash: str, rows: list[ManifestRow]) -> np.ndarray | None:
    path = SEARCH_CACHE / f"{model}_{config_hash}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    index = {str(f): i for i, f in enumerate(z["files"])}
    if not all(r.file in index for r in rows):
        return None
    return z["features"][[index[r.file] for r in rows]]


def cross_venue_recall(rows: list[ManifestRow], X: np.ndarray) -> float:
    """The metric the search ranked on, recomputed so the arm can be identified."""
    pos = {r.file: i for i, r in enumerate(rows)}
    folds = [f for f in leave_one_group_out(rows, group_key="venue")
             if not f.name.endswith("venue_01")]
    recalls = []
    for fold in folds:
        probe = LinearProbe("s", seed=SEED).fit(X[[pos[r.file] for r in fold.train]], fold.train)
        pred = probe.predict(X[[pos[r.file] for r in fold.test]], fold.test)
        truth = [r.class3 for r in fold.test]
        play = [(p, t) for p, t in zip(pred, truth, strict=True) if t == PLAY]
        recalls.append(sum(p == PLAY for p, _ in play) / len(play) if play else float("nan"))
    return float(np.mean(recalls))


def predictions(split: Split, rows: list[ManifestRow], X: np.ndarray) -> list[str]:
    pos = {r.file: i for i, r in enumerate(rows)}
    probe = LinearProbe("s", seed=SEED).fit(
        X[[pos[r.file] for r in split.train]], split.train
    )
    return probe.predict(X[[pos[r.file] for r in split.test]], split.test)


def lighting_subsets(split: Split) -> dict[str, list[int]]:
    """Positions in the test set for each lighting condition, venue_01 only.

    `lighting` is a brightness proxy and WP3-T5 found it wrong for at least three clip
    venues; within venue_01 it is not a proxy, because the two recording days are a daylight
    morning and a floodlit night. The grouped split's test set is venue_01 anyway.
    """
    out: dict[str, list[int]] = {name: [] for name in SUBSET}
    for i, row in enumerate(split.test):
        if row.venue != "venue_01":
            continue
        for name, value in SUBSET.items():
            if row.lighting == value:
                out[name].append(i)
    return {k: v for k, v in out.items() if v}


def distinct_scenes(rows: list[ManifestRow]) -> list[int]:
    import cv2

    hashes: dict[int, int] = {}
    for i, r in enumerate(rows):
        image = cv2.imread(str(settings.dataset_dir / r.file))
        if image is not None:
            hashes[i] = dhash(image)
    return distinct_subset(hashes, threshold=DEFAULT_THRESHOLD)


def compare(truth, base_pred, arm_pred, positions) -> dict:
    """One switch against the baseline on one subset, paired throughout."""
    t = [truth[i] for i in positions]
    a = [arm_pred[i] for i in positions]
    b = [base_pred[i] for i in positions]
    yt, ya, yb = _evaluable_triple(t, a, b)
    ci = paired_bootstrap_metric_diff(yt, ya, yb, macro_f1, resamples=RESAMPLES, seed=SEED)
    ok_a = np.array([p == x for p, x in zip(a, t, strict=True)])
    ok_b = np.array([p == x for p, x in zip(b, t, strict=True)])
    m = mcnemar(ok_a, ok_b)
    return {
        "n": len(t),
        "baseline_macro_f1": round(macro_f1(yt, yb), 4),
        "arm_macro_f1": round(macro_f1(yt, ya), 4),
        "delta": round(ci.estimate, 4),
        "ci_low": round(ci.low, 4), "ci_high": round(ci.high, 4),
        "mcnemar_p": m.p_value, "cohens_g": round(m.effect_size, 3),
        "n_discordant": m.n_discordant,
    }


def _evaluable_triple(truth, pred_a, pred_b):
    """Cut all three vectors the same way, or the comparison stops being paired."""
    yt, _, dropped = evaluable_subset(truth, pred_a)
    if not dropped:
        return list(truth), list(pred_a), list(pred_b)
    keep = [i for i, t in enumerate(truth) if t not in dropped]
    return ([truth[i] for i in keep], [pred_a[i] for i in keep], [pred_b[i] for i in keep])


def report_roi_clause() -> dict:
    """Why the first clause cannot be run, established from the code rather than asserted."""
    blob = json.loads(SEARCH_JSON.read_text(encoding="utf-8"))
    roi_values = {ev["config"].get("roi") for ev in blob["evaluations"]}
    polygons = (settings.results_dir.parent / "configs" / "cameras.json").exists()
    return {
        "roi_in_search_space": "roi" in SWITCHES,
        "roi_values_evaluated": sorted(str(v) for v in roi_values),
        "polygon_file_exists": polygons,
        "n_evaluations": len(blob["evaluations"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-scene-audit", action="store_true")
    args = ap.parse_args()

    rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    blob = json.loads(SEARCH_JSON.read_text(encoding="utf-8"))
    published = {(e["model"], e["config"]["clahe"], e["round"]): e["play_recall"]
                 for e in blob["evaluations"] if e["round"] <= 1}

    # --- clause 1 -----------------------------------------------------------------
    roi = report_roi_clause()
    print("=== H5, clause 1: ROI masking - UNRUNNABLE ===")
    print(f"  `roi` in the search space: {roi['roi_in_search_space']}")
    print(f"  values evaluated across {roi['n_evaluations']} runs: "
          f"{', '.join(roi['roi_values_evaluated'])}")
    print(f"  configs/cameras.json exists: {roi['polygon_file_exists']}")
    print("  `roi_mask` returns the frame untouched without a polygon, so an ROI arm would")
    print("  measure a transform that never ran. Reported unrunnable, not refuted - a null")
    print("  result here would be manufactured. Needs WP3-T1: one polygon per camera.\n")

    # --- clause 2 -----------------------------------------------------------------
    found = arms()
    if not found:
        raise SystemExit("no CLAHE arm found in the search record")

    records: list[dict] = []
    tests: list[tuple[str, float]] = []
    seeds = [SEED + i for i in range(args.seeds)]

    print("reproducing each arm's published recall before extending it")
    loaded: dict[tuple[str, str], np.ndarray] = {}
    for model, by_value in sorted(found.items()):
        for value, config_hash in sorted(by_value.items()):
            X = features(model, config_hash, rows)
            if X is None:
                print(f"  ({model}/{value}: no usable cache - skipped)")
                continue
            loaded[(model, value)] = X
            want = published.get((model, value, 0 if value == "off" else 1))
            got = cross_venue_recall(rows, X)
            if want is not None and abs(got - want) > TOLERANCE:
                raise SystemExit(
                    f"{model}/{value}: recomputed {got:.4f} against published {want:.4f}"
                )
            print(f"  {model:11} clahe={value:5} recall {got:.4f} reproduces")

    print(f"\n=== H5, clause 2: CLAHE on the night subset (grouped split, seed {SEED}) ===")
    header = (f"{'model':11}{'switch':8}{'subset':7}{'n':>6}{'baseline':>10}{'arm':>9}"
              f"{'delta':>9}{'95% CI':>20}{'p':>10}{'g':>7}")
    print(header)

    for (model, value), X in sorted(loaded.items()):
        if value == "off":
            continue
        base = loaded.get((model, "off"))
        if base is None:
            continue
        for seed in seeds:
            split = grouped_split(rows, group_key="slot_id", seed=seed)
            truth = [r.class3 for r in split.test]
            base_pred = predictions(split, rows, base)
            arm_pred = predictions(split, rows, X)
            subsets = lighting_subsets(split)
            for subset, positions in sorted(subsets.items()):
                stats = compare(truth, base_pred, arm_pred, positions)
                row = {"model": model, "switch": f"clahe={value}", "subset": subset,
                       "seed": seed, **stats}
                records.append(row)
                if seed == SEED:
                    tests.append((f"{model}/{value}/{subset}", stats["mcnemar_p"]))
                    print(f"{model:11}{value:8}{subset:7}{stats['n']:>6}"
                          f"{stats['baseline_macro_f1']:>10.4f}{stats['arm_macro_f1']:>9.4f}"
                          f"{stats['delta']:>+9.4f}"
                          f"  [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]"
                          f"{stats['mcnemar_p']:>10.2e}{stats['cohens_g']:>7.3f}")

    adjusted, rejected = holm_bonferroni([p for _, p in tests])
    print(f"\n  realised family: {len(tests)} comparisons "
          f"(the hypothesis anticipated 5 switches)")
    for (name, _), p_adj, sig in zip(tests, adjusted, rejected, strict=True):
        print(f"    {name:28} p (Holm) {p_adj:.3e}  {'significant' if sig else 'not'}")
    for rec in records:
        key = f"{rec['model']}/{rec['switch'].split('=')[1]}/{rec['subset']}"
        idx = next((i for i, (n, _) in enumerate(tests) if n == key), None)
        rec["p_holm"] = f"{adjusted[idx]:.3e}" if idx is not None and rec["seed"] == SEED else ""
        rec["significant"] = rejected[idx] if idx is not None and rec["seed"] == SEED else ""
        rec["mcnemar_p"] = f"{rec['mcnemar_p']:.3e}"

    # --- is the effect confined to night, as the clause says? ----------------------
    #
    # The clause needs a day column to compare against, and this corpus cannot supply one
    # inside a split. venue_01 has exactly two recording days - a daylight morning and a
    # floodlit night - and holding whole slots out fills the test side from one of them.
    print("\n=== is the effect confined to night, as the clause requires? ===")
    per_seed = {}
    for seed in seeds:
        split = grouped_split(rows, group_key="slot_id", seed=seed)
        per_seed[seed] = {k: len(v) for k, v in lighting_subsets(split).items()}
    mixed = [s for s, comp in per_seed.items() if len(comp) > 1]
    for seed, comp in sorted(per_seed.items()):
        print(f"  seed {seed}: " + ", ".join(f"{k} {v}" for k, v in sorted(comp.items())))

    if not mixed:
        print(f"\n  **No grouped split has both.** All {len(seeds)} test sets are a single")
        print("  lighting condition, so there is no within-split day control and the clause's")
        print("  'specifically' cannot be tested. The night and day figures below come from")
        print("  different partitions and are NOT paired - read them as two observations,")
        print("  never as a contrast.")
    else:
        at_seed = [r for r in records if r["seed"] in mixed]
        for model in sorted({r["model"] for r in at_seed}):
            for switch in sorted({r["switch"] for r in at_seed if r["model"] == model}):
                by_subset = {r["subset"]: r for r in at_seed
                             if r["model"] == model and r["switch"] == switch}
                if "night" not in by_subset or "day" not in by_subset:
                    continue
                n, d = by_subset["night"]["delta"], by_subset["day"]["delta"]
                verdict = ("night only" if n > 0 >= d else
                           "both" if n > 0 and d > 0 else
                           "neither" if n <= 0 and d <= 0 else "day only")
                print(f"  {model:11} {switch:12} night {n:+.4f}  day {d:+.4f}   {verdict}")

    # --- across seeds --------------------------------------------------------------
    print(f"\n=== the same comparison across {len(seeds)} grouped splits ===")
    for model in sorted({r["model"] for r in records}):
        for switch in sorted({r["switch"] for r in records if r["model"] == model}):
            for subset in sorted({r["subset"] for r in records}):
                deltas = [r["delta"] for r in records
                          if r["model"] == model and r["switch"] == switch
                          and r["subset"] == subset]
                if not deltas:
                    continue
                # `ddof=1` on a single observation is nan, and a nan prints as a plausible
                # blank. One split is one observation and is labelled as such.
                sd = f"{np.std(deltas, ddof=1):.3f}" if len(deltas) > 1 else "  -  "
                print(f"  {model:11}{switch:12}{subset:7} mean {np.mean(deltas):+.4f}  "
                      f"sd {sd}  "
                      f"improves in {sum(d > 0 for d in deltas)} of {len(deltas)}"
                      + ("   (one split only)" if len(deltas) == 1 else ""))

    # --- the effective sample -------------------------------------------------------
    if not args.skip_scene_audit:
        split = grouped_split(rows, group_key="slot_id", seed=SEED)
        subsets = lighting_subsets(split)
        print("\n=== how many distinct scenes are those subsets? ===")
        for subset, positions in sorted(subsets.items()):
            picks = distinct_scenes([split.test[i] for i in positions])
            print(f"  {subset:7} {len(positions):>5} frames -> {len(picks)} distinct scenes")
            records.append({
                "model": "", "switch": "", "subset": subset, "seed": "",
                "n": len(positions), "baseline_macro_f1": "", "arm_macro_f1": "",
                "delta": "", "ci_low": "", "ci_high": "", "mcnemar_p": "",
                "cohens_g": "", "n_discordant": len(picks), "p_holm": "",
                "significant": "distinct_scenes",
            })

    fields = list(dict.fromkeys(k for r in records for k in r))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nwrote {OUT.name}")

    n_sig = sum(rejected)
    print("\n=== verdict on H5 ===")
    print("  clause 1 (ROI):   UNRUNNABLE - the switch has no polygon and never ran")
    worse = sum(1 for r in records if r["seed"] == SEED and r["delta"] != "" and r["delta"] < 0)
    total = sum(1 for r in records if r["seed"] == SEED and r["delta"] != "")
    print(f"  clause 2 (CLAHE): {n_sig} of {len(tests)} comparisons significant after Holm,")
    print(f"                    and CLAHE is *worse* in {worse} of {total} - the clause")
    print("                    predicted an improvement, so its direction is refuted.")
    if not mixed:
        print("                    Its 'on the night subset specifically' is untestable here:")
        print("                    no grouped split holds both lighting conditions.")
    record(
        "H5 preprocessing switches",
        "`python -m experiments.h5_preprocessing_switches`",
        f"`{OUT.name}`",
        f"ROI clause unrunnable (no polygon); CLAHE clause {n_sig}/{len(tests)} significant",
    )


if __name__ == "__main__":
    main()
