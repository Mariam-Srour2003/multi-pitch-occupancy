"""WP4-T1 - the same models under four split protocols, and what each protocol measures.

The headline the plan asks for is *"the random-vs-grouped delta is the thesis's first key
figure"*. It is here, and it is large. But running all four protocols side by side makes a
second point that the delta on its own cannot: **every protocol available on this corpus is
degenerate, and each one differently**. Reporting four numbers without saying that would
invite the reader to treat the lowest as the honest one, when in fact none of them measures
what its name suggests.

| protocol | what it is | how it fails here |
|---|---|---|
| `random` | frames shuffled | leaks: frames seconds apart land on both sides. Kept only as the control arm of H1. |
| `grouped_slot` | whole slots held out | honest, but the test side is ~99% one class, so accuracy on it is uninformative |
| `lo_venue_out` | one venue held out | every held-out clip venue is **100% ACTIVE_PLAY**, so recall is free and macro-F1 averages one class |
| `temporal` | trained before a date, tested after | day one is 97.6% EMPTY and all daylight, day two 98.9% ACTIVE_PLAY and all floodlit. It measures a class-and-lighting flip, not drift. |

So each row carries its own diagnostics - test size, how many classes are in it, the
majority class's share - beside the score. A number is not quotable without them.

The zero-shot control
---------------------

**A model that never trains cannot leak.** OpenCLIP is scored zero-shot under all four
protocols, so whatever its score does across them is caused entirely by *which frames are in
the test set*, never by what the model saw in training. That gives a subtraction the four
numbers cannot give on their own:

    leakage-attributable drop = (model random - model grouped)
                              - (clip  random - clip  grouped)

The second term is the part of the drop any model would show from the change in test-set
composition alone. It is a control, not a proof: it assumes the composition effect is
roughly additive and similar across models, which is an assumption and is stated as one.

Its prompt set is **fixed in advance** - the first descriptor of each class, all five
templates - and is deliberately *not* the winner of `prompt_search.py`. That winner was
selected on the same folds it is scored on, so importing it here would import its selection
bias into a table about protocols.

    uv run python experiments/benchmark_v2.py
    uv run python experiments/benchmark_v2.py --seeds 3      # quicker

**Reproduce before extending.** The `seed 42` rows of the random and grouped protocols are
checked against `h1_h2_baseline_floor.csv` before anything else is computed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.feature_cache import features_for, load_cache
from pitch_occupancy.data.manifest import ManifestRow, read_manifest
from pitch_occupancy.data.splits import (
    Split,
    check_split,
    development_rows,
    grouped_split,
    leave_one_group_out,
    random_split,
    temporal_split,
)
from pitch_occupancy.data.taxonomy import CLASS3_ORDER
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.metrics import evaluable_subset, evaluate
from pitch_occupancy.vision.backbones import BACKBONES
from pitch_occupancy.vision.cheap_features import build_cheap_features
from pitch_occupancy.vision.heads import ClockRule, LinearProbe, MajorityClass
from pitch_occupancy.vision.zeroshot import (
    DESCRIPTORS,
    TEMPLATES,
    PromptSet,
    classify,
    encode_prompts,
)

SEED = 42
DATASET = settings.dataset_dir
CACHE = settings.feature_cache_dir
RESULTS = settings.results_dir
OUT = RESULTS / "benchmark_v2.csv"
PUBLISHED = RESULTS / "h1_h2_baseline_floor.csv"
TOLERANCE = 5e-4

#: The date venue_01's two recording days fall either side of. It is the only cut this
#: corpus offers: nothing else is dated at all.
TEMPORAL_CUTOFF = "2026-07-12"

#: Fixed before any scoring, and deliberately not the prompt search's winner - that was
#: chosen on the folds it reports, and this table is about protocols, not prompts.
ZERO_SHOT_PROMPTS = PromptSet(
    descriptors={cls: options[0] for cls, options in DESCRIPTORS.items()},
    templates=tuple(TEMPLATES),
)
ZERO_SHOT = "clip_zeroshot"

#: Predictors that reach an answer without looking at the pitch. `clip_zeroshot` is not one
#: of them - it reads the image, it simply never trains - so it is the composition control,
#: not part of the floor.
TRIVIAL = ("majority", "clock_rule", "cheap_intensity", "cheap_histogram")


# --- features --------------------------------------------------------------------


def cheap_matrix(rows: list[ManifestRow], kind: str) -> np.ndarray:
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


def clip_predictions(rows: list[ManifestRow]) -> dict[str, str] | None:
    """Zero-shot label per frame, or None if CLIP's image cache is absent.

    Computed once for the whole corpus rather than per split, which is the point: the same
    prediction is scored under every protocol, so only the test set differs.
    """
    path = CACHE / "clip_image_features.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    index = {str(f): i for i, f in enumerate(z["files"])}
    if not all(r.file in index for r in rows):
        missing = sum(r.file not in index for r in rows)
        print(f"  ({ZERO_SHOT}: {missing} frame(s) not in the CLIP cache - skipping)")
        return None

    classes = list(CLASS3_ORDER)
    cached = CACHE / "clip_zeroshot_directions.npz"
    if cached.exists():
        directions = np.load(cached)["directions"]
    else:
        from experiments.prompt_search import make_text_encoder

        directions = encode_prompts(ZERO_SHOT_PROMPTS, classes, make_text_encoder())
        np.savez_compressed(cached, directions=directions)
    X = z["features"][[index[r.file] for r in rows]]
    return dict(zip([r.file for r in rows], classify(X, directions, classes), strict=True))


def predictors(rows: list[ManifestRow], clip: dict[str, str] | None):
    """Every candidate, cheapest first, each with its feature matrix."""
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
            print(f"  (warning: {key} covers {len(kept)}/{len(rows)} rows - skipped)")
            continue
        yield key, LinearProbe(key, seed=SEED), X
    if clip is not None:
        yield ZERO_SHOT, _FrozenLabels(clip), np.zeros((n, 1), np.float32)


class _FrozenLabels:
    """A predictor whose answers were decided before any split existed.

    Not a model wrapper for its own sake: it is what makes the zero-shot arm a *control*.
    Fitting is a no-op, so the same per-frame label is scored under every protocol and any
    movement in its score is test-set composition by construction.
    """

    name = ZERO_SHOT

    def __init__(self, labels: dict[str, str]) -> None:
        self._labels = labels

    def fit(self, X, rows) -> "_FrozenLabels":
        return self

    def predict(self, X, rows) -> list[str]:
        return [self._labels[r.file] for r in rows]


# --- one split -------------------------------------------------------------------


def describe(split: Split) -> dict:
    """What a reader needs before believing any score computed on this split."""
    truth = Counter(r.class3 for r in split.test)
    n = sum(truth.values()) or 1
    present = [c.value for c in CLASS3_ORDER if truth.get(c.value)]
    return {
        "n_train": len(split.train),
        "n_test": len(split.test),
        "n_classes_test": len(present),
        "majority_share_test": round(max(truth.values()) / n, 4) if truth else 0.0,
        "classes_in_test": ";".join(present),
        "warnings": " | ".join(check_split(split)),
    }


def run_split(split: Split, rows: list[ManifestRow], clip, protocol: str) -> list[dict]:
    pos = {r.file: i for i, r in enumerate(rows)}
    tr = [pos[r.file] for r in split.train]
    te = [pos[r.file] for r in split.test]
    y_true = [r.class3 for r in split.test]
    facts = describe(split)

    out = []
    for name, model, X in predictors(rows, clip):
        model.fit(X[tr], split.train)
        y_pred = model.predict(X[te], split.test)

        # Accuracy on the full test set; the macro average only over classes with enough
        # support to be averaged. See `evaluation.metrics.evaluable_subset`.
        rep_all = evaluate(y_true, y_pred)
        yt, yp, dropped = evaluable_subset(y_true, y_pred)
        rep = evaluate(yt, yp)
        out.append({
            "protocol": protocol,
            "replicate": split.name,
            "model": name,
            "accuracy": round(rep_all.accuracy, 4),
            "macro_f1": round(rep.macro_f1, 4),
            "balanced_acc": round(rep_all.balanced_accuracy, 4),
            "macro_over_classes": ";".join(c.label for c in rep.per_class),
            "excluded_low_support": ";".join(dropped),
            **facts,
        })
    return out


def protocols(rows: list[ManifestRow], seeds: list[int]) -> list[tuple[str, Split]]:
    """Every split this corpus supports, named by the protocol it belongs to."""
    out: list[tuple[str, Split]] = []
    for s in seeds:
        out.append(("random", random_split(rows, seed=s)))
    for s in seeds:
        out.append(("grouped_slot", grouped_split(rows, group_key="slot_id", seed=s)))
    for fold in leave_one_group_out(rows, group_key="venue"):
        # A fold whose training side holds one class cannot fit anything; venue_01 is the
        # whole of it, so training on "everything else" leaves only ACTIVE_PLAY.
        if len({r.class3 for r in fold.train}) < 2:
            print(f"  (skipping {fold.name}: single-class training side)")
            continue
        out.append(("lo_venue_out", fold))
    out.append(("temporal", temporal_split(rows, cutoff_date=TEMPORAL_CUTOFF)))
    return out


# --- reproduction guard ------------------------------------------------------------


def check_reproduces(records: list[dict]) -> None:
    """The seed-42 random and grouped rows must match the published H1/H2 table."""
    if not PUBLISHED.exists():
        print("  (h1_h2_baseline_floor.csv absent - nothing to check against)")
        return
    published = {
        (r["split"], r["model"]): (float(r["accuracy"]), float(r["macro_f1"]))
        for r in csv.DictReader(PUBLISHED.open(encoding="utf-8"))
    }
    bad, checked = [], 0
    for r in records:
        key = (r["replicate"], r["model"])
        if key not in published:
            continue
        checked += 1
        for i, field in enumerate(("accuracy", "macro_f1")):
            if abs(r[field] - published[key][i]) > TOLERANCE:
                bad.append(f"{key} {field}: {r[field]} vs published {published[key][i]}")
    for line in bad:
        print(f"  MISMATCH {line}")
    if bad:
        raise SystemExit(
            f"{len(bad)} of {checked} published numbers did not reproduce. A table that "
            "disagrees with the one it extends is measuring something else."
        )
    print(f"  {checked} published numbers reproduce to {TOLERANCE:g}")


# --- aggregation -------------------------------------------------------------------


def summarise(records: list[dict], metric: str) -> dict[tuple[str, str], tuple[float, float, int]]:
    """``(mean, std, n_replicates)`` per (protocol, model).

    A single split is a sample of size one and reviewers read it as such, so the spread
    over replicates is reported beside the mean wherever a protocol has more than one.
    """
    grouped: dict[tuple[str, str], list[float]] = {}
    for r in records:
        grouped.setdefault((r["protocol"], r["model"]), []).append(r[metric])
    return {
        k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
        for k, v in grouped.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5,
                    help="replicates for the random and grouped protocols")
    args = ap.parse_args()

    rows = development_rows(read_manifest(DATASET / "manifest.csv"))
    seeds = [SEED + i for i in range(args.seeds)]
    print(f"development rows: {len(rows)}  venues: {len({r.venue for r in rows})}")
    print(f"seeds: {seeds}\n")

    clip = clip_predictions(rows)
    if clip is None:
        print("  zero-shot control unavailable - the leakage subtraction will be skipped\n")

    records: list[dict] = []
    for protocol, split in protocols(rows, seeds):
        results = run_split(split, rows, clip, protocol)
        records.extend(results)
        facts = results[0]
        print(f"{protocol:14} {split.name:38} train {facts['n_train']:>5} "
              f"test {facts['n_test']:>5}  {facts['n_classes_test']} class(es), "
              f"majority {facts['majority_share_test']:.1%}")

    print("\nreproducing the published table before extending it")
    check_reproduces(records)

    # --- the protocols, side by side -------------------------------------------
    macro = summarise(records, "macro_f1")
    acc = summarise(records, "accuracy")
    order = ["random", "grouped_slot", "lo_venue_out", "temporal"]
    models = [m for m in dict.fromkeys(r["model"] for r in records)]

    print("\n=== macro-F1 by protocol (mean +- sd over replicates) ===")
    print(f"{'model':16}" + "".join(f"{p:>22}" for p in order))
    for m in models:
        cells = ""
        for p in order:
            if (p, m) not in macro:
                cells += f"{'-':>22}"
                continue
            mu, sd, n = macro[(p, m)]
            cells += f"{mu:>15.4f} +-{sd:.3f}" if n > 1 else f"{mu:>22.4f}"
        print(f"{m:16}{cells}")

    print("\n=== what each protocol's test set actually contains ===")
    print(f"{'protocol':16}{'replicates':>11}{'median n_test':>14}{'classes':>9}"
          f"{'majority share':>16}")
    for p in order:
        facts = [r for r in records if r["protocol"] == p and r["model"] == models[0]]
        print(f"{p:16}{len(facts):>11}{int(np.median([f['n_test'] for f in facts])):>14}"
              f"{int(np.median([f['n_classes_test'] for f in facts])):>9}"
              f"{np.mean([f['majority_share_test'] for f in facts]):>15.1%}")

    # --- the floor, per protocol ------------------------------------------------
    #
    # WP4-T10 established that trivial baselines set a floor the deep probes have to clear.
    # Doing it per protocol turns "this protocol is degenerate" from a description into a
    # number: where a rule that never looks at a pixel matches or beats the best backbone,
    # the protocol is not measuring occupancy, whatever the backbone scores.
    print("\n=== the floor: can a model that ignores the image keep up? ===")
    print(f"{'protocol':16}{'best trivial':>26}{'best backbone':>26}{'gap':>10}")
    floor_rows: list[dict] = []
    for p in order:
        trivials = {m: macro[(p, m)][0] for m in TRIVIAL if (p, m) in macro}
        deeps = {m: macro[(p, m)][0] for m in BACKBONES if (p, m) in macro}
        if not trivials or not deeps:
            continue
        tm, tv = max(trivials.items(), key=lambda kv: kv[1])
        dm, dv = max(deeps.items(), key=lambda kv: kv[1])
        flag = "  <- floor reached" if tv >= dv else ""
        print(f"{p:16}{tm + ' ' + format(tv, '.4f'):>26}"
              f"{dm + ' ' + format(dv, '.4f'):>26}{dv - tv:>+10.4f}{flag}")
        floor_rows.append({
            "protocol": p, "replicate": "SUMMARY", "model": f"floor:{tm}_vs_{dm}",
            "accuracy": "", "macro_f1": round(dv - tv, 4), "balanced_acc": "",
            "macro_over_classes": "best_backbone_minus_best_trivial",
            "excluded_low_support": "", "n_train": "", "n_test": "", "n_classes_test": "",
            "majority_share_test": "", "classes_in_test": "",
            "warnings": "trivial baseline matches or beats every backbone" if tv >= dv else "",
        })
    reached = [r["protocol"] for r in floor_rows if r["warnings"]]
    if reached:
        print(f"\n  On {', '.join(reached)} a model that never looks at the image is not")
        print("  beaten by any backbone. Those protocols rank models by something other")
        print("  than whether they recognise occupancy.")

    # --- the leakage subtraction ------------------------------------------------
    trained = [m for m in models if m in set(BACKBONES) | {"cheap_intensity", "cheap_histogram"}]
    subtraction: list[dict] = list(floor_rows)
    if clip is not None and ("random", ZERO_SHOT) in macro and ("grouped_slot", ZERO_SHOT) in macro:
        clip_drop = macro[("random", ZERO_SHOT)][0] - macro[("grouped_slot", ZERO_SHOT)][0]
        print(f"\n=== random -> grouped, with the zero-shot control subtracted ===")
        print(f"  the untrained model drops {clip_drop:+.4f} on the same change of test set,")
        print( "  and it cannot be leaking, so that much of every drop is composition")
        print(f"\n{'model':16}{'raw drop':>12}{'composition':>14}{'attributable':>14}")
        for m in trained:
            if ("random", m) not in macro or ("grouped_slot", m) not in macro:
                continue
            raw = macro[("random", m)][0] - macro[("grouped_slot", m)][0]
            print(f"{m:16}{raw:>12.4f}{clip_drop:>14.4f}{raw - clip_drop:>14.4f}")
            subtraction.append({
                "protocol": "random_minus_grouped", "replicate": "SUMMARY", "model": m,
                "accuracy": "", "macro_f1": round(raw, 4), "balanced_acc": "",
                "macro_over_classes": "raw_drop", "excluded_low_support": "",
                "n_train": "", "n_test": "", "n_classes_test": "",
                "majority_share_test": round(clip_drop, 4),
                "classes_in_test": "composition_drop_from_zero_shot_control",
                "warnings": f"leakage_attributable={raw - clip_drop:+.4f}",
            })
        print("\n  This is a control, not a proof: it assumes the composition effect is")
        print("  additive and similar across models. It is stated as an assumption.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
        w.writerows(subtraction)
    print(f"\nwrote {OUT.name}  ({len(records) + len(subtraction)} rows)")

    (RESULTS / "benchmark_v2_protocols.json").write_text(
        json.dumps({
            p: {
                "replicates": len({r["replicate"] for r in records if r["protocol"] == p}),
                "median_n_test": int(np.median(
                    [r["n_test"] for r in records if r["protocol"] == p])),
                "mean_majority_share": round(float(np.mean(
                    [r["majority_share_test"] for r in records if r["protocol"] == p])), 4),
            } for p in order
        }, indent=2), encoding="utf-8",
    )

    record(
        "WP4-T1 benchmark v2",
        "`python experiments/benchmark_v2.py`",
        f"`{OUT.name}`",
        f"{len(models)} models x {len(order)} protocols; "
        f"every protocol degenerate, each differently",
    )


if __name__ == "__main__":
    main()
