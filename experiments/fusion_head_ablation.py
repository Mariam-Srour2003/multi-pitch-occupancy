"""WP5-T2: does a learned gate beat averaging, and is the gate reading the clock? (RQ5)

This is the M4 experiment. `logit_average_baseline.py` established the terrain: DINOv2 leads
the seven venue folds at 0.9297, a per-fold oracle reaches 0.9603, and a parameter-free mean
of two backbones' probabilities reaches 0.9473-0.9524 - most of the headroom, for twenty
lines. The question left open was whether the gated module the plan specifies does better
than that, and this answers it.

**The gate is ablated on a three-rung ladder, not with an on/off switch.** ``uniform`` fixes
the weights at 1/K, ``constant`` learns two weights shared by every frame, ``mlp`` learns them
per frame from the statistics. Head, trainer, seed, epochs and regularisation are identical
across all three, so:

* ``constant`` - ``uniform`` is the value of **learned mixing**;
* ``mlp`` - ``constant`` is the value of **routing**, which is the only thing the gate is for.

The first version of this experiment collapsed those into one comparison, and would have
credited routing with an effect that is a learned constant: the gate puts about 0.70 of its
weight on DINOv2 in every one of the seven folds and moves it by 0.086 between frames.
**Routing is worth -0.0238 recall on one informative fold out of seven (p = 1.000).** WP5-T2
does not earn its place in the pipeline.

Two further comparisons, neither of which isolates anything:

* ``mlp`` vs ``gated_lighting`` - **the confound test.** The gate reads brightness, contrast
  and edge density; at `venue_01` brightness is nearly a day/night indicator and day/night is
  nearly the class label. A gate fed a single lighting bit has strictly less information, so
  if it matches, the gate has re-learned the clock rule and the project's recurring confound
  has appeared again. Logged either way, along with the correlation between the gate's own
  weights and a night indicator. The answer is *partly*: the gate's DINOv2 weight correlates
  -0.653 with night, so lighting is much of what it reads, but the lighting-only gate is
  0.1015 worse on recall, so it is not all of it. One detail is worth its own line - the
  lighting-only gate's false-play rate is 0.0206, which is the published clock rule's rate to
  four decimals, both being 5 of 243.
* ``torch_single_dinov2`` vs the published ``dinov2`` probe - the trainer control. Adam with
  a shared head is not lbfgs with an independent probe, and this says how much of any
  torch-versus-published gap is the optimiser rather than the model.

**The larger finding is about the false-play axis itself, and it is not good news.** Asking
what the models answer *instead* of ACTIVE_PLAY - a column this experiment added to check
whether a rate of 0.000 was real - shows that on the 243 held-out empty frames:

* DINOv2 answers ACTIVE_PLAY 75 times and MAINTENANCE 168 times. **It is correct 0 times.**
* The gated head answers MAINTENANCE on all 243. False-play 0.000, accuracy 0.000.
* ViT is the best model on the axis at 40 of 243, and no model exceeds 0.165.

So **the complement of the false-play rate is not correctness**, and on this test set it is
nowhere near it: the third class absorbs the difference. The rate ranks models by *where they
put their errors*, not by whether they are right, and DINOv2's 0.309 - cited across this
repository as dominating ConvNeXtV2's 0.992 - is the rate at which it makes one kind of
mistake rather than another. Both published numbers are correct as stated; what does not
follow is the reading that a low rate means the model can recognise an empty pitch.

The mechanism is visible in the training set: this protocol trains on 518 ACTIVE_PLAY, 251
EMPTY and **6 MAINTENANCE** frames, and ``class_weight="balanced"`` gives that six-frame class
a weight of 43. Out-of-distribution frames land in it. That is a property of the dataset, not
of the fusion head, and it is why the gated head's apparently perfect 0.000 is worthless.

An EMPTY-accuracy column is therefore reported next to the false-play rate, and the joint
summary is built from accuracy rather than from ``1 - false_play``. Built the other way, the
gated head ranked first in the table on the strength of 243 wrong answers.

**Both axes, never recall alone - and neither axis alone either.** The cross-venue folds are
100% ACTIVE_PLAY, so recall rises by answering "playing" more often; the empty set is 100%
EMPTY. Each axis alone is gameable in the opposite direction, and the joint column is a
*summary across two protocols*, not a metric on one test set.

**That empty set has three distinct scenes, not 243 frames.** Both counts are reported. Every
difference in those columns, including the large ones, rests on three independent
observations, and no test on that axis can reach 0.05.

**The single-backbone rows are reproduced from the published CSV before anything is added.**
A harness that cannot reproduce the number it extends has no business adding a new one.

The recall axis is tested with an exact sign-flip test over the seven folds, which reports its
own resolution floor: 2/2**7 = 0.0156 when all seven differ, and worse when they tie.

    uv run python experiments/fusion_head_ablation.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.dedup import dhash, distinct_subset
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.stats import holm_bonferroni, sign_flip_test
from pitch_occupancy.slots.fusion_head import (
    GATE_STATISTIC_NAMES,
    FusionHead,
    build_gate_statistics,
)
from pitch_occupancy.vision.heads import LinearProbe

PLAY, EMPTY = "C2_ACTIVE_PLAY", "C1_EMPTY"
SEED = 42
OUT = settings.results_dir / "fusion_head_ablation.csv"
COMPARISONS_OUT = settings.results_dir / "fusion_head_comparisons.csv"
GATE_OUT = settings.results_dir / "fusion_head_gate.csv"
PUBLISHED = settings.results_dir / "logit_average_baseline.csv"
GATE_CACHE = settings.feature_cache_dir / "gate_stats.npz"

#: The plan names {ConvNeXtV2, DINOv2} and marks ViT optional. The pair is the headline; ViT
#: is carried only in the single-backbone rows, because adding a third block to the fusion
#: triples the head's width without adding a backbone that wins any fold.
PAIR = ("convnextv2", "dinov2")
CACHES = {"convnextv2": "convnextv2.npz", "dinov2": "dinov2.npz", "vit": "vit.npz"}

#: Reported in this order. The grouping matters more than the ordering: everything from
#: `torch_single_dinov2` down shares one trainer, so those four rows are comparable with each
#: other in a way none of them is comparable with the rows above.
ORDER = (
    "convnextv2",
    "dinov2",
    "vit",
    "ens_convnextv2_dinov2",
    "torch_single_dinov2",
    "fusion_uniform",
    "fusion_constant",
    "fusion_gated_stats",
    "fusion_gated_lighting",
)


def load_features(rows):
    """Rows every cache covers, with one aligned feature matrix per backbone.

    Every model must see the same frames in the same order, and the folds must be cut from
    that shared list rather than from three slightly different ones.
    """
    loaded = {}
    for key, name in CACHES.items():
        d = np.load(settings.feature_cache_dir / name, allow_pickle=True)
        loaded[key] = ({str(f): i for i, f in enumerate(d["files"])}, d["features"])
    keep = [r for r in rows if all(r.file in idx for idx, _ in loaded.values())]
    dropped = len(rows) - len(keep)
    if dropped:
        print(f"  {dropped} row(s) missing from at least one cache - dropped, not zero-filled")
    X = {k: feats[[idx[r.file] for r in keep]] for k, (idx, feats) in loaded.items()}
    return keep, X


def gate_matrix(rows) -> np.ndarray:
    """The three cheap statistics per frame, cached like every other feature family.

    Cached because it costs a full pass over 1,692 JPEGs and the ablation refits seven folds
    four ways; recomputing it per fit would make the experiment slow enough to run less often
    than it should be.
    """
    files = [r.file for r in rows]
    if GATE_CACHE.exists():
        z = np.load(GATE_CACHE, allow_pickle=True)
        idx = {str(f): i for i, f in enumerate(z["files"])}
        if all(f in idx for f in files):
            return z["features"][[idx[f] for f in files]]
    print(f"  building {GATE_CACHE.name} ({len(files)} frames)...")
    X = build_gate_statistics(files, settings.dataset_dir)
    GATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GATE_CACHE, files=np.array(files, dtype=object), features=X)
    return X


def lighting_matrix(rows) -> np.ndarray:
    """One column: 1.0 at night, 0.0 by day. The confound ablation's entire input."""
    return np.array([[1.0 if r.lighting == "night" else 0.0] for r in rows])


def fit_all(train_rows, test_rows, Xtr, Xte, Gtr, Gte, Ltr, Lte):
    """Every model's predictions on one fold, plus the gate's report on the test frames."""
    preds: dict[str, list[str]] = {}

    order: list[str] | None = None
    proba: dict[str, np.ndarray] = {}
    for key in CACHES:
        probe = LinearProbe(key, seed=SEED).fit(Xtr[key], train_rows)
        classes = [str(c) for c in probe.classes_]
        if order is None:
            order = sorted(classes)
        pos = {c: i for i, c in enumerate(classes)}
        proba[key] = probe.predict_proba(Xte[key])[:, [pos[c] for c in order]]
        preds[key] = [order[i] for i in proba[key].argmax(1)]
    assert order is not None

    mean_p = np.mean([proba[k] for k in PAIR], axis=0)
    preds["ens_convnextv2_dinov2"] = [order[i] for i in mean_p.argmax(1)]

    y = [r.class3 for r in train_rows]
    pair_tr, pair_te = {k: Xtr[k] for k in PAIR}, {k: Xte[k] for k in PAIR}

    single = FusionHead(["dinov2"], gate="uniform", seed=SEED).fit(
        {"dinov2": Xtr["dinov2"]}, Gtr, y
    )
    preds["torch_single_dinov2"] = single.predict({"dinov2": Xte["dinov2"]}, Gte)

    # The ladder. Same head, same trainer, same seed; one capability added per rung.
    fitted: dict[str, FusionHead] = {}
    for rung, label in (("uniform", "fusion_uniform"),
                        ("constant", "fusion_constant"),
                        ("mlp", "fusion_gated_stats")):
        head = FusionHead(PAIR, gate=rung, seed=SEED).fit(pair_tr, Gtr, y)
        preds[label] = head.predict(pair_te, Gte)
        fitted[label] = head

    lit = FusionHead(PAIR, gate="mlp", gate_dim=1, seed=SEED).fit(pair_tr, Ltr, y)
    preds["fusion_gated_lighting"] = lit.predict(pair_te, Lte)
    fitted["fusion_gated_lighting"] = lit

    return preds, fitted


def cross_venue(rows, X, G, L):
    """Per-fold ACTIVE_PLAY recall for every model, plus what the gate did on each fold."""
    pos = {r.file: i for i, r in enumerate(rows)}
    per_fold: dict[str, dict[str, float]] = {}
    gate_log: list[dict] = []
    for fold in leave_one_group_out(rows):
        play = [r for r in fold.test if r.class3 == PLAY]
        if not play or len({r.class3 for r in fold.train}) < 2:
            # The venue_01 fold trains on the clip venues alone, all ACTIVE_PLAY - one class,
            # nothing to fit. Seven folds, not eight; a dataset gap, not a code limitation.
            continue
        tr = [pos[r.file] for r in fold.train]
        te = [pos[r.file] for r in play]
        preds, fitted = fit_all(
            fold.train, play,
            {k: v[tr] for k, v in X.items()}, {k: v[te] for k, v in X.items()},
            G[tr], G[te], L[tr], L[te],
        )
        for model, pred in preds.items():
            per_fold.setdefault(model, {})[fold.name] = sum(p == PLAY for p in pred) / len(play)
        # What the gate learned is asked on the *training* frames, which are the only ones
        # carrying both classes and both lighting conditions; the test frames of a
        # cross-venue fold are all ACTIVE_PLAY, so a correlation measured there would be
        # computed inside one class.
        gated = fitted["fusion_gated_stats"]
        report = gated.report(G[tr])
        night = np.array([r.lighting == "night" for r in fold.train], dtype=float)
        gate_log.append({
            "fold": fold.name,
            "spread": report.spread,
            "mean_weight": report.mean_weight,
            "corr_with_night": _corr(gated.gate_weights(G[tr])[:, 1], night),
            "constant_weight": fitted["fusion_constant"].report(G[tr]).mean_weight,
            "n_params": gated.n_parameters,
        })
        print(f"  fold {fold.name[:44]:<44} n={len(play):<4} gate spread {report.spread:.3f}")
    return per_fold, gate_log


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, or 0.0 when either side is constant - a constant gate correlates with
    nothing, and numpy would return nan and print a warning saying so less clearly."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def false_play(rows, X, G, L):
    """How often each model calls a held-out empty pitch a match, and on how many scenes.

    Trained on venue_01 camera A, scored on camera B's empty frames - the protocol
    `h3_with_false_play.py` and `logit_average_baseline.py` both use, so the three tables can
    be read together. The clip venues are excluded from training because all their
    development frames are ACTIVE_PLAY and including them drives every model to 1.000, which
    measures the dataset rather than the model.
    """
    pos = {r.file: i for i, r in enumerate(rows)}
    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    venue = [r for r in rows if r.venue == "venue_01"]
    train = [r for r in venue if cam(r) == "camera_A"]
    empties = [r for r in venue if cam(r) == "camera_B" and r.class3 == EMPTY]
    if not empties or len({r.class3 for r in train}) < 2:
        return {}, {}, {}, {}, 0, 0
    tr = [pos[r.file] for r in train]
    te = [pos[r.file] for r in empties]
    preds, _ = fit_all(
        train, empties,
        {k: v[tr] for k, v in X.items()}, {k: v[te] for k, v in X.items()},
        G[tr], G[te], L[tr], L[te],
    )
    rates = {m: float(np.mean([p == PLAY for p in pred])) for m, pred in preds.items()}
    # The complement of the false-play rate is *not* correctness, and on this test set it is
    # nowhere near it. Every frame here is EMPTY, so accuracy is the share answered EMPTY;
    # anything answered MAINTENANCE is also not-play, also wrong, and sends the slot to a
    # human. A model can therefore reach a false-play rate of 0.000 with 0 correct.
    correct = {m: float(np.mean([p == EMPTY for p in pred])) for m, pred in preds.items()}
    scenes = _distinct_scenes(empties)
    on_scenes = {
        m: float(np.mean([pred[i] == PLAY for i in scenes])) for m, pred in preds.items()
    }
    answers = {m: Counter(pred) for m, pred in preds.items()}
    return rates, correct, on_scenes, answers, len(empties), len(scenes)


def _distinct_scenes(rows) -> list[int]:
    """Indices of a greedy maximal set of pairwise-distinct held-out empty frames.

    243 frames sampled every fifteen seconds from one fixed camera are not 243 observations,
    and every false-play number in this table inherits that. Both counts are reported: the
    243-frame rate so the column matches the published tables, and the rate on one frame per
    scene so the reader can see how many observations a difference of 0.30 actually rests on.
    """
    import cv2

    hashes = {}
    for i, row in enumerate(rows):
        image = cv2.imread(str(settings.dataset_dir / row.file))
        if image is not None:
            hashes[i] = dhash(image)
    return distinct_subset(hashes)


def published() -> dict[str, float]:
    if not PUBLISHED.exists():
        return {}
    return {
        r["model"]: float(r["play_recall"])
        for r in csv.DictReader(PUBLISHED.open(encoding="utf-8"))
        if r["held_out_venue"] == "MEAN_ACROSS_FOLDS"
    }


#: (label, model, reference), in the order the docstring introduces them. The first two are
#: the ablation - adjacent rungs of the ladder, so each names exactly one cause. The rest are
#: context, and are labelled as such in the output rather than left to look equivalent.
COMPARISONS = (
    ("routing: mlp vs constant gate", "fusion_gated_stats", "fusion_constant"),
    ("learned mixing: constant vs uniform", "fusion_constant", "fusion_uniform"),
    ("gate vs mean-probability ensemble", "fusion_gated_stats", "ens_convnextv2_dinov2"),
    ("confound: stats gate vs lighting gate", "fusion_gated_stats", "fusion_gated_lighting"),
    ("trainer control: torch vs published probe", "torch_single_dinov2", "dinov2"),
)


def _cohens_d(deltas: list[float]) -> float:
    """Standardised mean of the paired fold deltas.

    Reported beside every p-value because WP0-T6 says so, and because a mean delta over
    seven folds carries no sense of how wide those folds were. `-0.0238` and `+0.1015` read
    as different sizes of thing; standardised, the second is a fold-to-fold difference the
    design cannot separate from noise either.

    Zero variance gives 0.0 rather than an infinity: identical deltas across folds mean the
    comparison found nothing to vary, which is a null and not an unbounded effect.
    """
    import statistics

    if len(deltas) < 2:
        return 0.0
    sd = statistics.stdev(deltas)
    return 0.0 if sd == 0 else statistics.fmean(deltas) / sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    rows = read_manifest(settings.dataset_dir / "manifest.csv")
    rows, X = load_features(development_rows(rows))
    G, L = gate_matrix(rows), lighting_matrix(rows)
    print(f"development rows: {len(rows)}")
    print(f"gate input: {', '.join(GATE_STATISTIC_NAMES)}\n")

    print("=== cross-venue folds ===")
    per_fold, gate_log = cross_venue(rows, X, G, L)
    folds = sorted(next(iter(per_fold.values())))
    means = {m: float(np.mean([per_fold[m][f] for f in folds])) for m in per_fold}
    oracle = float(np.mean([max(per_fold[k][f] for k in CACHES) for f in folds]))

    print("\n=== reproduction check against the published baseline table ===")
    ref = published()
    ok = True
    for key in (*CACHES, "ens_convnextv2_dinov2"):
        want = ref.get(key)
        agree = want is None or abs(means[key] - want) < 0.005
        ok &= agree
        shown = "-" if want is None else f"{want:.4f}"
        print(f"  {key:<24} {means[key]:.4f}  published {shown}  {'ok' if agree else 'MISMATCH'}")
    if not ok:
        print("\nWARNING: a baseline row does not reproduce logit_average_baseline.csv.")
        print("Do not read the fusion rows until that is explained.")

    fp, fp_correct, fp_scenes, fp_answers, n_empty, n_scenes = false_play(rows, X, G, L)
    # A summary across two protocols, not a metric on one test set: play recall on the
    # cross-venue folds averaged with EMPTY accuracy on the held-out camera. Built from
    # accuracy rather than from the complement of the false-play rate, because those are not
    # the same number here and using the complement ranked the gated head first on the
    # strength of 243 wrong answers.
    joint = {m: (means[m] + fp_correct[m]) / 2 for m in ORDER if m in fp_correct}

    print(f"\n=== both axes (recall over {len(folds)} folds; the empty set is n={n_empty} "
          f"frames = {n_scenes} distinct scenes) ===")
    print(f"{'model':<26}{'recall':>9}{'worst':>8}{'false-play':>12}{'(scenes)':>10}"
          f"{'EMPTY acc':>11}{'joint':>8}")
    for m in ORDER:
        print(f"{m:<26}{means[m]:9.4f}{min(per_fold[m].values()):8.3f}"
              f"{fp.get(m, float('nan')):12.4f}{fp_scenes.get(m, float('nan')):10.4f}"
              f"{fp_correct.get(m, float('nan')):11.4f}{joint.get(m, float('nan')):8.4f}")
    print(f"{'ORACLE (per-fold best)':<26}{oracle:9.4f}{'-':>8}{'-':>12}{'-':>10}"
          f"{'-':>11}{'-':>8}")
    print("\n  joint = (play recall + EMPTY accuracy) / 2, averaged across two protocols")
    print("  false-play and EMPTY accuracy are not complements: the third class absorbs the")
    print("  difference, and a model can score 0.000 false-play with 0.000 accuracy")
    print(f"  both empty-set columns rest on {n_scenes} distinct scenes; no test on that "
          f"axis can reach 0.05")

    print(f"\n=== what the models answer on the {n_empty} held-out empty frames ===")
    print("  a false-play rate of 0.000 reached by answering MAINTENANCE is not a clean")
    print("  answer: it is not-play, it is not right, and it sends the slot to review")
    for m in ORDER:
        counts = fp_answers.get(m)
        if counts:
            print(f"  {m:<26}" + "  ".join(
                f"{c.split('_', 1)[1].lower()} {counts[c]}" for c in sorted(counts)))

    print("\n=== paired sign-flip tests over the folds ===")
    tests = []
    fold_deltas: dict[str, list[float]] = {}
    for label, model, reference in COMPARISONS:
        deltas = [per_fold[model][f] - per_fold[reference][f] for f in folds]
        fold_deltas[label] = deltas
        r = sign_flip_test(deltas)
        floor = "" if r.can_reach() else "  (0.05 UNREACHABLE at this resolution)"
        print(f"  {label:<40} d={r.estimate:+.4f}  p={r.p_value:.4f}  "
              f"floor={r.min_achievable_p:.4f}  informative={r.n_informative}/{r.n_pairs}{floor}")
        tests.append((label, model, reference, r))

    # Holm over the declared family, and an effect size beside every p - WP0-T6's standing
    # rule, which this report was quoting p-values without. It changes no conclusion here,
    # and that is worth saying rather than leaving it to be assumed: nothing in the family
    # was significant uncorrected, so nothing can become significant corrected. The reason
    # to apply it anyway is that "five comparisons, one of them p=0.125" is a family whether
    # or not it is declared one, and a reader cannot tell which unless the report says.
    #
    # The effect size is Cohen's d over the fold deltas rather than the mean delta alone:
    # seven folds with a spread as wide as these have means that look decisive and
    # distributions that do not.
    adjusted, rejected = holm_bonferroni([r.p_value for _, _, _, r in tests])
    print("\n=== the same family, Holm-corrected ===")
    print(f"  family: {len(tests)} paired sign-flip tests over {len(folds)} venue folds, "
          f"declared before running them (COMPARISONS)")
    for (label, _, _, r), p_adj, rej in zip(tests, adjusted, rejected, strict=True):
        deltas = fold_deltas[label]
        d = _cohens_d(deltas)
        print(f"  {label:<40} p={r.p_value:.4f} -> p_holm={p_adj:.4f}  "
              f"d={d:+.3f}  {'rejected' if rej else 'not rejected'}")
    if not any(rejected):
        print("  none rejected, and none was significant before correction either - the "
              "correction is reported for completeness, not because it changed an outcome")

    print("\n=== what the gate did ===")
    spreads = [g["spread"] for g in gate_log]
    corrs = [g["corr_with_night"] for g in gate_log]
    for g in gate_log:
        weights = ", ".join(f"{k} {w:.3f}" for k, w in zip(PAIR, g["mean_weight"], strict=True))
        print(f"  {g['fold'][:44]:<44} spread {g['spread']:.4f}  r(night) "
              f"{g['corr_with_night']:+.3f}   {weights}")
    print(f"  {'mean':<44} spread {np.mean(spreads):.4f}  r(night) {np.mean(corrs):+.3f}")
    print(f"  parameters: {gate_log[0]['n_params']} in the gated pair head")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "held_out_venue", "play_recall", "false_play_rate",
                    "false_play_rate_distinct_scenes", "empty_accuracy",
                    "joint_recall_and_empty_accuracy", "n_held_out_empty",
                    "n_distinct_scenes"])
        for m in ORDER:
            for f in folds:
                w.writerow([m, f, f"{per_fold[m][f]:.4f}", "", "", "", "", "", ""])
            w.writerow([m, "MEAN_ACROSS_FOLDS", f"{means[m]:.4f}",
                        f"{fp.get(m, float('nan')):.4f}",
                        f"{fp_scenes.get(m, float('nan')):.4f}",
                        f"{fp_correct.get(m, float('nan')):.4f}",
                        f"{joint.get(m, float('nan')):.4f}", n_empty, n_scenes])
        w.writerow(["ORACLE_per_fold_best", "MEAN_ACROSS_FOLDS", f"{oracle:.4f}"])

    # Three files rather than three sections of one. A multi-section CSV has one header, so
    # `csv.DictReader` maps every later section onto the first section's keys and returns
    # rows that look fine and mean nothing - which is exactly how the gate criterion for this
    # experiment failed to find the comparison it was written to check.
    with COMPARISONS_OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        adjusted, rejected = holm_bonferroni([r.p_value for _, _, _, r in tests])
        w.writerow(["comparison", "model", "reference", "mean_delta", "p_value",
                    "p_holm", "rejected_holm", "cohens_d",
                    "min_achievable_p", "n_informative", "n_pairs", "isolates_one_cause"])
        for (label, model, reference, r), p_adj, rej in zip(tests, adjusted, rejected,
                                                            strict=True):
            w.writerow([label, model, reference, f"{r.estimate:+.4f}", f"{r.p_value:.4f}",
                        f"{p_adj:.4f}", rej, f"{_cohens_d(fold_deltas[label]):+.3f}",
                        f"{r.min_achievable_p:.4f}", r.n_informative, r.n_pairs,
                        label.startswith(("routing:", "learned mixing:"))])

    with GATE_OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "gate_weight_spread", "corr_gate_weight_with_night",
                    *[f"mlp_mean_weight_{k}" for k in PAIR],
                    *[f"constant_weight_{k}" for k in PAIR]])
        for g in gate_log:
            w.writerow([g["fold"], f"{g['spread']:.4f}", f"{g['corr_with_night']:+.4f}",
                        *[f"{x:.4f}" for x in g["mean_weight"]],
                        *[f"{x:.4f}" for x in g["constant_weight"]]])
    print(f"\nwrote {OUT.name}, {COMPARISONS_OUT.name}, {GATE_OUT.name}")

    isolating = next(r for label, _, _, r in tests if label.startswith("routing:"))
    record(
        "WP5-T2 gated fusion head, ablated",
        "`python experiments/fusion_head_ablation.py`",
        f"`{OUT.name}`",
        f"routing worth {isolating.estimate:+.4f} recall, p={isolating.p_value:.3f} "
        f"(floor {isolating.min_achievable_p:.3f}); no model exceeds "
        f"{max(fp_correct.values()):.4f} EMPTY accuracy on the held-out camera",
    )


if __name__ == "__main__":
    main()
