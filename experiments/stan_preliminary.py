"""WP5-T1: STAN against four tuned baselines, on a test set that cannot carry it yet. (RQ5)

The second M4 criterion. STAN is the plan's headline novelty - learn the slot verdict from the
ordered per-minute sequence instead of applying three hand-set ratio thresholds - and this
runs it against the baselines WP5-T7 names, on the slot data WP5-T8 composes.

**It is labelled preliminary by the code, not by a promise.** WP5-T8 sets a hard gate: no
headline number below 30 real labelled slots. There are **two**. So
`pitch_occupancy.slots.stan` holds that rule as `MIN_REAL_SLOTS` and this script calls
`assert_preliminary`, catches the refusal, and prints the caveat next to every table. A note
in a plan reading "remember to call this preliminary" is precisely the kind of guard this
project has repeatedly found not guarding.

**Two structural facts about the data, and neither is a code limitation.**

*First*, the real test set is two slots. Two. No test of any kind survives that, and none is
attempted: the real slots are reported as a smoke test - does the pipeline reach a verdict at
all - and the model comparison is run on held-out composed slots.

*Second*, and worse: **the two real slots contain every EMPTY and every MAINTENANCE frame in
the dataset.** 1,296 of 1,692 frames belong to them; the other 396 are highlight clips and all
396 are ACTIVE_PLAY. So there is no frame anywhere outside the real slots that can teach a
model what an empty pitch looks like, and "train on synthetic, test on real slots only"
*cannot* be made frame-disjoint with this dataset. It is not that the split was done
carelessly; the split does not exist. That is the sharpest available statement of why WP2-T8
blocks WP5-T1, and it is measured here rather than asserted.

What *is* clean is the synthetic comparison. The frame pool is split in two before any slot is
composed, training slots are built only from the first half and test slots only from the
second, so no frame appears on both sides. The five predictors are compared there.

**And the result there is a ceiling in three draws of five, which is a finding about the
benchmark.** STAN scores 1.0000 on 200 held-out composed slots, beating a tuned HMM by 0.105
(p < 0.0001) - and the right reading is not "STAN wins" but "this test set can be exhausted".
The composed label is a deterministic function of the template, and the five templates stay
separable under jitter, so a model that reads contiguity - one long block of play versus
scattered short runs, the thing a play *ratio* throws away - can recover the generating
process exactly. That STAN does so is worth knowing; it is necessary for the architecture to
be worth anything. It is not evidence that it beats an HMM on real slots, where the verdict is
not a function of five shapes. Composing more slots cannot break this tie. Only labelling real
ones can.

**The ceiling is a property of the draw, and the ordering is not** (`--seeds`,
`stan_draw_spread.csv`). The pool split and both compositions move with the seed; the probe
does not, so a difference between draws is the construction rather than the fit. Over five
draws STAN scores 1.0000, 1.0000, 1.0000, 0.9100, 0.8000 - mean 0.9420, sd 0.0884 - so
"saturated" describes particular compositions rather than the design. **STAN is first in all
five** and no baseline matches it in any, which is the part that replicates. The baselines
move more than the published table suggests: `summary_logistic` runs 0.7350-0.9550 and is the
best baseline in one draw and the worst in another, so the 0.105 margin over the HMM is one
draw's margin and should not be quoted alone.

This replication exists because the augmentation experiment's headline was a single
construction draw and did not survive being drawn again. Two results in this project consumed
a draw once; that was one and this was the other.

**Composed labels come from the template, not from the threshold rule.** A full match is USED
because it is a full match. Labelling by `aggregate_slot` would make the threshold baseline
correct by construction and every comparison against it circular.

    uv run python experiments/stan_preliminary.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.stats import sign_flip_test
from pitch_occupancy.slots.fusion import PRIORITY
from pitch_occupancy.slots.stan import (
    CLASS_ORDER,
    PREDICTORS,
    NotEnoughRealSlots,
    assert_preliminary,
    preliminary_caveat,
)
from pitch_occupancy.slots.synthetic import SlotSequence, compose_dataset
from pitch_occupancy.vision.heads import LinearProbe

SEED = 42
BACKBONE = "dinov2"
OUT = settings.results_dir / "stan_preliminary.csv"

#: The verdicts the two recorded slots actually carry, from `results/end_to_end_slots.csv`,
#: which derives them from the human labels rather than from any model. Written here rather
#: than read from that CSV because that file also holds synthetic booking fixtures, and one
#: slot appears in it twice under two different booking cases.
REAL_TRUTH = {
    "venue_01_2026-07-11_1000": SlotStatus.NOTUSED,
    "venue_01_2026-07-12_2030": SlotStatus.USED,
}


def load(cache: str = BACKBONE):
    """Manifest rows aligned with their cached features."""
    d = np.load(settings.feature_cache_dir / f"{cache}.npz", allow_pickle=True)
    index = {str(f): i for i, f in enumerate(d["files"])}
    rows = [r for r in read_manifest(settings.dataset_dir / "manifest.csv") if r.file in index]
    return rows, d["features"][[index[r.file] for r in rows]]


def split_frame_pool(rows, seed: int = SEED):
    """Two disjoint frame pools, stratified by class.

    Composed slots draw minutes from a pool, so training and test slots share frames unless
    the pool is split first. Without this the comparison would measure how well each model
    memorises 1,578 frames, and the sequence models - which see 60 of them per slot - would
    look strongest for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    a: dict[str, list[int]] = defaultdict(list)
    b: dict[str, list[int]] = defaultdict(list)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_class[r.class3].append(i)
    for cls, idx in by_class.items():
        order = rng.permutation(idx)
        half = len(order) // 2
        a[cls] = list(order[:half])
        b[cls] = list(order[half:])
    return (
        {k: np.array(v) for k, v in a.items()},
        {k: np.array(v) for k, v in b.items()},
    )


def scorer(probe: LinearProbe, features: np.ndarray):
    """Row indices -> an (n, 3) probability matrix in `CLASS_ORDER`.

    The reordering is not a formality: the probe's classes come back in sklearn's sorted
    order and `CLASS_ORDER` is EMPTY, PLAY, MAINTENANCE. Averaging or arg-maxing a matrix
    whose columns mean something else produces a plausible table rather than an error.
    """
    classes = [str(c) for c in probe.classes_]
    columns = []
    for cls in CLASS_ORDER:
        columns.append(classes.index(cls.value) if cls.value in classes else None)

    def score(indices: np.ndarray) -> np.ndarray:
        raw = probe.predict_proba(features[np.asarray(indices)])
        out = np.zeros((len(indices), 3))
        for j, col in enumerate(columns):
            if col is not None:
                out[:, j] = raw[:, col]
        return out

    return score


def real_slots(rows, features, probe) -> list[SlotSequence]:
    """The two recorded slots, per minute, scored by the model rather than read from labels.

    Reconstructed from the manifest the way `end_to_end_slots.py` does it - frames carry a
    camera tag and an offset in seconds, so grouping by minute rebuilds what the sampler saw -
    but with the probe's probabilities in place of the ground-truth class, because the
    question here is what the *pipeline* decides, not what the decision layer does given
    perfect perception. That comparison already exists and is a different experiment.

    The two cameras are fused by the max-activity priority `slots.fusion` uses, keeping the
    winning camera's probability vector rather than averaging the two: a half that saw nothing
    would otherwise dilute the evidence of the half that saw the match.
    """
    score = scorer(probe, features)
    grid: dict[str, dict[int, list[tuple[Class3, int]]]] = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        if r.source == "clip" or r.slot_id not in REAL_TRUTH:
            continue
        grid[r.slot_id][r.t_s // 60].append((Class3(r.class3), i))

    out = []
    for slot_id, minutes in grid.items():
        rows_for_minutes = []
        for minute in sorted(minutes):
            frames = minutes[minute]
            predicted = score(np.array([i for _, i in frames]))
            winner = min(
                range(len(frames)),
                key=lambda k: PRIORITY.index(CLASS_ORDER[int(predicted[k].argmax())]),
            )
            rows_for_minutes.append(predicted[winner])
        out.append(SlotSequence(
            slot_id=slot_id,
            probabilities=np.stack(rows_for_minutes),
            truth=REAL_TRUTH[slot_id],
            synthetic=False,
            template="recorded",
        ))
    return out


def accuracy(predicted, truth) -> float:
    return float(np.mean([p is t for p, t in zip(predicted, truth, strict=True)]))


def per_slot_correct(predicted, truth) -> np.ndarray:
    return np.array([1.0 if p is t else 0.0 for p, t in zip(predicted, truth, strict=True)])


SPREAD = settings.results_dir / "stan_draw_spread.csv"


def parse_seeds(text: str) -> list[int]:
    seeds = [int(part) for part in str(text).replace(",", " ").split()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds needs at least one integer")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"--seeds repeats a value: {seeds}")
    #: `seed + 1` composes the test slots, so a pair of seeds one apart would build one
    #: draw's test set from the next draw's training pool - correlated draws reported as
    #: independent, which is the opposite of what replication is for.
    collisions = sorted(set(seeds) & {s + 1 for s in seeds})
    if collisions:
        raise argparse.ArgumentTypeError(
            f"seeds {collisions} collide with another seed's test composition (seed + 1); "
            f"space them out"
        )
    return seeds


def one_draw(rows, features, slots_per_template: int, seed: int):
    """Everything the construction seed touches, for one draw.

    The pool split and both compositions move with ``seed``; the probe's own seed does not,
    so a difference between two draws is the *construction* and cannot be the fit. That is
    the same separation `augmentation_transfer.py` uses, and for the same reason - it was
    the un-replicated construction there that produced a headline which did not survive
    being drawn again.
    """
    train_pool, test_pool = split_frame_pool(rows, seed=seed)
    train_idx = np.concatenate(list(train_pool.values()))
    probe = LinearProbe(BACKBONE, seed=SEED).fit(
        features[train_idx], [rows[i] for i in train_idx]
    )
    score = scorer(probe, features)
    train_slots = compose_dataset(train_pool, score, n_per_template=slots_per_template,
                                  seed=seed)
    test_slots = compose_dataset(test_pool, score, n_per_template=slots_per_template,
                                 seed=seed + 1)
    return train_slots, test_slots, real_slots(rows, features, probe)


def evaluate_draw(train_slots, test_slots, recorded):
    """Fit every predictor on one draw. Returns its rows and the per-slot correctness."""
    truth = [s.truth for s in test_slots]
    results, correctness = [], {}
    for factory in PREDICTORS:
        model = factory()
        model.fit(train_slots)
        predicted = model.predict(test_slots)
        correctness[model.name] = per_slot_correct(predicted, truth)
        on_real = model.predict(recorded)
        results.append((model.name, accuracy(predicted, truth),
                        accuracy(on_real, [s.truth for s in recorded]), on_real))
    return results, correctness


def save_spread(per_draw: dict[str, dict[int, float]], n_test: int) -> None:
    """One row per model: its accuracy in each draw, and the spread across them.

    A separate artefact from `stan_preliminary.csv`, which stays the published draw. The
    question this answers is not "how good is STAN" but "does the answer depend on which
    slots happened to be composed", and mixing the two tables would make the second easy to
    read as the first.
    """
    import statistics

    seeds = sorted({s for by_seed in per_draw.values() for s in by_seed})
    SPREAD.parent.mkdir(parents=True, exist_ok=True)
    with SPREAD.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n_draws", "n_composed_test_per_draw", "mean", "sd", "min", "max"]
                   + [f"seed_{s}" for s in seeds])
        for name, by_seed in per_draw.items():
            vals = [by_seed[s] for s in seeds if s in by_seed]
            sd = f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else ""
            w.writerow([name, len(vals), n_test, f"{statistics.fmean(vals):.4f}", sd,
                        f"{min(vals):.4f}", f"{max(vals):.4f}"]
                       + [f"{by_seed[s]:.4f}" if s in by_seed else "" for s in seeds])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slots-per-template", type=int, default=40)
    ap.add_argument(
        "--seeds", default=str(SEED),
        help="comma-separated construction draws. The first is the published one and gets "
             "the full write-up; the rest are replicates that feed the spread table.",
    )
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    rows, features = load()
    print(f"{len(rows)} frames; draws: {', '.join(str(s) for s in seeds)}")
    print(f"composing {args.slots_per_template} slots per template from each pool...")

    # The frame classifier is fitted on the training pool only. It is the same probe the rest
    # of the project uses, so the sequence models are handed the perception this system
    # actually has rather than an idealised one.
    train_slots, test_slots, recorded = one_draw(
        rows, features, args.slots_per_template, seeds[0]
    )

    print(f"  train {len(train_slots)} composed, test {len(test_slots)} composed, "
          f"{len(recorded)} recorded")
    print(f"  verdict mix (test): "
          f"{ {s.value: sum(q.truth is s for q in test_slots) for s in SlotStatus} }")

    # The gate, exercised rather than described. It must fail here, and the run must continue
    # with the caveat attached rather than pretending the number is a headline.
    try:
        assert_preliminary(recorded)
        gate = "MET"
    except NotEnoughRealSlots as exc:
        gate = f"NOT MET - {exc}"
    print(f"\nWP5-T8 gate: {gate}")
    print(preliminary_caveat(recorded))

    print("\n=== held-out composed slots (the only comparison with a usable n) ===")
    results, correctness = evaluate_draw(train_slots, test_slots, recorded)
    for name, acc, real_acc, _ in results:
        print(f"  {name:<20} composed {acc:.4f}   recorded {real_acc:.3f} "
              f"({len(recorded)} slots)")

    # A ceiling score is not a good result, it is an exhausted benchmark, and saying so is
    # the difference between "STAN is better" and "this test set can no longer tell".
    saturated = [name for name, acc, _, _ in results if acc >= 1.0]
    if saturated:
        print(f"\n  SATURATED: {', '.join(saturated)} scores 1.0000 on the composed set.")
        print("  The composed label is a deterministic function of five templates, and the")
        print("  templates stay separable under jitter, so the ceiling is 1.0000 and reaching")
        print("  it shows the generating process was recovered - not that the model would win")
        print("  on real slots, where no such function exists. Composing more slots cannot")
        print("  break this tie; only labelled real ones can (WP2-T8).")

    print("\n=== STAN against each baseline, paired over the composed test slots ===")
    tests = []
    stan = correctness["stan"]
    for name, arr in correctness.items():
        if name == "stan":
            continue
        r = sign_flip_test(stan - arr)
        floor = "" if r.can_reach() else "  (0.05 unreachable)"
        print(f"  stan - {name:<20} d={r.estimate:+.4f}  p={r.p_value:.4f}  "
              f"informative={r.n_informative}/{r.n_pairs}{floor}")
        tests.append((name, r))

    print("\n=== the two recorded slots, one line each ===")
    print("  a smoke test, not an evaluation: n=2, and see the caveat below")
    for name, _, _, on_real in results:
        verdicts = ", ".join(
            f"{s.slot_id.split('_', 1)[1]} -> {p.value} (truth {s.truth.value})"
            for s, p in zip(recorded, on_real, strict=True)
        )
        print(f"  {name:<20} {verdicts}")

    print("\n=== why the recorded slots cannot be a clean test set ===")
    non_clip = [r for r in rows if r.source != "clip"]
    in_real = [r for r in non_clip if r.slot_id in REAL_TRUTH]
    outside = [r for r in rows if r.slot_id not in REAL_TRUTH]
    print(f"  {len(in_real)} of {len(rows)} frames belong to the two recorded slots")
    for cls in ("C1_EMPTY", "C3_MAINTENANCE_NON_SPORTING", "C2_ACTIVE_PLAY"):
        n_out = sum(1 for r in outside if r.class3 == cls)
        print(f"  {cls:<30} {n_out:>4} frame(s) exist outside those slots")
    print("  so no frame-disjoint 'train on composed, test on real' split exists in this")
    print("  dataset. WP2-T8 (>=30 labelled slots) is the blocker, not the code.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "composed_test_accuracy", "n_composed_test",
                    "recorded_accuracy", "n_recorded", "is_headline"])
        for name, acc, real_acc, _ in results:
            w.writerow([name, f"{acc:.4f}", len(test_slots), f"{real_acc:.4f}",
                        len(recorded), "no - below the WP5-T8 gate"])
        w.writerow([])
        w.writerow(["comparison", "mean_delta", "p_value", "min_achievable_p", "n_informative"])
        for name, r in tests:
            w.writerow([f"stan - {name}", f"{r.estimate:+.4f}", f"{r.p_value:.4f}",
                        f"{r.min_achievable_p:.4f}", r.n_informative])
        w.writerow([])
        w.writerow(["caveat", preliminary_caveat(recorded)])
    print(f"\nwrote {OUT.name}")

    # --- the other draws -------------------------------------------------------------
    #
    # Everything above is one construction: one pool split, one set of composed slots. The
    # augmentation experiment was exactly that shape, and its headline did not survive being
    # drawn again - so the same question is asked here rather than left to be asked later.
    per_draw = {name: {seeds[0]: acc} for name, acc, _, _ in results}
    for seed in seeds[1:]:
        print(f"\n  replicate draw {seed}...", end="", flush=True)
        rep_train, rep_test, rep_recorded = one_draw(
            rows, features, args.slots_per_template, seed
        )
        rep_results, _ = evaluate_draw(rep_train, rep_test, rep_recorded)
        for name, acc, _, _ in rep_results:
            per_draw[name][seed] = acc
        print("\r  replicate draw {}: {}".format(
            seed, "  ".join(f"{n} {a:.4f}" for n, a, _, _ in rep_results)))

    save_spread(per_draw, len(test_slots))
    if len(seeds) > 1:
        import statistics

        print(f"\n=== does the comparison depend on which slots were composed? "
              f"({len(seeds)} draws) ===")
        print(f"{'model':<20}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}")
        for name, by_seed in per_draw.items():
            vals = list(by_seed.values())
            sd = f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else "-"
            print(f"{name:<20}{statistics.fmean(vals):>9.4f}{sd:>9}"
                  f"{min(vals):>9.4f}{max(vals):>9.4f}")
        stan_by_seed = per_draw["stan"]
        beaten = [
            s for s in stan_by_seed
            if any(by_seed[s] >= stan_by_seed[s] for n, by_seed in per_draw.items()
                   if n != "stan" and s in by_seed)
        ]
        print(f"\n  stan is matched or beaten by a baseline in {len(beaten)} of "
              f"{len(stan_by_seed)} draws"
              + (f" (seeds {sorted(beaten)})" if beaten else ""))
        print(f"  wrote {SPREAD.name}")

    best_baseline = max(
        (r for r in results if r[0] != "stan"), key=lambda r: r[1]
    )
    stan_acc = next(r[1] for r in results if r[0] == "stan")
    record(
        "WP5-T1 STAN, preliminary",
        "`python experiments/stan_preliminary.py`",
        f"`{OUT.name}`",
        f"composed test: stan {stan_acc:.4f} vs best baseline {best_baseline[0]} "
        f"{best_baseline[1]:.4f}; {len(recorded)} real slots, below the 30-slot gate",
    )


if __name__ == "__main__":
    main()
