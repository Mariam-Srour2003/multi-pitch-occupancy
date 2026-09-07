"""Does class balancing help, and can its acceptance criterion even be checked? (WP3-T7)

WP3-T7 asks for `class_weight='balanced'` by default and accepts on *"C3 recall improves on
validation vs unweighted"*. The first half is already true. The second half turns out to be
unanswerable with this dataset, and the reason is worth a paragraph rather than a shrug.

**C3 has six frames, and they are all the same moment.** Every one comes from
`venue_01`, camera A, the 2026-07-11 10:00 slot, in daylight; four of the six are seconds
apart. So under any leakage-free split the class lands entirely on one side:

* grouped split — 6 in train, **0 in test**;
* leave-one-venue-out — 6/0 for all seven clip venues, and 0/6 for the `venue_01` fold,
  which therefore has no C3 training data at all.

There is no partition where C3 can be trained on *and* scored on. A number could be produced
by shuffling frames randomly, but four of the six are near-duplicates, so it would measure
memorisation of one 30-second window. **The acceptance criterion cannot be met, and reporting
a random-split figure instead would be reporting an artifact.**

## What *is* answerable, and it justifies the setting for a different reason

`balanced` gives those six frames a weight of about **88x**, so the obvious worry is that the
weight is spent on a class the model cannot learn while the cost lands on C1 and C2 — the
classes the billing decision actually uses. Measured, that worry looks confirmed: balancing
costs **0.0286** mean cross-venue ACTIVE_PLAY recall and 0.1167 on the worst fold.

**It is not confirmed. The recall was free, and the control is what shows it.** Every
cross-venue fold is 100% ACTIVE_PLAY, so recall rises simply by predicting PLAY more often.
On held-out EMPTY frames the unweighted probe calls **46.5%** of empty pitches a match,
against **23.1%** balanced. Dropping the weight does not improve the model; it moves the
boundary and collects recall on a single-class test set.

So `balanced=True` stays the default — earned by halving the false-play rate, not by the C3
recall the task asked for, which cannot be measured at all.

**The control nearly failed to catch it.** A grouped split inside `venue_01` splits on
`slot_id`, and `venue_01` has two slots, so it is nearly all-or-nothing: it left **nine**
EMPTY frames, on which both settings scored 0.000 and the recall gain looked real. Splitting
on the physical camera instead — two genuinely different views, across both slots — leaves
243 EMPTY frames and reverses the conclusion. A control too small to distinguish 0.0 from 0.5
is not a control.

Runs on the existing `ablate_dinov2_full.npz` cache, so it needs no embedding pass and does
not compete with a running search.

    uv run python experiments/class_balancing.py
"""

from __future__ import annotations

import csv
from collections import Counter

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, grouped_split, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe

CACHE = settings.feature_cache_dir / "ablate_dinov2_full.npz"
OUT = settings.results_dir / "class_balancing.csv"
C3 = "C3_MAINTENANCE_NON_SPORTING"
PLAY = "C2_ACTIVE_PLAY"


def load_features(rows):
    data = np.load(CACHE, allow_pickle=True)
    index = {str(f): i for i, f in enumerate(data["files"])}
    features = data["features"]
    keep = [(r, index[r.file]) for r in rows if r.file in index]
    return [r for r, _ in keep], features[[i for _, i in keep]]


def report_c3_is_unmeasurable(rows) -> list[str]:
    """Show the criterion cannot be met, rather than asserting it."""
    lines = []
    split = grouped_split(rows)
    a = sum(1 for r in split.train if r.class3 == C3)
    b = sum(1 for r in split.test if r.class3 == C3)
    lines.append(f"grouped split: C3 train={a} test={b}")
    both = 0
    for fold in leave_one_group_out(rows):
        x = sum(1 for r in fold.train if r.class3 == C3)
        y = sum(1 for r in fold.test if r.class3 == C3)
        both += bool(x and y)
        lines.append(f"  {fold.name}: C3 train={x} test={y}")
    lines.append(f"folds with C3 on both sides: {both}")
    return lines


EMPTY = "C1_EMPTY"


def _false_play_control(rows, features, index) -> list[tuple[str, float, int]]:
    """How often each weighting calls an empty pitch a match.

    Needed because the cross-venue folds are entirely ACTIVE_PLAY: on a single-class test set
    recall rises simply by predicting that class more often, so a recall gain means nothing
    until the cost on EMPTY frames is known. Only `venue_01` has empty frames - that is the
    dataset gap the whole project keeps running into - so the control uses a grouped split
    inside it, which keeps whole slots on one side.
    """
    venue = [r for r in rows if r.venue == "venue_01"]
    # Split on the *physical camera*, not the slot. venue_01 has only two slots, so a
    # slot-grouped split is nearly all-or-nothing: it left nine EMPTY frames to test on,
    # which cannot distinguish a false-play rate of 0 from one of 0.1. The two cameras see
    # different halves of the pitch across both slots, so holding one out keeps the split
    # honest and leaves hundreds of empty frames to measure against.
    train = [r for r in venue if PHYSICAL_CAMERA.get(r.camera, r.camera) == "camera_A"]
    test = [r for r in venue if PHYSICAL_CAMERA.get(r.camera, r.camera) == "camera_B"]
    empties = [r for r in test if r.class3 == EMPTY]
    out = []
    if not empties or len({r.class3 for r in train}) < 2:
        return [("(no held-out EMPTY frames available)", float("nan"), 0)]
    Xtr = features[[index[r.file] for r in train]]
    Xte = features[[index[r.file] for r in empties]]
    for balanced in (True, False):
        probe = LinearProbe(balanced=balanced).fit(Xtr, train)
        pred = probe.predict(Xte, empties)
        rate = sum(p == PLAY for p in pred) / len(empties)
        out.append(("balanced" if balanced else "unweighted", rate, len(empties)))
    return out


def main() -> None:
    if not CACHE.exists():
        raise SystemExit(
            f"no cached features at {CACHE}. Run experiments/input_ablation.py first."
        )
    rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    rows, features = load_features(rows)
    print(f"{len(rows)} frames, {features.shape[1]}-d features")
    print(f"class counts: {dict(Counter(r.class3 for r in rows))}")

    print("\n--- can the acceptance criterion be checked? ---")
    for line in report_c3_is_unmeasurable(rows):
        print(line)
    n_c3 = sum(1 for r in rows if r.class3 == C3)
    print(f"'balanced' weights C3 about {len(rows) / (3 * n_c3):.0f}x")

    print("\n--- what balancing costs the classes that are used ---")
    index = {r.file: i for i, r in enumerate(rows)}
    results = []
    for balanced in (True, False):
        per_fold = []
        for fold in leave_one_group_out(rows):
            tr = [r for r in fold.train if r.file in index]
            te = [r for r in fold.test if r.file in index]
            play = [r for r in te if r.class3 == PLAY]
            if not play or len({r.class3 for r in tr}) < 2:
                continue
            Xtr = features[[index[r.file] for r in tr]]
            Xte = features[[index[r.file] for r in play]]
            probe = LinearProbe(balanced=balanced).fit(Xtr, tr)
            pred = probe.predict(Xte, play)
            per_fold.append((fold.name, sum(p == PLAY for p in pred) / len(play), len(play)))
        mean = float(np.mean([r for _, r, _ in per_fold]))
        worst = min(r for _, r, _ in per_fold)
        results.append((balanced, mean, worst, per_fold))
        label = "balanced" if balanced else "unweighted"
        print(f"{label:11} cross-venue play recall: mean {mean:.4f}  worst fold {worst:.4f}")

    (bal, bal_mean, bal_worst, bal_folds), (unw, unw_mean, unw_worst, unw_folds) = results
    print(f"\ndelta (balanced - unweighted): mean {bal_mean - unw_mean:+.4f}  "
          f"worst fold {bal_worst - unw_worst:+.4f}")

    false_play = _false_play_control(rows, features, index)
    print("\n--- the false-play control, which decides whether that delta is real ---")
    print("Every cross-venue fold is 100% ACTIVE_PLAY, so recall can be bought outright by")
    print("saying PLAY more often - and dropping the weight on a minority class is exactly")
    print("the kind of change that would do that. Measured on held-out EMPTY frames:")
    for label, rate, n in false_play:
        print(f"  {label:11} false-play rate on {n} held-out EMPTY frames: {rate:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["weighting", "fold", "n_play", "play_recall"])
        for balanced, _, _, folds in results:
            for name, recall, n in folds:
                w.writerow(["balanced" if balanced else "unweighted", name, n, f"{recall:.4f}"])
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
