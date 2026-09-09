"""WP4-T3: what does onboarding a new client site cost, in labels? (RQ3)

The task asks for leave-one-venue-out followed by adding {0, 25, 100, 300} frames from the
held-out venue - an adaptation curve per venue, answering what a new site costs. **That
experiment cannot be run on this dataset, and the reason is measured here rather than
asserted.** Exactly one of the eight venues has more than one class:

    venue_01                    3 classes  (494 EMPTY, 796 ACTIVE_PLAY, 6 MAINTENANCE)
    the seven clip venues       1 class    (ACTIVE_PLAY only, 282 frames between them)

So holding out `venue_01` leaves a single-class training set that nothing can be fitted to,
and holding out any clip venue leaves a single-class *test* set on which "adaptation" is
unmeasurable - a model improves its score there by answering ACTIVE_PLAY more often, which is
the trap the constant-predictor baseline already documents.

**The runnable form of the same question is one camera down.** `venue_01` has two cameras
looking at opposite halves of one pitch, both carrying EMPTY and ACTIVE_PLAY, and a probe
trained on camera A has never seen camera B. That is a genuine multi-class transfer boundary,
it is the boundary `h3_with_false_play.py` already scores across, and it is the one this
project's deployment story actually turns on: a facility adds a *camera*, not a venue.

So this reports an adaptation curve over **camera** onboarding, and is explicit that it is a
proxy. A camera on the same pitch is an easier target than a new site, so every number here is
an **optimistic bound** on what a new venue would cost.

Three things are measured at each budget *k*:

* **adapted** - camera A's frames plus *k* labelled frames of camera B.
* **target only** - the same *k* frames and nothing else. If this matches, camera A's 775
  frames are contributing nothing and "transfer" is the wrong word for what is happening.
* **source only** (*k* = 0) - the published protocol, for the left-hand end of the curve.

Five seeds per budget, because at *k* = 1 *which* frame you get matters more than anything
else, and a single draw would report that accident as a result.

**The result, and the control is the interesting half.** Macro-F1 on the new camera's unseen
frames goes from 0.441 / 0.357 / 0.508 (DINOv2 / ConvNeXtV2 / ViT) at *k* = 0 to roughly 0.99
at *k* = 1, for all three. But **from *k* = 5 the "target only" column matches the "adapted"
one exactly**, on every backbone: five labelled frames of the new camera, on their own, do as
well as those five plus 775 frames from the source camera.

So the source set stops contributing almost immediately. What buys the accuracy is having
*any* labels from the target camera, not having a large corpus somewhere else. That is a
weaker claim than "the model generalises across cameras" and it is the one these numbers
support - and it is good news operationally, because labelling five frames is cheap and
collecting a large corpus per site is not.

**Read the curve with its denominator.** The camera-B test set is a handful of distinct scenes,
not hundreds of independent frames, and every row carries the count of near-duplicate pairs
crossing the train/test boundary. That count is *why* small budgets work so well: a fixed
camera pointed at a pitch sees very few distinct views, so the first labelled frame of each
class already covers one of them. That is a real property of the deployment and also the reason
these numbers must not be read as a learning curve for a harder problem.

    uv run python experiments/onboarding_cost.py
"""

from __future__ import annotations

import csv
from collections import Counter

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.dedup import dhash, distinct_subset, hamming
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.metrics import evaluate
from pitch_occupancy.vision.heads import LinearProbe

SEED = 42
SEEDS = (0, 1, 2, 3, 4)
BUDGETS = (0, 1, 2, 5, 10, 25, 50)
BACKBONES = ("dinov2", "convnextv2", "vit")
OUT = settings.results_dir / "onboarding_cost.csv"


def camera(row) -> str:
    return PHYSICAL_CAMERA.get(row.camera, row.camera)


def load(backbone: str):
    d = np.load(settings.feature_cache_dir / f"{backbone}.npz", allow_pickle=True)
    index = {str(f): i for i, f in enumerate(d["files"])}
    rows = [
        r for r in development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
        if r.file in index
    ]
    return rows, d["features"][[index[r.file] for r in rows]]


def venue_feasibility(rows) -> list[dict]:
    """Which venues could carry a leave-one-venue-out adaptation curve. Measured, not assumed."""
    by_venue: dict[str, Counter] = {}
    for r in rows:
        by_venue.setdefault(r.venue, Counter())[r.class3] += 1
    out = []
    for venue in sorted(by_venue):
        counts = by_venue[venue]
        out.append({
            "venue": venue,
            "n_frames": sum(counts.values()),
            "n_classes": len(counts),
            "usable_as_target": len(counts) >= 2,
        })
    return out


def stratified_sample(rows, k: int, rng: np.random.Generator) -> set[str]:
    """``k`` frames spread across the classes present, not ``k`` frames of whichever is biggest.

    An operator onboarding a camera labels a few frames of each thing they care about; drawing
    uniformly would spend a budget of one on the majority class and measure nothing about the
    other. At k=1 one class necessarily goes unrepresented, and which one is a property of the
    draw - hence the five seeds.
    """
    if k <= 0:
        return set()
    by_class: dict[str, list] = {}
    for r in rows:
        by_class.setdefault(r.class3, []).append(r)
    classes = sorted(by_class)
    chosen: list[str] = []
    for i in range(k):
        pool = by_class[classes[i % len(classes)]]
        remaining = [r for r in pool if r.file not in chosen]
        if not remaining:
            continue
        chosen.append(remaining[int(rng.integers(len(remaining)))].file)
    return set(chosen)


def crossing_pairs(train_rows, test_rows, cache: dict[str, int], threshold: int = 6) -> int:
    a = [r.file for r in train_rows if r.file in cache]
    b = [r.file for r in test_rows if r.file in cache]
    return sum(1 for x in a for y in b if hamming(cache[x], cache[y]) <= threshold)


def score(X, pos, train_rows, test_rows, backbone: str):
    if len({r.class3 for r in train_rows}) < 2:
        return None
    tr = [pos[r.file] for r in train_rows]
    te = [pos[r.file] for r in test_rows]
    probe = LinearProbe(backbone, seed=SEED).fit(X[tr], train_rows)
    predicted = probe.predict(X[te], test_rows)
    return evaluate([r.class3 for r in test_rows], predicted)


def main() -> None:
    rows, _ = load(BACKBONES[0])

    print("=== can WP4-T3 be run as written? ===")
    feasibility = venue_feasibility(rows)
    for row in feasibility:
        mark = "usable" if row["usable_as_target"] else "single-class"
        print(f"  {row['venue']:<32} {row['n_frames']:>5} frames  "
              f"{row['n_classes']} class(es)  {mark}")
    usable = [r for r in feasibility if r["usable_as_target"]]
    print(f"\n  {len(usable)} of {len(feasibility)} venues carry more than one class.")
    print("  Leave-one-venue-out adaptation is therefore not runnable: holding out the only")
    print("  multi-class venue leaves nothing to fit, and holding out any other leaves a")
    print("  single-class test set where 'adaptation' is unmeasurable. The camera-level")
    print("  curve below is the runnable proxy, and an optimistic one.")

    venue = [r for r in rows if r.venue == "venue_01"]
    source = [r for r in venue if camera(r) == "camera_A"]
    target = [r for r in venue if camera(r) == "camera_B"]
    print(f"\n  source camera_A: {len(source)} frames "
          f"{dict(Counter(r.class3 for r in source))}")
    print(f"  target camera_B: {len(target)} frames "
          f"{dict(Counter(r.class3 for r in target))}")

    print(f"\nhashing {len(venue)} frames for the leakage counts...")
    import cv2

    cache = {}
    for r in venue:
        image = cv2.imread(str(settings.dataset_dir / r.file))
        if image is not None:
            cache[r.file] = dhash(image)
    scenes = len(distinct_subset({r.file: cache[r.file] for r in target if r.file in cache}))
    print(f"  camera_B's {len(target)} frames are {scenes} distinct scenes")

    results: list[dict] = []
    print(f"\n=== onboarding curve (macro-F1 on camera_B's unseen frames, "
          f"{len(SEEDS)} seeds) ===")
    print(f"{'backbone':<12}{'k':>4}  {'adapted':>21}  {'target only':>12}"
          f"{'n_test':>8}{'leak':>9}")

    for backbone in BACKBONES:
        rows_b, X = load(backbone)
        pos = {r.file: i for i, r in enumerate(rows_b)}
        for k in BUDGETS:
            adapted, alone, sizes, leaks = [], [], [], []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                picked = stratified_sample(target, k, rng)
                extra = [r for r in target if r.file in picked]
                test = [r for r in target if r.file not in picked]
                if not test:
                    continue
                a = score(X, pos, source + extra, test, backbone)
                if a is not None:
                    adapted.append(a.macro_f1)
                b = score(X, pos, extra, test, backbone)
                if b is not None:
                    alone.append(b.macro_f1)
                sizes.append(len(test))
                leaks.append(crossing_pairs(source + extra, test, cache))
            if not adapted:
                continue
            row = {
                "backbone": backbone, "k": k,
                "adapted_mean": float(np.mean(adapted)),
                "adapted_min": float(np.min(adapted)),
                "adapted_max": float(np.max(adapted)),
                "target_only_mean": float(np.mean(alone)) if alone else float("nan"),
                "n_test": int(np.mean(sizes)),
                "n_distinct_scenes": scenes,
                "crossing_pairs": int(np.mean(leaks)),
                "n_seeds": len(adapted),
            }
            results.append(row)
            spread = (f"{row['adapted_mean']:.3f} "
                      f"[{row['adapted_min']:.3f},{row['adapted_max']:.3f}]")
            alone_text = "-" if not alone else f"{row['target_only_mean']:.3f}"
            print(f"{backbone:<12}{k:>4}  {spread:>21}  {alone_text:>12}"
                  f"{row['n_test']:>8}{row['crossing_pairs']:>9,}")
        print()

    print("=== what a camera costs ===")
    for backbone in BACKBONES:
        mine = {r["k"]: r for r in results if r["backbone"] == backbone}
        if 0 not in mine:
            continue
        base = mine[0]["adapted_mean"]
        reached = next(
            (k for k in BUDGETS if k in mine and mine[k]["adapted_mean"] >= 0.9), None
        )
        best = max(mine.values(), key=lambda r: r["adapted_mean"])
        print(f"  {backbone:<12} k=0 {base:.3f} -> best {best['adapted_mean']:.3f} at "
              f"k={best['k']}; first k reaching 0.90 macro-F1: "
              f"{reached if reached is not None else 'none'}")
    # The control that reframes the transfer story, so it gets its own section rather than a
    # column somebody has to notice.
    print("\n=== does the source camera help at all? ===")
    for backbone in BACKBONES:
        mine = {r["k"]: r for r in results if r["backbone"] == backbone}
        crossover = None
        for k in BUDGETS:
            row = mine.get(k)
            if row is None or row["target_only_mean"] != row["target_only_mean"]:
                continue  # k too small to fit the target-only model at all
            if row["target_only_mean"] >= row["adapted_mean"] - 0.005:
                crossover = k
                break
        if crossover is None:
            print(f"  {backbone:<12} the source frames still help at every budget tested")
        else:
            print(f"  {backbone:<12} from k={crossover}, training on the target frames ALONE "
                  f"matches {len(source)} source frames plus them")
    print("\n  So what buys the accuracy is having *any* labels from the new camera, not")
    print("  having a large source set. That is a weaker transfer story than 'the model")
    print("  generalises', and it is the one the numbers support.")

    print("\n  A camera on the same pitch is an easier target than a new venue, so these are")
    print("  an optimistic bound on onboarding a site. Read them beside the leak column and")
    print(f"  the {scenes} distinct scenes: small budgets work here partly because a fixed")
    print("  camera sees very few distinct views, which is true in deployment too.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["backbone", "k", "adapted_mean", "adapted_min", "adapted_max",
              "target_only_mean", "n_test", "n_distinct_scenes", "crossing_pairs", "n_seeds"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})
    print(f"\nwrote {OUT.name}")

    dino = {r["k"]: r["adapted_mean"] for r in results if r["backbone"] == "dinov2"}
    record(
        "WP4-T3 onboarding cost, camera-level",
        "`python experiments/onboarding_cost.py`",
        f"`{OUT.name}`",
        f"leave-one-venue-out adaptation is not runnable ({len(usable)} of "
        f"{len(feasibility)} venues carry >1 class); at camera level DINOv2 goes "
        f"{dino.get(0, float('nan')):.3f} -> {dino.get(5, float('nan')):.3f} macro-F1 "
        f"with 5 labelled frames of the new camera",
    )


if __name__ == "__main__":
    main()
