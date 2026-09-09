"""WP3-T6: can augmentation buy what five labelled frames buy? (RQ3)

`onboarding_cost.py` found that a probe trained on camera A scores 0.441 macro-F1 on camera B
and 0.9895 once it has seen **one** labelled frame of B. That leaves an obvious question, and
it is the one that matters commercially: **can train-time augmentation close that gap without
any labels from the target camera?** If it can, onboarding costs nothing. If it cannot, the
five-frame recipe is the answer and augmentation is not a substitute for it.

This is the form of WP3-T6's outstanding "flag in benchmark" item worth running. A generic
augmentation switch across every protocol needs a backbone forward pass per view and its own
extraction budget (see `docs/IDEAS.md` #2); this asks one question, on one boundary, for one
backbone, and costs a few minutes.

**Why augmentation is a plausible fix here, so the null means something.** Camera B differs
from camera A in viewpoint, in what fills the frame, and in the photometric character a fixed
sensor gives a fixed scene. Three of the four presets act on exactly that: `colour` jitters
brightness, contrast, saturation and hue - and turf hue is the venue cue the input ablation
implicated; `light` varies brightness, gamma and noise; `weather` adds haze and streaks. If
cross-camera failure were photometric, these should recover some of it.

**What is held fixed.** Augmented views are added to camera A's training set; camera B
contributes nothing at any point, so the k = 0 column is comparable to `onboarding_cost.csv`'s
directly. The probe, the seed and the test set are unchanged. Only the training set grows.

**The honest null, stated before running it.** Augmentation multiplies the training set, and a
larger training set is not free of confounds: if augmentation helps, the cause might be the
extra views rather than their variety. So `views=N` with the `none` preset is run as a control -
N identical copies of each frame, which adds the same number of rows and no variety at all. A
gain that the `none` control reproduces is not augmentation.

The answer, and it is neither of the two clean ones
---------------------------------------------------

| preset | macro-F1 | vs baseline | EMPTY recall |
|---|---|---|---|
| no augmentation | 0.4406 | - | 0.000 |
| `none` *(duplicate rows, the control)* | 0.3625 | -0.0781 | 0.000 |
| `colour` | 0.3479 | -0.0926 | 0.000 |
| **`light`** | **0.8550** | **+0.4144** | **0.687** |
| `weather` | 0.4486 | +0.0080 | 0.099 |
| `full` *(everything at once)* | 0.3479 | -0.0926 | 0.000 |

**One preset works, and it works for the failure that mattered.** `light` - brightness, gamma
and sensor noise - takes empty-pitch recall from **0.000 to 0.687** with no labels from the
target camera at all, and closes **75%** of the gap that one labelled frame closes. The
duplicate-rows control moves *down*, so this is variety and not row count.

**Everything else is neutral or harmful, and `full` is the finding to take away.** `full`
contains every one of `light`'s effects and adds colour jitter, fog, rain and flip - and it
scores 0.3479, exactly what `colour` alone scores, with EMPTY recall back at zero. Turning
more augmentation on did not dilute the gain, it **erased** it. The practical rule is the
opposite of the usual instinct: pick the augmentation that matches the shift you are fighting,
and adding the rest costs you the benefit.

So the earlier framing was too strong in both directions. Augmentation is not a substitute for
the five-frame recipe - 0.855 is not 0.9895, and the labelled frames need no CPU. But
"augmentation does not help across cameras" is wrong: the right one recovers most of the
distance, and a deployment that cannot label a new camera immediately has something to use in
the meantime.

**One seed per preset.** The augmentation draw is seeded and reproducible, but only one draw
was taken, so nothing here bounds the variance of `light`'s 0.855. Read the ordering, not the
third decimal.

    uv run python experiments/augmentation_transfer.py
    uv run python experiments/augmentation_transfer.py --views 6
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.metrics import evaluate
from pitch_occupancy.vision.augment import AUGMENTATIONS, augment
from pitch_occupancy.vision.backbones import embed_batch, load_backbone
from pitch_occupancy.vision.heads import LinearProbe

SEED = 42
BACKBONE = "dinov2"
OUT = settings.results_dir / "augmentation_transfer.csv"

#: `none` first: it is the control, and reading the table top-down should start from what
#: adding rows alone does before any preset is credited with anything.
ORDER = ("none", "colour", "light", "weather", "full")


def camera(row) -> str:
    return PHYSICAL_CAMERA.get(row.camera, row.camera)


def load_rows():
    d = np.load(settings.feature_cache_dir / f"{BACKBONE}.npz", allow_pickle=True)
    index = {str(f): i for i, f in enumerate(d["files"])}
    rows = [
        r for r in development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
        if r.file in index
    ]
    return rows, index, d["features"]


def embed_augmented(rows, preset: str, views: int, model, processor, spec, *, batch: int = 16):
    """Embed ``views`` augmented copies of each row. Returns features and their labels.

    Every view goes through the same processor path the cache used, so an augmented feature
    and a cached one live in the same space. A separate path here would make the comparison
    against `onboarding_cost.csv` meaningless while looking fine.
    """
    import cv2
    from PIL import Image

    config = AUGMENTATIONS[preset]
    rng = np.random.default_rng(SEED)
    features: list[np.ndarray] = []
    labels: list[str] = []
    batch_images: list[Image.Image] = []
    batch_labels: list[str] = []

    def flush():
        if batch_images:
            features.append(embed_batch(model, processor, spec, batch_images))
            labels.extend(batch_labels)
            batch_images.clear()
            batch_labels.clear()

    for row in rows:
        image = cv2.imread(str(settings.dataset_dir / row.file))
        if image is None:
            continue
        for _ in range(views):
            view = augment(image, config, rng=rng)
            batch_images.append(Image.fromarray(view[:, :, ::-1]))
            batch_labels.append(row.class3)
            if len(batch_images) >= batch:
                flush()
    flush()
    return (np.concatenate(features) if features else np.zeros((0, 1))), labels


class Row:
    """A minimal manifest-row stand-in, so augmented views can carry a label into the probe."""

    __slots__ = ("file", "class3", "lighting")

    def __init__(self, file: str, class3: str) -> None:
        self.file, self.class3, self.lighting = file, class3, "day"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", type=int, default=4, help="augmented copies per source frame")
    args = ap.parse_args()

    rows, index, features = load_rows()
    venue = [r for r in rows if r.venue == "venue_01"]
    source = [r for r in venue if camera(r) == "camera_A"]
    target = [r for r in venue if camera(r) == "camera_B"]
    truth = [r.class3 for r in target]

    src_X = features[[index[r.file] for r in source]]
    tgt_X = features[[index[r.file] for r in target]]

    print(f"source camera_A: {len(source)} frames; target camera_B: {len(target)} frames")
    print(f"{args.views} augmented view(s) per source frame "
          f"= {len(source) * args.views:,} extra rows per preset\n")

    baseline = evaluate(
        truth, LinearProbe(BACKBONE, seed=SEED).fit(src_X, source).predict(tgt_X, target)
    )
    print(f"no augmentation (the published k=0 point):  macro-F1 {baseline.macro_f1:.4f}")

    model, processor, spec = load_backbone(BACKBONE)
    results: list[dict] = [{
        "preset": "baseline", "views": 0, "n_train": len(source),
        "macro_f1": baseline.macro_f1, "empty_recall": _recall(baseline, "C1_EMPTY"),
        "play_recall": _recall(baseline, "C2_ACTIVE_PLAY"),
    }]

    print(f"\n{'preset':<12}{'n_train':>9}{'macro-F1':>11}{'vs base':>10}"
          f"{'EMPTY rec':>11}{'PLAY rec':>10}")
    print(f"{'baseline':<12}{len(source):>9}{baseline.macro_f1:>11.4f}{'-':>10}"
          f"{_recall(baseline, 'C1_EMPTY'):>11.3f}{_recall(baseline, 'C2_ACTIVE_PLAY'):>10.3f}")

    for preset in ORDER:
        print(f"  embedding {preset}...", end="", flush=True)
        aug_X, aug_labels = embed_augmented(
            source, preset, args.views, model, processor, spec
        )
        train_rows = list(source) + [
            Row(f"aug_{preset}_{i}", label) for i, label in enumerate(aug_labels)
        ]
        train_X = np.concatenate([src_X, aug_X])
        report = evaluate(
            truth,
            LinearProbe(BACKBONE, seed=SEED).fit(train_X, train_rows).predict(tgt_X, target),
        )
        results.append({
            "preset": preset, "views": args.views, "n_train": len(train_rows),
            "macro_f1": report.macro_f1, "empty_recall": _recall(report, "C1_EMPTY"),
            "play_recall": _recall(report, "C2_ACTIVE_PLAY"),
        })
        save(results)
        delta = report.macro_f1 - baseline.macro_f1
        print(f"\r{preset:<12}{len(train_rows):>9}{report.macro_f1:>11.4f}{delta:>+10.4f}"
              f"{_recall(report, 'C1_EMPTY'):>11.3f}{_recall(report, 'C2_ACTIVE_PLAY'):>10.3f}")

    control = next(r for r in results if r["preset"] == "none")
    best = max((r for r in results if r["preset"] not in ("baseline", "none")),
               key=lambda r: r["macro_f1"])

    print("\n=== does augmentation substitute for labels? ===")
    print(f"  no augmentation           {baseline.macro_f1:.4f}")
    print(f"  duplicate-rows control    {control['macro_f1']:.4f}  "
          f"({control['macro_f1'] - baseline.macro_f1:+.4f}; same rows, no variety)")
    print(f"  best preset ({best['preset']})       {best['macro_f1']:.4f}  "
          f"({best['macro_f1'] - baseline.macro_f1:+.4f})")
    print(f"  attributable to variety   "
          f"{best['macro_f1'] - control['macro_f1']:+.4f}")
    print("\n  one labelled frame of the target camera    0.9895  (onboarding_cost.csv)")
    gap = 0.9895 - baseline.macro_f1
    closed = (best["macro_f1"] - baseline.macro_f1) / gap if gap else float("nan")
    print(f"  augmentation closes {closed:.0%} of the gap that one label closes")

    save(results)
    print(f"\nwrote {OUT.name}")

    record(
        "WP3-T6 augmentation across cameras",
        f"`python experiments/augmentation_transfer.py --views {args.views}`",
        f"`{OUT.name}`",
        f"best preset {best['preset']} {best['macro_f1']:.4f} vs {baseline.macro_f1:.4f} "
        f"unaugmented and {control['macro_f1']:.4f} for the duplicate-rows control; "
        f"closes {closed:.0%} of the gap one labelled frame closes",
    )


def save(results: list[dict]) -> None:
    """Write the table after every preset, not once at the end.

    Learned the hard way: the first full run embedded 15,500 views over roughly an hour, died
    during the fourth preset, and lost the three that had already finished because the CSV was
    written after the loop. A long experiment that keeps nothing until it finishes is one
    crash away from having done nothing.
    """
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "preset", "views", "n_train", "macro_f1", "empty_recall", "play_recall"])
        writer.writeheader()
        for row in results:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})


def _recall(report, label: str) -> float:
    for score in report.per_class:
        if score.label == label:
            return score.recall
    return float("nan")


if __name__ == "__main__":
    main()
