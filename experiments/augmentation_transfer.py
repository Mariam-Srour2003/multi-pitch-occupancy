"""WP3-T6: can augmentation buy what five labelled frames buy? (RQ3)

`onboarding_cost.py` found that a probe trained on camera A scores 0.441 macro-F1 on camera B
and 0.9895 once it has seen **one** labelled frame of B. That leaves an obvious question, and
it is the one that matters commercially: **can train-time augmentation close that gap without
any labels from the target camera?** If it can, onboarding costs nothing. If it cannot, the
five-frame recipe is the answer and augmentation is not a substitute for it.

This is the form of WP3-T6's outstanding "flag in benchmark" item worth running. A generic
augmentation switch across every protocol needs a backbone forward pass per view and its own
extraction budget (see `docs/IDEAS.md` #2); this asks one question, on one boundary, for one
backbone.

**Why augmentation is a plausible fix here, so the null means something.** Camera B differs
from camera A in viewpoint, in what fills the frame, and in the photometric character a fixed
sensor gives a fixed scene. Three of the four presets act on exactly that: `colour` jitters
brightness, contrast, saturation and hue - and turf hue is the venue cue the input ablation
implicated; `light` varies brightness, gamma and noise; `weather` adds haze and streaks. If
cross-camera failure were photometric, these should recover some of it.

**What is held fixed.** Augmented views are added to camera A's training set; camera B
contributes nothing at any point, so the k = 0 column is comparable to `onboarding_cost.csv`'s
directly. The probe, its seed and the test set are unchanged. Only the training set grows.

**The honest null, stated before running it.** Augmentation multiplies the training set, and a
larger training set is not free of confounds: if augmentation helps, the cause might be the
extra views rather than their variety. So `views=N` with the `none` preset is run as a control -
N identical copies of each frame, which adds the same number of rows and no variety at all. A
gain that the `none` control reproduces is not augmentation.

The answer: the first draw said one thing and the next four said another
------------------------------------------------------------------------

**The published result was a draw, not an effect.** One augmentation draw (seed 42) produced
`light` at 0.8550 macro-F1 with empty-pitch recall 0.687, against 0.4406 unaugmented — and it
was reported as a finding. Re-drawn, with the same rows, the same preset, the same probe seed
and the same test set, it does not hold:

| draw (seed) | macro-F1 | EMPTY recall | frames called EMPTY (of 521) |
|---|---|---|---|
| 99 | 0.3479 | 0.000 | 0 |
| 7 | 0.3501 | 0.000 | 0 |
| 123 | 0.3510 | 0.000 | 0 |
| 13 | 0.4136 | 0.062 | 15 |
| **42** *(the published one)* | **0.8550** | **0.687** | — |
| *no augmentation* | *0.4406* | *0.000* | *0* |

Nothing differs between those rows but which random brightness, gamma and noise values landed
on which of the 775 source frames. `light` applies each effect with probability 0.5, so a
draw is 3,100 views' worth of coin flips, and the aggregate distribution of a preset is
almost identical between seeds. The metric is not.

**What that means.** The number to read is the spread, not the mean. Over five draws `light`
runs 0.3479-0.8550, sd **0.2206**, median **0.3510**, and **four of the five score below the
0.4406 the probe reaches with no augmentation at all**. The published number is the maximum
of five. The mechanism is visible in the `pred_empty` column, which counts how many of camera
B's 521 frames the probe called EMPTY: three draws call **none** of them empty, one calls 15,
and the published draw found the boundary. What moves between draws is whether the
fitted boundary reaches camera B's empty pitch at all, and that is not a small perturbation
of a stable quantity — it is the difference between a working classifier and the trivial
one. Every preset that scores 0.3479 is that trivial predictor: 0.3479 is not a measurement
of colour jitter, it is the macro-F1 arithmetic of answering ACTIVE_PLAY to all 521 frames.

**So the honest reading is a negative result with one bright draw in it.** Augmentation
*can* recover empty-pitch recall across this camera boundary — one draw in five did, and
0.687 recall with no target labels at all is not noise. It cannot be relied on to: four draws
in five left the probe no better than not augmenting at all, three of the five produced the
trivial one-class predictor, and nothing in a run tells you in advance which one you are in.
A deployment cannot ship "usually nothing, occasionally 0.687". The five-frame recipe reaches
0.9895 every time, and it is the answer to the onboarding question.

**Why this was worth the hour.** The single-draw result was written up, tested, entered in the
claims ledger and pinned by two tests that would have failed loudly if anyone had *weakened*
it. None of that machinery could see the thing that was actually wrong, because all of it
described the same one draw. The check that found it is the cheapest possible: run it again
with a different seed.

    uv run python experiments/augmentation_transfer.py
    uv run python experiments/augmentation_transfer.py --seeds 42,7,13,99,123 --presets light
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from contextlib import contextmanager
from dataclasses import fields as dataclass_fields
from datetime import datetime

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
"""The published draw, and the probe's seed everywhere.

`--seeds` varies the *augmentation* draw and nothing else: the probe is fitted with this
seed on every run, so a difference between two seeds is the draw and cannot be the fit.
"""

BACKBONE = "dinov2"
OUT = settings.results_dir / "augmentation_transfer.csv"
SPREAD = settings.results_dir / "augmentation_transfer_spread.csv"
LOCK = settings.results_dir / ".augmentation_transfer.lock"

#: `none` first: it is the control, and reading the table top-down should start from what
#: adding rows alone does before any preset is credited with anything.
ORDER = ("none", "colour", "light", "weather", "full")

FIELDS = ("preset", "seed", "views", "n_train", "macro_f1", "empty_recall", "play_recall",
          "pred_empty", "n_test")

SPREAD_FIELDS = ("preset", "n_seeds", "seeds", "macro_f1_mean", "macro_f1_sd", "macro_f1_min",
                 "macro_f1_max", "empty_recall_mean", "empty_recall_min", "empty_recall_max")


@contextmanager
def only_one_run():
    """Refuse to start while another run of this script is going.

    Resuming made concurrent runs *data-losing* rather than merely wasteful: each process
    reads the CSV once at startup and rewrites the whole file after every preset, so the one
    that finishes second erases the rows the first added. Nearly done during development —
    a five-second re-run to check a flag would have deleted a draw that took a quarter of an
    hour to produce, and nothing would have said so.

    `O_EXCL` rather than a check-then-create: the gap between the two is exactly the race
    being closed. A stale lock left by a crash is removed by hand, and the message says so
    rather than timing it out - a lock that expires on a timer would silently permit the
    thing it exists to prevent on any run longer than the timer, and these runs are hours.
    """
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"another run holds {LOCK.name} ({LOCK.read_text(encoding='utf-8').strip()}).\n"
            f"Two runs would erase each other's rows: each rewrites the whole CSV from the "
            f"rows it read at startup. If that run died, delete the file and start again."
        ) from None
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(f"pid {os.getpid()} started {datetime.now().isoformat(timespec='seconds')}")
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)


def parse_seeds(text: str) -> list[int]:
    """`--seeds 42,7,13` -> [42, 7, 13], rejecting an empty list rather than running nothing."""
    seeds = [int(part) for part in str(text).replace(",", " ").split()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds needs at least one integer")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"--seeds repeats a value: {seeds}")
    return seeds


def draws_nothing(preset: str) -> bool:
    """True when a preset enables no effect, so `augment` never touches the generator.

    Only `none` does, and it is the point of it - the control adds rows and no variety. It
    also makes the control **seed-independent by construction**: with nothing enabled the
    views are byte-identical whatever the seed, so re-embedding it once per seed would buy
    thirteen minutes of identical numbers. This is read off the config rather than by name,
    so a preset that is quietly emptied is caught too.
    """
    config = AUGMENTATIONS[preset]
    return not any(getattr(config, f.name) for f in dataclass_fields(config) if f.name != "p")


def load_existing() -> list[dict]:
    """Rows already on disk, so a run that died - or a second run adding seeds - resumes.

    Rows written before `--seeds` existed carry no `seed` column. They were all run under
    the published draw, so they are backfilled with it rather than dropped: throwing away an
    hour of embedding to avoid a two-line backfill is the wrong trade.
    """
    if not OUT.exists():
        return []
    with OUT.open(encoding="utf-8", newline="") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    for row in rows:
        if not row.get("seed"):
            row["seed"] = str(SEED) if row["preset"] != "baseline" else ""
    return rows


def key(row: dict) -> tuple[str, str, str]:
    return (str(row["preset"]), str(row.get("seed", "")), str(row["views"]))


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


def embed_augmented(rows, preset: str, views: int, model, processor, spec, *,
                    seed: int = SEED, batch: int = 16):
    """Embed ``views`` augmented copies of each row. Returns features and their labels.

    Every view goes through the same processor path the cache used, so an augmented feature
    and a cached one live in the same space. A separate path here would make the comparison
    against `onboarding_cost.csv` meaningless while looking fine.
    """
    import cv2
    from PIL import Image

    config = AUGMENTATIONS[preset]
    rng = np.random.default_rng(seed)
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


def _f(row: dict, column: str) -> float:
    """One number out of a row that may have come from memory or read back from the CSV."""
    return float(row[column])


def row_for(results: list[dict], preset: str, seed: int, views: int) -> dict | None:
    """The row for one preset under one draw, falling back to a seed-independent one.

    `none` is run once and stands for every seed (see `draws_nothing`), so a per-seed
    comparison that insisted on a `none` row under its own seed would silently drop the
    control - the one row the headline is not allowed to lose.
    """
    exact = [r for r in results
             if r["preset"] == preset and str(r["seed"]) == str(seed)
             and str(r["views"]) == str(views)]
    if exact:
        return exact[0]
    if draws_nothing(preset):
        shared = [r for r in results
                  if r["preset"] == preset and str(r["views"]) == str(views)]
        if shared:
            return shared[0]
    return None


def summarise(results: list[dict], views: int) -> list[dict]:
    """Per-preset spread across the draws that were actually run.

    Mean, sample sd and the observed range - not a confidence interval. Three or four draws
    do not support an interval, and quoting one would claim a precision the run does not
    have. The range is what a reader can check against the per-draw rows.
    """
    out: list[dict] = []
    for preset in ORDER:
        rows = [r for r in results
                if r["preset"] == preset and str(r["views"]) == str(views)]
        if not rows:
            continue
        f1 = [_f(r, "macro_f1") for r in rows]
        empty = [_f(r, "empty_recall") for r in rows]
        out.append({
            "preset": preset,
            "n_seeds": len(rows),
            "seeds": " ".join(sorted(str(r["seed"]) for r in rows)),
            "macro_f1_mean": statistics.fmean(f1),
            "macro_f1_sd": statistics.stdev(f1) if len(f1) > 1 else float("nan"),
            "macro_f1_min": min(f1),
            "macro_f1_max": max(f1),
            "empty_recall_mean": statistics.fmean(empty),
            "empty_recall_min": min(empty),
            "empty_recall_max": max(empty),
        })
    return out


#: The claims the test suite pins, written as questions about a *single* draw so that a
#: re-draw can answer them one at a time.
CLAIM_KEYS = (
    "light beats no augmentation by >0.2",
    "light recovers EMPTY recall above 0.5",
    "the duplicate-rows control does not",
    "full does not overtake light",
    "light is the best preset",
)


def stability(results: list[dict], seeds: list[int], views: int, baseline_f1: float) -> list[dict]:
    """Does each published claim hold in every draw, or only in the one that was published?

    A claim that holds in four draws out of four is not proven, but a claim that holds in one
    out of four is refuted - and with a single draw there is no way to tell those apart. That
    is the entire reason this function exists.
    """
    checks: list[dict] = []
    for seed in seeds:
        rows = {p: row_for(results, p, seed, views) for p in ORDER}
        if any(rows[p] is None for p in ORDER):
            continue  # an incomplete draw answers nothing; a partial one would mislead
        light, control, full = rows["light"], rows["none"], rows["full"]
        best = max(ORDER, key=lambda p: _f(rows[p], "macro_f1"))
        checks.append({
            "seed": seed,
            "light_macro_f1": _f(light, "macro_f1"),
            "light_empty_recall": _f(light, "empty_recall"),
            CLAIM_KEYS[0]: _f(light, "macro_f1") > baseline_f1 + 0.2,
            CLAIM_KEYS[1]: _f(light, "empty_recall") > 0.5,
            CLAIM_KEYS[2]: _f(control, "macro_f1") <= baseline_f1,
            CLAIM_KEYS[3]: _f(full, "macro_f1") < _f(light, "macro_f1"),
            CLAIM_KEYS[4]: best == "light",
        })
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", type=int, default=4, help="augmented copies per source frame")
    ap.add_argument(
        "--seeds", default=str(SEED),
        help="comma-separated augmentation draws, e.g. 42,7,13. Each is a full pass over the "
             "presets. Rows already in the CSV are never recomputed, so adding a seed to an "
             "earlier run costs only the new seed.",
    )
    ap.add_argument(
        "--presets", default=None,
        help="comma-separated presets to run, e.g. light. Extra draws are worth spending on "
             "the number in dispute rather than on presets that are already settled - and "
             "`stability` scores only draws that have every preset, so a partial draw "
             "cannot answer a question it does not have the rows for.",
    )
    args = ap.parse_args()
    with only_one_run():
        run(args)


def run(args) -> None:
    seeds = parse_seeds(args.seeds)
    presets = ORDER
    if args.presets:
        presets = tuple(p.strip() for p in args.presets.split(",") if p.strip())
        unknown = [p for p in presets if p not in ORDER]
        if unknown:
            raise SystemExit(f"unknown preset(s): {unknown}; have {list(ORDER)}")

    rows, index, features = load_rows()
    venue = [r for r in rows if r.venue == "venue_01"]
    source = [r for r in venue if camera(r) == "camera_A"]
    target = [r for r in venue if camera(r) == "camera_B"]
    truth = [r.class3 for r in target]

    src_X = features[[index[r.file] for r in source]]
    tgt_X = features[[index[r.file] for r in target]]

    print(f"source camera_A: {len(source)} frames; target camera_B: {len(target)} frames")
    print(f"{args.views} augmented view(s) per source frame "
          f"= {len(source) * args.views:,} extra rows per preset")
    print(f"draws: {', '.join(str(s) for s in seeds)}\n")

    baseline = evaluate(
        truth, LinearProbe(BACKBONE, seed=SEED).fit(src_X, source).predict(tgt_X, target)
    )
    print(f"no augmentation (the published k=0 point):  macro-F1 {baseline.macro_f1:.4f}")

    results = load_existing()
    prior = next((r for r in results if r["preset"] == "baseline"), None)
    if prior is not None and abs(_f(prior, "macro_f1") - baseline.macro_f1) > 5e-5:
        # A free reproducibility check: the baseline needs no embedding, so it is recomputed
        # on every run. If it has moved, the feature cache or the manifest has moved with it,
        # and every stored row was scored against the old one - which makes mixing them and
        # the new rows into one table meaningless.
        print(f"  !! the baseline has MOVED: {_f(prior, 'macro_f1'):.4f} on disk vs "
              f"{baseline.macro_f1:.4f} now. The stored rows predate that change; delete "
              f"{OUT.name} and re-run rather than mixing the two.")
    results = [r for r in results if r["preset"] != "baseline"]
    results.insert(0, {
        "preset": "baseline", "seed": "", "views": 0, "n_train": len(source),
        "macro_f1": baseline.macro_f1, "empty_recall": _recall(baseline, "C1_EMPTY"),
        "play_recall": _recall(baseline, "C2_ACTIVE_PLAY"),
        "pred_empty": 0, "n_test": len(target),
    })
    done = {key(r) for r in results}

    header = (f"\n{'preset':<10}{'seed':>6}{'n_train':>9}{'macro-F1':>11}{'vs base':>10}"
              f"{'EMPTY rec':>11}{'PLAY rec':>10}{'pred EMPTY':>12}")
    print(header)
    print(f"{'baseline':<10}{'-':>6}{len(source):>9}{baseline.macro_f1:>11.4f}{'-':>10}"
          f"{_recall(baseline, 'C1_EMPTY'):>11.3f}{_recall(baseline, 'C2_ACTIVE_PLAY'):>10.3f}"
          f"{0:>12}")

    backbone = None
    for seed in seeds:
        for preset in presets:
            if (preset, str(seed), str(args.views)) in done:
                kept = row_for(results, preset, seed, args.views)
                print(f"{preset:<10}{seed:>6}{kept['n_train']:>9}{_f(kept, 'macro_f1'):>11.4f}"
                      f"{_f(kept, 'macro_f1') - baseline.macro_f1:>+10.4f}"
                      f"{_f(kept, 'empty_recall'):>11.3f}{_f(kept, 'play_recall'):>10.3f}"
                      f"{str(kept.get('pred_empty', '')):>12}   (already on disk)")
                continue
            if draws_nothing(preset) and row_for(results, preset, seed, args.views) is not None:
                print(f"{preset:<10}{seed:>6}   no effect is enabled, so this draw is "
                      f"byte-identical to the one already run")
                continue
            if backbone is None:
                backbone = load_backbone(BACKBONE)
            model, processor, spec = backbone
            print(f"  embedding {preset} (seed {seed})...", end="", flush=True)
            aug_X, aug_labels = embed_augmented(
                source, preset, args.views, model, processor, spec, seed=seed
            )
            train_rows = list(source) + [
                Row(f"aug_{preset}_{i}", label) for i, label in enumerate(aug_labels)
            ]
            train_X = np.concatenate([src_X, aug_X])
            predicted = LinearProbe(BACKBONE, seed=SEED).fit(train_X, train_rows).predict(
                tgt_X, target
            )
            report = evaluate(truth, predicted)
            pred_empty = sum(1 for p in predicted if p == "C1_EMPTY")
            results.append({
                "preset": preset, "seed": seed, "views": args.views, "n_train": len(train_rows),
                "macro_f1": report.macro_f1, "empty_recall": _recall(report, "C1_EMPTY"),
                "play_recall": _recall(report, "C2_ACTIVE_PLAY"),
                "pred_empty": pred_empty, "n_test": len(target),
            })
            done.add((preset, str(seed), str(args.views)))
            save(results)
            delta = report.macro_f1 - baseline.macro_f1
            print(f"\r{preset:<10}{seed:>6}{len(train_rows):>9}{report.macro_f1:>11.4f}"
                  f"{delta:>+10.4f}{_recall(report, 'C1_EMPTY'):>11.3f}"
                  f"{_recall(report, 'C2_ACTIVE_PLAY'):>10.3f}{pred_empty:>12}")

    save(results)
    spread = summarise(results, args.views)
    save_spread(spread)

    control = next(r for r in spread if r["preset"] == "none")
    best = max((r for r in spread if r["preset"] != "none"), key=lambda r: r["macro_f1_mean"])
    n_draws = max(r["n_seeds"] for r in spread)

    if n_draws > 1:
        print(f"\n=== spread across draws (views={args.views}) ===")
        print(f"{'preset':<10}{'n':>4}{'mean':>10}{'sd':>10}{'min':>10}{'max':>10}"
              f"{'EMPTY mean':>12}{'EMPTY range':>20}")
        for r in spread:
            sd = "-" if r["n_seeds"] < 2 else f"{r['macro_f1_sd']:.4f}"
            span = f"{r['empty_recall_min']:.3f} - {r['empty_recall_max']:.3f}"
            print(f"{r['preset']:<10}{r['n_seeds']:>4}{r['macro_f1_mean']:>10.4f}{sd:>10}"
                  f"{r['macro_f1_min']:>10.4f}{r['macro_f1_max']:>10.4f}"
                  f"{r['empty_recall_mean']:>12.3f}{span:>20}")

    checks = stability(results, seeds, args.views, baseline.macro_f1)
    if checks:
        print(f"\n=== does each published claim survive a re-draw? "
              f"({len(checks)} complete draw(s)) ===")
        for claim in CLAIM_KEYS:
            held = sum(1 for c in checks if c[claim])
            mark = "holds" if held == len(checks) else ("REFUTED" if held == 0 else "NOT ALWAYS")
            print(f"  {claim:<44}{held}/{len(checks)}   {mark}")

    print("\n=== does augmentation substitute for labels? ===")
    print(f"  no augmentation           {baseline.macro_f1:.4f}")
    print(f"  duplicate-rows control    {control['macro_f1_mean']:.4f}  "
          f"({control['macro_f1_mean'] - baseline.macro_f1:+.4f}; same rows, no variety)")
    over = "" if best["n_seeds"] < 2 else f", mean of {best['n_seeds']} draws"
    print(f"  best preset ({best['preset']})       {best['macro_f1_mean']:.4f}  "
          f"({best['macro_f1_mean'] - baseline.macro_f1:+.4f}{over})")
    print(f"  attributable to variety   "
          f"{best['macro_f1_mean'] - control['macro_f1_mean']:+.4f}")
    print("\n  one labelled frame of the target camera    0.9895  (onboarding_cost.csv)")
    gap = 0.9895 - baseline.macro_f1
    closed = (best["macro_f1_mean"] - baseline.macro_f1) / gap if gap else float("nan")
    print(f"  augmentation closes {closed:.0%} of the gap that one label closes")

    print(f"\nwrote {OUT.name} and {SPREAD.name}")

    spread_note = ("one draw only" if best["n_seeds"] < 2 else
                   f"{best['n_seeds']} draws, {best['macro_f1_min']:.4f}-"
                   f"{best['macro_f1_max']:.4f}")
    record(
        "WP3-T6 augmentation across cameras",
        f"`python experiments/augmentation_transfer.py --views {args.views} "
        f"--seeds {','.join(str(s) for s in seeds)}`",
        f"`{OUT.name}`",
        f"best preset {best['preset']} {best['macro_f1_mean']:.4f} ({spread_note}) vs "
        f"{baseline.macro_f1:.4f} unaugmented and {control['macro_f1_mean']:.4f} for the "
        f"duplicate-rows control; closes {closed:.0%} of the gap one labelled frame closes",
    )


def save(results: list[dict]) -> None:
    """Write the table after every preset, not once at the end.

    Learned the hard way: the first full run embedded 15,500 views over roughly an hour, died
    during the fourth preset, and lost the three that had already finished because the CSV was
    written after the loop. A long experiment that keeps nothing until it finishes is one
    crash away from having done nothing. With several draws that matters more, not less - the
    run is now hours long, and a resumed run reads this file back rather than recomputing.
    """
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        # restval, because a row written by an older version of this script - or by a test -
        # legitimately has no `seed` or `pred_empty`, and dropping such a row to keep the
        # schema tidy would throw away the very hour of embedding this function exists for.
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS), restval="", extrasaction="ignore")
        writer.writeheader()
        for row in sorted(results, key=_order):
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})


def _order(row: dict) -> tuple[int, int, str]:
    """Baseline first, then preset order, then seed - so the file reads like the table."""
    preset = str(row.get("preset", ""))
    rank = ORDER.index(preset) if preset in ORDER else -1
    seed = str(row.get("seed", ""))
    return (rank, int(seed) if seed.lstrip("-").isdigit() else -1, preset)


def save_spread(spread: list[dict]) -> None:
    """The across-draw summary, as its own artefact.

    Separate from the per-draw table because a claim should cite one number and its spread
    without a selector that quietly picks a seed. `nan` is written as an empty cell: a
    single-draw preset has no standard deviation, and writing 0.0000 there would state the
    opposite of what is known.
    """
    SPREAD.parent.mkdir(parents=True, exist_ok=True)
    with SPREAD.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SPREAD_FIELDS))
        writer.writeheader()
        for row in spread:
            writer.writerow({
                k: ("" if isinstance(v, float) and v != v
                    else f"{v:.4f}" if isinstance(v, float) else v)
                for k, v in row.items()
            })


def _recall(report, label: str) -> float:
    for score in report.per_class:
        if score.label == label:
            return score.recall
    return float("nan")


if __name__ == "__main__":
    main()
