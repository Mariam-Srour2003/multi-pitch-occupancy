"""WP5-T9: does a naive ensemble already capture the multi-backbone headroom? (RQ5)

The gated-fusion module (WP5-T2) is the one WP5 module this data can actually support, so
it carries the M4 gate. Its premise is that the backbones are *complementary*: on
`h3_with_false_play.csv` DINOv2 leads the unweighted fold mean at 0.9297 but is strictly
beaten on three of seven venue folds, and a per-fold oracle reaches 0.9603 - **+3.1 points**
of headroom a gate could in principle capture.

**But a learned gate is only a contribution if something simpler does not get there first.**
Averaging the probes' probabilities is twenty lines and no parameters. If that already
reaches the oracle, the ensemble is the contribution and the gate is decoration - and it is
far better to establish that here, cheaply, than to spend six weeks and be asked at the
defence. This is the sanity baseline the plan calls for before 5.B is built.

Three things are reported side by side, on the two axes that matter:

* each single backbone, so the published numbers are visible;
* **mean probability** and **mean log-probability** ensembles - the two naive forms;
* the **per-fold oracle**, the ceiling that assumes a perfect per-venue choice no gate has.

Both axes, never recall alone. Cross-venue folds are 100% ACTIVE_PLAY, so recall rises by
answering "playing" more often - the mistake the repaired search control caught once already.
The second column is false-play on held-out empty frames, where DINOv2 (0.309) dominates
ConvNeXtV2 (0.992) outright, so a recall gain bought by routing away from DINOv2 on empty
pitches is not a gain.

**The single backbones are checked against the published CSV first.** A harness that cannot
reproduce the number it is extending has no business adding a new one.

    uv run python experiments/logit_average_baseline.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe

PLAY, EMPTY = "C2_ACTIVE_PLAY", "C1_EMPTY"
SEED = 42
OUT = settings.results_dir / "logit_average_baseline.csv"
PUBLISHED = settings.results_dir / "h3_cross_venue_recall.csv"

#: Which cache family to read. The default is the published one, built from *raw* frames
#: handed to the HF processor. `--cache-dir data/cache/geom_probe` reads the letterboxed
#: caches the WP3-T3 probe built instead, and that matters here more than anywhere else:
#: this experiment's second finding - that blending is disqualified because ConvNeXtV2
#: destroys DINOv2's false-play advantage - rests entirely on ConvNeXtV2 scoring 0.9918,
#: and under letterboxing it scores 0.0206. The premise may simply invert.
CACHE_DIR = settings.feature_cache_dir

CACHES = {
    "convnextv2": "convnextv2.npz",
    "dinov2": "dinov2.npz",
    "vit": "vit.npz",
}

#: The plan names {ConvNeXtV2, DINOv2} with ViT optional, so all three combinations are
#: reported: the pair the plan proposes, and the triple, which is the same idea with more
#: of the same. Nothing is tuned - that is the point of a sanity baseline.
ENSEMBLES = {
    "ens_convnextv2_dinov2": ("convnextv2", "dinov2"),
    "ens_all_three": ("convnextv2", "dinov2", "vit"),
}


def common_rows_and_features(rows):
    """Rows every cache covers, with one aligned feature matrix per backbone.

    The ensemble compares predictions frame by frame, so every backbone must see exactly
    the same rows in exactly the same order, and the folds must be built from that shared
    list rather than from three slightly different ones.
    """
    loaded = {}
    for key, name in CACHES.items():
        d = np.load(CACHE_DIR / name, allow_pickle=True)
        loaded[key] = ({str(f): i for i, f in enumerate(d["files"])}, d["features"])

    keep = [r for r in rows if all(r.file in idx for idx, _ in loaded.values())]
    dropped = len(rows) - len(keep)
    if dropped:
        print(f"  {dropped} row(s) missing from at least one cache - dropped, not zero-filled")
    X = {k: feats[[idx[r.file] for r in keep]] for k, (idx, feats) in loaded.items()}
    return keep, X


def _proba(probe: LinearProbe, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    return probe.predict_proba(X), [str(c) for c in probe.classes_]


def _align(p: np.ndarray, classes: list[str], order: list[str]) -> np.ndarray:
    """Reorder a probability matrix onto a shared class order.

    Every probe here is fitted on the same labels so the orders already agree, but relying
    on that is how a silent mislabelling gets in: averaging column 0 of one model with
    column 0 of another is only meaningful if both mean the same class.
    """
    pos = {c: i for i, c in enumerate(classes)}
    return p[:, [pos[c] for c in order]]


def fit_fold(train_rows, test_rows, Xtr, Xte):
    """Fit one probe per backbone and return aligned probabilities on the test rows."""
    order: list[str] | None = None
    out: dict[str, np.ndarray] = {}
    for key in CACHES:
        probe = LinearProbe(key, seed=SEED).fit(Xtr[key], train_rows)
        p, classes = _proba(probe, Xte[key])
        if order is None:
            order = sorted(classes)
        out[key] = _align(p, classes, order)
    assert order is not None
    return out, order


def predictions(probs: dict[str, np.ndarray], order: list[str]) -> dict[str, list[str]]:
    """Single-backbone, mean-probability and mean-log-probability predictions."""
    preds = {k: [order[i] for i in p.argmax(1)] for k, p in probs.items()}
    for name, members in ENSEMBLES.items():
        mean_p = np.mean([probs[m] for m in members], axis=0)
        preds[name] = [order[i] for i in mean_p.argmax(1)]
        # Averaging log-probabilities is a geometric mean: it lets any confident member
        # veto, where the arithmetic mean lets a confident member carry. They disagree
        # exactly where the backbones disagree, which is the case in question.
        mean_l = np.mean([np.log(np.clip(probs[m], 1e-12, None)) for m in members], axis=0)
        preds[f"{name}_logit"] = [order[i] for i in mean_l.argmax(1)]
    return preds


def cross_venue(rows, X) -> dict[str, dict[str, float]]:
    """Per-fold ACTIVE_PLAY recall for every model, on the shared folds."""
    pos = {r.file: i for i, r in enumerate(rows)}
    per_fold: dict[str, dict[str, float]] = {}
    for fold in leave_one_group_out(rows):
        play = [r for r in fold.test if r.class3 == PLAY]
        if not play or len({r.class3 for r in fold.train}) < 2:
            # The venue_01 fold trains on clip venues alone, all ACTIVE_PLAY - a single
            # class no classifier can be fitted to. Seven folds, not eight; the dataset
            # gap, not a code limitation.
            continue
        tr = [pos[r.file] for r in fold.train]
        te = [pos[r.file] for r in play]
        probs, order = fit_fold(
            fold.train, play, {k: v[tr] for k, v in X.items()}, {k: v[te] for k, v in X.items()}
        )
        for model, pred in predictions(probs, order).items():
            per_fold.setdefault(model, {})[fold.name] = sum(
                p == PLAY for p in pred
            ) / len(play)
        print(f"  fold {fold.name[:44]:<44} n={len(play)}")
    return per_fold


def false_play(rows, X) -> dict[str, float]:
    """How often each model calls a held-out empty pitch a match.

    Trained on venue_01 camera A, scored on camera B's empty frames - the same protocol as
    `h3_with_false_play.py`, so the two tables can be read together. Clip venues are
    excluded from training on purpose: all their development frames are ACTIVE_PLAY, and
    including them drives every model to 1.000, which measures the dataset, not the model.
    """
    pos = {r.file: i for i, r in enumerate(rows)}
    cam = lambda r: PHYSICAL_CAMERA.get(r.camera, r.camera)  # noqa: E731
    venue = [r for r in rows if r.venue == "venue_01"]
    train = [r for r in venue if cam(r) == "camera_A"]
    empties = [r for r in venue if cam(r) == "camera_B" and r.class3 == EMPTY]
    if not empties or len({r.class3 for r in train}) < 2:
        return {}
    tr = [pos[r.file] for r in train]
    te = [pos[r.file] for r in empties]
    probs, order = fit_fold(
        train, empties, {k: v[tr] for k, v in X.items()}, {k: v[te] for k, v in X.items()}
    )
    out = {
        m: float(np.mean([p == PLAY for p in pred]))
        for m, pred in predictions(probs, order).items()
    }
    out["_n"] = float(len(empties))
    return out


def published_means() -> dict[str, float]:
    if not PUBLISHED.exists():
        return {}
    return {
        r["model"]: float(r["play_recall"])
        for r in csv.DictReader(PUBLISHED.open(encoding="utf-8"))
        if r["held_out_venue"] == "MEAN_ACROSS_FOLDS"
    }


def main() -> None:
    global CACHE_DIR, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-dir", type=Path, default=None,
        help="feature cache family to read; default is the published raw-frame one",
    )
    ap.add_argument(
        "--suffix", default="",
        help="appended to the output filename, so two cache families do not overwrite",
    )
    args = ap.parse_args()
    if args.cache_dir:
        CACHE_DIR = args.cache_dir
        print(f"reading caches from {CACHE_DIR}")
    if args.suffix:
        OUT = settings.results_dir / f"logit_average_baseline{args.suffix}.csv"

    rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    rows, X = common_rows_and_features(rows)
    print(f"development rows: {len(rows)}\n")

    print("=== cross-venue folds ===")
    per_fold = cross_venue(rows, X)
    folds = sorted(next(iter(per_fold.values())))
    means = {m: float(np.mean([v[f] for f in folds])) for m, v in per_fold.items()}

    # The ceiling: best single backbone per fold, which needs the venue identity a gate
    # does not have. It bounds what any router could achieve, learned or otherwise.
    oracle = float(np.mean([max(per_fold[k][f] for k in CACHES) for f in folds]))
    best_single = max(CACHES, key=lambda k: means[k])

    # Reproducing the published H3 table is only meaningful for the cache family that
    # produced it. On a different family the numbers *should* differ - that is the point of
    # running it there - so the check is skipped rather than reported as a failure, which
    # would train the reader to ignore a warning that matters on the default path.
    if CACHE_DIR == settings.feature_cache_dir:
        print("\n=== reproduction check against the published H3 table ===")
        ref = published_means()
        ok = True
        for k in CACHES:
            want = ref.get(k)
            agree = want is None or abs(means[k] - want) < 0.005
            ok &= agree
            print(f"  {k:<22} {means[k]:.4f}  published {want if want is None else f'{want:.4f}'}"
                  f"  {'ok' if agree else 'MISMATCH'}")
        if not ok:
            print("\nWARNING: a single-backbone mean does not reproduce the published CSV.")
            print("Do not read the ensemble rows until that is explained.")
    else:
        print(f"\n=== reproduction check skipped: reading {CACHE_DIR.name}, not the "
              f"published cache family ===")
        ref = published_means()
        for k in CACHES:
            want = ref.get(k)
            shown = "-" if want is None else f"{want:.4f}"
            print(f"  {k:<22} {means[k]:.4f}  (published raw-frame value {shown})")

    fp = false_play(rows, X)
    n_empty = int(fp.pop("_n", 0))

    print(f"\n=== both axes (recall over {len(folds)} folds; false-play on n={n_empty}) ===")
    print(f"{'model':<28}{'recall':>9}{'worst fold':>12}{'false-play':>12}{'vs best single':>16}")
    ordering = [*CACHES, *[k for k in means if k not in CACHES]]
    for m in ordering:
        delta = means[m] - means[best_single]
        print(f"{m:<28}{means[m]:9.4f}{min(per_fold[m].values()):12.3f}"
              f"{fp.get(m, float('nan')):12.4f}{delta:+16.4f}")
    print(f"{'ORACLE (per-fold best)':<28}{oracle:9.4f}{'-':>12}{'-':>12}"
          f"{oracle - means[best_single]:+16.4f}")

    print(f"\nbest single backbone: {best_single} at {means[best_single]:.4f}")
    print(f"oracle headroom:      {oracle - means[best_single]:+.4f}")
    captured = {
        m: (means[m] - means[best_single]) / (oracle - means[best_single])
        for m in means if m not in CACHES
    } if oracle - means[best_single] > 1e-9 else {}
    for m, share in sorted(captured.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<28} captures {share * 100:6.1f}% of it")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "held_out_venue", "play_recall", "false_play_rate",
                    "n_held_out_empty", "is_ensemble"])
        for m in ordering:
            for f in folds:
                w.writerow([m, f, f"{per_fold[m][f]:.4f}", "", "", m not in CACHES])
            w.writerow([m, "MEAN_ACROSS_FOLDS", f"{means[m]:.4f}",
                        f"{fp.get(m, float('nan')):.4f}", n_empty, m not in CACHES])
        w.writerow(["ORACLE_per_fold_best", "MEAN_ACROSS_FOLDS", f"{oracle:.4f}", "", "", "ceiling"])
    print(f"\nwrote {OUT.name}")

    with (settings.results_dir / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n- {datetime.now(timezone.utc):%Y-%m-%d} | WP5-T9 logit-average baseline | "
            f"`python experiments/logit_average_baseline.py` | `{OUT.name}` | "
            f"best single {best_single} {means[best_single]:.4f}, oracle {oracle:.4f}\n"
        )


if __name__ == "__main__":
    main()
