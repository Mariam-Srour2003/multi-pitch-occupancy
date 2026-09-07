"""H3's cross-venue result, with the false-play control it never had (RQ3, RQ7).

H3 is the headline generalisation claim: across unseen venues the frozen backbones hold
above 0.86 ACTIVE_PLAY recall while a clock rule collapses to 0.219. It is reported on
**recall alone**, and every cross-venue test fold is 100% ACTIVE_PLAY.

That combination is exactly what the preprocessing search got wrong. On a single-class test
set recall rises by answering "playing" more often, so a model that has learned nothing but
the majority answer scores perfectly. The search's own control was broken and read 0.0000
for 52 evaluations; when repaired, configurations tied at 1.0000 recall separated into
false-play rates of 0.021, 0.663 and 0.979. H3 has no such column at all.

So this re-reports H3 with both numbers. It runs on the existing feature caches, so it costs
seconds rather than an embedding pass.

**The recall column is reproduced first and checked against the published CSV.** A harness
that cannot reproduce the number it is extending has no business adding a second one.

    uv run python experiments/h3_with_false_play.py
"""

from __future__ import annotations

import csv

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import ClockRule, LinearProbe

PLAY, EMPTY = "C2_ACTIVE_PLAY", "C1_EMPTY"
SEED = 42
OUT = settings.results_dir / "h3_with_false_play.csv"
PUBLISHED = settings.results_dir / "h3_cross_venue_recall.csv"

#: model key -> cached feature file. The clock rule reads no pixels, so it gets a dummy.
CACHES = {
    "convnextv2": "convnextv2.npz",
    "dinov2": "dinov2.npz",
    "vit": "vit.npz",
}


def load(name: str, rows):
    data = np.load(settings.feature_cache_dir / name, allow_pickle=True)
    index = {str(f): i for i, f in enumerate(data["files"])}
    kept = [r for r in rows if r.file in index]
    return kept, data["features"][[index[r.file] for r in kept]]


def make_head(model: str):
    return ClockRule() if model == "clock_rule" else LinearProbe(model, seed=SEED)


def cross_venue_recall(model, rows, X) -> list[tuple[str, float, int]]:
    pos = {r.file: i for i, r in enumerate(rows)}
    out = []
    for fold in leave_one_group_out(rows):
        play = [r for r in fold.test if r.class3 == PLAY]
        if not play:
            continue
        if len({r.class3 for r in fold.train}) < 2:
            # The venue_01 fold trains on the clip venues alone, and all 282 of their
            # development frames are ACTIVE_PLAY - a single class, which no classifier can
            # be fitted to. This is why H3 reports seven folds and not eight, and it is the
            # dataset gap rather than a code limitation.
            continue
        head = make_head(model).fit(X[[pos[r.file] for r in fold.train]], fold.train)
        pred = head.predict(X[[pos[r.file] for r in play]], play)
        out.append((fold.name, sum(p == PLAY for p in pred) / len(play), len(play)))
    return out


def false_play(model, rows, X) -> tuple[float, int]:
    """How often the model calls a *held-out* empty pitch a match.

    Trained on venue_01 camera A and scored on camera B's empty frames. The clip venues are
    excluded from this training set on purpose: all 282 of their development frames are
    ACTIVE_PLAY and none are EMPTY, and including them drives the rate to 1.000 for every
    model, which measures the dataset gap rather than the model.
    """
    pos = {r.file: i for i, r in enumerate(rows)}
    cam = lambda r: PHYSICAL_CAMERA.get(r.camera, r.camera)  # noqa: E731
    venue = [r for r in rows if r.venue == "venue_01"]
    train = [r for r in venue if cam(r) == "camera_A"]
    empties = [r for r in venue if cam(r) == "camera_B" and r.class3 == EMPTY]
    if not empties or len({r.class3 for r in train}) < 2:
        return float("nan"), 0
    head = make_head(model).fit(X[[pos[r.file] for r in train]], train)
    pred = head.predict(X[[pos[r.file] for r in empties]], empties)
    return float(np.mean([p == PLAY for p in pred])), len(empties)


def published_means() -> dict[str, float]:
    if not PUBLISHED.exists():
        return {}
    out = {}
    for row in csv.DictReader(PUBLISHED.open(encoding="utf-8")):
        if row["held_out_venue"] == "MEAN_ACROSS_FOLDS":
            out[row["model"]] = float(row["play_recall"])
    return out


def main() -> None:
    all_rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    reference = published_means()
    results = []

    for model in ["clock_rule", *CACHES]:
        if model == "clock_rule":
            rows = all_rows
            X = np.zeros((len(rows), 1), dtype=np.float32)  # reads lighting, not pixels
        else:
            path = settings.feature_cache_dir / CACHES[model]
            if not path.exists():
                print(f"  {model}: no cache at {path.name}, skipped")
                continue
            rows, X = load(CACHES[model], all_rows)

        folds = cross_venue_recall(model, rows, X)
        mean = float(np.mean([r for _, r, _ in folds]))
        worst = min(r for _, r, _ in folds)
        fp, n_empty = false_play(model, rows, X)
        results.append((model, mean, worst, fp, n_empty, folds))

    print(f"{'model':13} {'recall':>8} {'worst':>7} {'false-play':>11} "
          f"{'published':>10} {'match':>7}")
    ok = True
    for model, mean, worst, fp, n_empty, _ in results:
        want = reference.get(model)
        agree = want is None or abs(mean - want) < 0.005
        ok &= agree
        shown = f"{want:.3f}" if want is not None else "-"
        print(f"{model:13} {mean:8.4f} {worst:7.3f} {fp:11.4f} {shown:>10} "
              f"{'yes' if agree else 'NO':>7}")

    if not ok:
        print("\nWARNING: a recall column does not reproduce the published CSV.")
        print("Do not read the false-play column until that is explained.")
    else:
        print(f"\nRecall reproduces the published H3 table, so the false-play column is")
        print(f"measured on the same footing. Held-out empty frames: {results[0][4]}.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "held_out_venue", "n_play", "play_recall",
                    "false_play_rate", "n_held_out_empty"])
        for model, mean, worst, fp, n_empty, folds in results:
            for name, recall, n in folds:
                w.writerow([model, name, n, f"{recall:.4f}", "", ""])
            w.writerow([model, "MEAN_ACROSS_FOLDS", "", f"{mean:.4f}",
                        f"{fp:.4f}", n_empty])
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
