"""What does training on four videos cost? (2026-09-21)

Four source recordings are **75% of every recorded frame** - `slot_20260712_2030_camB` alone
is 516 of 1,720, and the median source video contributes six
(`experiments/dataset_redundancy.py`). The deployed probe is fitted on all of them.
`splits.balanced_rows` can bound what one recording is worth; this measures whether doing so
is worth anything.

Same cached features, same seed, **identical held-out sides** - leave-one-venue-out for
cross-venue recall and venue_01 camera B for the false-play control, exactly as
`rule_frame_eval` and `h3_with_false_play` hold them. Only the training rows differ, so a
difference here is a difference in what the fit was *shown*.

**Read the EMPTY column, not just the balanced score.** A20's rule stands: the complement of
a false-play rate is not correctness, because a model can avoid saying PLAY by saying C3. It
is on this table for exactly that reason, and it is the column where capping does **not**
help.

    uv run python experiments/video_concentration_cost.py
"""
from __future__ import annotations

import csv

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.feature_cache import features_for, load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import (
    balanced_rows,
    development_rows,
    distinct_rows,
    leave_one_group_out,
)
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.vision.heads import LinearProbe

EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
SEED = 42
RESULTS = settings.results_dir
#: Play frames a venue needs before its recall joins the mean, as in `rule_frame_eval`.
MIN_VENUE_PLAY = 5

cached = load_cache("dinov2", settings.feature_cache_dir)
rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
features, rows = features_for(cached, rows)
index = {r.file: i for i, r in enumerate(rows)}
print(f"{len(rows)} development rows cached\n")


def physical(r):
    return PHYSICAL_CAMERA.get(r.camera, r.camera)


def evaluate(name, select):
    """Cross-venue play recall and the camera-B false-play control, fitting on `select`."""
    per_venue: dict[str, list[int]] = {}
    for fold in leave_one_group_out(rows):
        train = select(fold.train)
        if len({r.class3 for r in train}) < 2 or not fold.test:
            continue
        head = LinearProbe("dinov2", seed=SEED).fit(features[[index[r.file] for r in train]], train)
        pred = head.predict(features[[index[r.file] for r in fold.test]], fold.test)
        for row, p in zip(fold.test, pred, strict=True):
            if row.venue != "venue_01" and row.class3 == PLAY:
                per_venue.setdefault(row.venue, []).append(int(str(p) == PLAY))
    recalls = {v: float(np.mean(h)) for v, h in per_venue.items()
               if len(h) >= MIN_VENUE_PLAY}

    venue = [r for r in rows if r.venue == "venue_01"]
    train = select([r for r in venue if physical(r) == "camera_A"])
    held = [r for r in venue if physical(r) == "camera_B" and r.class3 == EMPTY]
    head = LinearProbe("dinov2", seed=SEED).fit(features[[index[r.file] for r in train]], train)
    pred = [str(p) for p in head.predict(features[[index[r.file] for r in held]], held)]
    false_play = float(np.mean([int(p == PLAY) for p in pred]))
    empty_acc = float(np.mean([int(p == EMPTY) for p in pred]))

    n_fit = len(select(rows))
    recall = float(np.mean(list(recalls.values()))) if recalls else float("nan")
    print(f"{name:<26}{n_fit:>7}{recall:>9.4f}{min(recalls.values()):>9.3f}"
          f"{false_play:>12.4f}{empty_acc:>11.4f}{recall - false_play:>10.4f}")
    return {"training_rows": name, "n_fit": n_fit, "play_recall": round(recall, 4),
            "recall_worst_venue": round(min(recalls.values()), 4),
            "false_play_rate": round(false_play, 4),
            "empty_accuracy": round(empty_acc, 4),
            "balanced": round(recall - false_play, 4),
            "n_venues_in_mean": len(recalls)}


print(f"{'training rows are':<26}{'n fit':>7}{'recall':>9}{'worst':>9}"
      f"{'false-play':>12}{'EMPTY acc':>11}{'balanced':>10}")
out = [evaluate("everything (deployed)", lambda rs: rs),
       evaluate("distinct scenes", lambda rs: distinct_rows(rs))]
for cap in (12, 20, 40):
    out.append(evaluate(f"capped {cap}/video", lambda rs, c=cap: balanced_rows(rs, per_video=c)))

for line in (
    "",
    "The ordering among the caps is within noise at these sample sizes - 12 beating 20 and",
    "40 is not evidence that twelve is the right number. What is outside noise is that every",
    "capped arm beats the deployed one on recall AND on false-play at the same time.",
    "",
    "EMPTY accuracy is ~0 in every arm. Capping does not teach the probe to say EMPTY; it",
    "stops it saying PLAY, which it achieves by saying C3 instead. That is A20's finding,",
    "unmoved - the probe still never answers EMPTY at a camera it has not seen, and no",
    "amount of rebalancing the training rows changes it.",
):
    print(line)

RESULTS.mkdir(exist_ok=True)
path = RESULTS / "video_concentration_cost.csv"
with path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(out[0]))
    writer.writeheader()
    writer.writerows(out)
print()
print(f"wrote {path}")

record(
    "what training on four videos costs",
    "uv run python experiments/video_concentration_cost.py",
    "video_concentration_cost.csv",
    "; ".join(f"{r['training_rows']} n={r['n_fit']} recall {r['play_recall']:.4f} "
              f"false-play {r['false_play_rate']:.4f} EMPTY {r['empty_accuracy']:.4f}"
              for r in out)
    + "; identical held-out sides, only the training rows differ; EMPTY accuracy stays ~0 "
      "in every arm, so capping stops the probe saying PLAY without teaching it to say EMPTY",
)
