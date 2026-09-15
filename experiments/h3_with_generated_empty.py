"""H3 re-run with the generated EMPTY frames in training — and what its recall was measuring.

H3 asks whether ACTIVE_PLAY recall transfers to an unseen venue, and its own docstring says
why it reports recall rather than macro-F1: **every held-out venue fold is 100% ACTIVE_PLAY**,
because no clip venue contains an empty pitch. A model that answers PLAY to everything scores
1.000 on such a fold.

`a13_false_play_repair` showed that is not hypothetical. Trained on venue_01 camera A plus the
clip venues, DINOv2 calls **99.6% of unseen empty pitches a match** while holding play-recall
at 1.000. H3's folds cannot see that, by construction.

So this runs H3's protocol twice — training as H3 does, and again with the 31 generated EMPTY
frames added to the training side — and reports, beside each fold's recall, the false-play
rate of that same fitted model on **243 recorded EMPTY frames from venue_01 camera B**. Those
frames are never in any fold's training set here, and they are real.

The expected shape of the result is a recall that falls. That is not a regression: a model
that stops answering PLAY to everything must lose recall on an all-play test set. The
`balanced` column is what says whether the trade was worth making.

This does not amend H3. H3 is pre-registered and its numbers stand as reported; this is the
control it could not carry, run alongside.

    uv run python experiments/h3_with_generated_empty.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import features_for, load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
SEED = 42
PLAY = "C2_ACTIVE_PLAY"
EMPTY = "C1_EMPTY"
SYNTHETIC = "synthetic"


def main() -> None:
    every = read_manifest(DATASET / "manifest.csv")

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    # The false-play control: recorded, held out of every training set below, and the only
    # thing in this script that can tell a transferring model from one that says PLAY always.
    control = [
        r for r in every
        if r.source != SYNTHETIC and r.venue == "venue_01"
        and cam(r) == "camera_B" and r.class3 == EMPTY
    ]

    gen_empty = [
        r for r in every
        if r.source == SYNTHETIC and r.class3 == EMPTY and r.venue != "venue_01"
    ]

    rows = development_rows(every)
    rows = [r for r in rows if r not in control]
    pos = {r.file: i for i, r in enumerate(rows)}

    folds = [
        f for f in leave_one_group_out(rows, group_key="venue")
        if not f.name.endswith("venue_01")
    ]
    print(f"{len(rows)} development rows | {len(folds)} folds")
    print(f"false-play control: {len(control)} recorded EMPTY frames (venue_01 camera B)")
    print(f"generated EMPTY available to add: {len(gen_empty)}\n")

    cached = load_cache("dinov2", CACHE)
    feats = {f: v for f, v in zip(cached.files, cached.features)}
    missing = [r for r in rows + control + gen_empty if r.file not in feats]
    if missing:
        raise SystemExit(
            f"{len(missing)} frame(s) missing from the dinov2 cache. "
            f"Run experiments/a13_false_play_repair.py first, which builds them."
        )
    Xc = np.stack([feats[r.file] for r in control])

    records: list[dict] = []
    for arm in ("H3 as pre-registered", "+ generated EMPTY"):
        recalls, fps = [], []
        print(f"=== {arm} ===")
        for fold in folds:
            venue = fold.name.split("__")[-1]
            train = list(fold.train)
            if arm != "H3 as pre-registered":
                train = train + [g for g in gen_empty if g.venue != venue]
            if len({r.class3 for r in train}) < 2:
                continue
            Xtr = np.stack([feats[r.file] for r in train])
            Xte = np.stack([feats[r.file] for r in fold.test])

            probe = LinearProbe("dinov2", seed=SEED).fit(Xtr, train)
            pred = probe.predict(Xte, fold.test)
            truth = [r.class3 for r in fold.test]
            play = [(p, t) for p, t in zip(pred, truth, strict=True) if t == PLAY]
            recall = sum(p == PLAY for p, _ in play) / len(play) if play else float("nan")

            fp = sum(p == PLAY for p in probe.predict(Xc, control)) / len(control)
            recalls.append(recall)
            fps.append(fp)
            print(f"  {venue:<28} recall {recall:.3f}   false-play {fp:.3f}   "
                  f"balanced {recall - fp:+.3f}")
            records.append({
                "arm": arm, "held_out_venue": venue, "n_train": len(train),
                "play_recall": round(recall, 4), "false_play": round(fp, 4),
                "balanced": round(recall - fp, 4),
            })
        mr, mf = float(np.mean(recalls)), float(np.mean(fps))
        print(f"  --> mean recall {mr:.3f}   mean false-play {mf:.3f}   "
              f"balanced {mr - mf:+.3f}\n")
        records.append({
            "arm": arm, "held_out_venue": "MEAN_ACROSS_FOLDS", "n_train": "",
            "play_recall": round(mr, 4), "false_play": round(mf, 4),
            "balanced": round(mr - mf, 4),
        })

    out = RESULTS / "h3_with_generated_empty.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
