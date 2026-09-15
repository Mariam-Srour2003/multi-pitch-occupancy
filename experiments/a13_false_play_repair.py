"""Do the generated EMPTY frames repair the false-play collapse? (A13 condition 3)

`EXPERIMENT_LOG.md` records the most serious consequence of the dataset gap: adding the clip
venues to training takes false-play on held-out EMPTY frames from **0.231 to 1.000**. Every
clip-venue development frame is ACTIVE_PLAY and none is EMPTY, so those frames widen the PLAY
region of feature space until it swallows an unseen camera's empty pitches. A probe trained
that way calls *every* empty pitch a match - and because H3's folds are all-play, it still
scores beautifully.

That is the defect the generated EMPTY frames were made for: 31 of them, across six clip
venues that had none. This measures whether they fix it.

Four training sets, one held-out test set of real EMPTY frames from venue_01's camera B:

1. `camera_A` - the baseline, 0.231 in the log
2. `camera_A + clip` - the collapse, 1.000 in the log
3. `camera_A + clip + generated EMPTY` - the repair, if there is one
4. `camera_A + clip + all generated` - the control that separates "the empty frames fixed
   it" from "any extra data fixed it"

**The test set is entirely real.** Generated frames appear only on training sides, which is
A13 condition 1, and `splits.check_split` enforces it elsewhere. A repair measured against
generated empties would be the model agreeing with its own generator.

    uv run python experiments/a13_false_play_repair.py --backbone dinov2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pitch_occupancy.config import RESULTS_DIR
from pitch_occupancy.data.feature_cache import build_cache, load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe

ROOT = Path(__file__).resolve().parents[1]
EMPTY = "C1_EMPTY"
PLAY = "C2_ACTIVE_PLAY"
SYNTHETIC = "synthetic"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifest.csv")
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "a13_false_play_repair.json")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)

    cached = load_cache(args.backbone, args.cache_dir)
    feats = {f: v for f, v in zip(cached.files, cached.features)}
    missing = [r for r in rows if r.file not in feats]
    if missing:
        print(f"embedding {len(missing)} frame(s) absent from the cache ...")
        extra_dir = args.cache_dir / "a13_gate"
        extra_dir.mkdir(parents=True, exist_ok=True)
        extra = build_cache(missing, args.backbone, args.dataset_dir, extra_dir)
        feats.update(dict(zip(extra.files, extra.features)))

    rows = [r for r in rows if r.file in feats]
    X = {r.file: feats[r.file] for r in rows}

    def cam(r) -> str:
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    real = [r for r in rows if r.source != SYNTHETIC]
    gen = [r for r in rows if r.source == SYNTHETIC]

    v01 = [r for r in real if r.venue == "venue_01"]
    cam_a = [r for r in v01 if cam(r) == "camera_A"]
    test_empty = [r for r in v01 if cam(r) == "camera_B" and r.class3 == EMPTY]

    clip = [r for r in real if r.venue != "venue_01"]
    gen_clip = [r for r in gen if r.venue != "venue_01"]
    gen_clip_empty = [r for r in gen_clip if r.class3 == EMPTY]

    print(f"\nheld-out REAL empty frames (venue_01 camera B): {len(test_empty)}")
    print(f"camera_A train pool: {len(cam_a)}   real clip frames: {len(clip)}")
    print(f"generated at clip venues: {len(gen_clip)}  of which EMPTY: {len(gen_clip_empty)}")
    if not test_empty:
        print("no held-out empty frames - nothing to measure")
        return 1

    arms = {
        "camera_A": cam_a,
        "camera_A + clip": cam_a + clip,
        "camera_A + clip + generated EMPTY": cam_a + clip + gen_clip_empty,
        "camera_A + clip + all generated": cam_a + clip + gen_clip,
    }

    # The other half of the measurement, and it is not optional. False-play alone is a
    # single-class test, and a probe that answers EMPTY to everything scores 0.0000 on it
    # while being useless. That is H3's flaw with the classes swapped, and reporting a
    # repair without this column would repeat the error this project exists to document.
    test_play = [r for r in v01 if cam(r) == "camera_B" and r.class3 == PLAY]
    print(f"held-out REAL play frames  (venue_01 camera B): {len(test_play)}")

    Xte = np.stack([X[r.file] for r in test_empty])
    Xtp = np.stack([X[r.file] for r in test_play]) if test_play else None
    out: dict[str, dict] = {}
    print()
    for name, train in arms.items():
        classes = {r.class3 for r in train}
        if len(classes) < 2:
            print(f"  {name:38s} skipped - one class only")
            continue
        Xtr = np.stack([X[r.file] for r in train])
        probe = LinearProbe(balanced=True).fit(Xtr, train)
        pred = probe.predict(Xte, test_empty)
        rate = sum(p == PLAY for p in pred) / len(test_empty)
        play_frac = sum(1 for r in train if r.class3 == PLAY) / len(train)
        recall = float("nan")
        if Xtp is not None:
            pp = probe.predict(Xtp, test_play)
            recall = sum(p == PLAY for p in pp) / len(test_play)
        balanced = recall - rate
        out[name] = {"n_train": len(train), "play_fraction": play_frac,
                     "false_play": rate, "play_recall": recall, "balanced": balanced}
        print(f"  {name:38s} n={len(train):5d}  false-play={rate:.4f}  "
              f"play-recall={recall:.4f}  balanced={balanced:+.4f}")

    base = out.get("camera_A + clip", {}).get("false_play")
    rep = out.get("camera_A + clip + generated EMPTY", {}).get("false_play")
    print()
    if base is not None and rep is not None:
        delta = base - rep
        print(f"collapse {base:.4f} -> with generated EMPTY {rep:.4f}   (change {-delta:+.4f})")
        rep_bal = out["camera_A + clip + generated EMPTY"].get("balanced", float("nan"))
        base_bal = out["camera_A + clip"].get("balanced", float("nan"))
        print(f"balanced (recall - false-play)  {base_bal:+.4f} -> {rep_bal:+.4f}")
        if delta > 0.05 and rep_bal > base_bal:
            print(
                "\nThe generated EMPTY frames reduce false-play at an unseen camera. They were\n"
                "made for exactly this defect, and on this measure they address it. Note what\n"
                "it is not: the test set is one camera at one venue, so this is evidence the\n"
                "frames help the known failure, not that the model generalises."
            )
        else:
            print(
                "\nThe generated EMPTY frames do not repair the collapse. That is a result, and\n"
                "under A13 it is reported rather than retried until it comes out differently."
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "backbone": args.backbone,
                "n_test_empty_real": len(test_empty),
                "test_set": "venue_01 camera B, EMPTY, recorded only",
                "arms": out,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
