"""Every headline number, refitted on one frame per scene (A15).

`distinct_rows` prunes the training side to 290 frames from 1,881. This re-runs the results
that depend on what the probe was fitted on, full against pruned, so the change is reported as
a pair rather than a replacement:

- **H3 cross-venue play-recall**, with the false-play control beside it, because H3's folds
  are 100% ACTIVE_PLAY and recall alone cannot distinguish transfer from answering PLAY always
- **venue_01 camera B**, the only recorded split with both classes present
- **the unseen clip**, hand-labelled, where the model actually fails

Test sides are never pruned. Deduplicating a test set changes what its number means.

    uv run python experiments/rerun_on_distinct_scenes.py --video CLIP --truth 9,12,15
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, distinct_rows, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
SEED = 42


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--truth", default="")
    ap.add_argument("--roi-pooled", action="store_true",
                    help="fit on the ROI-pooled cache, matching what serving computes")
    args = ap.parse_args()

    every = read_manifest(DATASET / "manifest.csv")
    print(f"training features: {'ROI-pooled' if args.roi_pooled else 'whole frame'}"
          f"   (serving always pools inside the boundary)")
    cached = load_cache(args.backbone, CACHE, roi_pooled=args.roi_pooled)
    feats = {f: v for f, v in zip(cached.files, cached.features)}

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    def fit(rows_):
        return LinearProbe(args.backbone, seed=SEED).fit(
            np.stack([feats[r.file] for r in rows_]), rows_)

    def score(probe, test):
        pred = probe.predict(np.stack([feats[r.file] for r in test]), test)
        truth = [r.class3 for r in test]
        n_e = sum(1 for t in truth if t == EMPTY)
        n_p = len(truth) - n_e
        rec = sum(p == PLAY for p, t in zip(pred, truth) if t == PLAY) / n_p if n_p else float("nan")
        fp = sum(p == PLAY for p, t in zip(pred, truth) if t == EMPTY) / n_e if n_e else float("nan")
        return rec, fp

    records = []

    # --- 1. H3 cross-venue, with the false-play control -------------------------------
    dev = [r for r in development_rows(every) if r.file in feats]
    control = [r for r in every
               if r.source != "synthetic" and r.venue == "venue_01"
               and cam(r) == "camera_B" and r.class3 == EMPTY and r.file in feats]
    cf = {r.file for r in control}
    dev = [r for r in dev if r.file not in cf]
    folds = [f for f in leave_one_group_out(dev, group_key="venue")
             if not f.name.endswith("venue_01")]

    print(f"H3: {len(folds)} folds | false-play control {len(control)} recorded EMPTY frames")
    print(f"\n{'arm':<10}{'mean play-recall':>18}{'mean false-play':>18}{'balanced':>11}")
    for arm in ("full", "pruned"):
        recs, fps = [], []
        for fold in folds:
            train = list(fold.train)
            if arm == "pruned":
                train = distinct_rows(train)
            if len({r.class3 for r in train}) < 2:
                continue
            probe = fit(train)
            r1, _ = score(probe, list(fold.test))
            _, f1 = score(probe, control)
            recs.append(r1)
            fps.append(f1)
        mr, mf = float(np.mean(recs)), float(np.mean(fps))
        print(f"{arm:<10}{mr:>18.4f}{mf:>18.4f}{mr - mf:>11.4f}")
        records.append({"test": "H3 cross-venue (7 folds)", "arm": arm,
                        "play_recall": round(mr, 4), "false_play": round(mf, 4),
                        "balanced": round(mr - mf, 4)})

    # --- 2. venue_01 camera B, both classes present -----------------------------------
    keep = [r for r in every if r.file in feats and r.class3 in (EMPTY, PLAY)]
    real = [r for r in keep if r.source != "synthetic"]
    gen_e = [r for r in keep if r.source == "synthetic" and r.class3 == EMPTY
             and r.venue != "venue_01"]
    v01 = [r for r in real if r.venue == "venue_01"]
    test_b = [r for r in v01 if cam(r) == "camera_B"]
    base = ([r for r in v01 if cam(r) == "camera_A"]
            + [r for r in real if r.venue != "venue_01"] + gen_e)

    from sklearn.metrics import f1_score

    print(f"\nvenue_01 camera B: {len(test_b)} recorded frames, "
          f"{sum(1 for r in test_b if r.class3 == EMPTY)} EMPTY")
    print(f"\n{'arm':<10}{'n_train':>9}{'macro-F1':>11}{'recall':>9}{'false-play':>12}")
    fitted = {}
    for arm in ("full", "pruned"):
        train = distinct_rows(base) if arm == "pruned" else base
        probe = fit(train)
        fitted[arm] = probe
        pred = probe.predict(np.stack([feats[r.file] for r in test_b]), test_b)
        truth = [r.class3 for r in test_b]
        f1 = float(f1_score(truth, pred, average="macro", zero_division=0))
        rec, fp = score(probe, test_b)
        print(f"{arm:<10}{len(train):>9}{f1:>11.4f}{rec:>9.4f}{fp:>12.4f}")
        records.append({"test": "venue_01 camera B", "arm": arm, "n_train": len(train),
                        "macro_f1": round(f1, 4), "play_recall": round(rec, 4),
                        "false_play": round(fp, 4)})

    # --- 3. the unseen clip ------------------------------------------------------------
    if args.video:
        import sys

        import cv2
        from PIL import Image

        sys.path.insert(0, str(ROOT / "scripts"))
        from derive_roi import polygon_from_mask, turf_mask

        from pitch_occupancy.vision.backbones import embed_batch, load_backbone
        from pitch_occupancy.vision.motion import MotionGate, motion_cue

        cap = cv2.VideoCapture(str(args.video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(15 * fps)))
        frames, pool, i = [], [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % 40 == 0 and len(pool) < 40:
                pool.append(cv2.resize(fr, (320, 180)))
            if i % step == 0:
                frames.append(fr)
            i += 1
        cap.release()
        poly = polygon_from_mask(turf_mask(np.median(np.stack(pool), axis=0).astype(np.uint8)))
        person = {int(x) for x in args.truth.split(",") if x.strip().isdigit()}
        model, processor, spec = load_backbone(args.backbone)
        X = embed_batch(model, processor, spec,
                        [Image.fromarray(f[:, :, ::-1]) for f in frames], roi_polygon=poly)
        gate = MotionGate()
        print(f"\nunseen clip: {len(frames)} minutes, boundary + motion gate, "
              f"{len(frames) - len(person)} empty")
        print(f"\n{'arm':<10}{'false-play':>12}{'says PLAY':>12}{'of those, on a person':>24}")
        for arm, probe in fitted.items():
            proba = probe.predict_proba(X)
            classes = list(probe.classes_)
            raw = [classes[int(k)] for k in proba.argmax(1)]
            out = [raw[0]]
            for k in range(1, len(raw)):
                cue = motion_cue(frames[k - 1], frames[k], poly)
                out.append(EMPTY if (raw[k] == PLAY and cue < gate.threshold) else raw[k])
            empt = [k for k in range(len(out)) if k not in person]
            fp = [k for k in empt if out[k] == PLAY]
            said = sum(1 for o in out if o == PLAY)
            on_person = sum(1 for k in person if k < len(out) and out[k] == PLAY)
            print(f"{arm:<10}{f'{len(fp)}/{len(empt)} = {len(fp)/len(empt):.2f}':>12}"
                  f"{said:>12}{on_person:>24}")
            records.append({"test": "unseen clip", "arm": arm,
                            "false_play": round(len(fp) / len(empt), 4),
                            "says_play": said, "of_those_on_person": on_person})

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "distinct_scenes_rerun.csv"
    cols = sorted({k for r in records for k in r})
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
