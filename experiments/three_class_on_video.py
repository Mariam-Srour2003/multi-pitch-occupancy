"""Can the model say "people, not playing"? (C3, WP2-T11)

The corpus holds **six recorded C3 frames**, three distinct scenes, one venue, one moment. On
that the class cannot be trained and `rq_matrix.md` records the recommendation as "report
two-class and state the scope reduction". The deployed probe is three-class in name and
two-class in behaviour, which is why a lone walker on an empty pitch comes back ACTIVE_PLAY.

158 generated C3 frames now exist, 88 distinct scenes across seven venues. This asks the only
question that matters about them: **does a three-class probe fitted with them say C3 on a real
person walking across a real pitch it has never seen?**

The test is the unseen clip, hand-labelled: 16 minutes, three with one person on the pitch and
no ball. There is no recorded C3 test set anywhere - six frames at one venue is not one - so
this is the only honest test available, and it is three positives.

Three arms:

- **2-class**: EMPTY and ACTIVE_PLAY only, as deployed
- **3-class, recorded C3 only**: the six frames
- **3-class, + generated C3**: what the generation was for

    uv run python experiments/three_class_on_video.py --video CLIP --truth 9,12,15
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import distinct_rows
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
EMPTY, PLAY, C3 = "C1_EMPTY", "C2_ACTIVE_PLAY", "C3_MAINTENANCE_NON_SPORTING"
SEED = 42


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--truth", default="", help="minutes with a person on the pitch")
    args = ap.parse_args()

    every = read_manifest(DATASET / "manifest.csv")
    cached = load_cache(args.backbone, CACHE)
    feats = {f: v for f, v in zip(cached.files, cached.features)}

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    keep = [r for r in every if r.file in feats]
    real = [r for r in keep if r.source != "synthetic"]
    syn = [r for r in keep if r.source == "synthetic"]
    v01 = [r for r in real if r.venue == "venue_01"]

    two = ([r for r in v01 if cam(r) == "camera_A" and r.class3 in (EMPTY, PLAY)]
           + [r for r in real if r.venue != "venue_01" and r.class3 in (EMPTY, PLAY)]
           + [r for r in syn if r.class3 == EMPTY and r.venue != "venue_01"])
    c3_real = [r for r in real if r.class3 == C3]
    c3_gen = [r for r in syn if r.class3 == C3]

    arms = {
        "2-class": two,
        "3-class, recorded C3": two + c3_real,
        "3-class, + generated C3": two + c3_real + c3_gen,
    }
    print(f"recorded C3 frames: {len(c3_real)}   generated C3 frames: {len(c3_gen)}")

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
    cues = [None] + [motion_cue(frames[k - 1], frames[k], poly) for k in range(1, len(frames))]

    print(f"\nunseen clip: {len(frames)} minutes, {len(person)} with a person on the pitch "
          f"({sorted(person)})\n")
    records = []
    for name, train_rows in arms.items():
        train = distinct_rows(train_rows)
        probe = LinearProbe(args.backbone, seed=SEED).fit(
            np.stack([feats[r.file] for r in train]), train)
        classes = list(probe.classes_)
        raw = [classes[int(k)] for k in probe.predict_proba(X).argmax(1)]
        # The gate only ever overrules PLAY; a C3 verdict is left alone, since "people are
        # here and not playing" is not a claim about movement.
        out = [raw[0]] + [EMPTY if (raw[k] == PLAY and cues[k] < gate.threshold) else raw[k]
                          for k in range(1, len(raw))]

        empt = [k for k in range(len(out)) if k not in person]
        fp = sum(1 for k in empt if out[k] == PLAY)
        on_person = [out[k] for k in sorted(person) if k < len(out)]
        got_c3 = sum(1 for v in on_person if v == C3)
        # C3 on an empty minute is a different error and has to be counted separately: it is
        # a person reported where there is none, not a match reported where there is none.
        c3_on_empty = sum(1 for k in empt if out[k] == C3)
        print(f"{name:<26} n_train {len(train):>4}   "
              f"false-play {fp}/{len(empt)}   "
              f"C3 on the person minutes {got_c3}/{len(person)}   "
              f"C3 on empty minutes {c3_on_empty}/{len(empt)}")
        print(f"{'':<26} person minutes -> {on_person}")
        records.append({"arm": name, "n_train": len(train), "false_play": fp,
                        "n_empty": len(empt), "c3_on_person": got_c3,
                        "n_person": len(person), "c3_on_empty": c3_on_empty,
                        "person_verdicts": "|".join(on_person)})

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "three_class_on_video.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {p}")
    print("\nThree positives. A class that works on three frames at one venue has not been "
          "shown to work; a class that fails on them has been shown to fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
