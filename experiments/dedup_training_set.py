"""Does training on distinct scenes beat training on every frame? (WP2)

The corpus is 1,692 recorded frames and **179 distinct scenes** - 11%. The EMPTY class is the
extreme: **494 frames carrying 5 distinct scenes**, one venue, two cameras, two days. A probe
fitted on that sees five empty pitches, each about a hundred times.

Two questions, and they are different:

1. **Does pruning to distinct scenes cost anything?** If 494 frames and 50 frames fit the same
   model, the other 444 were never evidence, and every count in this thesis that reads as
   dataset size is really a sampling rate.
2. **Does pruning help?** Repeating five scenes a hundred times each is not neutral: it tells
   the fit that those five backgrounds are what EMPTY looks like, with a confidence a hundred
   frames would justify and five do not.

Evaluated on two test sets, because the corpus one cannot see the failure that matters:

- **venue_01 camera B**, recorded, both classes present - where the model already works
- **an unseen clip**, hand-labelled, where it does not

    uv run python experiments/dedup_training_set.py --video PATH --truth 9,12,15
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pitch_occupancy.data.dedup import distinct_subset
from pitch_occupancy.data.feature_cache import load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
SEED = 42


def dhash(path: Path, size: int = 8) -> int:
    from PIL import Image

    im = np.asarray(Image.open(path).convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (im[:, 1:] > im[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--truth", default="")
    ap.add_argument("--threshold", type=int, default=5, help="dHash distance for 'the same'")
    args = ap.parse_args()

    rows = read_manifest(DATASET / "manifest.csv")
    cached = load_cache(args.backbone, CACHE)
    feats = {f: v for f, v in zip(cached.files, cached.features)}

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    keep = [r for r in rows if r.file in feats and r.class3 in (EMPTY, PLAY)]
    real = [r for r in keep if r.source != "synthetic"]
    gen_empty = [r for r in keep if r.source == "synthetic"
                 and r.class3 == EMPTY and r.venue != "venue_01"]
    v01 = [r for r in real if r.venue == "venue_01"]
    test = [r for r in v01 if cam(r) == "camera_B"]
    full = ([r for r in v01 if cam(r) == "camera_A"]
            + [r for r in real if r.venue != "venue_01"] + gen_empty)

    # Distinct within (venue, class), not globally: two venues that happen to look alike are
    # still two venues, and collapsing across them would throw away the only cross-venue
    # variation the corpus has.
    print("hashing ...")
    h = {r.file: dhash(DATASET / r.file) for r in full}
    groups: dict[tuple, list] = {}
    for r in full:
        groups.setdefault((r.venue, r.class3), []).append(r)
    pruned = []
    for key, members in groups.items():
        kept = distinct_subset({r.file: h[r.file] for r in members}, threshold=args.threshold)
        pruned.extend([r for r in members if r.file in kept])

    print(f"\nfull    {len(full)} frames")
    print(f"pruned  {len(pruned)} frames  ({len(pruned) / len(full):.0%})")
    for label, s in (("full", full), ("pruned", pruned)):
        n_e = sum(1 for r in s if r.class3 == EMPTY)
        print(f"  {label:<8} EMPTY {n_e:>5}  PLAY {len(s) - n_e:>5}")

    Xte = np.stack([feats[r.file] for r in test])
    yte = [r.class3 for r in test]
    n_e = sum(1 for t in yte if t == EMPTY)

    from sklearn.metrics import f1_score

    fitted = {}
    print(f"\n{'training set':<12}{'macro-F1':>10}{'recall':>9}{'false-play':>12}{'balanced':>11}")
    for label, s in (("full", full), ("pruned", pruned)):
        probe = LinearProbe(args.backbone, seed=SEED).fit(
            np.stack([feats[r.file] for r in s]), s)
        fitted[label] = probe
        pred = probe.predict(Xte, test)
        f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
        rec = sum(p == PLAY for p, t in zip(pred, yte) if t == PLAY) / (len(yte) - n_e)
        fp = sum(p == PLAY for p, t in zip(pred, yte) if t == EMPTY) / n_e
        print(f"{label:<12}{f1:>10.4f}{rec:>9.4f}{fp:>12.4f}{rec - fp:>11.4f}")

    if args.video:
        import sys

        import cv2

        sys.path.insert(0, str(ROOT / "scripts"))
        from derive_roi import polygon_from_mask, turf_mask

        from pitch_occupancy.vision.backbones import load_backbone
        from pitch_occupancy.vision.motion import MotionGate

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
        from PIL import Image

        from pitch_occupancy.vision.backbones import embed_batch

        pil = [Image.fromarray(f[:, :, ::-1]) for f in frames]
        X = embed_batch(model, processor, spec, pil, roi_polygon=poly)
        gate = MotionGate()
        print(f"\non the unseen clip ({len(frames)} minutes, boundary + motion gate):")
        print(f"{'training set':<12}{'false-play on empty minutes':>30}")
        for label, probe in fitted.items():
            proba = probe.predict_proba(X)
            classes = list(probe.classes_)
            raw = [classes[int(i)] for i in proba.argmax(1)]
            out = []
            for k, cls in enumerate(raw):
                if k == 0:
                    out.append(cls)
                    continue
                from pitch_occupancy.vision.motion import motion_cue

                cue = motion_cue(frames[k - 1], frames[k], poly)
                out.append(EMPTY if (cls == PLAY and cue < gate.threshold) else cls)
            empt = [k for k in range(len(out)) if k not in person]
            fp = [k for k in empt if out[k] == PLAY]
            # A probe that answers EMPTY to everything scores a perfect false-play and is
            # useless. The play-side count is what separates a fix from a collapsed prior.
            n_play_said = sum(1 for o in out if o == PLAY)
            said_on_person = sum(1 for k in person if k < len(out) and out[k] == PLAY)
            print(f"{label:<12} false-play {len(fp)}/{len(empt)} = "
                  f"{len(fp)/len(empt):.2f}  | says PLAY {n_play_said}/{len(out)} times, "
                  f"{said_on_person}/{len(person)} of them on the person minutes")

    RESULTS.mkdir(exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
