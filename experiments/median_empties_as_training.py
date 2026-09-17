"""Do manufactured empty-night frames repair cross-venue false-play? (A30)

A13 found that **generated** EMPTY frames took DINOv2's false-play from 0.9959 to 0.0000 on the
venue_01 control, and that generated C3 frames did nothing - because the first were subtraction
(people removed from a real frame) and the second addition. The 31 generated EMPTY frames have
been in every training set since.

A29 produced 26 EMPTY frames by the same *kind* of operation with none of the model: a
whole-clip temporal median, real pixels, real exposure, at three venues that are not venue_01
and in the lighting the corpus has almost no empty frames of. If subtraction is what worked,
these should work at least as well, and they answer a question the generated frames cannot -
whether the repair survives when the empty frames come from **somewhere other than venue_01**.

**The leak this has to avoid.** A median frame is built from a clip's own pixels, so adding
venue X's medians to a fit that is then tested on venue X is training on the test venue.
Each fold therefore drops the medians belonging to its own held-out venue, and the generated
frames are dropped by venue the same way.

**The generated frames are an arm here, not an assumption.** `development_rows` excludes
synthetic frames entirely, so H3's folds have never contained them - A13's repair was measured
in a different train assembly. Putting both on the same seven folds asks two questions at once:
whether that repair reproduces under this protocol, and whether real pixels beat drawn ones at
the job the drawn ones were introduced for. The two sets are the same size to within five
frames, 31 against 26, and both come only from clip venues.

Two measurements, the same seven folds as H3:

- **play-recall** on the held-out venue, which extra EMPTY frames can only hurt
- **false-play** on venue_01 camera B's 243 recorded EMPTY frames, which is what they are for

    uv run python experiments/median_empties_as_training.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.feature_cache import load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, distinct_rows, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
MEDIANS = ROOT / "data" / "interim" / "median_empties_full"
RESULTS = ROOT / "results"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
SEED = 42


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    args = ap.parse_args()

    index = MEDIANS / "index.csv"
    if not index.exists():
        raise SystemExit(f"{index} does not exist. Run scripts/median_empties_from_clips.py")
    with index.open(newline="", encoding="utf-8") as fh:
        medians = [r for r in csv.DictReader(fh) if r["venue"].startswith("clipvenue_")]

    every = read_manifest(DATASET / "manifest.csv")
    cached = load_cache(args.backbone, CACHE)
    feats = {f: v for f, v in zip(cached.files, cached.features, strict=False)}

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    dev = [r for r in development_rows(every) if r.file in feats]
    control = [r for r in every
               if r.source != "synthetic" and r.venue == "venue_01"
               and cam(r) == "camera_B" and r.class3 == EMPTY and r.file in feats]
    cf = {r.file for r in control}
    dev = [r for r in dev if r.file not in cf]
    folds = [f for f in leave_one_group_out(dev, group_key="venue")
             if not f.name.endswith("venue_01")]

    # Embedded once, pooled inside each frame's own boundary, exactly as serving would.
    from PIL import Image

    from pitch_occupancy.vision import roi
    from pitch_occupancy.vision.backbones import embed_batch, load_backbone

    model, processor, spec = load_backbone(args.backbone)
    med_feats, med_rows = [], []
    template = dev[0]
    for r in medians:
        img = cv2.imread(str(MEDIANS / r["file"]))
        if img is None:
            continue
        vec = embed_batch(model, processor, spec, [Image.fromarray(img[:, :, ::-1])],
                          roi_polygon=roi.get(r["camera"]))[0]
        med_feats.append(vec)
        med_rows.append(replace(template, file=r["file"], venue=r["venue"],
                                camera=r["camera"], lighting="night", class3=EMPTY,
                                source="median"))
    generated = [r for r in every
                 if r.source == "synthetic" and r.class3 == EMPTY and r.file in feats]
    print(f"{len(med_rows)} manufactured EMPTY frames from "
          f"{len({r.venue for r in med_rows})} venues, all night")
    print(f"{len(generated)} generated EMPTY frames from "
          f"{len({r.venue for r in generated})} venues (A13's repair)")
    print(f"{len(folds)} folds | false-play control {len(control)} recorded EMPTY frames\n")

    look = {r.file: v for r, v in zip(med_rows, med_feats, strict=True)}
    look.update(feats)
    Xc = np.stack([look[r.file] for r in control])

    def run(extra: str, prune: bool) -> tuple[float, float]:
        recs, fps = [], []
        for fold in folds:
            train = distinct_rows(list(fold.train)) if prune else list(fold.train)
            # Drop this fold's own venue from whatever is added: a median is built from that
            # venue's pixels, and a generated frame was conditioned on them.
            if extra in ("median", "both"):
                train = train + [r for r in med_rows if not fold.name.endswith(r.venue)]
            if extra in ("generated", "both"):
                train = train + [r for r in generated if not fold.name.endswith(r.venue)]
            if extra == "generated_3":
                # The generated set restricted to the three venues the medians cover, so the
                # comparison is not "31 frames from 6 venues against 26 from 3". If venue
                # coverage is what matters, this arm should fail the way the medians do.
                train = train + [r for r in generated
                                 if r.venue in med_venues and not fold.name.endswith(r.venue)]
            if len({r.class3 for r in train}) < 2:
                continue
            probe = LinearProbe(args.backbone, seed=SEED).fit(
                np.stack([look[r.file] for r in train]), train)
            test = list(fold.test)
            pred = probe.predict(np.stack([look[r.file] for r in test]), test)
            recs.append(sum(p == PLAY for p in pred) / len(pred))
            pc = probe.predict(Xc, control)
            fps.append(sum(p == PLAY for p in pc) / len(pc))
        return float(np.mean(recs)), float(np.mean(fps))

    med_venues = {r.venue for r in med_rows}
    n_gen3 = sum(1 for r in generated if r.venue in med_venues)
    labels = {"none": "", "median": " + median empties (26)",
              "generated": " + generated empties (31)",
              "generated_3": f" + generated, same 3 venues ({n_gen3})",
              "both": " + both"}
    records = []
    print(f"{'arm':<40}{'play-recall':>13}{'false-play':>13}{'balanced':>11}")
    for prune in (False, True):
        for extra in ("none", "generated", "generated_3", "median", "both"):
            rec, fp = run(extra, prune)
            name = f"{'pruned' if prune else 'full'}{labels[extra]}"
            print(f"{name:<40}{rec:>13.4f}{fp:>13.4f}{rec - fp:>11.4f}")
            records.append({"arm": name, "pruned": prune, "extra_empties": extra,
                            "play_recall": round(rec, 4), "false_play": round(fp, 4),
                            "balanced": round(rec - fp, 4)})
        print()

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "median_empties_as_training.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {out}")
    print("\nfalse-play is measured on venue_01, which is the only venue with recorded empty\n"
          "frames. A repair that shows up there has still not been shown to transfer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
