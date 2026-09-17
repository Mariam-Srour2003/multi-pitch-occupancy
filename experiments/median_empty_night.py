"""An empty pitch at night, at a venue the model has never seen (A28).

A25 named the two cells this corpus does not have: **empty at night** (9 frames) and **play in
daylight** (6). Everything that cannot be decided here - whether a probe recognises an empty
pitch or the time of day, whether the gates earn their place, whether the clock rule's perfect
cross-venue recall means anything - is undecidable because those cells are empty.

One of them can be manufactured, and by the operation this project has already shown works.
Generated EMPTY frames repaired the false-play collapse and generated C3 frames did nothing,
because the first were **subtraction** - people removed from a real frame, leaving real pitch -
and the second addition. A temporal median is subtraction with no model at all: players move,
the pitch does not, so the per-pixel median of a clip is that venue's pitch with the people
gone, in its own pixels at its own exposure.

`scripts/make_median_empties.py` produces **5** such frames from `clipvenue_a_blue_barrier`,
floodlit night, after three checks - a residual, a spread, and a person detector that rejected
four frames the first two passed.

**What this can and cannot test, stated before the numbers.**

- It **cannot** test the person gate, or the count rule, or anything built on the detector.
  The frames were *selected* by that detector; it will score perfectly on them by construction
  and the figure would mean nothing. It is not reported.
- It **can** test a probe and the clock rule, neither of which had any say in the selection.
- It is **not a recorded empty pitch** and no number from it should be reported as one. Every
  frame is an average of frames that had people in them.

The probe is fitted with `clipvenue_a` held out, so the venue is unseen in the ordinary sense
as well.

    uv run python experiments/median_empty_night.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.feature_cache import load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows
from pitch_occupancy.vision.heads import ClockRule, LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
MEDIANS = ROOT / "data" / "interim" / "median_empties"
RESULTS = ROOT / "results"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
SEED = 42


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--venue", default="clipvenue_a_blue_barrier")
    args = ap.parse_args()

    index = MEDIANS / "index.csv"
    if not index.exists():
        raise SystemExit(f"{index} does not exist. Run scripts/make_median_empties.py first.")
    with index.open(newline="", encoding="utf-8") as fh:
        medians = [r for r in csv.DictReader(fh) if r["venue"] == args.venue]
    if not medians:
        raise SystemExit(f"no median frames for {args.venue}")

    every = read_manifest(DATASET / "manifest.csv")
    cached = load_cache(args.backbone, CACHE)
    feats = {f: v for f, v in zip(cached.files, cached.features, strict=False)}

    # Held out properly: the venue the frames come from contributes nothing to the fit.
    train = [r for r in development_rows(every) if r.file in feats and r.venue != args.venue]
    print(f"{len(medians)} median EMPTY frames from {args.venue}, "
          f"lighting {medians[0]['lighting']}")
    print(f"probe trained on {len(train)} frames with {args.venue} held out "
          f"({len({r.venue for r in train})} venues)\n")

    from PIL import Image

    from pitch_occupancy.vision import roi
    from pitch_occupancy.vision.backbones import embed_batch, load_backbone

    images, polygons = [], []
    for r in medians:
        img = cv2.imread(str(MEDIANS / r["file"]))
        images.append(img)
        polygons.append(roi.get(r["camera"]))

    model, processor, spec = load_backbone(args.backbone)
    # Pooled inside each frame's own boundary, which is what serving does.
    X = np.stack([
        embed_batch(model, processor, spec, [Image.fromarray(im[:, :, ::-1])],
                    roi_polygon=poly)[0]
        for im, poly in zip(images, polygons, strict=True)
    ])

    template = train[0]
    from dataclasses import replace

    rows = [replace(template, file=r["file"], venue=args.venue,
                    lighting=r["lighting"], class3=EMPTY) for r in medians]

    probe = LinearProbe(args.backbone, seed=SEED).fit(
        np.stack([feats[r.file] for r in train]), train)
    probe_pred = list(probe.predict(X, rows))
    clock = ClockRule().fit(np.zeros((len(train), 1)), train)
    clock_pred = list(clock.predict(np.zeros((len(rows), 1)), rows))

    print(f"{'frame':<38}{'probe':>26}{'clock rule':>26}")
    for r, p, c in zip(medians, probe_pred, clock_pred, strict=True):
        print(f"{r['file']:<38}{p.split('_', 1)[1]:>26}{c.split('_', 1)[1]:>26}")

    records = []
    print(f"\n{'system':<24}{'calls it EMPTY':>17}{'calls it PLAY':>16}")
    for name, pred in (("probe (" + args.backbone + ")", probe_pred),
                       ("clock rule", clock_pred)):
        n_e = sum(1 for p in pred if p == EMPTY)
        n_p = sum(1 for p in pred if p == PLAY)
        print(f"{name:<24}{n_e:>10}/{len(pred):<6}{n_p:>10}/{len(pred):<6}")
        records.append({"system": name, "venue": args.venue, "n_frames": len(pred),
                        "says_empty": n_e, "says_play": n_p,
                        "says_other": len(pred) - n_e - n_p})

    print("\nthe person gate is deliberately absent: these frames were selected by the "
          "detector\nit runs, so it scores perfectly on them by construction and the figure "
          "would mean nothing.")
    print("these are manufactured frames, not a recorded empty pitch at night, and no number "
          "here\nsubstitutes for one.")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "median_empty_night.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
