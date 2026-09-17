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
MEDIANS_FULL = ROOT / "data" / "interim" / "median_empties_full"
RESULTS = ROOT / "results"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
SEED = 42


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--venue", default=None,
                    help="restrict to one venue; default is every venue in the index")
    ap.add_argument("--dir", type=Path, default=None,
                    help="median directory; default is the full-clip set if it exists")
    args = ap.parse_args()

    # The full-clip set (A29) is preferred when present: it medians 250-350 frames per clip
    # rather than the six the dataset sampled, so a player must stand still for ten seconds to
    # survive it. `--dir` overrides.
    base = args.dir or (MEDIANS_FULL if (MEDIANS_FULL / "index.csv").exists() else MEDIANS)
    index = base / "index.csv"
    if not index.exists():
        raise SystemExit(f"{index} does not exist. Run "
                         f"scripts/median_empties_from_clips.py first.")
    with index.open(newline="", encoding="utf-8") as fh:
        medians = [r for r in csv.DictReader(fh)
                   if r["venue"].startswith("clipvenue_")
                   and (args.venue is None or r["venue"] == args.venue)]
    if not medians:
        raise SystemExit(f"no clip-venue median frames in {index}")
    venues = sorted({r["venue"] for r in medians})

    every = read_manifest(DATASET / "manifest.csv")
    cached = load_cache(args.backbone, CACHE)
    feats = {f: v for f, v in zip(cached.files, cached.features, strict=False)}

    # Leave *one* venue out, per venue, rather than all of them at once.
    #
    # Holding out all three shrinks the fit from 1410 frames and 7 venues to 1362 and 5, so a
    # score measured that way confounds "this venue is unseen" with "there is less training
    # data". H3 holds out one venue at a time and this matches it: each venue's median frames
    # are scored by a probe that saw the other venues and not that one.
    def fit_without(held: str):
        rows_ = [r for r in development_rows(every) if r.file in feats and r.venue != held]
        return LinearProbe(args.backbone, seed=SEED).fit(
            np.stack([feats[r.file] for r in rows_]), rows_), len(rows_)

    train = [r for r in development_rows(every) if r.file in feats and r.venue not in venues]
    print(f"{base.name}: {len(medians)} median EMPTY frames, night, "
          f"from {len(venues)} venues")
    for v in venues:
        print(f"    {v:<34}{sum(1 for r in medians if r['venue'] == v):>4}")
    print("leave-one-venue-out: one probe per venue, each blind to its own\n")

    from PIL import Image

    from pitch_occupancy.vision import roi
    from pitch_occupancy.vision.backbones import embed_batch, load_backbone

    images, polygons = [], []
    for r in medians:
        img = cv2.imread(str(base / r["file"]))
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

    rows = [replace(template, file=r["file"], venue=r["venue"],
                    lighting=r["lighting"], class3=EMPTY) for r in medians]

    # One probe per held-out venue; each frame is scored by the probe that never saw its
    # venue. `n_train` is printed so the fits are visibly comparable to each other.
    probe_pred: list[str] = [""] * len(rows)
    for v in venues:
        probe_v, n_train = fit_without(v)
        idx = [i for i, r in enumerate(medians) if r["venue"] == v]
        preds = probe_v.predict(X[idx], [rows[i] for i in idx])
        for i, pr in zip(idx, preds, strict=True):
            probe_pred[i] = pr
        print(f"  probe for {v:<34}fitted on {n_train} frames")
    print()
    clock = ClockRule().fit(np.zeros((len(train), 1)), train)
    clock_pred = list(clock.predict(np.zeros((len(rows), 1)), rows))

    records = []
    print(f"{'system':<22}{'venue':<32}{'says EMPTY':>13}{'says PLAY':>12}")
    for name, pred in (("probe (" + args.backbone + ")", probe_pred),
                       ("clock rule", clock_pred)):
        for v in [*venues, "ALL"]:
            idx = ([i for i, r in enumerate(medians) if r["venue"] == v]
                   if v != "ALL" else list(range(len(medians))))
            n_e = sum(1 for i in idx if pred[i] == EMPTY)
            n_p = sum(1 for i in idx if pred[i] == PLAY)
            label = name if v == venues[0] else ""
            print(f"{label:<22}{v:<32}{n_e:>7}/{len(idx):<5}{n_p:>7}/{len(idx):<5}")
            records.append({"system": name, "venue": v, "n_frames": len(idx),
                            "says_empty": n_e, "says_play": n_p,
                            "says_other": len(idx) - n_e - n_p})

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
