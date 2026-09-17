"""The clock rule beats the backbones on the corpus. Put it on real footage (A26).

A25 corrected the lighting labels and the trivial baseline overtook every frozen backbone on
the cross-venue protocol: play-recall **1.000** against DINOv2's 0.930, at a false-play rate of
**0.0206** against DINOv2's 0.3086. Taken at face value that says the deep half of this project
is unnecessary, and the gates on top of it doubly so.

It should not be taken at face value, and the reason is checkable rather than rhetorical. The
rule wins because of a confound - across the recorded corpus night is 99% ACTIVE_PLAY and day
is 98% EMPTY, so "night means play" is right on 99.1% of frames - and **the corpus contains
almost no frames where that shortcut fails**: 9 empty frames at night, 6 play frames in
daylight.

The unseen clip is exactly such a frame, sixteen times over. It is floodlit night football's
lighting with nobody playing: 13 empty minutes, 3 with one person walking, no match at any
point. A rule keyed on lighting has to call all 16 of them ACTIVE_PLAY.

So this runs three systems on the same 16 minutes and reports them together:

- **the clock rule**, fitted on the development set exactly as H3 fits it
- **the probe alone**, DINOv2, the same fit H3 scores
- **the deployed path**, probe plus boundary plus motion gate plus person gate

    uv run python experiments/clock_rule_on_video.py VIDEO --truth 9,12,15
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.feature_cache import load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision.heads import ClockRule, LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
SEED = 42


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--every", type=float, default=15.0)
    ap.add_argument("--truth", default="", help="minutes with a person on the pitch")
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--lighting", default="night",
                    help="what the clock rule is told about this footage")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(args.every * fps)))
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

    sys.path.insert(0, str(ROOT / "scripts"))
    from derive_roi import polygon_from_mask, turf_mask

    polygon = polygon_from_mask(turf_mask(np.median(np.stack(pool), axis=0).astype(np.uint8)))
    person = {int(x) for x in args.truth.split(",") if x.strip().isdigit()}
    brightness = float(np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames]))

    every = read_manifest(DATASET / "manifest.csv")
    dev = development_rows(every)
    cached = load_cache(args.backbone, CACHE)
    feats = {f: v for f, v in zip(cached.files, cached.features, strict=False)}
    train = [r for r in dev if r.file in feats]

    # The clock rule needs a `lighting` per test frame and nothing else. The footage is
    # floodlit night by inspection - black sky, lit fixtures - and `extract.py`'s brightness
    # rule would call it day at mean 108, which is the very error A25 corrected. It is told
    # the truth here, which is the *charitable* setting: told "day" it would answer EMPTY to
    # everything and score 13/13 on the empty minutes for exactly the wrong reason.
    print(f"clip: {len(frames)} minutes, mean brightness {brightness:.0f}, "
          f"told lighting={args.lighting!r}")
    print(f"truth: {len(frames) - len(person)} empty, {len(person)} with one person walking, "
          f"0 with a match\n")

    template = train[0]
    test_rows = [replace(template, file=f"video_{k:03d}.jpg", lighting=args.lighting)
                 for k in range(len(frames))]
    clock = ClockRule().fit(np.zeros((len(train), 1)), train)
    clock_pred = clock.predict(np.zeros((len(frames), 1)), test_rows)

    from PIL import Image

    from pitch_occupancy.vision.backbones import embed_batch, load_backbone

    model, processor, spec = load_backbone(args.backbone)
    X = embed_batch(model, processor, spec,
                    [Image.fromarray(f[:, :, ::-1]) for f in frames], roi_polygon=polygon)
    probe = LinearProbe(args.backbone, seed=SEED).fit(
        np.stack([feats[r.file] for r in train]), train)
    probe_pred = list(probe.predict(X, test_rows))

    from pitch_occupancy.vision.motion import MotionGate
    from pitch_occupancy.vision.people import PersonGate

    mgate, pgate = MotionGate(), PersonGate()
    gated = []
    for k, cls in enumerate(probe_pred):
        state = Class3(cls)
        state, _ = mgate.apply(state, frames[k - 1] if k else None, frames[k], polygon)
        state, _ = pgate.inspect(state, frames[k], polygon)
        gated.append(state.value)

    arms = {"clock rule": clock_pred, f"probe alone ({args.backbone})": probe_pred,
            "deployed path (probe + boundary + gates)": gated}
    empty_minutes = [k for k in range(len(frames)) if k not in person]

    print(f"{'system':<44}{'says PLAY':>11}{'false-play':>13}{'right':>9}")
    records = []
    for name, pred in arms.items():
        n_play = sum(1 for p in pred if p == PLAY)
        fp = sum(1 for k in empty_minutes if pred[k] == PLAY)
        # No minute of this clip is a match, so every correct verdict is a non-PLAY one and
        # the empty minutes must read EMPTY specifically.
        right = sum(1 for k in empty_minutes if pred[k] == EMPTY)
        print(f"{name:<44}{n_play:>11}{fp / len(empty_minutes):>12.2f}"
              f"{right:>6}/{len(empty_minutes)}")
        records.append({"system": name, "minutes": len(frames), "says_play": n_play,
                        "false_play": round(fp / len(empty_minutes), 4),
                        "empty_correct": right, "empty_minutes": len(empty_minutes)})

    print("\nper minute:")
    print(f"{'#':>3}{'truth':>10}" + "".join(f"{n[:22]:>24}" for n in arms))
    for k in range(len(frames)):
        row = f"{k:>3}{'person' if k in person else 'empty':>10}"
        for pred in arms.values():
            row += f"{pred[k].split('_', 1)[1][:22]:>24}"
        print(row)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "clock_rule_on_video.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
