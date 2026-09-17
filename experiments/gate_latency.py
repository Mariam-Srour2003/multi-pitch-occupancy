"""What the gates cost the 60-second cycle (A21).

`efficiency_latency.csv` reports 20 cameras in 2.5-5.7 s against a 60 s cycle, and it measures
the **probe**. Since A14 the deployed path is probe plus motion gate plus person gate, and the
person gate runs a detector - so the reported throughput describes something the system stopped
being three amendments ago.

The gate's cost is not a constant, which is why this is a table and not a number: the detector
only runs when the verdict is ACTIVE_PLAY, because that is the only verdict it can change. A
site with nobody on it pays nothing. A site mid-match pays for every camera. The honest answer
is a cost per play-rate, and the deployment claim has to hold at 100%.

**Measured on the machine that happens to be here, like every other timing in this project, and
therefore not the deployment claim.** WP7-T1's run on the target Mini-PC is what settles that.
Run this on an idle machine and never inside a batch - `efficiency_latency.py` documents why in
detail, and the same applies with more force here, since a detector is the heaviest thing this
system runs.

    uv run python experiments/gate_latency.py
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.vision.explain import detect_objects
from pitch_occupancy.vision.motion import motion_cue
from pitch_occupancy.vision.people import BALL_CONFIDENCE, DETECT_CONFIDENCE, DETECT_IMGSZ

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CYCLE_SECONDS = 60.0
N_CAMERAS = 20


def timed(fn, frames, *, warmup: int = 2) -> tuple[float, float]:
    """(median ms, p95 ms) over the frames, warm-up discarded."""
    for f in frames[:warmup]:
        fn(f)
    took = []
    for f in frames:
        start = time.perf_counter()
        fn(f)
        took.append((time.perf_counter() - start) * 1000.0)
    a = np.array(took)
    return float(np.median(a)), float(np.percentile(a, 95))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--backbone", default="dinov2")
    args = ap.parse_args()

    rows = [r for r in read_manifest(DATASET / "manifest.csv") if r.source != "synthetic"]
    frames = []
    for r in rows:
        img = cv2.imread(str(DATASET / r.file))
        if img is not None:
            frames.append(img)
        if len(frames) >= args.frames:
            break
    print(f"{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}\n")

    det_med, det_p95 = timed(
        lambda f: detect_objects(f, confidence=min(DETECT_CONFIDENCE, BALL_CONFIDENCE),
                                 imgsz=DETECT_IMGSZ, classes=(0, 32)), frames)
    prev = frames[0]
    mot_med, mot_p95 = timed(lambda f: motion_cue(prev, f, None), frames)

    # The probe's own per-frame cost, so the comparison is against this machine rather than
    # against a figure measured on another one on another day.
    from PIL import Image

    from pitch_occupancy.vision.backbones import embed_batch, load_backbone

    model, processor, spec = load_backbone(args.backbone)
    emb_med, emb_p95 = timed(
        lambda f: embed_batch(model, processor, spec, [Image.fromarray(f[:, :, ::-1])]),
        frames)

    print(f"{'stage':<40}{'median ms':>12}{'p95 ms':>10}")
    print(f"{'backbone embed (' + args.backbone + ')':<40}{emb_med:>12.0f}{emb_p95:>10.0f}")
    print(f"{'motion cue (160x90 difference)':<40}{mot_med:>12.1f}{mot_p95:>10.1f}")
    print(f"{'person gate detector (imgsz ' + str(DETECT_IMGSZ) + ')':<40}"
          f"{det_med:>12.0f}{det_p95:>10.0f}")

    # One round is: embed every camera, motion-cue every camera, and run the detector on the
    # cameras whose verdict survived as ACTIVE_PLAY.
    print(f"\none round of {N_CAMERAS} cameras against the {CYCLE_SECONDS:.0f} s cycle, "
          f"by how many of them are showing play:")
    print(f"{'cameras in play':<20}{'round (s)':>12}{'of the cycle':>15}{'verdict':>10}")
    records = []
    for rate in (0.0, 0.25, 0.5, 1.0):
        n_play = int(round(rate * N_CAMERAS))
        seconds = (N_CAMERAS * (emb_med + mot_med) + n_play * det_med) / 1000.0
        ok = seconds < CYCLE_SECONDS
        print(f"{n_play:>3} of {N_CAMERAS:<14}{seconds:>12.1f}{seconds / CYCLE_SECONDS:>14.0%}"
              f"{'  fits' if ok else '  OVER':>10}")
        records.append({"backbone": args.backbone, "cameras": N_CAMERAS,
                        "cameras_in_play": n_play, "round_seconds": round(seconds, 2),
                        "cycle_fraction": round(seconds / CYCLE_SECONDS, 4),
                        "fits": ok, "embed_ms": round(emb_med, 1),
                        "motion_ms": round(mot_med, 2), "detect_ms": round(det_med, 1)})

    print("\nthe round is serial on purpose: the cameras are sampled one after another and "
          "\nthe detector is the only stage that could be batched, which would change the "
          "\nshape of the worker rather than its arithmetic.")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "gate_latency.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
