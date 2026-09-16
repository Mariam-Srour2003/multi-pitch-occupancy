"""Does the motion cue fix what the boundary could not, on unseen footage? (A14, second look)

`motion_feature_ablation` found the cue worthless - as a 769th feature and as an override
rule - and measured that on venue_01, where the model already scores 0.9386 and there is
nothing to repair. The boundary looked equally worthless on that split and then halved
false-play on a clip from an unseen venue. So the motion result deserves the same retest for
the same reason.

The design is a **transfer test, not a fit**. The threshold is chosen on the corpus, on
venue_01's recorded EMPTY and PLAY frames at a 15-second gap, as the value maximising
(recall - false-play) there. It is then applied to the video unchanged. Nothing about the
video informs it, so the number it produces is an honest estimate of what the rule would do
on arrival at a new site.

The video is sampled every 15 seconds to match the gap the threshold was fitted at. A cue
compared against a threshold learned at a different spacing is a different quantity.

    uv run python experiments/motion_override_on_video.py VIDEO --truth 139,179,209,224
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
SIZE = (160, 90)
GAP_S = 15


def _gray_bgr(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), SIZE).astype(np.float32)


def corpus_threshold() -> tuple[float, float, float]:
    """Pick the motion threshold on venue_01, and report what it achieves there."""
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    from motion_cue_probe import _auc  # noqa: F401  (kept for parity of imports)
    from motion_feature_ablation import motion_cue

    from pitch_occupancy.data.manifest import read_manifest

    rows = [r for r in read_manifest(DATASET / "manifest.csv")
            if r.source != "synthetic" and r.venue == "venue_01"
            and r.class3 in (EMPTY, PLAY)]
    cue = motion_cue(rows, GAP_S)
    have = [r for r in rows if r.file in cue]
    m = np.array([cue[r.file] for r in have])
    y = np.array([r.class3 for r in have])
    best, best_bal = 0.0, -2.0
    for thr in np.quantile(m, np.linspace(0.02, 0.95, 120)):
        pred_play = m >= thr
        rec = float((pred_play & (y == PLAY)).sum() / max(1, (y == PLAY).sum()))
        fp = float((pred_play & (y == EMPTY)).sum() / max(1, (y == EMPTY).sum()))
        if rec - fp > best_bal:
            best, best_bal = float(thr), rec - fp
    pred_play = m >= best
    rec = float((pred_play & (y == PLAY)).sum() / (y == PLAY).sum())
    fp = float((pred_play & (y == EMPTY)).sum() / (y == EMPTY).sum())
    print(f"threshold fitted on venue_01 ({len(have)} frames, {GAP_S}s gap): {best:.3f}")
    print(f"  on venue_01 it gives recall {rec:.3f}, false-play {fp:.3f}, "
          f"balanced {rec - fp:+.3f}")
    return best, rec, fp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--truth", default="",
                    help="comma-separated sample indices that contain a person")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    thr, _, _ = corpus_threshold()

    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from derive_roi import polygon_from_mask, turf_mask

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(GAP_S * fps)))
    frames, times, pool = [], [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % 40 == 0 and len(pool) < 40:
            pool.append(cv2.resize(fr, (320, 180)))
        if i % step == 0:
            frames.append(fr)
            times.append(i / fps)
        i += 1
    cap.release()

    polygon = polygon_from_mask(turf_mask(np.median(np.stack(pool), axis=0).astype(np.uint8)))
    print(f"\nboundary derived from the video: {len(polygon)} points")
    print(f"{len(frames)} samples every {GAP_S}s")

    from pitch_occupancy.vision.classifier import load_classifier

    clf = load_classifier(args.model)
    verdicts = clf.classify_batch(frames, polygon=polygon)

    small = [_gray_bgr(f) for f in frames]
    motion = [float("nan")] + [float(np.abs(b - a).mean()) for a, b in zip(small, small[1:])]

    person = {int(x) for x in args.truth.split(",") if x.strip().isdigit()}
    print(f"\n{'#':>3}{'t':>8}{'verdict':>14}{'motion':>9}{'+override':>12}{'truth':>10}")
    rows = []
    for k, ((cls, conf), t, m) in enumerate(zip(verdicts, times, motion)):
        name = cls.name if hasattr(cls, "name") else str(cls)
        # A frame with no earlier partner has no motion value and the rule cannot speak.
        over = name
        if name == "ACTIVE_PLAY" and m == m and m < thr:
            over = "EMPTY"
        truth = "person" if k in person else "empty"
        print(f"{k:>3}{t:>8.1f}{name:>14}"
              + (f"{m:>9.3f}" if m == m else f"{'-':>9}")
              + f"{over:>12}{truth:>10}")
        rows.append({"idx": k, "t_s": round(t, 1), "verdict": name,
                     "motion": round(m, 4) if m == m else "", "with_override": over,
                     "truth": truth})

    empt = [r for r in rows if r["truth"] == "empty"]
    for label, key in (("boundary only", "verdict"), ("boundary + motion rule", "with_override")):
        fp = sum(1 for r in empt if r[key] == "ACTIVE_PLAY")
        print(f"\n{label:<26} false-play {fp}/{len(empt)} = {fp / len(empt):.2f}")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "motion_override_on_video.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {p}  (threshold {thr:.3f}, fitted on venue_01, never on this video)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
