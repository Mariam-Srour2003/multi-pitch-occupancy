"""Fit the motion gate's threshold, on the cue it will actually be compared against.

`MOTION_THRESHOLD` was fitted on **whole-frame** motion. Restricting the cue to inside the
pitch boundary - so that a person walking behind the goal cannot keep the gate silent - lowers
every value by roughly a tenth, and a threshold compared against a different quantity is not
the threshold that was measured. This refits it.

Fitted on venue_01's recorded EMPTY and ACTIVE_PLAY frames at a 15-second gap, each camera
using its own derived boundary, as the value maximising (play-recall - false-play). The
reported figures are on venue_01 and are what the number is calibrated to; whether it
transfers is a separate question that only unseen footage answers.

    uv run python experiments/fit_motion_threshold.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.vision.motion import WORK_SIZE, motion_cue

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
GAP_S = 15


def main() -> int:
    polys = {k: v for k, v in
             json.loads((ROOT / "configs" / "roi_derived.json").read_text(encoding="utf-8"))
             .items() if not k.startswith("_")}

    rows = [r for r in read_manifest(DATASET / "manifest.csv")
            if r.source != "synthetic" and r.venue == "venue_01"
            and r.class3 in (EMPTY, PLAY) and str(r.t_s).isdigit()]
    seq = defaultdict(list)
    for r in rows:
        seq[(r.slot_id, r.camera)].append(r)
    for k in seq:
        seq[k].sort(key=lambda r: int(r.t_s))

    cache: dict[str, np.ndarray | None] = {}

    def img(r):
        if r.file not in cache:
            im = cv2.imread(str(DATASET / r.file))
            cache[r.file] = im
        return cache[r.file]

    lo, hi = GAP_S * 0.6, GAP_S * 1.6
    for name, use_polygon in (("whole frame", False), ("inside the boundary", True)):
        cues, labels = [], []
        for (slot, camera), members in seq.items():
            poly = polys.get(camera) if use_polygon else None
            times = [int(m.t_s) for m in members]
            for i, r in enumerate(members):
                best = None
                for j in range(i - 1, -1, -1):
                    d = times[i] - times[j]
                    if d > hi:
                        break
                    if lo <= d and (best is None or abs(d - GAP_S) <
                                    abs(times[i] - int(best.t_s) - GAP_S)):
                        best = members[j]
                if best is None:
                    continue
                a, b = img(best), img(r)
                if a is None or b is None:
                    continue
                cues.append(motion_cue(a, b, poly))
                labels.append(r.class3)

        m = np.array(cues)
        y = np.array(labels)
        best_thr, best_bal = 0.0, -2.0
        for thr in np.quantile(m, np.linspace(0.02, 0.95, 200)):
            play = m >= thr
            rec = float((play & (y == PLAY)).sum() / max(1, (y == PLAY).sum()))
            fp = float((play & (y == EMPTY)).sum() / max(1, (y == EMPTY).sum()))
            if rec - fp > best_bal:
                best_thr, best_bal = float(thr), rec - fp
        play = m >= best_thr
        rec = float((play & (y == PLAY)).sum() / (y == PLAY).sum())
        fp = float((play & (y == EMPTY)).sum() / (y == EMPTY).sum())
        med_e = float(np.median(m[y == EMPTY]))
        med_p = float(np.median(m[y == PLAY]))
        print(f"\n{name}: {len(m)} pairs  (EMPTY median {med_e:.3f}, PLAY median {med_p:.3f})")
        print(f"  threshold {best_thr:.3f}  ->  recall {rec:.3f}, false-play {fp:.3f}, "
              f"balanced {rec - fp:+.3f}")
    print(f"\nframe size for the cue: {WORK_SIZE[0]}x{WORK_SIZE[1]}, gap {GAP_S}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
