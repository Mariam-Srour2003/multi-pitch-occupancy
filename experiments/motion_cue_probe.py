"""Does frame-to-frame motion separate play from not-play, as appearance alone cannot?

The proposal: a pitch with people on it but **nothing moving** should not read as a match.
That is exactly the boundary this corpus is worst at - `data/interim/_pending_4d/` holds
frames of two to four people with a ball, held unlabelled because §2.2 and §2.6 call any
athletic activity ACTIVE_PLAY while a small static kickabout arguably is not.

Appearance features cannot see it. A single frame of four people standing with a ball and a
single frame of four people mid-rally are nearly the same picture; what differs is what
happened in the second before it.

This measures whether a one-number motion cue - mean absolute difference against the nearest
earlier frame of the same slot and camera - separates the classes at all. If it does not, no
amount of wiring it into the model will help, and that is worth knowing before building it.

Two things it reports honestly:

- **Coverage.** A motion cue needs a previous frame. Frames without one get nothing, and the
  production sampler takes **one frame per camera per minute**, where this corpus has gaps of
  1-15 seconds. A cue that works here may not survive the deployment sampling rate, so the
  gap distribution is printed alongside.
- **Separation, not accuracy.** AUC between class pairs, on the raw cue. A probe would do
  better; the question here is whether there is any signal to give it.

    uv run python experiments/motion_cue_probe.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from pitch_occupancy.data.manifest import read_manifest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
EMPTY, PLAY, OTHER = "C1_EMPTY", "C2_ACTIVE_PLAY", "C3_MAINTENANCE_NON_SPORTING"
MAX_GAP_S = 30
SIZE = (160, 90)


def _gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(SIZE), dtype=np.float32)


def _auc(pos: list[float], neg: list[float]) -> float:
    """Rank AUC. 0.5 is no separation; below 0.5 means the cue runs the other way."""
    if not pos or not neg:
        return float("nan")
    allv = np.array(pos + neg, dtype=float)
    ranks = allv.argsort().argsort().astype(float) + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    rows = [r for r in read_manifest(DATASET / "manifest.csv") if r.source != "synthetic"]

    # Group by the actual recording: same slot, same camera. A "previous frame" from another
    # camera or another day is not a previous frame.
    seq: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        if isinstance(r.t_s, int) or str(r.t_s).isdigit():
            seq[(r.slot_id, r.camera)].append(r)
    for k in seq:
        seq[k].sort(key=lambda r: int(r.t_s))

    cue: dict[str, float] = {}
    gaps: list[int] = []
    for members in seq.values():
        prev = None
        for r in members:
            if prev is not None:
                gap = int(r.t_s) - int(prev.t_s)
                if 0 < gap <= MAX_GAP_S:
                    a, b = DATASET / prev.file, DATASET / r.file
                    if a.exists() and b.exists():
                        cue[r.file] = float(np.abs(_gray(b) - _gray(a)).mean())
                        gaps.append(gap)
            prev = r

    by_class: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.file in cue:
            by_class[r.class3].append(cue[r.file])

    print(f"\n{len(rows)} recorded frames | {len(cue)} have a previous frame "
          f"within {MAX_GAP_S}s ({len(cue) / len(rows):.0%} coverage)")
    if gaps:
        g = np.array(gaps)
        print(f"gap to previous frame: median {np.median(g):.0f}s  "
              f"p90 {np.percentile(g, 90):.0f}s  max {g.max()}s")
    print("production sampling is one frame per camera per MINUTE - see "
          "thesis/data_requests.md §7\n")

    print(f"{'class':<32}{'n':>6}{'median':>10}{'mean':>10}")
    for c in (EMPTY, PLAY, OTHER):
        v = by_class.get(c, [])
        if v:
            print(f"{c:<32}{len(v):>6}{np.median(v):>10.3f}{np.mean(v):>10.3f}")
        else:
            print(f"{c:<32}{0:>6}{'-':>10}{'-':>10}")

    print("\nseparation (AUC, motion cue alone)")
    pairs = [("PLAY vs EMPTY", PLAY, EMPTY), ("PLAY vs C3", PLAY, OTHER),
             ("C3 vs EMPTY", OTHER, EMPTY)]
    out = []
    for label, a, b in pairs:
        auc = _auc(by_class.get(a, []), by_class.get(b, []))
        print(f"  {label:<18} {auc:.3f}"
              + ("   (no signal)" if not np.isnan(auc) and abs(auc - 0.5) < 0.05 else ""))
        out.append({"pair": label, "auc": round(auc, 4) if not np.isnan(auc) else "",
                    "n_a": len(by_class.get(a, [])), "n_b": len(by_class.get(b, []))})

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "motion_cue_probe.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "auc", "n_a", "n_b"])
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
