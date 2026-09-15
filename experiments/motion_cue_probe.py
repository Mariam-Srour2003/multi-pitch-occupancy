"""Does frame-to-frame motion separate play from not-play, as appearance alone cannot?

The proposal: a pitch with people on it but **nothing moving** should not read as a match.
That is exactly the boundary this corpus is worst at - `data/interim/_pending_4d/` holds
frames of two to four people with a ball, held unlabelled because §2.2 and §2.6 call any
athletic activity ACTIVE_PLAY while a small static kickabout arguably is not.

Appearance features cannot see it. A single frame of four people standing with a ball and a
single frame of four people mid-rally are nearly the same picture; what differs is what
happened in the second before it.

This measures whether a one-number motion cue - mean absolute difference against an earlier
frame of the same slot and camera - separates the classes, **and how that decays as the two
frames are pulled apart**. The second half is the part that decides anything: this corpus has
frames seconds apart, and the production sampler takes one per camera per minute.

So the cue is computed at several target gaps, pairing each frame with the earlier one closest
to that gap rather than the nearest one available. Taking "any frame within 60 seconds" would
quietly keep using the 8-second pairs and answer the wrong question.

Separation is reported as AUC on the raw cue - no model, no training. A probe would do better;
the question here is whether there is signal to give it.

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
SIZE = (160, 90)

#: Gaps to measure at. 15s is what this corpus offers; 60s is what the production sampler
#: actually takes (`thesis/data_requests.md` §7), and is therefore the only one that speaks
#: to deployment. A cue that works at 8 seconds and dies at 60 is a cue this system cannot use.
GAPS_S = (15, 30, 60, 120)


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

    # One cache of decoded frames; the four gap settings reuse it rather than re-reading.
    grays: dict[str, np.ndarray] = {}

    def gray(r) -> np.ndarray | None:
        if r.file not in grays:
            p = DATASET / r.file
            grays[r.file] = _gray(p) if p.exists() else None
        return grays[r.file]

    def cue_at(target_gap: int) -> tuple[dict[str, float], list[int]]:
        """Pair each frame with the earlier frame closest to `target_gap` seconds before it.

        Not "within" the gap - *at* it. Taking the nearest frame under 60s would quietly keep
        using the 8-second pairs and answer the wrong question.
        """
        out: dict[str, float] = {}
        seen: list[int] = []
        lo, hi = target_gap * 0.6, target_gap * 1.6
        for members in seq.values():
            times = [int(m.t_s) for m in members]
            for i, r in enumerate(members):
                t = int(r.t_s)
                best, best_err = None, None
                for j in range(i - 1, -1, -1):
                    d = t - times[j]
                    if d > hi:
                        break
                    if d < lo:
                        continue
                    err = abs(d - target_gap)
                    if best_err is None or err < best_err:
                        best, best_err = members[j], d
                if best is None:
                    continue
                a, b = gray(best), gray(r)
                if a is None or b is None:
                    continue
                out[r.file] = float(np.abs(b - a).mean())
                seen.append(best_err)
        return out, seen

    out_rows = []
    for target in GAPS_S:
        cue, gaps = cue_at(target)
        by_class: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r.file in cue:
                by_class[r.class3].append(cue[r.file])
        cov = len(cue) / len(rows)
        med = float(np.median(gaps)) if gaps else float("nan")
        print(f"=== target gap {target}s ===  {len(cue)} pairs "
              f"({cov:.0%} coverage), actual median {med:.0f}s")
        for c, nm in ((EMPTY, "EMPTY"), (PLAY, "PLAY"), (OTHER, "C3")):
            v = by_class.get(c, [])
            print(f"    {nm:<6} n={len(v):>5}  median {np.median(v):.3f}" if v
                  else f"    {nm:<6} n=    0")
        for label, a, b in (("PLAY vs EMPTY", PLAY, EMPTY), ("PLAY vs C3", PLAY, OTHER)):
            auc = _auc(by_class.get(a, []), by_class.get(b, []))
            print(f"    {label:<15} AUC {auc:.3f}")
            out_rows.append({"target_gap_s": target, "actual_median_gap_s": round(med, 1),
                             "coverage": round(cov, 4), "pair": label,
                             "auc": round(auc, 4) if not np.isnan(auc) else "",
                             "n_a": len(by_class.get(a, [])), "n_b": len(by_class.get(b, []))})
        print()

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "motion_cue_probe.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
