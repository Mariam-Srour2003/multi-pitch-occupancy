"""RQ6 on a test set that is not 99% one class (WP4-T5).

`rq_matrix.md` records RQ6 as **blocked by data**: "machinery built and tested; the answer is
blocked by the same degenerate split. With a test set that is 99% one class, a 99% precision
target is met before confidence is consulted, so the risk-coverage curve has nothing to trade
against." `evaluation/calibration.py` has been ready and waiting for "a test set with a real
class mix".

One exists, and it always did - it was simply not the split any experiment used. Holding out
venue_01's **physical camera B** leaves 237 recorded EMPTY frames and 275 recorded PLAY
frames, a 46/54 mix. The project's splits are grouped by slot or by venue, and venue_01 has
only two slots, so neither protocol produces this; `class_balancing._false_play_control`
reached for the same camera split for the same reason.

Training is camera A plus the clip venues plus the 31 generated EMPTY frames - the
configuration `a13_false_play_repair` found necessary to stop the probe calling every empty
pitch a match. Without those frames the curve would describe a model with a false-play rate
near 1.0, and its operating point would be meaningless.

**Scope, stated once and meant.** This is one venue, two cameras, two days. It answers "what
does review buy *here*", not "what does review buy". RQ1's blocker is untouched: the test
frames are real, but they are venue_01's, and a threshold set on them has not been shown to
transfer.

    uv run python experiments/rq6_on_a_real_class_mix.py --backbone dinov2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import load_cache
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
SEED = 42


def risk_coverage_band(conf: np.ndarray, correct: np.ndarray):
    """Accuracy against coverage, as a band rather than a line.

    Ties are the reason for the band. "The most confident k" is undefined when the k-th and
    (k+1)-th confidences are equal, and sorting them imposes an order the model never
    expressed. At each coverage the best and worst accuracy consistent with the ties are both
    reported, and an operating point is read off the **lower** bound - a promise that holds
    whichever way they fall.
    """
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], correct[order]
    out = []
    for k in range(1, len(c) + 1):
        thr = c[k - 1]
        strictly_above = ok[c > thr]
        tied = ok[c == thr]
        n_from_tied = k - len(strictly_above)
        base = int(strictly_above.sum())
        lo = (base + max(0, n_from_tied - (len(tied) - int(tied.sum())))) / k
        hi = (base + min(n_from_tied, int(tied.sum()))) / k
        out.append((k / len(c), lo, hi, float(thr)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--roi-pooled", action="store_true")
    ap.add_argument("--gates", action="store_true",
                    help="apply the A16 person gate before reading confidence")
    args = ap.parse_args()

    rows = read_manifest(DATASET / "manifest.csv")
    cached = load_cache(args.backbone, CACHE, roi_pooled=args.roi_pooled)
    feats = {f: v for f, v in zip(cached.files, cached.features)}

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    keep = [r for r in rows if r.file in feats and r.class3 in (EMPTY, PLAY)]
    real = [r for r in keep if r.source != "synthetic"]
    gen_empty = [r for r in keep if r.source == "synthetic"
                 and r.class3 == EMPTY and r.venue != "venue_01"]
    v01 = [r for r in real if r.venue == "venue_01"]
    train = ([r for r in v01 if cam(r) == "camera_A"]
             + [r for r in real if r.venue != "venue_01"] + gen_empty)
    test = [r for r in v01 if cam(r) == "camera_B"]

    n_e = sum(1 for r in test if r.class3 == EMPTY)
    print(f"\ntest: {len(test)} recorded frames - EMPTY {n_e}, PLAY {len(test) - n_e} "
          f"({n_e / len(test):.0%} / {1 - n_e / len(test):.0%})")
    print(f"train: {len(train)} (+{len(gen_empty)} generated EMPTY)")

    probe = LinearProbe(args.backbone, seed=SEED).fit(
        np.stack([feats[r.file] for r in train]), train)
    Xte = np.stack([feats[r.file] for r in test])
    proba = probe.predict_proba(Xte)
    classes = list(probe.classes_)
    pred = [classes[int(i)] for i in proba.argmax(1)]
    conf = proba.max(1)

    if args.gates:
        # The gates run before confidence is read, because that is the order the deployed
        # path uses: `run_slot` overrules the verdict and the verdict is what reaches the
        # REVIEW band. Reading confidence from the probe and the class from the gate would
        # describe a system nobody runs.
        #
        # An overruled frame keeps the probe's confidence. The gate has no calibrated
        # confidence of its own - it is a count and a threshold - and inventing one (1.0, say)
        # would put every gated frame at the top of the risk-coverage curve on no evidence.
        import cv2

        from pitch_occupancy.vision import roi
        from pitch_occupancy.vision.people import PersonGate

        gate = PersonGate()
        n_over = 0
        for i, r in enumerate(test):
            if pred[i] != PLAY:
                continue
            frame = cv2.imread(str(DATASET / r.file))
            if frame is None:
                continue
            state, _count = gate.apply(Class3(pred[i]), frame, roi.get(r.camera))
            if state.value != pred[i]:
                pred[i] = state.value
                n_over += 1
        print(f"person gate overruled {n_over} of {len(test)} verdicts")
    truth = [r.class3 for r in test]
    correct = np.array([p == t for p, t in zip(pred, truth)])

    print(f"\naccuracy at full coverage: {correct.mean():.4f}")
    print(f"confidences that are exactly 1.0: {(conf >= 0.9999).sum()} of {len(conf)}")

    band = risk_coverage_band(conf, correct)
    print(f"\n{'coverage':>10}{'acc (worst)':>14}{'acc (best)':>13}{'threshold':>12}"
          f"{'review':>9}")
    marks = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    records = []
    for m in marks:
        k = max(1, int(round(m * len(band))))
        cov, lo, hi, thr = band[k - 1]
        print(f"{cov:>10.2f}{lo:>14.4f}{hi:>13.4f}{thr:>12.4f}{1 - cov:>9.0%}")
        # `gates` is part of the row, not just the console output. Two arms whose numbers
        # come out identical are indistinguishable once appended, and a results file where
        # you cannot tell which run a row came from is not a record of anything.
        records.append({"backbone": args.backbone, "roi_pooled": args.roi_pooled,
                        "gates": args.gates,
                        "coverage": round(cov, 4), "acc_worst": round(lo, 4),
                        "acc_best": round(hi, 4), "threshold": round(thr, 4),
                        "review_rate": round(1 - cov, 4)})

    for target in (0.95, 0.99):
        ok = [b for b in band if b[1] >= target]
        if ok:
            cov, lo, hi, thr = max(ok, key=lambda b: b[0])
            print(f"\nto guarantee {target:.0%} accuracy on automated verdicts: "
                  f"answer {cov:.0%}, review {1 - cov:.0%}  (confidence >= {thr:.4f})")
        else:
            print(f"\n{target:.0%} accuracy is not reachable at any coverage on this split")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "rq6_real_class_mix.csv"
    # An append only makes sense onto a file with the same columns. When the header predates
    # a column, the old rows are carried over with it filled in rather than silently mixed.
    exists = p.exists()
    if exists:
        with p.open(newline="", encoding="utf-8") as fh:
            prior = list(csv.DictReader(fh))
        if prior and set(prior[0]) != set(records[0]):
            with p.open("w", newline="", encoding="utf-8") as fh:
                w0 = csv.DictWriter(fh, fieldnames=list(records[0]))
                w0.writeheader()
                for row in prior:
                    w0.writerow({k: row.get(k, "") for k in records[0]})
    with p.open("a" if exists else "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        if not exists:
            w.writeheader()
        w.writerows(records)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
