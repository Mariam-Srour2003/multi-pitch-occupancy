"""Does a motion feature add anything the frozen backbone does not already have? (A14)

`motion_cue_probe` showed the cue separates PLAY from EMPTY at AUC 0.864 even at the
deployment sampling rate. That is separation *on its own*. It is a different question whether
it adds anything **on top of** 768 backbone dimensions, and the answer could easily be no: a
frame with people moving on it may already look different enough that the appearance features
carry the same information.

So: the same probe, the same split, one extra column.

- **without** - 768 backbone dims, as everywhere else in this project
- **with** - 769 dims, the motion scalar standardised on the training side and appended

Train on venue_01 camera A, test on venue_01 camera B. Both cameras are real and the split is
by physical camera, which is what `class_balancing._false_play_control` uses and for the same
reason: venue_01 has only two slots, so a slot-grouped split leaves too few EMPTY frames to
measure against.

Reported together, because either alone can be gamed on this data:

- **macro-F1** over the classes present in the test set
- **play-recall** and **false-play**, and `balanced` = recall - false-play

**Frames without a motion value are dropped, not imputed.** A frame with no earlier partner
has no motion measurement, and filling one in - zero, or the mean - would be inventing the
feature's value for exactly the frames it cannot describe. The restriction is reported.

    uv run python experiments/motion_feature_ablation.py --gap 60
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from pitch_occupancy.data.feature_cache import load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
SIZE = (160, 90)
SEED = 42


def _gray(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L").resize(SIZE), dtype=np.float32)


def motion_cue(rows, target_gap: int) -> dict[str, float]:
    """Mean absolute difference against the earlier frame closest to `target_gap` seconds."""
    seq: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        if str(r.t_s).isdigit():
            seq[(r.slot_id, r.camera)].append(r)
    for k in seq:
        seq[k].sort(key=lambda r: int(r.t_s))

    grays: dict[str, np.ndarray | None] = {}

    def gray(r):
        if r.file not in grays:
            p = DATASET / r.file
            grays[r.file] = _gray(p) if p.exists() else None
        return grays[r.file]

    lo, hi = target_gap * 0.6, target_gap * 1.6
    out: dict[str, float] = {}
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
                    best, best_err = members[j], err
            if best is None:
                continue
            a, b = gray(best), gray(r)
            if a is None or b is None:
                continue
            out[r.file] = float(np.abs(b - a).mean())
    return out


def _macro_f1(truth, pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(truth, pred, average="macro", zero_division=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap", type=int, default=60, help="target gap in seconds")
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument(
        "--train-set", choices=("camera_A", "camera_A+clip"), default="camera_A",
        help="camera_A is the saturated split; camera_A+clip is the one that collapses",
    )
    args = ap.parse_args()

    every = [r for r in read_manifest(DATASET / "manifest.csv") if r.source != "synthetic"]
    cue = motion_cue(every, args.gap)

    cached = load_cache(args.backbone, CACHE)
    feats = {f: v for f, v in zip(cached.files, cached.features)}

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    usable = [r for r in every
              if r.venue == "venue_01" and r.file in cue and r.file in feats
              and r.class3 in (EMPTY, PLAY)]
    train = [r for r in usable if cam(r) == "camera_A"]
    test = [r for r in usable if cam(r) == "camera_B"]

    if args.train_set == "camera_A+clip":
        # The setting where appearance alone fails: adding the all-ACTIVE_PLAY clip venues
        # took false-play on these same test frames from 0.31 to 0.9959
        # (`a13_false_play_repair`). If the motion scalar is worth a dimension anywhere, it
        # is here - it is computed from two frames of one camera and so carries nothing about
        # which venue the scene is, which is exactly what misleads the appearance features.
        clip = [r for r in every
                if r.venue != "venue_01" and r.file in cue and r.file in feats
                and r.class3 in (EMPTY, PLAY)]
        print(f"adding {len(clip)} clip-venue frames with a motion value")
        train = train + clip

    v01 = [r for r in every if r.venue == "venue_01" and r.class3 in (EMPTY, PLAY)]
    print(f"\nvenue_01 frames in EMPTY/PLAY: {len(v01)}")
    print(f"with a motion value at ~{args.gap}s: {len(usable)} "
          f"({len(usable) / len(v01):.0%}) - the rest are dropped, not imputed")
    print(f"train (camera A) {len(train)}   test (camera B) {len(test)}")
    for name, s in (("train", train), ("test", test)):
        n_e = sum(1 for r in s if r.class3 == EMPTY)
        print(f"  {name}: EMPTY {n_e}  PLAY {len(s) - n_e}")
    if not train or not test or len({r.class3 for r in train}) < 2:
        print("\nsplit is degenerate - nothing to measure")
        return 1

    m_tr = np.array([cue[r.file] for r in train], dtype=np.float32)
    mu, sd = float(m_tr.mean()), float(m_tr.std() or 1.0)

    def design(rows_, with_motion: bool) -> np.ndarray:
        X = np.stack([feats[r.file] for r in rows_])
        if not with_motion:
            return X
        m = np.array([[(cue[r.file] - mu) / sd] for r in rows_], dtype=np.float32)
        return np.hstack([X, m])

    truth = [r.class3 for r in test]
    n_empty = sum(1 for t in truth if t == EMPTY)
    n_play = len(truth) - n_empty

    records = []
    print()
    for label, wm in (("without motion (768)", False), ("with motion (769)", True)):
        probe = LinearProbe(args.backbone, seed=SEED).fit(design(train, wm), train)
        pred = probe.predict(design(test, wm), test)
        f1 = _macro_f1(truth, pred)
        recall = sum(p == PLAY for p, t in zip(pred, truth) if t == PLAY) / n_play
        fp = sum(p == PLAY for p, t in zip(pred, truth) if t == EMPTY) / n_empty
        print(f"  {label:<22} macro-F1 {f1:.4f}   play-recall {recall:.4f}   "
              f"false-play {fp:.4f}   balanced {recall - fp:+.4f}")
        records.append({"gap_s": args.gap, "backbone": args.backbone, "arm": label,
                        "n_train": len(train), "n_test": len(test),
                        "macro_f1": round(f1, 4), "play_recall": round(recall, 4),
                        "false_play": round(fp, 4), "balanced": round(recall - fp, 4)})

    # Third arm: motion as a *rule*, which is what was actually proposed - "nothing moved,
    # so not a match" - rather than as one more number for the probe to weigh. A single
    # standardised scalar among 768 highly predictive dimensions is shrunk to nothing by the
    # regulariser; an override sits outside the probe and cannot be shrunk.
    #
    # The threshold is chosen on the TRAINING frames only, as the value maximising
    # (recall - false-play) there, and then applied unchanged to the test side.
    m_tr_raw = np.array([cue[r.file] for r in train])
    tr_truth = [r.class3 for r in train]
    base_tr = LinearProbe(args.backbone, seed=SEED).fit(design(train, False), train)
    tr_pred = base_tr.predict(design(train, False), train)
    best_thr, best_bal = 0.0, -2.0
    for thr in np.quantile(m_tr_raw, np.linspace(0.01, 0.60, 60)):
        adj = [EMPTY if (p == PLAY and m < thr) else p
               for p, m in zip(tr_pred, m_tr_raw)]
        ne = sum(1 for t in tr_truth if t == EMPTY)
        np_ = len(tr_truth) - ne
        if not ne or not np_:
            break
        rc = sum(a == PLAY for a, t in zip(adj, tr_truth) if t == PLAY) / np_
        fpr = sum(a == PLAY for a, t in zip(adj, tr_truth) if t == EMPTY) / ne
        if rc - fpr > best_bal:
            best_thr, best_bal = float(thr), rc - fpr

    m_te = np.array([cue[r.file] for r in test])
    base_probe = LinearProbe(args.backbone, seed=SEED).fit(design(train, False), train)
    raw = base_probe.predict(design(test, False), test)
    over = [EMPTY if (p == PLAY and m < best_thr) else p for p, m in zip(raw, m_te)]
    f1 = _macro_f1(truth, over)
    recall = sum(p == PLAY for p, t in zip(over, truth) if t == PLAY) / n_play
    fp = sum(p == PLAY for p, t in zip(over, truth) if t == EMPTY) / n_empty
    print(f"  {'motion override':<22} macro-F1 {f1:.4f}   play-recall {recall:.4f}   "
          f"false-play {fp:.4f}   balanced {recall - fp:+.4f}   (thr={best_thr:.3f} from train)")
    records.append({"gap_s": args.gap, "backbone": args.backbone, "arm": "motion override",
                    "n_train": len(train), "n_test": len(test), "macro_f1": round(f1, 4),
                    "play_recall": round(recall, 4), "false_play": round(fp, 4),
                    "balanced": round(recall - fp, 4)})

    d = records[1]["macro_f1"] - records[0]["macro_f1"]
    db = records[1]["balanced"] - records[0]["balanced"]
    print(f"\nmotion adds {d:+.4f} macro-F1 and {db:+.4f} balanced")
    if abs(d) < 0.005 and abs(db) < 0.005:
        print(
            "\nNo material change. The backbone already carries whatever the motion scalar\n"
            "knows on this split, and one extra dimension does not add to 768. A cue that\n"
            "separates on its own is not the same as a cue that contributes."
        )

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "motion_feature_ablation.csv"
    write_header = not p.exists()
    with p.open("a" if p.exists() else "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        if write_header:
            w.writeheader()
        w.writerows(records)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
