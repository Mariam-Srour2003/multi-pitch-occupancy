"""Is C=1.0 the right regularisation for the probe? (WP5)

`LinearProbe` fits `LogisticRegression` at scikit-learn's default `C=1.0` and nothing in the
project has ever varied it. With 768 features and on the order of a thousand training frames
- most of them near-duplicates of each other, 95 distinct scenes in a 907-frame test set -
the penalty is doing more work than the default was chosen for.

Two numbers are reported and they answer different questions:

- **the sweep**, which says how much C matters at all. Read as a shape, not a maximum: its
  best value is chosen with the test set in view and is therefore optimistic.
- **the nested choice**, C selected by 5-fold cross-validation *inside the training set*,
  which is the only number quotable as a result. If it lands near the sweep's best, the
  selection is reliable; if far, the curve is noise.

Test set is venue_01 camera B - real EMPTY and real PLAY, both present, so macro-F1 means
something and false-play is measurable beside it.

    uv run python experiments/probe_regularisation.py --backbone dinov2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.db.seed import PHYSICAL_CAMERA

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
SEED = 42


def _fit(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=C, class_weight="balanced", random_state=SEED),
    )
    m.fit(X, y)
    return m


def _score(model, X, y):
    from sklearn.metrics import f1_score

    pred = model.predict(X)
    f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    ne = sum(1 for t in y if t == EMPTY)
    np_ = len(y) - ne
    rec = sum(p == PLAY for p, t in zip(pred, y) if t == PLAY) / np_ if np_ else float("nan")
    fp = sum(p == PLAY for p, t in zip(pred, y) if t == EMPTY) / ne if ne else float("nan")
    return f1, rec, fp


def _blank(C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=C, class_weight="balanced", random_state=SEED),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--roi-pooled", action="store_true")
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
             + [r for r in real if r.venue != "venue_01"]
             + gen_empty)
    test = [r for r in v01 if cam(r) == "camera_B"]

    Xtr = np.stack([feats[r.file] for r in train])
    ytr = [r.class3 for r in train]
    Xte = np.stack([feats[r.file] for r in test])
    yte = [r.class3 for r in test]
    print(f"\ncache: {'ROI-pooled' if args.roi_pooled else 'whole frame'} | "
          f"train {len(train)} (+{len(gen_empty)} generated EMPTY) | test {len(test)}")

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    records, best_nested, best_cv = [], None, -1.0
    print(f"\n{'C':>8}{'macro-F1':>11}{'recall':>9}{'false-play':>12}{'balanced':>11}"
          f"{'inner CV':>10}")
    for C in GRID:
        m = _fit(Xtr, ytr, C)
        f1, rec, fp = _score(m, Xte, yte)
        cv = float(np.mean(cross_val_score(
            _blank(C), Xtr, ytr,
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1_macro")))
        if cv > best_cv:
            best_cv, best_nested = cv, C
        print(f"{C:>8}{f1:>11.4f}{rec:>9.4f}{fp:>12.4f}{rec - fp:>11.4f}{cv:>10.4f}")
        records.append({"backbone": args.backbone, "roi_pooled": args.roi_pooled, "C": C,
                        "macro_f1": round(f1, 4), "play_recall": round(rec, 4),
                        "false_play": round(fp, 4), "balanced": round(rec - fp, 4),
                        "inner_cv_f1": round(cv, 4)})

    chosen = next(r for r in records if r["C"] == best_nested)
    default = next(r for r in records if r["C"] == 1.0)
    print(f"\nnested choice C={best_nested} (inner CV {best_cv:.4f}) -> "
          f"macro-F1 {chosen['macro_f1']:.4f}, balanced {chosen['balanced']:+.4f}")
    print(f"default      C=1.0                     -> "
          f"macro-F1 {default['macro_f1']:.4f}, balanced {default['balanced']:+.4f}")
    d = chosen["macro_f1"] - default["macro_f1"]
    print(f"tuning C is worth {d:+.4f} macro-F1" +
          ("  (nothing - the default is fine)" if abs(d) < 0.005 else ""))

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "probe_regularisation.csv"
    exists = p.exists()
    with p.open("a" if exists else "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        if not exists:
            w.writeheader()
        w.writerows(records)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
