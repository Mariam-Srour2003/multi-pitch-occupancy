"""H3's cross-venue numbers with the gates in the loop, not the probe alone (A20).

Every cross-venue figure this project reports describes **the probe**. The deployed path has
not been the probe alone since A14: `run_slot` classifies, then a motion gate and a person gate
may overrule the verdict, and it is the overruled verdict that reaches a user. So the headline
failure - **false-play 0.6173 across held-out venues, even pruned to distinct scenes** - has
never been measured on the thing that actually runs.

It is worth measuring precisely because the gates were built for this error and cannot be shown
to fix it anywhere else: venue_01 camera B has false-play 0.0000 already, so RQ6 and A16 both
came back "0 verdicts overruled" (A16, A18). H3 is the one protocol whose false-play is large
enough for a gate to have something to do.

**What is applied and what is not.** The person gate and its A18 extension, which need one
frame. The motion gate is **not** applied: it compares a frame to the previous minute from the
same camera, and these are single frames from a manifest, not a sequence. So this is a lower
bound on what the deployed path does - the motion gate can only remove further play verdicts.

**The detections are the cached ones** from `results/ball_detection_rule.csv`, same detector,
same imgsz, same confidences, same boundaries. Re-running them would produce the same numbers
and take an hour; what is lost is the chance for this script to disagree with that file, which
is why it fails loudly if a frame it needs is missing rather than quietly scoring fewer.

    uv run python experiments/h3_with_gates.py --backbone dinov2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pitch_occupancy.data.feature_cache import load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, distinct_rows, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe
from pitch_occupancy.vision.people import PersonGate

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
SEED = 42


def load_detections() -> dict[str, tuple[int, bool]]:
    """file -> (people inside the boundary, a ball was seen inside it)."""
    path = RESULTS / "ball_detection_rule.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Run:\n"
            f"    uv run python experiments/ball_detection_rule.py --clip-venues"
        )
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["file"]: (int(r["people"]), int(r["balls"]) > 0)
                for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    args = ap.parse_args()

    every = read_manifest(DATASET / "manifest.csv")
    cached = load_cache(args.backbone, CACHE)
    feats = {f: v for f, v in zip(cached.files, cached.features, strict=False)}
    seen = load_detections()
    gate = PersonGate()

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    dev = [r for r in development_rows(every) if r.file in feats]
    control = [r for r in every
               if r.source != "synthetic" and r.venue == "venue_01"
               and cam(r) == "camera_B" and r.class3 == EMPTY and r.file in feats]
    cf = {r.file for r in control}
    dev = [r for r in dev if r.file not in cf]
    folds = [f for f in leave_one_group_out(dev, group_key="venue")
             if not f.name.endswith("venue_01")]

    missing = [r.file for r in control if r.file not in seen]
    print(f"H3: {len(folds)} folds | false-play control {len(control)} recorded EMPTY frames")
    if missing:
        raise SystemExit(f"{len(missing)} control frames have no cached detection, "
                         f"e.g. {missing[0]} - rerun ball_detection_rule.py")

    def gated(rows_, pred: list[str]) -> list[str]:
        """Apply the person gate to a probe's verdicts, from the cached detections.

        The gate's own logic is not reimplemented here - `PersonGate.inspect` is called with a
        stand-in `Counted`, so if the rule changes this experiment changes with it. A rule
        copied into an experiment is a rule that silently stops matching the system.
        """
        import unittest.mock

        from pitch_occupancy.data.taxonomy import Class3
        from pitch_occupancy.vision import people

        out = []
        for r, p in zip(rows_, pred, strict=True):
            if r.file not in seen:
                out.append(p)
                continue
            n, ball = seen[r.file]
            counted = people.Counted(people=n, ball=ball)
            # `inspect` would run the detector; the detection is already in hand, so the
            # decision is taken from the same code path with only that pass stubbed out.
            with unittest.mock.patch.object(people, "detect_inside",
                                            lambda *_a, _c=counted, **_k: _c):
                state, _ = gate.inspect(Class3(p), None, None)
            out.append(state.value)
        return out

    def score(rows_, pred: list[str]) -> tuple[float, float]:
        truth = [r.class3 for r in rows_]
        n_e = sum(1 for t in truth if t == EMPTY)
        n_p = len(truth) - n_e
        rec = (sum(p == PLAY for p, t in zip(pred, truth, strict=True) if t == PLAY) / n_p
               if n_p else float("nan"))
        fp = (sum(p == PLAY for p, t in zip(pred, truth, strict=True) if t == EMPTY) / n_e
              if n_e else float("nan"))
        return rec, fp

    records = []
    print(f"\n{'arm':<22}{'play-recall':>13}{'false-play':>13}{'balanced':>11}")
    for prune in (False, True):
        for use_gate in (False, True):
            recs, fps = [], []
            for fold in folds:
                train = distinct_rows(list(fold.train)) if prune else list(fold.train)
                if len({r.class3 for r in train}) < 2:
                    continue
                probe = LinearProbe(args.backbone, seed=SEED).fit(
                    np.stack([feats[r.file] for r in train]), train)
                test = list(fold.test)
                p_test = list(probe.predict(np.stack([feats[r.file] for r in test]), test))
                p_ctrl = list(probe.predict(
                    np.stack([feats[r.file] for r in control]), control))
                if use_gate:
                    p_test = gated(test, p_test)
                    p_ctrl = gated(control, p_ctrl)
                recs.append(score(test, p_test)[0])
                fps.append(score(control, p_ctrl)[1])
            mr, mf = float(np.mean(recs)), float(np.mean(fps))
            name = f"{'pruned' if prune else 'full'}, {'gated' if use_gate else 'probe'}"
            print(f"{name:<22}{mr:>13.4f}{mf:>13.4f}{mr - mf:>11.4f}")
            records.append({"arm": name, "pruned": prune, "gated": use_gate,
                            "play_recall": round(mr, 4), "false_play": round(mf, 4),
                            "balanced": round(mr - mf, 4)})

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "h3_with_gates.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {out}")
    print("\nthe motion gate is not applied - these are single frames, not a sequence - so "
          "the gated rows are a lower bound on what the deployed path does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
