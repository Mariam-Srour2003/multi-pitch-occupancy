"""Choose the detector the rule stands on, by a rule written before the numbers (A36, WP9-T2).

The detector-first path (A36) counts people inside the boundary and decides from the count,
so the detector's counting error is the method's error, and "which detector" is the one
model-selection question the rebuild has. It is answered here with the selection rule
registered in the amendment *before* this ran:

    Among the candidates whose round - 30 cameras x 3 burst frames - fits in 30 s on this
    machine, pick the best rate of |detected - truth| <= 1 on the hand-counted frames; break
    ties on the false-person rate on venue_01's recorded EMPTY frames; then on latency. A
    segmentation variant is used only for the one masks pass per minute, if its own latency
    fits; otherwise the overlay falls back to boxes.

Written down first because the alternative - looking at fourteen rows and choosing - is the
after-the-fact selection every gate in this project exists to prevent.

**Three measurements per (model, input size, tiling):**

- **count agreement** against `results/hand_counts.csv` (`file, people_inside,
  ball_visible`): MAE, the |Δ| ≤ 1 rate, recall of "≥ 5 when the truth is ≥ 5" and of "0 when
  the truth is 0", overall and per venue. Without that file this section is skipped and the
  audit says so - a selection made on latency alone is a selection made blind, and the CSV
  records that it was.
- **false persons on EMPTY**: the share of venue_01's recorded EMPTY frames with ≥ 1 person
  detected inside the boundary, and the height distribution of those detections, which is
  what sets the minimum-height filter (WP9-T5). And **ball recall** on recorded play frames
  per venue, with the false-ball rate on the empties beside it (A17's 0.06-0.89 table).
- **latency**: median and p95 per frame on an idle machine, warm-up discarded, one thread
  of work at a time, and the round that implies. `machine_dependent`, like every timing here:
  never inside a batch (`efficiency_latency.py` documents why), never the deployment claim
  (WP7-T1 is the Mini-PC).

    uv run python experiments/detector_audit.py                       # every candidate at 1280
    uv run python experiments/detector_audit.py --keys yolov8n yolo11n --imgsz 960 1280 --tiles 1 2
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.vision import roi
from pitch_occupancy.vision.counting import count_inside
from pitch_occupancy.vision.detector import DETECTORS, Detector
from pitch_occupancy.vision.people import BALL_CONFIDENCE, DETECT_CONFIDENCE

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
#: Tracked, not under `data/`: it holds frame names and numbers, no pixels, and a truth file
#: that a clone loses is a selection nobody can re-derive. `scripts/sample_hand_counts.py`
#: chooses the frames; a person fills the counts in.
HAND_COUNTS = RESULTS / "hand_counts.csv"
EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"

#: The round the selection rule is stated against: the busiest deployment the thesis names,
#: every camera read as a burst, and half the cycle left for everything else.
CAMERAS, BURST, BUDGET_S = 30, 3, 30.0
SEED = 42


def _timed(fn, frames, *, warmup: int = 2) -> tuple[float, float]:
    """(median ms, p95 ms) over the frames, warm-up discarded - `gate_latency.timed`."""
    for f in frames[:warmup]:
        fn(f)
    took = []
    for f in frames:
        start = time.perf_counter()
        fn(f)
        took.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(took)), float(np.percentile(took, 95))


def _hand_counts() -> dict[str, tuple[int, bool]]:
    """``file -> (people_inside, ball_visible)`` from the hand-count audit, or empty."""
    if not HAND_COUNTS.exists():
        return {}
    out = {}
    with HAND_COUNTS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["file"]] = (int(row["people_inside"]),
                                    str(row.get("ball_visible", "")).strip().lower()
                                    in {"1", "true", "yes", "y"})
            except (KeyError, ValueError):
                continue
    return out


def _sample(rows, per_group: int, key) -> list:
    """Up to ``per_group`` rows per ``key(row)``, drawn with the fixed seed."""
    rng = random.Random(SEED)
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    chosen = []
    for _, members in sorted(groups.items()):
        members = sorted(members, key=lambda r: r.file)
        rng.shuffle(members)
        chosen += members[:per_group]
    return chosen


def _load(row) -> np.ndarray | None:
    return cv2.imread(str(DATASET / row.file))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", nargs="*", default=None, help="detector keys; default all")
    ap.add_argument("--imgsz", nargs="*", type=int, default=[1280])
    ap.add_argument("--tiles", nargs="*", type=int, default=[1])
    ap.add_argument("--empties", type=int, default=60,
                    help="recorded EMPTY frames per venue_01 camera (0 = all)")
    ap.add_argument("--plays", type=int, default=15,
                    help="recorded play frames per venue for ball recall (0 = all)")
    ap.add_argument("--latency-frames", type=int, default=12)
    ap.add_argument("--conf", type=float, default=DETECT_CONFIDENCE)
    ap.add_argument("--ball-conf", type=float, default=BALL_CONFIDENCE)
    args = ap.parse_args()

    keys = args.keys or sorted(DETECTORS)
    unknown = sorted(set(keys) - set(DETECTORS))
    if unknown:
        raise SystemExit(f"unknown detector(s) {unknown}; known {sorted(DETECTORS)}")

    rows = [r for r in read_manifest(DATASET / "manifest.csv") if r.source != "synthetic"]
    by_file = {r.file: r for r in rows}

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    truth = _hand_counts()
    hand_rows = [by_file[f] for f in sorted(truth) if f in by_file]
    empties = [r for r in rows if r.venue == "venue_01" and r.class3 == EMPTY]
    if args.empties:
        empties = _sample(empties, args.empties, cam)
    plays = [r for r in rows if r.class3 == PLAY]
    if args.plays:
        plays = _sample(plays, args.plays, lambda r: r.venue)
    latency_frames = [f for f in (_load(r) for r in rows[: args.latency_frames * 3])
                      if f is not None][: args.latency_frames]

    print(f"hand counts: {len(hand_rows)} frame(s)"
          + ("" if hand_rows else "  - NONE: the selection below is blind to counting "
                                  "accuracy and the CSV says so"))
    print(f"empties: {len(empties)} recorded venue_01 EMPTY frames; play frames for the "
          f"ball: {len(plays)} over {len({r.venue for r in plays})} venues")
    print(f"latency on {len(latency_frames)} frames; round = {CAMERAS} cameras x {BURST} "
          f"frames against {BUDGET_S:.0f} s\n")

    # Frames are decoded once and reused across configurations.
    frames: dict[str, np.ndarray] = {}
    for r in hand_rows + empties + plays:
        if r.file not in frames:
            img = _load(r)
            if img is not None:
                frames[r.file] = img
    polygons = {r.file: roi.resolve(r.camera) for r in hand_rows + empties + plays}

    audit_rows: list[dict] = []
    latency_rows: list[dict] = []
    heights: dict[str, list[int]] = defaultdict(list)

    for key in keys:
        det = Detector.load(key)
        if not det.available:
            print(f"{key:<14} weights missing - run `uv run pitch fetch-weights {key}`; skipped")
            continue
        for imgsz in args.imgsz:
            for tiles in args.tiles:
                label = f"{key} @{imgsz} x{tiles}"
                t0 = time.perf_counter()

                def run(frame, *, masks=False, _det=det, _imgsz=imgsz, _tiles=tiles):
                    return _det.detect(frame, confidence=min(args.conf, args.ball_conf),
                                       imgsz=_imgsz, tiles=_tiles, masks=masks)

                # --- latency, first and alone, so the counting below does not warm anything
                med, p95 = _timed(run, latency_frames)
                round_s = CAMERAS * BURST * med / 1000.0
                seg_med = seg_p95 = None
                if det.spec.segment:
                    seg_med, seg_p95 = _timed(lambda f: run(f, masks=True), latency_frames)
                latency_rows.append({
                    "detector": key, "imgsz": imgsz, "tiles": tiles,
                    "median_ms": round(med, 1), "p95_ms": round(p95, 1),
                    "masks_median_ms": None if seg_med is None else round(seg_med, 1),
                    "masks_p95_ms": None if seg_p95 is None else round(seg_p95, 1),
                    "round_s": round(round_s, 2), "fits_budget": round_s <= BUDGET_S,
                    "cameras": CAMERAS, "burst": BURST, "budget_s": BUDGET_S,
                })

                # --- counts
                def counted(r):
                    found = run(frames[r.file])
                    if found is None:
                        return None
                    return count_inside(found, polygons[r.file], frames[r.file].shape[:2],
                                        person_conf=args.conf, ball_conf=args.ball_conf)

                # hand counts
                deltas, ge5_hits, ge5_n, zero_hits, zero_n = [], 0, 0, 0, 0
                per_venue: dict[str, list[int]] = defaultdict(list)
                for r in hand_rows:
                    if r.file not in frames:
                        continue
                    c = counted(r)
                    if c is None:
                        continue
                    people_truth, _ball_truth = truth[r.file]
                    delta = c.people_inside - people_truth
                    deltas.append(delta)
                    per_venue[r.venue].append(abs(delta) <= 1)
                    if people_truth >= 5:
                        ge5_n += 1
                        ge5_hits += c.people_inside >= 5
                    if people_truth == 0:
                        zero_n += 1
                        zero_hits += c.people_inside == 0

                # empties: false persons, and the heights of what was found
                false_person = 0
                for r in empties:
                    if r.file not in frames:
                        continue
                    c = counted(r)
                    if c is None:
                        continue
                    if c.raw_inside:
                        false_person += 1
                        heights[label] += [d.height for d in c.people]
                false_ball = 0
                for r in empties:
                    if r.file in frames:
                        c = counted(r)
                        if c is not None and c.ball_seen:
                            false_ball += 1

                # play frames: ball recall per venue
                ball_hits: dict[str, list[bool]] = defaultdict(list)
                for r in plays:
                    if r.file not in frames:
                        continue
                    c = counted(r)
                    if c is None:
                        continue
                    ball_hits[r.venue].append(c.ball_seen)
                ball_recall = {v: float(np.mean(h)) for v, h in ball_hits.items() if h}

                n_hand = len(deltas)
                audit_rows.append({
                    "detector": key, "imgsz": imgsz, "tiles": tiles,
                    "n_hand": n_hand,
                    "count_mae": round(float(np.mean(np.abs(deltas))), 3) if deltas else None,
                    "within_one_rate": round(float(np.mean(np.abs(deltas) <= 1)), 4)
                    if deltas else None,
                    "ge5_recall": round(ge5_hits / ge5_n, 4) if ge5_n else None,
                    "ge5_n": ge5_n,
                    "zero_recall": round(zero_hits / zero_n, 4) if zero_n else None,
                    "zero_n": zero_n,
                    "n_empty": len(empties),
                    "false_person_rate": round(false_person / len(empties), 4) if empties else None,
                    "false_ball_rate": round(false_ball / len(empties), 4) if empties else None,
                    "n_play": sum(len(h) for h in ball_hits.values()),
                    "ball_recall_mean": round(float(np.mean(list(ball_recall.values()))), 4)
                    if ball_recall else None,
                    "ball_recall_min": round(min(ball_recall.values()), 4) if ball_recall else None,
                    "ball_recall_max": round(max(ball_recall.values()), 4) if ball_recall else None,
                    "median_ms": round(med, 1), "round_s": round(round_s, 2),
                    "fits_budget": round_s <= BUDGET_S,
                    "within_one_by_venue": ";".join(
                        f"{v}={np.mean(h):.2f}" for v, h in sorted(per_venue.items())),
                    "ball_recall_by_venue": ";".join(
                        f"{v}={x:.2f}" for v, x in sorted(ball_recall.items())),
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                })
                a = audit_rows[-1]
                within = a["within_one_rate"]
                recall = a["ball_recall_mean"]
                print(f"{label:<24} {med:>6.0f} ms  round {round_s:>5.1f} s "
                      f"{'fits' if a['fits_budget'] else 'OVER'}  "
                      f"|d|<=1 {'  -  ' if within is None else f'{within:.3f}'}  "
                      f"false-person {a['false_person_rate']:.3f}  "
                      f"ball recall {'-' if recall is None else f'{recall:.2f}'}"
                      f"  ({a['elapsed_s']:.0f} s)")

    if not audit_rows:
        print("nothing measured: no weights present. `uv run pitch fetch-weights` first.")
        return 1

    # --- the selection, by the registered rule
    fitting = [a for a in audit_rows if a["fits_budget"]]
    pool = fitting or audit_rows
    blind = all(a["within_one_rate"] is None for a in pool)

    def rank(a):
        return (-(a["within_one_rate"] or 0.0), a["false_person_rate"] or 1.0, a["median_ms"])

    chosen = sorted(pool, key=rank)[0]
    print(f"\nselection rule: fits the {BUDGET_S:.0f} s round -> best |d|<=1 on hand counts -> "
          f"lowest false-person rate on empties -> fastest")
    if not fitting:
        print("  WARNING: no configuration fits the round; ranking among all of them")
    if blind:
        print("  WARNING: no hand counts - this selection is on false-person rate and latency "
              "only, and the CSV column `selection_blind` records it")
    print(f"  chosen: {chosen['detector']} @{chosen['imgsz']} x{chosen['tiles']}")
    for a in audit_rows:
        a["chosen"] = a is chosen
        a["selection_blind"] = blind

    # --- the height distribution of false persons, for the minimum-height filter (WP9-T5)
    height_rows = []
    for label, hs in sorted(heights.items()):
        if hs:
            q = np.percentile(hs, [10, 50, 90])
            height_rows.append({"config": label, "n": len(hs), "p10": round(q[0], 1),
                                "p50": round(q[1], 1), "p90": round(q[2], 1)})
    if height_rows:
        print("\nheights of detections inside the boundary on EMPTY frames (px), by config:")
        for h in height_rows:
            print(f"  {h['config']:<24} n={h['n']:<4} p10 {h['p10']:>5} p50 {h['p50']:>5} "
                  f"p90 {h['p90']:>5}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "detector_audit.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit_rows[0]))
        w.writeheader()
        w.writerows(audit_rows)
    lat = RESULTS / "detector_latency.csv"
    with lat.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(latency_rows[0]))
        w.writeheader()
        w.writerows(latency_rows)
    if height_rows:
        hp = RESULTS / "detector_false_person_heights.csv"
        with hp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(height_rows[0]))
            w.writeheader()
            w.writerows(height_rows)
    print(f"\nwrote {out}, {lat}" + (f", {hp}" if height_rows else ""))

    record(
        "WP9-T2 detector audit",
        "uv run python experiments/detector_audit.py"
        + (f" --keys {' '.join(keys)}" if args.keys else "")
        + f" --imgsz {' '.join(map(str, args.imgsz))} --tiles {' '.join(map(str, args.tiles))}",
        "detector_audit.csv",
        f"chosen {chosen['detector']} @{chosen['imgsz']} x{chosen['tiles']}"
        + (" on false-person and latency only (no hand counts)" if blind
           else f" at |d|<=1 {chosen['within_one_rate']:.3f} on {chosen['n_hand']} "
                f"hand-counted frames")
        + f"; false-person {chosen['false_person_rate']:.3f}; {chosen['median_ms']:.0f} "
          f"ms/frame; round {chosen['round_s']:.1f} s of {BUDGET_S:.0f}; machine-dependent",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
