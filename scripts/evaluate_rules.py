"""Score the detector + rules on the labelled frames, and compare detector settings.

Two steps, so the slow part runs once per detector setting:

    uv run python scripts/evaluate_rules.py detect --detector yolo11n --imgsz 960
    uv run python scripts/evaluate_rules.py score

``detect`` runs YOLO on every labelled frame (``data/processed/manifest.csv``, synthetic frames
left out) at a low confidence floor and caches the raw detections plus the seconds each frame
took, in ``data/interim/eval_detections/<detector>@<imgsz>.json``. ``score`` replays the rule
(`vision/counting.count_inside` + `vision/rules.decide`) on every cache it finds, with the
settings in ``configs/rules.json`` or those given on the command line, and prints accuracy per
class, false play on empty frames, the worst venue and the CPU seconds per frame.

Frames are stills, so motion and the ball-in-play clause are not measured here.
``--every N`` keeps one frame in N of the venue that dominates the set, to save CPU time.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
FRAMES = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "interim" / "eval_detections"
STATES = {"C1_EMPTY": "EMPTY", "C2_ACTIVE_PLAY": "PLAY", "C3_MAINTENANCE_NON_SPORTING": "C3"}


def _rows(every: int) -> list[dict[str, str]]:
    rows = [r for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))
            if r["source"] != "synthetic" and r["class3"] in STATES]
    big = Counter(r["venue"] for r in rows).most_common(1)[0][0]
    kept, seen = [], Counter()
    for r in rows:
        if r["venue"] == big and r["class3"] != "C3_MAINTENANCE_NON_SPORTING":
            seen[r["class3"]] += 1
            if (seen[r["class3"]] - 1) % every:
                continue
        kept.append(r)
    return kept


def detect(args: argparse.Namespace) -> None:
    import cv2

    from pitch_occupancy.vision.detector import PERSON, SPORTS_BALL, VEHICLES, Detector

    detector = Detector.load(args.detector)
    out = CACHE / f"{args.detector}@{args.imgsz}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    rows = _rows(args.every)
    for i, r in enumerate(rows):
        if r["file"] in done:
            continue
        image = cv2.imread(str(FRAMES / r["file"]))
        if image is None:
            continue
        start = time.perf_counter()
        found = detector.detect(image, confidence=0.05, imgsz=args.imgsz,
                                classes=(PERSON, SPORTS_BALL, *VEHICLES))
        took = time.perf_counter() - start
        if found is None:
            raise SystemExit(f"{args.detector} did not load - is its weights file present?")
        done[r["file"]] = {"shape": list(image.shape[:2]), "seconds": round(took, 3),
                           "dets": [[d.cls, *d.box, round(d.conf, 3)] for d in found]}
        if i % 25 == 0:
            out.write_text(json.dumps(done), encoding="utf-8")
            print(f"{i + 1}/{len(rows)} {took:.2f}s", flush=True)
    out.write_text(json.dumps(done), encoding="utf-8")
    print(f"wrote {out} ({len(done)} frames)")


def _settings(args: argparse.Namespace):
    from pitch_occupancy.vision.rules import RuleConfig

    cfg = RuleConfig.load()
    overrides = {k: v for k, v in {
        "person_conf": args.person_conf, "play_min": args.play_min,
        "small_group_max": args.small_group_max, "edge_margin": args.edge_margin,
        "cluster_max": args.cluster_max, "dark_luma": args.dark_luma}.items() if v is not None}
    return replace(cfg, **overrides)


def score(args: argparse.Namespace) -> None:
    import cv2

    from pitch_occupancy.vision import roi
    from pitch_occupancy.vision.counting import count_inside
    from pitch_occupancy.vision.detector import Detection
    from pitch_occupancy.vision.pitch_classifier import brightness
    from pitch_occupancy.vision.rules import decide

    cfg = _settings(args)
    labels = {r["file"]: r for r in _rows(1)}
    caches = sorted(CACHE.glob("*.json")) if not args.cache else [CACHE / args.cache]
    print(f"rule: person_conf {cfg.person_conf}, play_min {cfg.play_min}, "
          f"small_group_max {cfg.small_group_max}, "
          f"edge_margin {cfg.edge_margin}, cluster_max {cfg.cluster_max}, "
          f"dark_luma {cfg.dark_luma}")
    lights: dict[str, float] = {}
    for path in caches:
        cache = json.loads(path.read_text(encoding="utf-8"))
        hits, totals = Counter(), Counter()
        venue_hits, venue_totals = Counter(), Counter()
        confusion: dict[str, Counter] = defaultdict(Counter)
        seconds = []
        for file, entry in cache.items():
            row = labels.get(file)
            if row is None:
                continue
            truth = STATES[row["class3"]]
            dets = [Detection(int(c), (x1, y1, x2, y2), float(p))
                    for c, x1, y1, x2, y2, p in entry["dets"]]
            polygon = roi.resolve(row["camera"], venue=row["venue"]) if row["camera"] else None
            count = count_inside(dets, polygon, tuple(entry["shape"]),
                                 person_conf=cfg.person_conf, ball_conf=cfg.ball_conf,
                                 min_height_at=cfg.min_height_at(entry["shape"][0]),
                                 edge_margin=cfg.edge_margin)
            light = None
            if cfg.dark_luma is not None:
                if file not in lights:
                    lights[file] = brightness(cv2.imread(str(FRAMES / file)), polygon)
                light = lights[file]
            verdict = decide(count, motion=None, cfg=cfg, require_boundary=False,
                             brightness=light)
            said = {"EMPTY": "EMPTY", "ACTIVE_PLAY": "PLAY"}.get(verdict.state.name, "C3")
            if verdict.state.name == "UNCERTAIN":
                said = "UNCERTAIN"
            if verdict.state.name == "MAINTENANCE_NON_SPORTING":
                said = "C3"
            confusion[truth][said] += 1
            totals[truth] += 1
            hits[truth] += said == truth
            venue_totals[row["venue"]] += 1
            venue_hits[row["venue"]] += said == truth
            seconds.append(entry["seconds"])
        n = sum(totals.values())
        if not n:
            continue
        acc = sum(hits.values()) / n
        balanced = sum(hits[k] / totals[k] for k in totals) / len(totals)
        false_play = confusion["EMPTY"]["PLAY"] / max(totals["EMPTY"], 1)
        sizable = [v for v in venue_totals if venue_totals[v] >= 10] or list(venue_totals)
        worst = min(sizable, key=lambda v: venue_hits[v] / venue_totals[v])
        seconds.sort()
        print(f"\n{path.stem}: {n} frames, accuracy {acc:.3f}, balanced {balanced:.3f}, "
              f"false play on empty {false_play:.3f}, median {seconds[len(seconds) // 2]:.2f}s"
              f"/frame")
        for k in ("EMPTY", "PLAY", "C3"):
            if totals[k]:
                print(f"  {k:5} recall {hits[k] / totals[k]:.3f}  ({dict(confusion[k])})")
        print(f"  worst venue {worst}: {venue_hits[worst] / venue_totals[worst]:.3f} "
              f"of {venue_totals[worst]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--detector", default="yolov8n")
    d.add_argument("--imgsz", type=int, default=1280)
    d.add_argument("--every", type=int, default=3)
    s = sub.add_parser("score")
    s.add_argument("--cache", help="one cache file name; default every cache")
    for name in ("person_conf", "edge_margin", "cluster_max", "dark_luma"):
        s.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float)
    for name in ("play_min", "small_group_max"):
        s.add_argument(f"--{name.replace('_', '-')}", dest=name, type=int)
    args = parser.parse_args()
    {"detect": detect, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
