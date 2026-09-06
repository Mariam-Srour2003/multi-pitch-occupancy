"""End-to-end slot evaluation on recorded footage (VIDEO_SIM).

For each slot found in data/ (a pair of videos = camera A/B of one field):
  sample 1 frame/min from both cameras -> Tier-1 classify -> dual-camera fusion
  -> slot aggregation -> USED / NOTUSED / REVIEW + 3 evidence JPEGs
  -> persisted to SQLite (frame_samples, slot_evaluations).

Run one model or compare several:
    python tools/run_slot_eval.py                       # default model from config
    python tools/run_slot_eval.py --models convnextv2 vit dinov2 openclip
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.db import connect
from engine.frame_source import VideoSlotSource, discover_slots
from engine.fusion import fuse
from engine.roi import apply_roi, load_rois
from engine.slot_aggregator import aggregate
from engine.vision_pipeline import Classifier

CONFIG = json.loads((ROOT / "config" / "system_config.json").read_text())
EVIDENCE_DIR = ROOT / "data" / "evidence_cache"


def run_slot(model_key: str, clf: Classifier, slot_key: str, videos: dict,
             rois: dict, con) -> None:
    interval = CONFIG["SAMPLE_INTERVAL_S"]
    src = VideoSlotSource(videos, interval_s=interval)
    fused_samples = []            # (t_s, label, conf)
    frame_cache = {}              # t_s -> (cam_tag, BGR frame) of fused-state camera
    t0 = time.perf_counter()

    for s in src:
        per_cam = {}
        tags, imgs = [], []
        for tag, frame in s.frames.items():
            masked = apply_roi(frame, rois.get(tag))
            tags.append((tag, frame))
            imgs.append(Image.fromarray(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)))
        preds = clf.predict_batch(imgs)
        for (tag, _), (label, conf, _) in zip(tags, preds):
            per_cam[tag] = (label, conf)
            con.execute(
                "INSERT INTO frame_samples (field_id, slot_id, camera_id, t_s, "
                "model_key, predicted_category, confidence_score) VALUES (?,?,?,?,?,?,?)",
                ("field_01", slot_key, tag, s.t_s, model_key, label, conf))
        f_label, f_conf = fuse(per_cam)
        fused_samples.append((s.t_s, f_label, f_conf))
        # remember the frame of the camera that determined the fused state
        for (tag, frame) in tags:
            if per_cam[tag][0] == f_label:
                frame_cache[s.t_s] = (tag, frame)
                break
    src.close()

    result = aggregate(fused_samples)

    ev_paths = []
    out_dir = EVIDENCE_DIR / "field_01" / slot_key / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    for (t_s, label, conf) in result.evidence:
        tag, frame = frame_cache[t_s]
        p = out_dir / f"t{t_s:04d}_{label}_{conf:.2f}_{tag[-4:]}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        ev_paths.append(str(p.relative_to(ROOT)))
    ev_paths += [None] * (3 - len(ev_paths))

    r = result.ratios
    con.execute(
        "INSERT INTO slot_evaluations (field_id, slot_id, evaluation_date, model_key, "
        "final_status, confidence_score, decision_reason, ratio_playing, ratio_empty, "
        "ratio_people, ratio_maintenance, evidence_image_1, evidence_image_2, "
        "evidence_image_3) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("field_01", slot_key, date.today().isoformat(), model_key,
         result.status, result.confidence, result.reason,
         r.get("2_playing"), r.get("1_empty"), r.get("3_people_not_playing"),
         r.get("4_maintenance"), *ev_paths))
    con.commit()

    icon = {"USED": "USED  [#]", "NOTUSED": "NOTUSED[ ]", "REVIEW": "REVIEW [?]"}[result.status]
    dt = time.perf_counter() - t0
    print(f"  {slot_key}  ->  {icon}  conf {result.confidence:.2f}  "
          f"(play {r['2_playing']:.0%} / empty {r['1_empty']:.0%} / "
          f"people {r['3_people_not_playing']:.0%})  "
          f"[{result.n_samples} samples, {dt:.0f}s]", flush=True)
    print(f"      reason: {result.reason}", flush=True)
    for pth in [e for e in ev_paths if e]:
        print(f"      evidence: {pth}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="model keys from system_config MODELS (default: DEFAULT_MODEL)")
    args = ap.parse_args()
    keys = args.models or [CONFIG["DEFAULT_MODEL"]]

    slots = discover_slots()
    if not slots:
        sys.exit("no videos in data/")
    rois = load_rois()
    print(f"slots: {list(slots)} | ROIs: {list(rois) or 'none drawn yet'}")

    con = connect()
    for key in keys:
        print(f"\n=== model: {key} ({CONFIG['MODELS'][key]['note']}) ===", flush=True)
        clf = Classifier.from_config(key)
        for slot_key, videos in slots.items():
            run_slot(key, clf, slot_key, videos, rois, con)
    con.close()
    print("\nDone. Results in data/db/pitch_monitor.db, evidence in data/evidence_cache/")


if __name__ == "__main__":
    main()
