# Pitch Occupancy System

See [PLAN.md](PLAN.md) for the full implementation plan.

## Phase 0 workflow (dataset building)

```bash
# 1. Extract frames from the videos in data/ (1 per 15s + motion-mined extras)
python tools/extract_frames.py

# 2. Label them (keyboard: 1=empty 2=playing 3=people-not-playing 4=maintenance)
python tools/label_tool.py --shuffle

# 3. Draw the pitch ROI polygon for each camera view (excludes neighboring
#    pitches, benches, parking from all inference)
python tools/draw_roi.py
```

## Phase 2 — model bake-off

```bash
# Zero-shot models (no training needed)
python tools/benchmark.py --family zeroshot

# Embedding backbones + logistic-regression head (needs labels)
python tools/benchmark.py --family embed

# Honest cross-lighting evaluation: train on the day slot, test on the night slot
python tools/benchmark.py --family all --split slot

# Quick pass on a subset
python tools/benchmark.py --limit 200
```

Results accumulate in `results/leaderboard.csv` (sorted by macro-F1).
Trained heads are saved as `results/head_<model>.pkl` for reuse by the engine.

## Data layout

```
data/
├── *.mp4                    # StatBox replay exports (2 cameras × 2 slots)
├── ss data/                 # 14 annotated reference screenshots
└── dataset/
    ├── unlabeled/<camera_tag>/   # extracted frames awaiting labels
    ├── 1_empty/ 2_playing/ 3_people_not_playing/ 4_maintenance/
    └── labels.csv
```
