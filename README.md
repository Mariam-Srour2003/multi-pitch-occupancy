# Pilot — Model Selection Bake-Off

> **This branch is the first step of the project, kept as a reference.**
>
> Its only purpose was to answer one question: **which vision models are worth building the
> real system on?** It extracted and labelled a dataset from the recorded footage, ran seven
> candidate models through one shared harness, and produced the leaderboard that selected the
> four models the thesis carries forward.
>
> That question is answered — see [Results](#results). The real implementation starts fresh on
> `main`, following `TODO.md`. **Nothing here is production code**, and the numbers below are a
> feasibility signal, not thesis results (see [Honest reading](#honest-reading-of-these-numbers)).

---

## What this pilot produced

| Output | Where | Used for |
|---|---|---|
| 1,296 hand-labelled frames | `data/dataset/` *(not in git)* | Training the classifier heads |
| Model leaderboard, 7 candidates | `results/leaderboard.csv` | Choosing the four models to carry forward |
| 3 trained logistic-regression heads | `results/head_*.pkl` *(not in git)* | End-to-end pipeline runs |
| ROI polygons per camera | `config/cameras.json` *(not in git)* | Masking out adjacent pitches |
| Two slots evaluated end to end | `data/db/pitch_monitor.db` *(not in git)* | Proving the full pipeline works |

Data, model weights and trained heads are excluded from the repository — see
[Reproducing this](#reproducing-this).

---

## Requirements

Python 3.12, CPU only — no GPU is needed or used anywhere in this project.

```bash
pip install -r requirements.txt
```

Main dependencies: `torch` (CPU build), `transformers`, `open_clip_torch`, `opencv-python`,
`scikit-learn`, `pandas`, `pillow`, `ultralytics`.

Backbones download from HuggingFace on first use (~2 GB total) and cache in
`~/.cache/huggingface/hub`.

---

## How to run it

The tools are meant to run in this order. Steps 1–3 build the dataset, step 4 is the bake-off
itself, step 5 proves the end-to-end pipeline.

### 1. Extract frames from the videos

Samples one frame every 15 s from each video in `data/`, plus extra "high-motion" frames mined
with MOG2 background subtraction so the playing class gets diverse poses rather than mostly
mid-field standing.

```bash
python tools/extract_frames.py                      # defaults: 15 s interval, motion mining on
python tools/extract_frames.py --interval 10        # denser sampling
python tools/extract_frames.py --no-motion          # regular interval only
```

Output: `data/dataset/unlabeled/<camera_tag>/<camera_tag>_tXXXXXX.jpg`
(`XXXXXX` = timestamp in seconds within the video).

### 2. Label the frames

Shows each unlabelled frame; one keypress files it into a class folder. Progress is appended to
`data/dataset/labels.csv`, so you can stop and resume at any time.

```bash
python tools/label_tool.py --shuffle
```

| Key | Class |
|---|---|
| `1` | `1_empty` |
| `2` | `2_playing` — match, warm-up, drills, academy training |
| `3` | `3_people_not_playing` — people inside the ROI but idle, strolling, photos |
| `4` | `4_maintenance` — hi-vis staff, brooms, mowers, line painting |
| `s` | skip (shown again next run) · `u` undo · `d` delete (blurry/broken) · `q` quit |

Optional — build contact sheets (8×6 grids of 320×180 thumbnails) to review a camera's frames
quickly before labelling them individually:

```bash
python tools/make_sheets.py <camera_folder_name>
```

### 3. Draw the ROI polygon for each camera

**This step is not optional.** The footage shows neighbouring pitches, benches, walkways and
parking; without masking, a classifier happily fires "playing" on the *adjacent* field. Polygons
are stored normalised to 0–1 so they survive any resolution change.

```bash
python tools/draw_roi.py                                      # iterate all camera folders
python tools/draw_roi.py --image path/to/frame.jpg --name camA
```

Left-click adds a point, right-click removes the last. `ENTER` saves and moves to the next camera,
`r` resets, `v` previews the mask, `q` quits without saving. Written to `config/cameras.json`.

### 4. Run the bake-off

Every candidate gets the same ROI-masked frames and the same split, and appends one row to
`results/leaderboard.csv`: accuracy, macro-F1, per-class precision/recall, CPU ms/frame.

```bash
python tools/benchmark.py --family zeroshot     # CLIP / SigLIP / OpenCLIP, no training needed
python tools/benchmark.py --family embed        # frozen backbones + logistic-regression heads
python tools/benchmark.py --family all --split slot   # honest split: train on day, test on night
python tools/benchmark.py --limit 200           # quick pass on a subset
python tools/benchmark.py --models zs:openai/clip-vit-base-patch32   # one specific model
python tools/benchmark.py --no-roi              # ablation: skip ROI masking
```

**Two model families:**
- `zeroshot` — CLIP / SigLIP / OpenCLIP prompted with class descriptions, no training.
- `embed` — frozen backbone → mean-pooled embedding → logistic-regression head (trains in seconds).
  Heads are saved to `results/head_<model>.pkl` for reuse by the engine.

**Two split strategies:**
- `--split random` — stratified 80/20. Fast sanity check, **but frames are correlated**, so it
  flatters every model. This is the split the headline pilot numbers below come from.
- `--split slot` — train on one recorded slot day, test on the other. Honest and cross-lighting
  (day vs night).

### 5. End-to-end slot evaluation

For each slot in `data/` (a pair of videos = cameras A and B of one field): sample 1 frame/min
from both, classify, fuse the two halves, aggregate to a verdict, pick three evidence images,
and persist everything to SQLite.

```bash
python tools/run_slot_eval.py                                      # default model from config
python tools/run_slot_eval.py --models convnextv2 vit dinov2 openclip   # compare several
```

Writes `frame_samples` and `slot_evaluations` to `data/db/pitch_monitor.db`, and three evidence
JPEGs per slot per model to `data/evidence_cache/`.

---

## How the pieces fit

```
data/*.mp4
    │
    ├─ extract_frames.py ──→ data/dataset/unlabeled/
    │                              │
    │                        label_tool.py ──→ data/dataset/{1_empty,2_playing,...}
    │                                                │
    │        draw_roi.py ──→ config/cameras.json     │
    │                              │                 │
    │                              └────────┬────────┘
    │                                       ▼
    │                                 benchmark.py ──→ results/leaderboard.csv
    │                                                  results/head_*.pkl
    │                                       │
    └───────────────────────────────────────┤
                                            ▼
                                    run_slot_eval.py
                        (frame_source → vision_pipeline → fusion
                         → slot_aggregator → SQLite + evidence)
```

`engine/` holds the runtime pieces that `run_slot_eval.py` drives:

| Module | Role |
|---|---|
| `frame_source.py` | Serves frames from the videos as if they were a live camera |
| `vision_pipeline.py` | Classifier registry (`embed_head` + `zeroshot`), pooling-stamp assertion |
| `roi.py` | Polygon masking from `config/cameras.json` |
| `fusion.py` | Combines the two camera halves into one field state |
| `slot_aggregator.py` | Ratios → USED / NOTUSED / REVIEW + top-3 evidence selection |
| `db.py` | SQLite schema (fields, cameras, rental_slots, frame_samples, slot_evaluations) |

---

## Results

Seven models, 1,296 frames, stratified random 80/20 split (1,036 train / 260 test, seed 42):

| # | Model | Trained on our data | Accuracy | ms/frame | Outcome |
|---|---|---|---|---|---|
| 1 | ViT-Base + head | head only | **99.2%** | 203 | **Carried forward** — accuracy reference |
| 2 | DINOv2-Base + head | head only | **98.5%** | 254 | **Carried forward** — robustness reference |
| 3 | ConvNeXtV2-Tiny + head | head only | **98.1%** | **102** | **Carried forward** — production lead |
| 4 | OpenCLIP B/32 | no — zero-shot | **75.8%** | 247 | **Carried forward** — cold-start baseline |
| 5 | CLIP B/32 | no — zero-shot | 65.4% | 212 | Dropped |
| 6 | CLIP L/14 | no — zero-shot | 35.4% | 1,404 | Dropped |
| 7 | SigLIP2-Base | no — zero-shot | 31.9% | 1,035 | Dropped |

**End-to-end:** all three trained heads returned the correct verdict on both recorded slots
(morning `NOTUSED`, 97% empty; night `USED`, 100% playing). OpenCLIP returned `REVIEW` for the
morning slot — an acceptable escalation rather than a wrong answer.

### What the pilot decided

- **ConvNeXtV2-Tiny is the production lead** — statistically tied with ViT on accuracy at half the
  latency and a third of the parameters. On a CPU serving 20–30 cameras, that is the whole game.
- **A frozen backbone plus a logistic-regression head is enough.** No fine-tuning, no GPU, and the
  head retrains in seconds on cached features.
- **Zero-shot is a cold-start option, not a solution** — a ~22-point gap to the trained heads.
- **Model size does not predict transfer.** OpenCLIP B/32 beat architecturally *identical* OpenAI
  CLIP B/32 by ~10 points, purely from richer training data, while the much larger CLIP L/14 and
  SigLIP2 collapsed on foggy night CCTV. Leaderboard reputation did not survive contact with this
  domain.

---

## Honest reading of these numbers

These are a **feasibility signal, not thesis results**. Three reasons, all of which the main
project is designed to fix:

1. **The split leaks.** Frames sampled 15 s apart from the same video are near-identical, and the
   random split puts them on both sides. Real evaluation needs grouped (venue × day × slot) and
   leave-one-venue-out splits.
2. **Two classes are starved.** The held-out set had 6 samples of `people_not_playing` and **0** of
   `maintenance`. That is why macro-F1 sits near 0.66 despite 98–99% accuracy. The accuracy number
   is essentially a two-class number wearing a four-class label.
3. **One venue, two days.** Nothing here demonstrates generalisation to another facility, another
   camera angle, or another season.

## Gotchas worth keeping

1. **Mean pooling, never `pooler_output`.** Heads are trained on mean-pooled frozen features
   (`last_hidden_state.mean(dim=1)` for transformers, `.mean(dim=(2,3))` for convnets). HF's
   `pooler_output` is randomly initialised on plain ViT — using it produced a **false 38% score**
   (visible as the last row of `results/leaderboard.csv`) and a silent pipeline failure. Every
   saved head carries `pooling:"mean"` and `vision_pipeline.py` asserts it. Change the feature
   extraction and you must retrain every head and bump the stamp.
2. **If a result shocks you, suspect the harness before the model.** See above — that is how the
   38% was caught.
3. **`results/leaderboard.csv` may be locked** if it is open in Excel. `benchmark.py` falls back to
   `leaderboard_new.csv` rather than crashing; merge the two afterwards.
4. **The reference screenshots have their labels burned into the pixels** in red text. They cannot
   be used to evaluate any vision-language model without cropping or inpainting first — the model
   can literally read the answer.
5. **The `(1).mp4` ↔ camera-side mapping is not consistent across days.** Identify cameras by what
   they show, not by filename order.

---

## Reproducing this

The repository holds code only. To re-run the pilot you need the source footage placed in `data/`,
after which steps 1–5 above regenerate every artefact — the dataset, the ROI polygons, the trained
heads, the leaderboard and the database.

Footage is not distributed with the repository: it shows identifiable people at a client facility.
See the data-release decision task (`WP1-T5`) in `TODO.md` on `main`.
