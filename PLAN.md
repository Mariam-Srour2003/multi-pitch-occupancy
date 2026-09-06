# Pitch Occupancy System — Implementation Plan

Based on: *Master Technical Blueprint & System Engineering Specification* (PDF)
Corrected + grounded against the real data in `data/`.

---

## 1. Correction to the blueprint

The PDF says "20–30 cameras, possibly 3 per field." **This is wrong.**

> **Ground truth: every playground (pitch) has exactly 2 fixed cameras. Each camera sees one half of the pitch.**

This is not a detail — it changes the core decision logic:

- Classification must be made **per-field, not per-camera**, by fusing both halves.
- The labeled data proves why: `"Playing, but players is another side"` — one camera shows an
  almost-empty half **while a real match is running on the other half** (only the goalkeeper visible).
  A per-camera classifier would call that half EMPTY and be wrong at field level.
- **Fusion rule (per sampled minute):** `field_state = strongest activity across the 2 cameras`
  - PLAYING on either half ⇒ field is PLAYING
  - EMPTY requires **both** halves empty
  - Person counts are summed across halves before applying any people-based rules.
- DB schema change: replace `primary_camera_rtsp / secondary_camera_rtsp` with a proper
  `cameras` table (`camera_id, field_id, side ∈ {A, B}, rtsp_url, roi_polygon`), exactly 2 rows per field.

## 2. What we actually have (data inventory)

`data/` — 4 videos (StatBox replay exports, 1920×1080 @ 20 fps, ~60 min each):

| File | Slot | Notes |
|---|---|---|
| `StatBox_Replay_<venue>_2026-07-11_10-00.mp4` + `(1)` | Sat 10:00 (daylight) | 2 files = the 2 cameras of the same pitch/slot |
| `StatBox_Replay_<venue>_2026-07-12_20-30.mp4` + `(1)` | Sun 20:30 (night, floodlights) | 2 files = the 2 cameras of the same pitch/slot |

⇒ We effectively have **2 complete slots × 2 cameras** = ~4 hours of footage covering both lighting regimes (day + night/foggy floodlight).

`data/ss data/` — 14 annotated reference screenshots (label burned into the image in red text). Observed real-world cases:

| Case seen in screenshots | Implication |
|---|---|
| `playing` (day + night, 5-a-side with bibs) | Core BOOKED class |
| `Playing, but players is another side` (only goalkeeper in view) | **Dual-camera fusion is mandatory** |
| `Empty field` (day, night, fog/glare) | Core EMPTY class, night images have heavy lens glare |
| `Maintenance` / `maintenance` (hi-vis workers, brooms, seed spreader) | MAINTENANCE class exists in real footage |
| `ACADEMI, not a standard match` (coach + kids, cones, many balls) | Organized training — counts as **usage**, but visually ≠ match |
| `not playing, there is people inside the field` (2–3 people strolling) | People-present ≠ playing — needs its own frame class |
| `not playing, but there is people outside the field` (people on benches/behind fence) | **ROI masking is mandatory** |
| Multiple frames show the *neighboring pitch with active players* in the upper part of the frame | **ROI masking is mandatory** (a naive classifier would fire "playing" on the wrong field) |

Key engineering consequences from the screenshots:
1. **Per-camera ROI polygon** (drawn once at install time) masking out: adjacent pitches, benches, walkways, parking. All inference happens on the masked frame.
2. Frame-level taxonomy needs to be richer than the PDF's 3 classes.

## 3. Taxonomy (revised)

**Frame-level classes** (what the model predicts per snapshot per camera):

1. `EMPTY` — clear turf (any lighting/weather)
2. `PLAYING` — match, warm-up, drills, academy training (ball + athletic activity)
3. `PEOPLE_NOT_PLAYING` — people inside ROI but idle/strolling/photo/yoga
4. `MAINTENANCE` — hi-vis staff, brooms, mowers, line-painting, seeding

**Slot-level statuses** (aggregated per field per slot): `USED` 🟢 / `NOTUSED` ⚪ / `REVIEW` 🟡 — same as blueprint.

Aggregation thresholds (from the blueprint, applied to **fused field-level** samples):
- `Ratio_PLAYING ≥ 0.35` ⇒ USED
- `Ratio_PLAYING < 0.10` AND `Ratio_EMPTY ≥ 0.75` ⇒ NOTUSED
- otherwise ⇒ REVIEW
- (Optional flag: distinguish `USED (match)` vs `USED (academy/training)` later; both are usage.)

## 4. Architecture (unchanged from blueprint, with fusion added)

```
football/
├── config/
│   ├── system_config.json        # SOURCE_TYPE: VIDEO_SIM | API_SIM | RTSP_LIVE
│   ├── slots_schedule.json
│   └── cameras.json              # 2 cams/field, RTSP URLs, ROI polygons
├── data/
│   ├── db/pitch_monitor.db       # SQLite
│   ├── evidence_cache/
│   ├── dataset/                  # extracted + labeled frames (Phase 0)
│   │   ├── 1_empty/  2_playing/  3_people_not_playing/  4_maintenance/
│   └── (existing videos + ss data)
├── engine/
│   ├── frame_source.py           # abstraction: video file / sim API / RTSP
│   ├── roi.py                    # polygon mask per camera
│   ├── vision_pipeline.py        # Tier-1 CLIP/DINOv2, Tier-2 YOLO fallback
│   ├── fusion.py                 # 2-camera → field-level state per sample  ← NEW
│   ├── slot_aggregator.py        # ratios → USED/NOTUSED/REVIEW + evidence pick
│   └── simulator_api.py
├── tools/
│   ├── extract_frames.py         # video → 1 frame/60s (and denser for dataset)
│   ├── label_tool.py             # keyboard-driven frame sorter (1/2/3/4)
│   └── draw_roi.py               # click-to-draw ROI polygon per camera
├── dashboard/app.py              # FastAPI + simple web UI
├── requirements.txt
└── main.py
```

## 5. Phased plan

### Phase 0 — Dataset building from our videos *(new phase, highest value now)*
1. `extract_frames.py`: sample the 4 videos every 15 s ⇒ ~950 frames (~240/video); also copy the 14 reference screenshots as a held-out truth set (crop out the red label text before use).
2. `draw_roi.py`: define ROI polygon for each of the 2 camera views (visible in the videos).
3. `label_tool.py`: hand-label extracted frames into the 4 classes (~30–45 min of work).
   Expected yield: hundreds of `playing` + `empty`, some `people_not_playing`; `maintenance` will be scarce → supplement with the screenshots and future exports.
4. Output: `data/dataset/` in the blueprint's folder taxonomy + a `labels.csv`.

### Phase 1 — Frame source + simulator
- `frame_source.py` with three backends; **VIDEO_SIM** treats the 2 videos of a slot as camera A/B of `field_01`, serving 1 frame per simulated minute (fast-forward: a 60-min slot replays in seconds).
- `simulator_api.py`: the blueprint's REST snapshot endpoint on top of it.

### Phase 2 — Model bake-off *(the decisive phase — run TODAY on the Phase-0 dataset)*

One shared benchmark harness (`tools/benchmark.py`): every candidate gets the **same ROI-masked
frames**, same train/hold-out split (hold-out includes the 14 cleaned screenshots), and produces one
row in `results/leaderboard.csv`: per-class precision/recall, macro-F1, accuracy, CPU ms/frame, RAM.

**A. Zero-shot vision-language (no training needed — can run before labeling finishes):**

| Model | Why try it | Expected CPU latency |
|---|---|---|
| `openai/clip-vit-large-patch14` | PDF's primary choice — the baseline to beat | ~280 ms |
| `openai/clip-vit-base-patch32` | 8× faster CLIP; often nearly as good for coarse scenes | ~40 ms |
| **`google/siglip2-base-patch16-224`** *(not in PDF)* | SigLIP 2 beats CLIP on zero-shot in most benchmarks; same usage pattern | ~70 ms |
| **OpenCLIP `ViT-B-32` (laion2b)** *(not in PDF)* | Different training data than OpenAI CLIP; sometimes wins on outdoor scenes | ~40 ms |
| `microsoft/Florence-2-base` | PDF's backup; dense captions useful for REVIEW explanations | ~450 ms |
| **Moondream2 / SmolVLM2** *(not in PDF)* | Tiny VLMs (0.5–2B) that answer "are people playing football?" directly; also generate the dashboard's human-readable explanation text | 1–3 s (use only for REVIEW frames, not every sample) |

**B. Embedding backbone + small trained head (needs Phase-0 labels; head trains in seconds):**

| Model | Why try it |
|---|---|
| `facebook/dinov2-base` + logistic regression | PDF's fine-tuned-phase choice — baseline |
| **`facebook/dinov3-vitb16`** *(not in PDF)* | DINOv3 (2025) — stronger features than v2, same recipe; likely best accuracy |
| **`google/vit-base-patch16-224`** *(the ViT you asked about — not in PDF)* | Plain supervised ViT; linear-probe or full fine-tune; well-supported, fast |
| **ConvNeXt-Tiny** *(not in PDF)* | Modern CNN, very CPU-friendly (~30–50 ms), strong with few labels |
| **EfficientNet-B0 / MobileNetV3-Large** *(not in PDF)* | Ultra-light (5–20 ms); if accuracy is close, wins on 20-camera throughput |

**C. Detectors (Tier-2 ambiguity resolver):**

| Model | Why try it |
|---|---|
| YOLOv8n / YOLOv10n | PDF's choice — person + ball counts inside ROI |
| **YOLO-World-S** *(not in PDF)* | Open-vocabulary: detect "person in high-visibility vest", "soccer ball", "lawnmower" by text prompt — no custom training for the maintenance check |
| **RT-DETR-R18** *(not in PDF)* | Transformer detector, competitive with YOLO, no NMS quirks |

**D. Tier-0 motion gate** *(not in PDF — classical CV, ~1 ms)*: OpenCV MOG2 background
subtraction / frame differencing inside the ROI across consecutive samples. If zero motion **and**
Tier-1 says EMPTY → record EMPTY without further checks; sustained motion contradicting an EMPTY
prediction → escalate. Nearly free and kills the "post-game clutter" false-positive class.

**Today's run order:**
1. Extract + label frames (Phase 0, ~1 h including labeling).
2. Run all of group A zero-shot (no training) → first leaderboard.
3. Train heads for all of group B (each trains in seconds on ~500 embeddings) → second leaderboard.
4. Spot-check group C on ambiguous frames (people-not-playing vs playing).
5. Pick: best Tier-1 (accuracy under 500 ms budget), best Tier-2, keep runner-ups noted.

Acceptance gate: ≥ 90% frame accuracy on hold-out; < 500 ms/frame CPU.

### Phase 3 — DB, fusion, slot aggregation
- SQLite schema = blueprint §5.2 **plus** the `cameras` table (2/field) and a `frame_samples.camera_id` column; fused per-minute field state stored alongside raw per-camera predictions.
- `fusion.py` (rules in §1) then `slot_aggregator.py` (ratios, thresholds, top-3 evidence: first/mid/last third of the slot).
- **Validation:** run the two recorded slots end-to-end; both should come out `USED` with sensible ratios; hand-check the chosen evidence frames.

### Phase 4 — Dashboard
- FastAPI + a single-page UI: global meters, field × slot matrix, evidence inspector with the 3 snapshots, 1-click override (audit-logged), schedule editor writing `slots_schedule.json`.

### Phase 5 — Live deployment
- Switch `SOURCE_TYPE` to `RTSP_LIVE` (OpenCV/ffmpeg snapshot grab per camera per minute), 20 cameras for 10 fields, retention/purge worker (7-day raw / 365-day evidence), systemd units, 48 h shadow run vs. manual log.

## 5b. Additional suggestions (beyond the PDF)

1. **OpenVINO / ONNX export for deployment** — the target is an Intel mini-PC: exporting the winning
   model to OpenVINO (or ONNX Runtime) typically gives 2–4× faster CPU inference and can use the
   Iris Xe iGPU. Benchmark PyTorch first for accuracy; convert the winner before Phase 5.
2. **Temporal smoothing** — classify each minute, but let the fused field state be a majority vote over
   a sliding 3–5 sample window. Removes single-frame flicker (a walker crossing the pitch ≠ playing).
3. **Confidence calibration** — the 75% / 40–74% Tier-2 escalation bands in the PDF assume calibrated
   scores; after the bake-off, calibrate the winner (temperature scaling on hold-out) so the bands mean
   what they say.
4. **Person-count feature fusion** — feed YOLO's ROI person count into the aggregator as a feature
   (e.g. PLAYING requires ≥ 4 people at least once in the slot); cheap guard against
   `people_not_playing` slots billed as USED.
5. **Hard-negative mining loop** — every operator override in the dashboard automatically copies its
   3 evidence frames into `data/dataset/` under the corrected label; the model improves from real
   mistakes with zero extra labeling effort.
6. **Night-glare handling** — the night footage has heavy floodlight bloom; add CLAHE / gamma
   normalization as a preprocessing option in the bake-off harness (test with and without).
7. **Camera-health watchdog** — per blueprint's lens-dirt case: alert when a camera's mean confidence
   or mean frame contrast drops persistently below its own 7-day baseline.
8. **Seed the dataset with video-mined frames, not only every-15s samples** — also grab frames at
   moments the motion gate flags as high-activity, so the PLAYING class gets diverse poses rather
   than mostly mid-field standing.

## 6. Risks & open items
- **Maintenance/edge-case data is thin** (a handful of screenshots). Zero-shot CLIP covers the gap initially; collect more StatBox exports of maintenance windows to train the head properly.
- The red burned-in label text on the screenshots must be cropped/inpainted before using them as eval images (it literally says "playing" — a vision-language model can read it).
- Videos are 20 fps replays, not RTSP — fine for simulation; the RTSP snapshot path needs a short on-site validation in Phase 5.
- Slot lengths in the PDF are 90 min; our exports are 60 min — keep slot duration fully config-driven.
- Confirm business rule: does ACADEMI/training count as USED for billing? (Plan assumes **yes**, with an optional sub-tag.)

## 7. Suggested immediate next steps
1. Phase 0 tooling (`extract_frames.py`, `draw_roi.py`, `label_tool.py`) and produce the labeled dataset.
2. Phase 2 zero-shot CLIP benchmark on that dataset — this tells us fastest whether the PDF's model choice holds on *our* footage.
