# SUPER PLAN — Multi-Pitch Occupancy & Booking Verification (Master's Thesis + Client System)

> **Purpose of this file.** Complete, self-contained execution plan. It merges the two planning
> documents ("Thesis_Plan_Multi-Pitch_Occupancy" and "Thesis_Schedule_and_Effort_Plan") with the
> current state of this repository, broken into small tasks a fresh Claude session (or any
> engineer) can execute one by one without any other context. Work top-to-bottom inside a work
> package; respect the `Depends` column across packages.

---

## 0. How to use this plan (read first)

> **→ The working checklist is [`TODO.md`](TODO.md).** This file is the reference (rationale, file
> layout, gotchas); `TODO.md` is what you tick off day to day. It contains every task below plus
> additions marked `★` that close gaps found when reviewing this file against the two source docx
> documents. If the two disagree, `TODO.md` is newer.

- Tasks are numbered `WPx-Ty`. Each has **Do**, **Files**, **Accept** (exit criterion), **Depends**.
- Never skip an **Accept** check. If a result looks shocking (very good or very bad), suspect the
  harness before the model — this project already produced a false 38% from a harness bug (§2).
- Every experiment: fixed seed(s), results appended to a CSV in `results/`, and one line added to
  `results/EXPERIMENT_LOG.md` (create it: date, task id, command, result file, one-line finding).
- Python is the system Python 3.12 (Windows). Main deps installed: torch (CPU), transformers 5.16,
  opencv-python, scikit-learn, pandas, pillow, ultralytics, open_clip_torch. No GPU — everything
  must run CPU-only; that is a thesis constraint, not a limitation to work around.

---

## 0.5 Research questions (carried over from the thesis-plan docx §1.3)

Every experiment must map to at least one of these; see `thesis/rq_matrix.md` (TODO 1.2).
Anything that maps to none is either a missing RQ or scope to cut.

- **RQ1** — Can a frozen-backbone + lightweight-head architecture classify pitch occupancy at
  production accuracy on CPU-only hardware under night and fog?
- **RQ2** — Which of the four candidate models offers the best accuracy–latency–memory trade-off for
  one Mini-PC serving 20–30 cameras?
- **RQ3** — How does leakage-free, multi-venue evaluation change apparent performance versus a
  same-scene split, and what data volume and diversity close the class-imbalance gap?
- **RQ4** — Can a slot-level aggregation and booking-reconciliation layer reliably detect
  discrepancies between staff-recorded bookings and observed occupancy?
- **RQ5** — Can purpose-designed lightweight architectures (learned slot-temporal aggregator,
  condition-gated multi-backbone fusion head, context-aware multimodal classifier) outperform
  single-backbone probes with hand-tuned threshold aggregation, within the CPU/edge budget?
- **RQ6** ★ *(proposed)* — What is the achievable trade-off between automated-verdict precision and
  REVIEW rate, and where is the operating point that maximises recovered revenue per hour of human
  review?
- **RQ7** ★ *(proposed)* — Do deep frozen backbones actually earn their cost over trivial baselines
  (colour histogram, motion energy, majority class) on this task?

## 1. Current state of the repository (verified working)

```
football/
├── PLAN.md                  # earlier engineering plan (superseded by this file, keep for history)
├── README.md                # tool usage quickstart
├── requirements.txt
├── config/
│   └── system_config.json   # model registry (4 models) + aggregation thresholds
├── data/
│   ├── *.mp4                # 4 StatBox exports: 2 cameras × 2 slots (2026-07-11 10:00 day, 2026-07-12 20:30 night)
│   ├── ss data/             # 14 reference screenshots with red burned-in labels (EVAL-POISONED — see §2)
│   ├── dataset/             # 1,296 labeled frames, FLAT class folders (no camera subfolders)
│   │   ├── 1_empty/ (494)  2_playing/ (796)  3_people_not_playing/ (6)  4_maintenance/ (0)
│   │   └── labels.csv
│   ├── db/pitch_monitor.db  # SQLite: frame_samples + slot_evaluations from pilot runs
│   └── evidence_cache/      # 3 evidence JPEGs per slot per model
├── engine/                  # WORKING end-to-end pipeline
│   ├── vision_pipeline.py   # Classifier registry (embed_head + zeroshot), pooling-stamp assertion
│   ├── frame_source.py      # VideoSlotSource (video sim); discover_slots()
│   ├── fusion.py            # 2-camera half-pitch fusion (priority: playing > maintenance > people > empty)
│   ├── slot_aggregator.py   # ratios → USED/NOTUSED/REVIEW + top-3 evidence
│   ├── roi.py               # polygon masking from config/cameras.json (NO ROIs DRAWN YET)
│   └── db.py                # schema (fields, cameras, rental_slots, frame_samples, slot_evaluations)
├── tools/
│   ├── extract_frames.py    # video → frames (1/15s + motion-mined)
│   ├── label_tool.py        # keyboard labeler
│   ├── make_sheets.py       # contact-sheet grids for fast visual review
│   ├── draw_roi.py          # click-polygon ROI editor → config/cameras.json
│   ├── benchmark.py         # model bake-off harness → results/leaderboard.csv + head_*.pkl
│   └── run_slot_eval.py     # end-to-end: videos → classify → fuse → aggregate → DB + evidence
├── results/
│   ├── leaderboard.csv (+ leaderboard_new.csv fallback if locked)  # pilot benchmark
│   ├── head_*.pkl           # 3 trained heads (ViT, DINOv2, ConvNeXtV2), pooling="mean" stamped
│   └── sheets/              # contact sheets used for labeling
└── yolov8n.pt               # detector used for label pre-screening
```

**Pilot results (same-scene split — feasibility signal, NOT thesis numbers):**
| Model | Trained | Acc | ms/frame | Role going forward |
|---|---|---|---|---|
| ViT-Base + head | head only | 99.2% | 203 | accuracy reference |
| DINOv2-Base + head | head only | 98.5% | 254 | robustness reference |
| ConvNeXtV2-Tiny + head | head only | 98.1% | 102 | production lean |
| OpenCLIP B/32 | zero-shot | 75.8% | 247 | cold-start baseline |
| CLIP B/32 / CLIP L/14 / SigLIP2 | zero-shot | 65 / 35 / 32% | — | appendix negative results |

**End-to-end pilot:** all 3 heads verdict both recorded slots correctly (morning NOTUSED 97% empty,
night USED 100% playing); OpenCLIP gives morning REVIEW (acceptable escalation).

---

## 2. Hard-won gotchas — DO NOT RELEARN THESE

1. **Feature/head consistency.** Heads are trained on **mean-pooled** frozen features
   (`last_hidden_state.mean(dim=1)` for transformers, `.mean(dim=(2,3))` for convnets). NEVER use
   HF `pooler_output` (randomly initialized on plain ViT; caused a false 38% score AND a silent
   production failure). Every saved head carries `pooling:"mean"`; `vision_pipeline.py` asserts it.
   If you change feature extraction, retrain ALL heads and bump the stamp.
2. **`results/leaderboard.csv` may be locked** (user opens it in Excel). `benchmark.py` falls back
   to `leaderboard_new.csv`. Merge the two when unlocked; never crash on PermissionError.
3. **The 14 screenshots in `data/ss data/` have labels burned into pixels** (red text). Never use
   them for evaluation of any vision-language model without cropping/inpainting the text first.
4. **Windows shell:** PowerShell 5.1 — no `&&`, no `$(...)` in the PowerShell tool. Bash tool is
   Git Bash. `pandoc`/`ffprobe` NOT installed. Extract docx text via python zipfile+regex.
   Screenshot filenames may contain U+202F (narrow no-break space) — glob, don't type them.
5. **Long background runs die when the laptop sleeps** (a stuck HF download froze a whole run).
   Keep runs resumable; write partial results incrementally.
6. **HF model cache** is at `C:\Users\maria\.cache\huggingface\hub` (symlinks disabled → duplicated
   blobs, larger on disk). DINOv3 is license-gated — needs `huggingface-cli login` after the user
   accepts Meta's license (ask the user; do not attempt workarounds).
7. **StatBox export naming:** `... (1).mp4` = second camera of the same slot, but the A/B ↔ view
   mapping is NOT consistent across days. Identify cameras by *view content*, key ROIs by camera
   tag per slot (already how `cameras.json` is keyed).
8. **Class imbalance is the #1 scientific risk:** 6 frames of people-not-playing, 0 of maintenance.
   Do not report macro-F1 improvements as model wins while support is ~0; fix the data first.

---

## 3. Decision to implement first: class taxonomy (thesis = 3 classes)

The thesis consolidates the pilot's 4 folders into 3 operational classes:

| Thesis class | From pilot folders | Maps to slot decision |
|---|---|---|
| **C1 EMPTY** | `1_empty` | NOTUSED |
| **C2 ACTIVE_PLAY** | `2_playing` | USED |
| **C3 MAINTENANCE_NON_SPORTING** | `3_people_not_playing` + `4_maintenance` | NOTUSED / REVIEW |

Implementation rule: **keep the 4 physical folders** (finer labels cost nothing and allow 4-class
ablation later) but add a `taxonomy.py` mapping layer; all thesis metrics report the 3-class view.

---

## WP0 — Repo & rigor hardening (Weeks 1–2, ~40h) — M-gate feeds M1

- **WP0-T1 Git init.** `git init`, commit everything except `data/*.mp4`, `data/dataset`, HF caches
  (use `.gitignore`; large data referenced by manifest instead). *Accept:* clean `git status`, first commit.
- **WP0-T2 Dataset manifest.** Script `tools/build_manifest.py` → `data/dataset/manifest.csv` with
  columns: `file, class4, class3, venue, camera, slot_date, slot_time, t_s, source(regular|motion),
  labeled_by(human|assisted), lighting(day|night)`. Derive from filenames + labels.csv.
  *Accept:* 1,296 rows, zero missing fields; groupby prints match §1 counts.
- **WP0-T3 Taxonomy layer.** `engine/taxonomy.py`: `to_class3(label4)`, class lists, display names.
  Wire into `benchmark.py` (flag `--classes 3|4`, default 3) and `slot_aggregator.py` (C3 counts
  toward NOTUSED unless it dominates → REVIEW; keep thresholds in config).
  *Accept:* pilot benchmark reruns with `--classes 3` and reproduces ≥ pilot accuracy.
- **WP0-T4 Split module.** `engine/splits.py`: (a) `grouped_split(manifest, group=venue×date×slot)`;
  (b) `leave_one_group_out(...)`; (c) legacy random split (for the leakage-comparison experiment
  only). Splits are **materialized to CSV** in `results/splits/` and referenced by name — never
  re-randomized silently. *Accept:* unit test: no group appears on both sides.
- **WP0-T5 Feature cache.** `engine/feature_cache.py`: embed every manifest image once per backbone
  → `data/cache/<backbone>.npz` keyed by file path + preprocessing hash. `benchmark.py` uses cache.
  *Accept:* second benchmark run of same backbone takes seconds, identical metrics.
- **WP0-T6 Stats utilities.** `engine/stats.py`: bootstrap 95% CI (accuracy & macro-F1, n=10,000
  resamples), paired McNemar test, slot-level paired bootstrap. *Accept:* unit tests against
  hand-computed toy examples.
- **WP0-T7 Experiment log.** Create `results/EXPERIMENT_LOG.md`; backfill pilot entries. *Accept:* exists, has pilot rows.

## WP1 — Literature & protocol writing support (Weeks 1–4, ~70h) — mostly human/writing

- **WP1-T1 Related-work skeleton.** `thesis/ch2_related_work.md` with the 7 strands (§2.1–2.7 of the
  thesis doc: occupancy classification, zero-shot VLMs, linear probing, detectors-as-resolvers,
  temporal/fusion/distillation, edge inference, research gap) — 3–5 bullet claims + candidate
  citations each (search literature; mark placeholders `[CITE]`). *Accept:* file reviewed by user.
- **WP1-T2 Protocol document.** `thesis/protocol.md`: the 3-class taxonomy with labeling rules
  (people outside ROI don't count; goalkeeper-only during live match = ACTIVE_PLAY; academy =
  ACTIVE_PLAY; coach carrying gear = C3; hi-vis + tool = C3), split definitions, metrics, seeds.
  This is the M1 gate artifact. *Accept:* user/supervisor approves.

## WP2 — Data collection & labeling (Weeks 3–9, ~150h) — M2 gate

- **WP2-T1 Collection matrix tracker.** `tools/coverage_report.py`: reads manifest, prints/updates a
  per-(class × lighting × venue) tally table → `results/coverage.md`. Highlight empty cells.
  *Accept:* current data shows e.g. `C3 × night = 0` visibly.
- **WP2-T2 Collection requests (HUMAN — write the request doc for the user).**
  `thesis/data_requests.md` listing exactly what footage to export: (a) maintenance windows
  (brooming, line painting, mowing, seeding) day+night; (b) idle-people slots (walk-throughs,
  events, photo sessions); (c) ≥ 2 more pitches at the same facility; (d) ≥ 2 external venues,
  2 slots per lighting regime; (e) rain/fog days when they occur. Targets (3-class):
  C1 ≥ 600 · C2 ≥ 800 · C3 ≥ 400 per venue, several hundred per condition cell.
  *Accept:* doc delivered; user confirms export plan.
- **WP2-T3 Ingest loop (repeat per new footage batch).** For each new video batch:
  `extract_frames.py` (extend: `--venue` and `--pitch` flags into filenames) → contact sheets →
  label (label_tool / sheets + full-res checks; YOLO pre-screen allowed, every flag human-verified)
  → update manifest → `coverage_report.py`. *Accept:* per batch: labels merged, coverage table updated.
- **WP2-T4 De-duplication.** Add perceptual-hash near-duplicate pass (`imagehash` or cv2 pHash) to
  the ingest loop; drop near-identical consecutive frames within the same class.
  *Accept:* duplicate rate reported per batch; manifest flags survivors only.
- **WP2-T5 Double-labeling & κ.** Randomly sample 10% of all labeled frames → second annotator
  (the user or a second session labeling blind) → `tools/kappa.py` computes Cohen's κ, logs
  disagreements → resolve, amend `protocol.md` rules. *Accept:* κ reported (target ≥ 0.85);
  disagreement list resolved.
- **WP2-T6 Screenshot rescue (optional).** Crop/inpaint the red text from the 14 `ss data`
  screenshots (text regions are axis-aligned; cv2 inpaint) → add to a held-out qualitative set
  (esp. maintenance examples). *Accept:* no legible label text remains (visually verified).
- **WP2-T7 M2 gate check.** Coverage table shows every class ≥ target in ≥ 2 lighting regimes and
  ≥ 2 venues; grouped splits regenerate cleanly. *Accept:* M2 checklist in EXPERIMENT_LOG.

## WP3 — Preprocessing pipeline + ablations (Weeks 6–10, ~80h)

Build each stage as an independent, switchable step in `engine/preprocess.py` (single entry:
`preprocess(frame_bgr, camera_tag, *, roi=True, letterbox=True, clahe='auto', ...)`) used by BOTH
`benchmark.py` and the live pipeline (one code path — see gotcha §2.1).

- **WP3-T1 ROI polygons (HUMAN-assisted).** User runs `tools/draw_roi.py` for every camera view
  (current 4 + each new venue camera). *Accept:* `config/cameras.json` has a polygon per camera tag;
  masked previews visually correct.
- **WP3-T2 Letterbox resize.** Aspect-preserving letterbox to model input (replaces naive thumbnail).
  *Accept:* unit test: output 224×224 (or 384), no distortion, gray padding.
- **WP3-T3 Photometric normalization audit.** Verify each model's HF processor stats are applied
  (they are, via processors) — document per model in protocol.md. *Accept:* table in protocol.md.
- **WP3-T4 Low-light/fog branch.** Contrast measure (RMS on ROI); below threshold → CLAHE/gamma
  variant; expose `clahe on|off|auto`. *Accept:* toggleable; visual before/after saved to results/.
- **WP3-T5 Quality filter + camera-health metric.** Detect over/under-exposure, blur (Laplacian
  var), lens-dirt proxy (persistent contrast drop vs camera's 7-day baseline) → flag frame
  `quality=bad` in manifest; emit `camera_health.csv`. *Accept:* known-bad frames flagged.
- **WP3-T6 Train-time augmentation.** For head training only: photometric jitter, synthetic
  fog (gaussian haze), night-gamma, horizontal flip. NO rotations/warps (fixed cameras).
  *Accept:* flag in benchmark; visual grid of augmented samples saved.
- **WP3-T7 Class balancing.** Class weights (sklearn `class_weight='balanced'`) + optional weighted
  sampling; default ON for 3-class runs. *Accept:* C3 recall improves on validation vs unweighted.
- **WP3-T8 Preprocessing ablation experiment (E-PRE).** On the best model + grouped split: toggle
  {ROI, letterbox-vs-thumbnail, CLAHE, augmentation, balancing} one at a time; report deltas with
  CIs → `results/ablation_preprocessing.csv`. *Accept:* table + one-paragraph finding per switch in
  EXPERIMENT_LOG.

## WP4 — Leakage-free four-model benchmark + statistics (Weeks 9–13, ~110h) — M3 gate

- **WP4-T1 Same-scene vs grouped comparison (headline experiment).** Four models × {random split,
  grouped split, leave-one-venue-out} × 3 classes. This quantifies the leakage inflation the pilot
  suspected. *Accept:* `results/benchmark_v2.csv`; the random-vs-grouped delta is the thesis's
  first key figure.
- **WP4-T2 Label-efficiency curves.** Training-set sizes {10, 25, 50, 100, 300, 1000, all} × 5
  seeds × 4 models (zero-shot OpenCLIP = the 0-label horizontal line). Cached features make this
  cheap. *Accept:* `results/label_efficiency.csv` + matplotlib figure `results/figs/label_curve.png`.
- **WP4-T3 Cross-venue few-shot adaptation.** Leave-one-venue-out, then add {0, 25, 100, 300}
  target-venue frames to training. *Accept:* adaptation curve per venue; answers "cost of
  onboarding a new client site".
- **WP4-T4 Statistical testing.** For every headline pair: bootstrap CIs + McNemar (frame level),
  paired bootstrap (slot level). *Accept:* every claim in benchmark_v2 has CI + p-value columns.
- **WP4-T5 Calibration study.** Reliability diagrams + ECE per model; temperature scaling fitted on
  validation (within training venues); effect on REVIEW-band volumes. *Accept:*
  `results/calibration.csv` + figs; calibrated heads saved with `temperature` in the pkl.
- **WP4-T6 Error taxonomy & explainability.** Categorize every grouped-split misclassification by
  (condition, confusion pair, camera) → `results/error_taxonomy.csv`. Attention rollout (ViT,
  DINOv2) and Grad-CAM (ConvNeXt) on 20 representative errors + 20 correct frames → save overlays
  to `results/figs/xai/`. Confirm reliance on pitch, not background. *Accept:* figures exist;
  finding sentence per error class.
- **WP4-T7 DINOv3 (if access granted by then).** Add as 5th row to every WP4 table. *Accept:* same
  protocol, or explicitly logged as blocked.
- **WP4-T8 M3 gate.** Benchmark v2 complete, stats attached, figures exported. *Accept:* checklist row.

## WP5 — Novel architectures (Weeks 12–18, ~150h) — M4 gate, the thesis's originality

All modules: frozen backbones; only small trainable parts; every module ablated vs baseline with
significance tests; latency/memory reported next to accuracy. Success = significant gain within
CPU budget; a rigorous negative result is still reportable.

- **WP5-T1 STAN — Slot-Temporal Aggregation Network (Core tier).**
  `engine/stan.py`: input = ordered per-minute fused class-probability sequence (optionally + per-
  minute embeddings); model = 1-D TCN or 2-layer Transformer encoder (< 100k params); output =
  slot status (3-way) + calibrated confidence. Training data: slots synthesized from labeled
  frames — build `tools/make_slot_dataset.py` that composes realistic slot sequences from manifest
  frames (bootstrap-sample per-minute states following templates: full-match, no-show, late-start,
  maintenance-window, intermittent) since real labeled slots are few; hold out the 2 real recorded
  slots + all new real slots as the test set. Baseline = threshold rules (§5.2 thesis doc), with
  thresholds also *tuned* on the same training slots for fairness.
  *Accept:* STAN vs tuned-threshold on slot macro-F1 with paired bootstrap p < 0.05 reported
  (either direction — report honestly); REVIEW rate ≤ baseline; latency ~negligible.
- **WP5-T2 Gated multi-backbone fusion head (Target tier).**
  `engine/fusion_head.py`: features from {ConvNeXtV2, DINOv2} (option: +ViT); gate = tiny MLP on
  cheap image stats (contrast, brightness, edge density) → fusion weights → shared linear head.
  Variant B (conditional compute): run ConvNeXt first; invoke DINOv2 only when confidence < τ;
  report accuracy AND average ms/frame vs always-both. *Accept:* ablation table single-vs-fused
  with CIs, p-values, latency; adopt/reject decision logged.
- **WP5-T3 Context-aware multimodal head (Stretch tier).**
  `engine/context_head.py`: concat visual embedding + context vector (hour-of-day sin/cos, day-of-
  week, booked-flag, previous-slot state, optional weather) → small MLP. **Booking-flag dropout**
  (p=0.5) during training + adversarial ablation (train with flag, eval without) + always keep the
  vision-only path in production (audit independence — thesis §4.4 rigor note).
  *Accept:* ablation vision-only vs +context vs +context-minus-flag; leakage check documented.
- **WP5-T4 Distillation + quantization (Stretch).** ViT-head soft labels distill into ConvNeXt head
  (KL on cached logits — cheap); then OpenVINO export FP32/INT8 of ConvNeXtV2 backbone; measure
  accuracy delta + latency on the actual target CPU. *Accept:* `results/efficiency.csv`
  (PyTorch vs OV-FP32 vs OV-INT8: ms, RAM, accuracy).
- **WP5-T5 M4 gate.** ≥ STAN + one fusion module fully ablated with significance. *Accept:* checklist.

## WP6 — System integration, reconciliation, dashboard (Weeks 14–19, ~120h) — M5 gate

- **WP6-T1 Simulator API (blueprint P1).** `engine/simulator_api.py` (FastAPI):
  `GET /api/v1/cameras/{camera_id}/snapshot?mode=simulation` serving recorded frames on a loop;
  Bearer token from config. *Accept:* curl returns JPEG + JSON metadata.
- **WP6-T2 Scheduler service.** `main.py`: reads `config/slots_schedule.json`; during active slots
  pulls 1 frame/min per camera (source per config: VIDEO_SIM | API_SIM | RTSP_LIVE), classifies,
  fuses, writes DB; at slot end runs aggregator (threshold or STAN per config) + evidence.
  *Accept:* runs continuously against simulator; DB fills; verdicts correct on recorded slots.
- **WP6-T3 RTSP source.** `frame_source.RTSPSource` (OpenCV/ffmpeg snapshot per minute, retry +
  timeout, per-camera thread or sequential loop). *Accept:* tested against any reachable RTSP
  (or a local test stream via ffmpeg loop of the mp4s).
- **WP6-T4 Booking schema + connector.** `engine/bookings.py`: normalized table
  `bookings(field_id, date, start, end, customer_ref, status, entered_by, source)`; importers:
  CSV first (ask client for a sample export — HUMAN task), SQL/REST stubs behind one interface.
  READ-ONLY: never write to client systems. *Accept:* sample CSV imports; unit tests.
- **WP6-T5 Reconciliation job.** `engine/reconcile.py`: after each slot closes, join
  slot_evaluations × bookings on (field, date, slot) → typed anomaly per the matrix:
  booked+recorded-used+NOTUSED → `NO_SHOW_OR_OVERRECORDED`; booked+recorded-unused+USED →
  `PLAYED_NOT_RECORDED`; unbooked+USED → `UNBOOKED_USAGE`; maintenance-dominated during booked →
  `BLOCKED_SLOT_SOLD`; REVIEW → route to inspector, never hard anomaly. Low-confidence/low-contrast
  slots → REVIEW (audit never over-accuses on poor footage). Store in `reconciliations` table with
  evidence links + severity. *Accept:* synthetic booking fixtures produce exactly the expected
  anomaly types; weekly summary query works (rates per field / per staff id).
- **WP6-T6 Dashboard.** FastAPI + single-page UI (`dashboard/`): global meters (fields, active
  slots, occupancy %, pending reviews), live field matrix (status + confidence chips), slot
  evidence inspector (3 photos, AI reason, 1-click override USED/NOTUSED/confirm-REVIEW with
  operator name → `is_overridden` audit fields), anomalies view (reconciliation list, filter by
  type/severity), schedule editor writing slots_schedule.json (sampler hot-reloads).
  *Accept:* manager daily review flow < 5 min: badge → inspector → override → analytics update.
- **WP6-T7 Override → retraining loop.** Overridden slots' evidence frames auto-copy into
  `data/dataset/_incoming/<corrected_class>/` for the next labeling pass; document the periodic
  head re-fit procedure (cached features → seconds). *Accept:* override produces the file + log row.
- **WP6-T8 Retention worker.** Purge raw frames > 7 days (keep evidence 365 days); disk usage
  bounded and logged. *Accept:* dry-run mode prints correct purge set.
- **WP6-T9 M5 gate.** Full pipeline incl. reconciliation runs end-to-end on mock feeds. *Accept:* checklist.

## WP7 — Deployment & live validation (Weeks 18–21, ~70h) — M6 gate

- **WP7-T1 Target-hardware bench.** Run `benchmark.py --limit` + `run_slot_eval.py` on the actual
  Intel Mini-PC (or the client's chosen box): confirm < 500 ms/frame budget with 20 cameras
  headroom (20 × ConvNeXt ≈ 2 s/min cycle). *Accept:* efficiency.csv row for target hardware.
- **WP7-T2 Install & services.** Ubuntu (or Windows service) install doc; systemd units (engine,
  dashboard, retention); camera VLAN config with client IT; `SOURCE_TYPE=RTSP_LIVE`.
  *Accept:* auto-start after reboot verified.
- **WP7-T3 Shadow mode (1–2 weeks).** System records verdicts silently; staff keep normal process;
  compare verdicts vs staff records daily. *Accept:* shadow report with agreement rate.
- **WP7-T4 48-hour acceptance run.** With facility staff: every slot verdict adjudicated; measure
  slot accuracy, REVIEW rate, reconciliation precision/recall. This doubles as thesis E10 case
  study data. *Accept:* signed-off M6; case-study tables exported for Chapter 7.

## WP8 — Writing & defense (ongoing → Week 24, ~140h) — M7 gate

- **WP8-T1** Keep `thesis/` chapter drafts updated as each WP closes (Ch3 after WP2/3, Ch4 after
  WP5, Ch5 after WP6, Ch6 after WP4, Ch7 after WP7).
- **WP8-T2** Figures: label-efficiency curve, leakage comparison bar, ablation tables, reliability
  diagrams, XAI overlays, reconciliation matrix with real (blurred) evidence examples.
- **WP8-T3** Threats-to-validity section from §2 gotchas + WP4 findings. Ethics section: no face
  recognition/re-ID, retention policy, staff-audit human-in-the-loop, consent signage, blurred
  figures. **Blur all faces in any published figure.**
- **WP8-T4** Defense deck: pilot story (spec's model lost, harness bug lesson, leakage number),
  novel modules, live demo of dashboard + reconciliation. *Accept:* submitted thesis + deck (M7).

---

## Milestone gates (supervisor checkpoints)

| Gate | Week | Exit criterion |
|---|---|---|
| M1 | 4 | Protocol, taxonomy, evaluation design approved |
| M2 | 9 | Dataset balanced across classes/conditions/venues, grouped-split-ready |
| M3 | 13 | Four-model leakage-free benchmark with CIs + significance |
| M4 | 18 | STAN + ≥1 fusion module ablated with significance |
| M5 | 19 | End-to-end system + reconciliation on mock feeds |
| M6 | 21 | 48-h live validation accepted |
| M7 | 24 | Thesis submitted, defense ready |

16-week compression option: drop WP5-T3/T4 to stretch list, run WP2 at 2 venues minimum,
merge shadow mode into the 48-h run.

## Extensions backlog (only if ahead of schedule, in this order)

1. DINOv3 integration + quantization study (cheap, high visibility)
2. Cold-start transfer curve as a standalone publishable result (already ~WP4-T3)
3. Active learning + pseudo-labeling on the unlabeled pool
4. Self-supervised domain adaptation on facility footage (flagship stretch; needs GPU time)
5. NL explanations for REVIEW slots (Florence-2 / Moondream on evidence frames)
6. Camera-health prediction from confidence drift
7. Test-time augmentation + temporal smoothing polish
8. Multi-tenant productization + mobile manager view

## Risk register (condensed — act, don't admire)

| Risk | Trigger to watch | Action |
|---|---|---|
| C3 data stays scarce | coverage.md cell < 100 by week 6 | escalate to user; schedule maintenance-window exports; Tier-2 YOLO-World fallback for C3 |
| Only 1 venue accessible | week 5 no partner venue | reduce claim to cross-pitch/cross-lighting; state scope |
| Booking DB access blocked | week 12 no sample export | switch to manual booking sheet for case study |
| Novel module no gain | WP5 ablation p > 0.05 | report as negative result with analysis (still a contribution) |
| Laptop-sleep kills runs | any multi-hour run | chunk runs; incremental writes; re-runnable via cache |

## Tasks only the human can do (surface these early, repeatedly)

1. Export more StatBox footage per `thesis/data_requests.md` (esp. **maintenance windows** — 0 frames).
2. Draw/confirm ROI polygons (`tools/draw_roi.py`) for every camera.
3. HuggingFace login + accept DINOv3 license.
4. Introduce ≥ 2 partner venues; arrange exports.
5. Ask the client which booking system they use; obtain a read-only sample export (CSV is fine).
6. Second annotator for the κ double-labeling subset.
7. Supervisor sign-offs at M1–M7.
