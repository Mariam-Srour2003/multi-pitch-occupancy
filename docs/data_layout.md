# Data layout

`data/` is gitignored in full — it holds footage of identifiable people at client facilities.
This file is the tracked record of what belongs where.

```
data/
├── raw/                          source footage — IMMUTABLE, never edited in place
│   ├── venue_01/                 4 slot recordings, ~60 min each, 2 cameras × 2 slots
│   └── highlights_2026-09-04/    66 match clips, 10–14 s each, multi-venue
├── interim/
│   └── frames/<camera_tag>/      extracted frames awaiting labels
├── processed/                    the labelled dataset
│   ├── 1_empty/  2_playing/  3_people_not_playing/  4_maintenance/
│   ├── labels.csv                per-frame provenance from the labelling tool
│   └── manifest.csv              generated — `uv run pitch manifest`
├── reference/                    14 annotated screenshots (labels burned into pixels)
├── cache/                        frozen-backbone embeddings, keyed by backbone (WP0-T5)
├── evidence/                     evidence images bound to slot verdicts
└── db/                           pitch_monitor.db
```

## Rules

1. **`raw/` is append-only.** Never rename, crop, re-encode or delete inside it. Everything
   downstream is regenerable; raw footage is not.
2. **One venue per folder under `raw/`.** Venue is the grouping key for leave-one-venue-out
   evaluation and it cannot be recovered from filenames, so the directory *is* the record.
3. **The class folder is the authoritative label**, not `labels.csv`. The manifest treats
   `labels.csv` as provenance only, and marks frames absent from it as `labeled_by=bulk`.
4. **`processed/` keeps four folders, metrics report three classes.** The 4→3 collapse lives
   in `pitch_occupancy.data.taxonomy`; keeping the finer labels costs nothing and preserves a
   4-class ablation.
5. **`reference/` screenshots must never be used to evaluate a vision-language model** without
   inpainting first — the class label is burned into the pixels in red text, and the model can
   read it.

## What is in `raw/` today

### `venue_01/` — 4 recordings, ~2.5 GB
Two slots, two cameras each, one facility.

| Slot | Lighting | Content |
|---|---|---|
| 2026-07-11 10:00 | day | almost entirely empty |
| 2026-07-12 20:30 | night (floodlit) | almost entirely active play |

This is the source of all 1,296 currently labelled frames — and of the confound recorded below.

### `highlights_2026-09-04/` — 66 clips, 1.2 GB, 14 minutes total
Exported 2026-09-04. Filenames (`statbox-watermarked-<ms-timestamp>.mp4`) encode only the
*export* time, not the recording time, camera, or venue — that metadata is unrecoverable from
the files and has to come from the facility.

Measured, not assumed:

- 10.2–14.0 s each, 30 fps, mixed 1280×720 and 1920×1080, heavy fisheye
- roughly 8–10 visually distinct venues; several carry burned-in venue watermarks
- brightness split ≈ 36 day / 30 night
- **YOLO person count: every clip has ≥ 3 people, median 8, minimum 3 — zero empty pitches**

They are match highlights. Useful for venue and daylight-play diversity; they contribute
nothing to `1_empty`, `3_people_not_playing` or `4_maintenance`.

## The confound this data still has

Run `uv run pitch manifest --check` to see it live.

`1_empty` exists almost exclusively in one morning recording at one venue. On the currently
labelled data a rule that reads only the clock — *"if night → ACTIVE_PLAY, else EMPTY"* — scores
**98.4%**, matching or beating every trained model in the pilot. Accuracy measured here cannot
distinguish classifying occupancy from recognising the scene.

Adding the 66 clips fixes daytime active play but **does not fix the confound** — it moves it
from lighting to venue, because `1_empty` still comes from a single camera-slot.

**What actually resolves it:** empty pitches from more than one venue and both lighting
conditions — above all, *full-length* recordings rather than highlight clips. A complete slot
naturally contains the empty periods before and after play, and is also the only thing that can
supply slot-level ground truth for STAN (WP5-T1), which highlight clips cannot.

## After extracting the clips (2026-09-06)

`uv run pitch extract-clips` samples 6 frames from the middle 80% of each clip — 396 frames —
which were verified as active play and filed into `2_playing` with `labeled_by=bulk`. The
manifest now holds **1,692 frames across 9 venues**.

What changed, and what did not:

- **ACTIVE_PLAY is no longer confounded.** It now spans 9 venues and both lighting conditions
  (222 day / 970 night), and its confound warning has cleared.
- **EMPTY is unchanged**: 494 frames, 98% from one morning recording at one venue. Its warning
  still fires, and always will with this footage.
- **Leave-one-venue-out is now viable for one question only.** Each of the 7 development
  clip-venues yields a fold whose test side is 100% ACTIVE_PLAY — which is precisely H3
  (cross-venue play recall), and the "class absent from test" warning on those folds is expected
  rather than a defect. The `venue_01` fold is degenerate in the opposite direction: it would
  train on 282 clip frames containing no EMPTY at all, then test on all 494 EMPTY frames. Exclude
  it from H3.

### A finding worth carrying into WP4

YOLOv8n **under-counts people on hazy, distant, fisheye footage**. Eight extracted frames were
flagged as having fewer than 3 people; on inspection every one contained a match in progress,
with players simply small, low-contrast or spread wide. Tier-2 person counting is the planned
ambiguity resolver, and it will be least reliable in exactly the conditions where Tier-1 is also
weakest. Worth measuring rather than assuming.
