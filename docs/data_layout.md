# Data layout

`data/` is gitignored in full — it holds footage of identifiable people at client facilities.
This file is the tracked record of what belongs where.

```
data/
├── raw/                          source footage — IMMUTABLE, never edited in place
│   ├── venue_01/                 4 slot recordings, ~60 min each, 2 cameras × 2 slots
│   ├── highlights_2026-09-04/    66 match clips, 10–14 s each, multi-venue
│   ├── venue_unseen_2026-09-15/  1 clip, 234 s, EMPTY floodlit pitch at night (A26/A36 eval)
│   ├── davinci_2026-09-21/       15 operator exports, 4-9 s each, + provenance.csv
│   └── public_<source>/          CC-licensed evaluation footage, each with provenance.csv (A36)
├── interim/
│   └── frames/<camera_tag>/      extracted frames awaiting labels
├── processed/                    the labelled dataset
│   ├── 1_empty/  2_playing/  3_maintenance_non_sporting/
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
4. **`processed/` keeps three folders and metrics report three classes — the same three.**
   ~~Four folders, collapsed 4→3 for reporting, keeping a 4-class ablation available.~~
   *(Amended 2026-09-21.)* The ablation was never runnable: the corpus reached 1,692 recorded
   frames holding **6 real `3_people_not_playing` and 0 real `4_maintenance`**. A40 dropped
   the split from the prediction path; `scripts/collapse_label_folders.py` finished it on
   disk. `taxonomy.Label.parse` still reads both retired folder names, because every artefact
   older than that date says one of them.
5. **A manifest rebuild carries rows it cannot name.** `build_manifest` parses filenames, and
   generated frames (`syn_<batch>_<n>.jpg`) match no pattern. A rebuild used to report them
   as problems and write the manifest without them — 200 rows, silently, exit code 0. `pitch
   manifest` now passes the manifest it is replacing as `carry_unparseable`. If you call
   `build_manifest` directly and intend to overwrite, pass it too.
6. **`reference/` screenshots must never be used to evaluate a vision-language model** without
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

### `venue_unseen_2026-09-15/` — 1 clip, 234 s, 24 MB *(added 2026-09-19)*
`empty_floodlit_night.mp4`, received 2026-09-15. A venue absent from the corpus, floodlit,
**nobody playing at any point**: 13 of 16 samples at 15 s spacing empty, 3 with one person
walking. The only footage holding the EMPTY × night cell. Development data, evaluation only.
Truth is tracked at `configs/unseen_clip_truth.csv`; provenance notes in the folder's README.
Previously referenced by an absolute Downloads path in `experiments/reproduce_all.py`.

### `public_<source>/` — CC-licensed evaluation footage *(reserved 2026-09-19, A36)*
The detector-first path has no training set, so evaluation footage is the only kind that
helps, and the operator cannot easily supply more. Public clips under a Creative Commons
licence - an empty pitch at night, a kickabout with four or fewer people, groundskeeping -
may be admitted **for evaluation only**, one folder per source, each carrying
`provenance.csv` (`file, url, licence, retrieved, truth_note`). They never join the labelled
corpus, never tune a threshold, and are redacted like any other frame if reproduced.

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

## Conditions the dataset does not cover

`lighting` (day/night) is the only condition axis. There is **no weather field and no rain
footage** - every frame is dry. Any claim about robustness to precipitation is therefore
unsupported, and the weather ideas in `IDEAS.md` cannot start until such footage exists.
Rain must arrive *crossed with occupancy* - rain-with-play and rain-without-play both
present - or it will reproduce the day/night confound in a new variable.

### `davinci_2026-09-21/` — 15 operator exports, 13 clips + 2 stills

Short DaVinci Resolve exports the operator supplied on 2026-09-20 and 2026-09-21, all
864×496 (well below the 1080p everything else was tuned on), 4–9 s each. Labelled by their
own filenames. `provenance.csv` records which frame came from which export.

**They are new moments, not new places.** Thirteen of the fifteen are venues already in the
corpus — `playing day 3` and `not playing night` are `clipvenue_g_netting`, `playing day` is
`clipvenue_b_floodlit_track`, `maint night` is `clipvenue_a_blue_barrier`, six are
`venue_01`. Each identification is recorded with its evidence in `configs/davinci_venues.csv`
and was made by looking at frames side by side, because a dHash at 8×8 cannot tell two
five-a-side pitches apart. Filing them under new names would have put the same camera on both
sides of a leave-one-venue-out fold. Only `davinci_j_maint_outdoor` and
`davinci_l_city_pitch` are new.

**What they are worth is C3.** Before them the class held 6 recorded frames, all at
`venue_01`; it now holds 16 across four venues, and two of the clips are the first real
groundskeeping footage this project has had. They also bring daylight to two venues the
corpus only had at night.

**Three clips are held out and never extracted** (`split_role=test` in the config): one
maintenance, one people-with-a-ball, one playing still. Their frames are not written at all,
so they cannot reach a training set by accident; the whole videos stay here.

Frames are sampled **by difference, not by clock**: a candidate is kept only if its dHash is
at least 6 bits from every frame already kept from that clip — one past the threshold at
which `dedup.py` calls two frames the same scene. A four-second clip of people standing still
yields one frame and a clip with a game in it yields seven. That is deliberate: sampling six
evenly from each would have produced 90 frames and roughly 15 observations.
