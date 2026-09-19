# Configuration

Camera definitions (ROI polygons, field/side mapping) and the slot schedule live here.
Written by `pitch draw-roi` (WP3-T1) and the dashboard schedule editor (WP6-T6);
secrets never do - those come from the environment via `PITCH_*` or `.env`.

## What is in here

| file | what it is | written by |
|---|---|---|
| `roi.json` | hand-drawn pitch boundaries, and the `_aliases` block | the editor at `/roi` |
| `roi_derived.json` | boundaries measured from the footage, read *underneath* `roi.json` | `scripts/derive_roi.py`, `worker --derive-roi` |
| `rules.json` | the detector-first decision rule's numbers (A36) | `experiments/fit_rule_thresholds.py` (WP9-T5) |
| `slots_schedule.json` | when a slot runs, on which pitch, with which cameras | the schedule editor |
| `cameras.example.json` | the shape of `cameras.json`, which is gitignored | by hand |
| `clip_venues.csv` | which clip belongs to which venue | the fingerprint audit |
| `unseen_clip_truth.csv` | per-sample truth for the unseen floodlit clip | by hand (A36) |
| `bookings_example.csv` | a booking export in the shape `bookings.py` reads | by hand |

## `roi.json` and the `_aliases` block

A boundary is stored per camera, as `[x, y]` fractions of the frame. **Production knows a
camera by several names and the store knew none of them until A36**, which is why the boundary
machinery was complete and applied to nothing:

- the worker replaying a recording calls a camera `file0` / `file1` (`discover_slots`);
- `cameras.json` and `slots_schedule.json` call it `camera_A` / `camera_B` under `venue_01`;
- the derived store keys it per recording, `slot_20260711_1000_camA`.

`roi.resolve(camera, venue=..., slot_key=...)` tries the most specific key first and looks each
candidate up through `_aliases`, one hop. `file0 -> camA` is applied only *within* one
recording, because the physical camera behind each file flips between days (`db/seed.py`
`PHYSICAL_CAMERA`) - the mapping was measured by IoU on all four recordings before being
relied on. `roi.get` remains the store's own vocabulary and does none of this; a caller
holding a production id wants `resolve`.

A hand-drawn outline wins over a derived one for the same key, and deleting the hand-drawn one
reveals the measurement rather than erasing it too.

## `rules.json`

Every number the decision table reads (`vision/rules.RuleConfig`, registered in A36).

- `play_min` (5) and `small_group_max` (4) are **the facility's requirement, not fits**.
- `detector`, `imgsz`, `tiles` name the model the counts come from, chosen by
  `experiments/detector_audit.py` under a rule written before it ran.
- Everything else is fitted on **venue_01 camera A only** and then frozen: `frozen_at`,
  `frozen_commit` and `tuned_on` record when, from what, and on which camera.
- **A `null` threshold means the cue it gates is not consulted** - which is how rows 4, 5 and 8
  of the table stay off until they have been measured rather than guessed.
- While `frozen_at` is `null` the file is not for deployment. Experiments run against it,
  because measuring an unfrozen rule is how it gets frozen.

## The boundary is mandatory on the deployed path

A camera whose boundary cannot be resolved contributes no observation: the minute is recorded
as `UNCERTAIN` and counts against the slot's capture floor, so the verdict moves toward REVIEW
and never toward a confident one. Without an outline the model scores the neighbouring pitch
and the car park as if they were this pitch, which A19 measured at 0.74 false-play against
0.38 with one. `worker --derive-roi` measures a boundary from the footage and stores it under
the production id; `/roi` is where a person confirms or redraws it.
