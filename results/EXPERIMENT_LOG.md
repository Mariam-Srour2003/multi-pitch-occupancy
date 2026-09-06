# Experiment log

One entry per run: date, hypothesis, command, seed, result file, and the finding in a
sentence. Appended automatically by the experiment scripts; findings written by hand.

---

## Pilot (2026-09-01/02) — superseded

Seven models on 1,296 frames, single random stratified split (seed 42, 1,036/260), one
venue. ViT-Base 99.2%, DINOv2 98.5%, ConvNeXtV2 98.1%, OpenCLIP zero-shot 75.8%.

**Not thesis results.** Same-scene split, two starved classes, one venue. Kept for the
methodological story: a shock 38% ViT score was traced to the harness reading a randomly
initialised `pooler_output` instead of real features. Code preserved on the
`pilot/model-selection` branch.

---

## 2026-09-06 · Dataset audit

`uv run pitch manifest`

Class and scene are the same variable in the labelled data: EMPTY is 98% daytime,
ACTIVE_PLAY 99% night. A rule reading only the clock — *"if night → ACTIVE_PLAY, else
EMPTY"* — scores **98.4%**, matching or beating every pilot model. The pilot's accuracy is
consistent with having learned day-vs-night rather than occupancy.

---

## 2026-09-06 · Clip ingest

`uv run pitch extract-clips` → 396 frames from 66 clips over 9 venues.

ACTIVE_PLAY's confound warning clears — it now spans 9 venues and both lighting
conditions. EMPTY is unchanged at 494 frames from a single morning recording, and cannot
be fixed with the available footage. Leave-one-venue-out becomes viable for H3 alone.

Incidental: YOLOv8n under-counts people on hazy, distant, fisheye footage. All 8 frames it
flagged as having <3 people contained a match in progress.

---

## 2026-09-06 · H1 / H2 — baseline floor and split leakage

`uv run python experiments/h1_h2_baseline_floor.py` · seed 42 · 2,000 CI resamples
→ `results/h1_h2_baseline_floor.csv`

*Partial run: ConvNeXtV2 only; ViT and DINOv2 caches still building.*

| split | model | accuracy | macro-F1 [95% CI] |
|---|---|---|---|
| random | **cheap_histogram** | 0.9594 | **0.6855** [0.665, 0.974] |
| random | convnextv2 | 0.9873 | 0.6573 [0.649, 0.997] |
| random | clock_rule | 0.9162 | 0.6050 [0.587, 0.931] |
| random | majority | 0.6954 | 0.2735 |
| grouped | convnextv2 | 0.9901 | 0.4975 [0.496, 0.499] |
| grouped | **clock_rule** | 0.9636 | **0.4907** [0.487, 0.494] |
| grouped | majority | 0.0099 | 0.0098 |

**H2 — confirmed, emphatically.**

- Under the random split a **16-bin colour histogram beats ConvNeXtV2 on macro-F1**
  (0.686 vs 0.657). A feature that cannot represent "are there people on the pitch" wins.
- Under the grouped split the **clock rule is 0.007 macro-F1 behind ConvNeXtV2** — well
  inside H2's pre-registered 2-point margin, using no image data whatsoever. McNemar calls
  the difference significant (p_holm ≈ 1.2e-07), and that is exactly why effect size is
  reported beside it: significant, and negligible.

**H1 — directionally confirmed, but not cleanly attributable.** ConvNeXtV2's macro-F1 falls
0.66 → 0.50 from random to grouped. Part of that is the leakage H1 predicts; part is that
the grouped test set is 99% single-class, so macro-F1 is near-degenerate by construction.
The two causes cannot be separated on this data, and the thesis must say so rather than
quote the drop as a pure leakage figure.

**Also worth noting:** `majority` scores 0.0099 on the grouped split. Training rows are
mostly EMPTY (the morning recording), test rows almost entirely ACTIVE_PLAY, so predicting
the training majority is wrong ~99% of the time. A vivid illustration of how degenerate
that split is.

`cheap_intensity` scores below `majority` on accuracy because balanced class weights on a
single feature push it to over-predict rare classes — behaving as intended, not a bug.

---

## 2026-09-06 - H3: cross-venue active-play recall

`uv run python experiments/h3_cross_venue_recall.py` | seed 42 | 7 held-out venue folds
-> `results/h3_cross_venue_recall.csv`

*Partial: ViT cache still building.*

| model | mean play-recall [95% CI] | worst fold | target 0.90 |
|---|---|---|---|
| clock_rule | **0.219** [0.029, 0.505] | 0.000 | fails |
| convnextv2 | **0.910** [0.821, 0.986] | 0.722 | meets |
| dinov2 | **0.930** [0.834, 0.993] | 0.667 | meets |

**H3 confirmed for the frozen backbones, and this is the finding that answers H2.**

H2 showed a clock rule matching ConvNeXtV2 *within* the confounded venue. H3 shows what
happens when the confound is removed: on venues never seen in training the clock rule
collapses to 0.22 mean recall - zero on four of seven folds - while the same frozen
backbones hold at 0.91 and 0.93.

Read together the two results say something the pilot could not: **the deep features are
genuinely detecting play rather than lighting; it was the evaluation that could not tell
the difference.** The pilot's 98-99% was uninformative, but the models underneath it were
not the problem. That distinction is the thesis's central methodological argument, and it
is now backed by two pre-registered experiments pointing opposite ways.

Caveats to carry into the write-up:

- Fold sizes are small - five of seven venues contribute 12-30 frames - so the CIs are
  wide and the mean is an unweighted average over folds, giving tiny venues equal weight.
  Report the per-fold distribution, not just the mean.
- Recall on one class only. Nothing here says whether an *empty* pitch is recognised at an
  unseen venue; there is no data for that (see the pre-registration's "not answerable").
- Both models are weakest on `clipvenue_h_teal_pitch` (0.72 / 0.67), which is one of the
  two venue groups flagged `confidence=low` in `configs/clip_venues.csv`. If that group is
  actually two facilities, this fold is measuring something other than it claims - worth
  resolving before the number is quoted.
