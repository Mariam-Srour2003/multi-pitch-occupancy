# Pre-registration of the analysis plan

**Written 2026-09-06, before any leakage-free experiment was run.** Committed to git so the date
is verifiable. Amendments go at the bottom, dated — earlier entries are never edited.

Purpose: fix what will be tested, how it will be judged, and what counts as a result, *before*
seeing outcomes. This is what lets the thesis answer "how many things did you try before one came
out significant?" with evidence rather than recollection.

---

## Standing rules

1. **Every hypothesis is reported whichever way it comes out.** A refuted prediction is a result
   and appears in the thesis with the same prominence as a confirmed one.
2. **Primary metric is macro-F1** over the 3-class taxonomy, with accuracy reported alongside.
   Macro-F1 leads because accuracy is dominated by the majority class in this data.
3. **Uncertainty on every number**: 95% bootstrap CI, 10,000 resamples.
4. **Paired significance tests**: McNemar at frame level, paired bootstrap at slot level.
5. **Multiple comparisons**: Holm–Bonferroni within each hypothesis family; the family is
   declared with the hypothesis below. Corrected p-values are the ones reported.
6. **Effect size beside every p-value.** A significant difference of 0.3 points is described as
   negligible, not as a win.
7. **Seeds fixed and recorded** (base seed 42); anything stochastic is run over ≥5 seeds and
   reported as mean ± std.
8. **The final test set is evaluated once**, after all development is complete. Any additional
   look is disclosed in the thesis.

---

## Data available at the time of writing

- `raw/venue_01/` — 4 recordings, one facility, 2 slots (day ≈ empty, night ≈ active play)
- `raw/highlights_2026-09-04/` — 66 clips, 10–14 s, ~8–10 distinct venues, **all containing
  active play**, zero empty pitches
- 1,296 labelled frames, all from venue_01

**No further data collection is planned.** The hypotheses below are scoped to what this data can
actually support, and §"Not answerable" records what it cannot.

---

## Hypotheses

### H1 — Same-scene evaluation inflates apparent accuracy
Random stratified splitting gives significantly higher macro-F1 than grouped splitting by
`slot_id`, because frames sampled seconds apart appear on both sides.
**Family:** split-strategy comparisons (4 models × 3 strategies).
**Test:** McNemar, paired per model; Holm-corrected across models.
**Predicted direction:** random > grouped.
**Decision rule:** report the delta per model with CI, whatever the sign.

### H2 — Trivial baselines are competitive on the current data
A classifier using no image content beyond global statistics — (a) the clock rule *"night →
ACTIVE_PLAY else EMPTY"*, (b) mean ROI brightness + logistic regression, (c) colour histogram +
logistic regression — reaches within 2 points of macro-F1 of the best frozen-backbone probe on
the venue_01 data.
**Family:** baseline comparisons (3 baselines vs best deep model).
**Predicted direction:** baselines competitive — i.e. **H2 is expected to be confirmed**, which
would be evidence that the dataset, not the models, is doing the work.
**Why declared in advance:** the clock rule already scores 98.4% on the labelled data. Stating
the prediction now prevents it being presented later as an incidental observation.

### H3 — Active-play detection transfers to unseen venues
A model trained on venue_01 plus a subset of clip venues achieves ACTIVE_PLAY recall ≥ 0.90 on
held-out venues never seen in training.
**Family:** leave-one-venue-out folds.
**Metric:** per-class recall for ACTIVE_PLAY only (see §Not answerable for why not macro-F1).
**Test:** bootstrap CI per fold; report the distribution across folds, not just the mean.

### H4 — The compact backbone is not significantly worse
ConvNeXtV2-Tiny + head is statistically indistinguishable from ViT-Base + head on macro-F1 under
grouped splitting, while being ≥ 2× faster per frame.
**Family:** model comparisons (4 models, all pairs).
**Decision rule:** a null result confirms H4; the production recommendation then rests on
latency, which is reported as median and p95 under a 20-camera concurrent load.

### H5 — Preprocessing contributes measurably
ROI masking improves macro-F1 significantly under grouped splitting; CLAHE improves it on the
night subset specifically.
**Family:** preprocessing ablations (5 switches).
**Decision rule:** each switch reported with delta and CI; adopted only if the gain is
significant after correction.

### H6 — Zero-shot lags trained probes
OpenCLIP zero-shot is significantly worse than every trained probe on macro-F1, quantifying the
cost of a no-label deployment at a new site.
**Family:** zero-shot vs trained (4 comparisons).

---

## Not answerable with the available data

Recorded here so these are stated as scope limits in the thesis rather than discovered as gaps
during examination.

1. **Three-class discrimination at an unseen venue.** `1_empty` exists almost exclusively in one
   morning recording at one facility, and none of the 66 clips contains an empty pitch. No split
   can produce a held-out venue containing both EMPTY and ACTIVE_PLAY. Cross-venue evaluation is
   therefore restricted to ACTIVE_PLAY recall (H3).
2. **Class/scene confounding cannot be fully separated.** Within venue_01, EMPTY is 98% daytime
   and ACTIVE_PLAY 99% night. Any accuracy figure on this data is partly a measure of scene
   recognition. H2 quantifies how much; it does not remove it.
3. **The C3 class.** 6 frames of `people_not_playing`, 0 of `maintenance`. No claim about C3
   performance is made. Metrics are reported 2-class where C3 support is zero, and this is stated
   in every table rather than hidden behind a macro average over an empty class.
4. **Slot-level aggregation and STAN (WP5-T1).** Two real labelled slots exist; the 66 clips are
   10–14 s highlights with no slot structure. Any STAN result is explicitly labelled preliminary
   and trained/evaluated on synthesised sequences, with the synthetic-to-real gap unquantifiable.
5. **Cross-venue few-shot adaptation for EMPTY**, and **temporal/seasonal drift** (2 recording
   days).

---

## The locked final test set

Fixed 2026-09-06, **before any model was fitted on the clip data**.

The 66 clips were grouped into 9 physical venues (`configs/clip_venues.csv`) by background
fingerprint plus visual confirmation — automated clustering alone over-split venues across
day/night lighting, which would have leaked one lighting condition of a venue into training while
the other was held out.

**Selection rule:** the single largest venue is retained for training (it holds 42% of the clips
and holding it out would leave too little to fit on); from the remaining 8 venues, 2 are drawn at
random with seed 42.

**Result:** `clipvenue_b_floodlit_track` and `clipvenue_c_teal_boards` — 19 clips, 29% of the
clip data — are the locked final test set (`results/splits/FINAL_TESTSET_venues.csv`). They are
not evaluated until all development is complete.

**Amendment, same day, disclosed:** the rule initially had no "retain the largest venue" clause.
The unconstrained draw selected the two largest venues, placing 62% of clips in the test set and
leaving 25 for training. The rule was amended and re-drawn. **No model had been fitted and no
accuracy had been observed at the time** — the change was driven by train/test proportion alone,
not by any result. Recorded here because an undisclosed re-draw is indistinguishable from a
convenient one.

---

## Amendments

*(append below, dated; never edit above this line)*
