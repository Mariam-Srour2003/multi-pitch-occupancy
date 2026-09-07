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

---

### 2026-09-07 — First amendment batch: reconciling this document with what was actually run

This log was empty while six hypotheses had been written and four experiments' worth of
results had landed. Standing rule 1 says *every hypothesis is reported whichever way it comes
out*, and rule 5 says *the family is declared with the hypothesis*. Both had drifted. Nothing
below changes a result; it records movement that had already happened in
`results/EXPERIMENT_LOG.md` and `thesis/rq_matrix.md` but never reached the document an
examiner will read first.

**A1 — H2 is refuted as pre-registered, and confirmed in a narrower form. Both are reported.**
H2 predicted trivial baselines would come *within 2 macro-F1 points* of the best probe, and
predicted confirmation. Outcome: **refuted at the pre-registered threshold** — DINOv2 clears
the baseline floor by 8.9 points. The narrower claim stands and is the more useful one: *two
of three* backbones fail to beat a clock rule that never looks at the image (grouped split,
clock rule 0.4907 vs ConvNeXtV2 0.4975), and under the leaky random split a 16-bin colour
histogram scores above ConvNeXtV2 (0.686 vs 0.657). The 2-point threshold is **not** revised
after the fact; it is reported as missed, with the narrower finding beside it.

**A2 — H1's finding stands; its wording and its p-value do not.**
Originally reported as *"a colour histogram beat the deep probe under the leaky protocol"*.
After the effective-sample audit (`experiments/effective_sample_audit.py`) the 394-frame
random test set is **62 distinct scenes**, and on those DINOv2 and the histogram both score
**1.0000** — identical, with p moving 7.4e-03 → 1.00. The claim becomes *"the leaky protocol
cannot tell them apart"*, which is a cleaner statement of the same point and no longer rests
on a difference that is not there. H2's grouped-split comparison survives but moves from
p 1.2e-03 to **2.2e-02** on 94 scenes — from comfortable to marginal.

**A3 — New standing rule (9): effective sample size beside every frame-level statistic.**
The dataset is 1,692 frames but roughly **150 distinct scenes**; 98.5% of frames have a near
neighbour. Frame-level CIs and p-values computed as if frames were independent are inflated,
and this was discovered *after* the first results were reported, not designed in. From now on
every frame-level number carries its effective sample size. H3 needs no correction for a
specific reason worth stating: its intervals bootstrap over the **seven venue folds**, not
over frames, so the resampling unit already matches the thing being generalised over.

**A4 — H3's conclusion stands; the figure 0.219 is withdrawn.**
The `lighting` column for the 396 clip frames is a brightness proxy, not a clock, and at least
`b_floodlit_track`, `f_outdoor_bldg` and `c_teal_boards` are night football filed as `day`.
The clock rule reads that column and nothing else, so its 0.219 cross-venue play-recall is a
lower bound partly produced by label error — correcting one twelve-frame fold alone moves it to
0.362. **The direction is what is claimed** (a lighting-only rule collapses across venues while
the backbones hold above 0.86); the value waits on hand relabelling.

**A5 — H4, H5 and H6 are declared OUTSTANDING, not quietly dropped.**
None of the three appears anywhere outside this document — not in the experiment log, the RQ
matrix, or the task list. Recorded now so they are visibly open rather than silently missing:

| | status | what it needs |
|---|---|---|
| **H4** compact backbone not significantly worse | **outstanding** | pairwise McNemar under the grouped split (WP4-T4b). Note it is now partly *superseded*: latency stopped being the binding constraint (10–24× headroom), so a null result no longer hands the decision to speed the way the decision rule assumed |
| **H5** preprocessing contributes measurably | **outstanding and complicated** | the 76-evaluation search ran, but WP3-T3 found the HF processor re-crops after `preprocess.py`, so every *geometric* switch was partly overwritten. H5 cannot be settled until the `processor_geometry` convention is decided by measurement |
| **H6** zero-shot lags trained probes | **outstanding** | the prompt search produced the zero-shot numbers; the pre-registered *comparison* against every trained probe, with correction, was never run |

**A6 — Two large search families were run outside the declared family structure. Disclosed.**
Rule 5 requires the multiple-comparison family to be declared with its hypothesis. Two searches
were not:

- the **preprocessing search** — 76 configurations scored (`results/preprocess_search.json`);
- the **zero-shot prompt search** — 375 prompt/model rows (`results/prompt_search.csv`).

That is **451 evaluations** whose selection was not corrected for, and both were *selection*
procedures — picking the best configuration — rather than hypothesis tests, which is why no
p-value was attached and also why the winner's margin is optimistically biased. The honest
treatment, and what the thesis will do: report any searched configuration as **selected, not
tested**; never quote a search-winning delta as a significance result; and re-measure any
configuration that is actually adopted on the held-out venues under its own pre-declared
comparison. The prompt search additionally has no task in any plan document, so it is being
mapped to H6/RQ1 rather than left as an orphan artefact.

**A7 — Disclosure of one further look at nothing.** No amendment above was prompted by a
result on the locked final test set. `results/splits/FINAL_TESTSET_venues.csv` has still never
been evaluated. During this review the lock was found to be **enforceable only from the repo
root** — a relative path meant a run launched from any other directory silently returned all
1,692 rows as development data. The path is now resolved from the package and a missing lock
file raises instead of unlocking everything (`tests/test_splits.py`). No evidence exists that
any reported run was affected — every experiment script is launched from the root — but the
guarantee was weaker than this document claimed, and that is worth recording rather than
quietly repairing.
