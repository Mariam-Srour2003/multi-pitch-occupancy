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

> **Superseded in part by A12.** The random-split figures quoted here are pre-A8 and the
> comparison reverses on the corrected estimand: ConvNeXtV2 0.9879 against the histogram's
> 0.9616. The grouped-split half is untouched. Kept as written, with the correction recorded
> rather than edited into the original.

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
| **H5** preprocessing contributes measurably | **reported 2026-09-08 — clause 1 unrunnable, clause 2 refuted** | A11. ROI masking has no polygon and never ran, so a null there would be manufactured; CLAHE is significantly **worse** on all 4 comparisons (DINOv2 −0.27 macro-F1). The "night specifically" clause is untestable: no grouped split holds both lighting conditions |
| **H6** zero-shot lags trained probes | **reported 2026-09-08 — inconclusive** | A10. Direction is against it (zero-shot beats all three probes, Holm-corrected, on 5 of 5 splits), but the declared prompt set is above 85% of the prompt space and the median set loses. Prompt choice moves macro-F1 by 0.726 against the backbones' 0.082 |

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

---

### 2026-09-07 — A8: H1's effect size is corrected upward, and the metric is pinned

Found while auditing `bootstrap_metric_ci`. `evaluate` was called inside the bootstrap, so it
re-derived which classes to macro-average **once per resample**. C3 has support 1 in the random
split's test set and appears in 63.4% of resamples, so two thirds of them averaged three
classes and one third averaged two — one confidence interval spanning two estimands. The
grouped split was unaffected (both its classes appear in every resample).

Standing rule 2 says the primary metric is macro-F1 over the 3-class taxonomy; §"Not
answerable" item 3 says metrics are reported 2-class where C3 support is zero. **Support 1 fell
between the two**, and the code resolved it differently in different resamples. Rule 2 is now
read as: *the class set is fixed once from the full test set, a class needs at least 2 test
frames to enter the macro average, and the retained and dropped classes are printed in every
table* (`macro_over_classes`, `excluded_low_support`).

**This moves a headline number, and it moves it against us being modest.** H1's drop from the
leaky to the honest protocol was reported as −0.159 macro-F1 for ConvNeXtV2. On a consistent
metric it is **−0.490** — about 3.1× larger. The pre-registered prediction (random > grouped)
is confirmed more strongly, not less. Affected intervals also narrowed ~15×, the mixed estimand
having been most of the apparent uncertainty.

**One reported claim is weakened and is restated rather than dropped.** The ranking inversion
(ViT selected under the leaky protocol, weakest generaliser under the honest one) **stands** as
a statement about rank. Its magnitude does not: ViT's apparent 0.339 lead under the leaky
protocol came almost entirely from correctly classifying the single C3 frame that the other
models missed. On equal footing the lead is **0.006**. The thesis must quote the inversion as a
reversal of *ordering*, never as a large leaky-protocol margin.

Nothing here touches H3 (which bootstraps over venue folds, not frames), the effective-sample
audit (which concerns independence between frames, not the class set), or any grouped-split
number.

---

### 2026-09-07 — A9: H4 reported, and its decision rule disowned

First of the three hypotheses A5 declared outstanding. `experiments/h4_model_equivalence.py`,
`results/h4_model_equivalence.csv`.

**The decision rule as written is unsafe and was not followed.** H4 says *"a null result
confirms H4"*. That is the absence-of-evidence error, and on a grouped test set that is 99%
one class a null result is close to guaranteed regardless of the models. It was replaced with
an equivalence test read off the interval against a margin **declared before looking** -
+/-0.02 macro-F1, taken from H2's own "within 2 points" threshold rather than invented - with
three possible outcomes instead of two: equivalent, different, or **inconclusive**.

That change matters here rather than in principle. Under the original rule all three model
pairs "confirm" equivalence, including the two where DINOv2 is **8.2 macro-F1 points** better
and the interval runs to -0.23. Read as intervals those two are inconclusive.

**Verdict: H4 is refuted as a whole; its accuracy clause is confirmed and uninformative.**

- *Accuracy clause - confirmed, degenerately.* ConvNeXtV2 and ViT differ by exactly 0.0000
  with a **zero-width** interval, because their predictions are *identical*: both predict
  ACTIVE_PLAY for all 907 frames and EMPTY zero times (C1 F1 = 0.000 on 9 frames). 0.4975 is
  (0.995 + 0)/2 - the score of a model that never gets an empty pitch right. They are
  equivalent to each other and equally equivalent to a constant predictor.
- *Speed clause - refuted on its own stated condition.* The rule specifies latency "under a
  20-camera concurrent load". There the ratio is **1.83x**, below the >= 2x the hypothesis
  requires. It reaches 2.01x only on the single-frame median, the measurement the
  pre-registration declined to rely on.

**Consequence for the production recommendation**, which this hypothesis existed to support:
it cannot. RQ2's choice of DINOv2 rests on the cross-venue evidence instead, where the three
models are not equivalent at all.

**Realised family declared:** 3 pairs, not the 6 this hypothesis anticipated from "4 models,
all pairs". OpenCLIP's image features cover 600 of 1,578 frames and its slice of this test set
is 339 ACTIVE_PLAY against 6 EMPTY, which cannot carry a macro-F1. Holm corrects over 3.
Declared here because a family that shrinks silently is a family that was chosen afterwards.

**A statistical gap this closed.** The comparison needed a paired bootstrap on *macro-F1*, and
no such function existed - `mcnemar` and `paired_bootstrap_diff` both work on per-frame
correctness, i.e. on accuracy. That absence is why H2 was published with a macro-F1 delta
beside an accuracy p-value that for one pair pointed the other way.
`paired_bootstrap_metric_diff` now resamples once per draw and scores both models on the same
frames; 8 tests, including two models with *identical accuracy* that macro-F1 separates.


---

### 2026-09-08 — A10: H6 reported; its family shrinks to 3 and its verdict is inconclusive

The last unreported hypothesis. `experiments/h6_zero_shot_gap.py` →
`results/h6_zero_shot_gap.csv`.

**Realised family: 3, not 4.** H6 anticipated "zero-shot vs trained (4 comparisons)" from
four carried-forward models, but OpenCLIP *is* the zero-shot arm, so the trained probes are
ConvNeXtV2, DINOv2 and ViT. Holm corrects over 3. Declared here because a family that shrinks
quietly is a family chosen after the fact — the same disclosure A9 made for H4.

**The zero-shot arm is a prompt set declared in advance**, not the prompt search's winner:
first descriptor per class, all five templates, fixed by position in
`vision.zeroshot.DECLARED_PROMPT_SET`. A6 requires a searched configuration to be treated as
selected rather than tested, and the winner was chosen as the best of 375 sets on the folds
it reports. Using it would have tested the search.

**Verdict: inconclusive**, and neither half of that is the obvious one.

- *Not confirmed.* On the pre-registered comparison zero-shot is significantly worse than
  **none** of the three probes and significantly **better** than all three (Δ +0.050 to
  +0.132 macro-F1, Holm p ≤ 1.5e−05). It wins on all five grouped splits. The hypothesis's
  direction is wrong.
- *Not refuted.* Scoring all 375 prompt sets in the declared space, the declared set is above
  **85%** of them, only **22.9%** beat the best trained probe, and the **median** prompt set
  (0.4967) loses to it. A verdict that depends on which prompt was declared is not a verdict.
- The frame-level significance also does not survive the effective sample: 907 frames are
  **95 distinct scenes**, and recounted on those, no comparison is significant.

**What replaces the hypothesis.** Prompt choice moves macro-F1 by **0.726** (0.021–0.747)
where the three trained backbones span **0.082**. The reportable finding is that variance,
not a ranking: the cost of a no-label deployment is not a fixed penalty but a wide
distribution whose position cannot be known at a new site *without* the labels that would
make it unnecessary. That is a sharper answer to RQ1's onboarding question than H6 posed, and
an operationally worse one.

**A6 quantified.** On the search's own protocol the selected prompt set leads the declared one
by **+0.4474** balanced score. A6 called such a margin "optimistically biased"; this is how
much. It is not an unbiased estimate of the bias — no held-out prompt data exists — but it
bounds what the search's headline is worth as a claim about the model.

**Rule 5 note.** The 375-set sweep run here is *not* a search and nothing is selected from
it: it is reported as a distribution, and the only statistic taken from it is the fraction
exceeding a threshold fixed by the trained probes. No correction is owed for it, and the
declared set's result is stated separately and first.

---

### 2026-09-08 — A11: H5 reported; its two clauses fail differently, and all six are now in

The last unreported hypothesis. `experiments/h5_preprocessing_switches.py` →
`results/h5_preprocessing_switches.csv`.

**The two clauses are reported separately, and must stay separate.** Collapsing them into one
H5 verdict would either manufacture a null result or bury a real one.

**Clause 1, ROI masking: unrunnable.** `configs/cameras.json` does not exist, WP3-T1 needs a
human to draw a polygon per camera, and `roi_mask` returns the frame untouched without one.
All 88 search evaluations carry `roi: False`, and `SWITCHES` deliberately excludes `roi` so
the search cannot record "ROI masking does not help" from a transform that never ran. This is
recorded as **unrunnable, not refuted** — the same distinction A9 drew for H4's decision rule
and A10 for H6's prompt dependence.

**Clause 2, CLAHE: refuted in the opposite direction.** On the grouped split, macro-F1,
paired, Holm-corrected over the realised family of **4** (not the 5 switches anticipated —
only CLAHE has arms): all four comparisons are significant and all four are **negative**.
ConvNeXtV2 loses 0.008–0.009; DINOv2 loses **0.27**, collapsing onto roughly the score of a
model that never recognises an empty pitch. Every interval lies entirely below zero, Cohen's
g ≈ 0.5 means the disagreements are wholly one-sided, and across five splits CLAHE improves
in 0 of 4 night test sets.

**The clause's "specifically" is untestable on this corpus, and that is a scope limit rather
than a result.** It needs a day column to contrast against, and venue_01 has exactly two
recording days; holding whole slots out fills the test side from one of them. Across five
seeds four test sets are entirely night and the fifth entirely day — **no grouped split holds
both**. The day figures are therefore reported as a separate observation from a different
partition and explicitly not as a contrast. This joins the "not answerable with the available
data" list rather than being counted against the hypothesis.

**All six pre-registered hypotheses are now reported**: H1 confirmed and its effect size
corrected upward (A8) then decomposed against a zero-shot control; H2 confirmed; H3 confirmed
and re-reported with the false-play control it lacked; H4 refuted with its decision rule
disowned (A9); H5 as above; H6 inconclusive with the prompt-space distribution as the finding
(A10). Three of the six required an amendment to the analysis as pre-registered, and every
amendment is recorded above rather than folded into the result.

---

### 2026-09-08 — A12: A8 corrected the numbers and A1 was never followed through

Found while building the trivial-baseline floor chart (WP4-T10), by reading the figures A1
quotes against the CSV they came from.

**A1 states that "under the leaky random split a 16-bin colour histogram scores above
ConvNeXtV2 (0.686 vs 0.657)". Both numbers were superseded by A8 and the comparison
reverses.** A8 fixed `bootstrap_metric_ci` re-deriving the class set inside the bootstrap,
which moved every random-split macro-F1. On the corrected estimand:

| random split, macro-F1 | as A1 quotes it | corrected (A8) |
|---|---|---|
| ConvNeXtV2 | 0.657 | **0.9879** |
| colour histogram | 0.686 | **0.9616** |
| verdict | histogram ahead by 0.029 | **ConvNeXtV2 ahead by 0.026** |

So the sentence is false on its own data. A8 was reported as leaving grouped-split numbers
unchanged, which is true, and the random-split correction was written up for H1 — but H2's
floor claim reads the same column and nobody re-read it. An amendment that corrects a number
has to be followed to every claim resting on it, and this one was not.

**What survives, and it is most of it.**

* **The grouped-split half is untouched**: the clock rule scores 0.4907 against ConvNeXtV2's
  0.4975 — within 0.0068, using no pixels at all. That is A1's narrower claim and the one
  that carries RQ7.
* **A2's restatement also survives, and is now the only form of the random-split claim worth
  making.** On the 62 distinct scenes in that test set the histogram and DINOv2 both score
  1.0000, so *the leaky protocol cannot tell them apart*. That was already the better
  sentence; it is now the only true one.
* **H2's pre-registered verdict is unchanged**: refuted at the 2-point threshold, since
  DINOv2 clears the floor by 8.9 points.

**What is withdrawn**: "a colour histogram beats the deep probe under the leaky protocol", in
its frame-level form, wherever it appears as a live claim. `thesis/rq_matrix.md` and
`TODO.md` are corrected; `docs/CODEBASE.md`'s branch description is corrected; the commit
subject on `exp/h1-h2-baseline-floor` is history and stays as written.

**And both numbers are now in the claims ledger**, which did not carry them. That is the
whole reason this went unnoticed for a day: the ledger checks what is in it, and a claim
outside it has nothing checking it. WP8-T5's standing instruction — add a claim when you
write the sentence — applies to amendments too.

---

### 2026-09-10 — A13: H4's speed clause was measured under load, and fails on both readings

The committed latency table had been measured **with other work in flight**, and the
codebase knew: `efficiency_latency.py`'s own docstring records that ConvNeXtV2 read 150.9 ms
in a batch against 101.2 ms idle, and `reproduce_all.py` excludes the stage from `--force` for
exactly that reason. The stage was never re-run. The number the warning describes is the
number that was published.

Re-measured on an idle machine (2026-09-10, same hardware, 4 threads):

| backbone | published (under load) | idle | concurrent x20, published | idle |
|---|---|---|---|---|
| ConvNeXtV2 | 150.9 ms | **100.5 ms** | 2388.1 ms | **1858.3 ms** |
| DINOv2 | 418.3 ms | **219.4 ms** | 5501.8 ms | **4021.5 ms** |
| ViT | 303.3 ms | **169.4 ms** | 4361.5 ms | **3023.1 ms** |

**H4's speed clause fails on both readings now, where it previously failed on one.** A9
reported 1.83x on the 20-camera concurrent median — the condition this document names — and
2.01x on the single-frame median, which it declined to rely on. Idle, those are **1.63x** and
**1.69x**. The pre-registered verdict is unchanged: the clause was refuted then and is refuted
now, more clearly.

**Why the direction matters more than the size.** Contention costs the heavier model more, so
load *inflates* a speed ratio between models of different weight. It flatters the compact
backbone — the direction that would have supported "ship ConvNeXtV2" — and it was enough to
carry the single-frame line over a 2x bar it does not clear. A benchmark run in a batch does
not simply add noise; here it added bias, with a sign.

**What else moves.** The 60-second-cycle headroom improves from 10–24x to **14–31x**, which
strengthens rather than weakens the finding that latency is not the binding constraint, so
RQ2's recommendation of DINOv2 on cross-venue evidence is untouched. `README.md`,
`thesis/rq_matrix.md`, `thesis/threats_to_validity.md` and `TODO.md` carry the corrected
figures; A9's text stands as written, with this beside it.

**Still this machine, not the deployment target.** WP7-T1 remains open: every figure here is
an AMD development laptop, and the Mini-PC has never been measured.
