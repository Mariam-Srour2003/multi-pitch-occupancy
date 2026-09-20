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

---

### 2026-09-13 — A13: generated frames are admitted, for training only

**What changes.** §"Data available" states *"No further data collection is planned."* That
sentence now has an exception: frames **generated** by an external image model, conditioned
on frames from this corpus, may enter the **training side only**. Ethics approval for the
upload was obtained from the supervisor and from the facility operator before any frame left
the machine, and is recorded in `thesis/ethics.md`.

**Why it is being considered at all.** `results/coverage.md` reports
`MAINTENANCE_NON_SPORTING` at **6 frames**, one venue, one camera, one moment, daylight only,
with `MAINTENANCE × night` empty. Six frames cannot train a class and cannot test one. The
third class is currently decorative: every macro-average in this thesis is a two-class
average carrying a three-class name.

**The three conditions, and none of them is optional.**

1. **Training only, enforced in code.** `splits.SYNTHETIC_SOURCE` marks generated rows and
   `splits.check_split` reports any that reach a test side. A model scored against its own
   generator's output measures the generator. This project's recurring defect is *guards that
   exist and do not operate*, so this one is a check, not a convention.
2. **No synthetic venue, ever.** Generated frames inherit the venue of the frame that
   conditioned them and are never given a venue of their own. A generated "venue B" is
   venue_01's pixels wearing a filter; admitting one into leave-one-venue-out would
   manufacture exactly the inflation this project exists to remove — and would do it
   invisibly, because the number would look better.
3. **Reported as an ablation, not folded in.** Every headline figure is quoted on real data.
   The synthetic contribution appears as a separate with/without row. If no such row appears
   in the thesis, the augmentation was not used.

**What it does not fix, stated here so it is not quietly forgotten.** Generation cannot
supply the two things most needed: an empty pitch **at a second venue** (item 1 of
`thesis/data_requests.md`) and **complete slots with a real verdict** (item 2, still n=2).
Both are test-side needs, and the test side must be real. A13 makes the third class trainable.
It does not make it *evaluable*, and RQ1 and RQ6 remain blocked on real footage.

**Pre-declared failure condition.** If a real-vs-generated probe on frozen features separates
the two sets at macro-F1 > 0.90, the generated frames are a distinguishable distribution
rather than an augmentation of this one, and the augmentation is withdrawn. Declared before
the frames exist so the threshold cannot be chosen after seeing them.

---

### 2026-09-16 — A14: the pitch boundary and a motion gate enter the prediction path

**What changes.** Two things are added to the deployed path, both outside the frozen backbone
and the linear probe, so the model this project studies is unchanged:

1. **The pitch boundary is applied.** `embed_batch` has always been able to pool only over
   positions inside an outline, and `classify_batch` has always accepted one. It was never
   applied to anything: `configs/roi.json` holds outlines named `cam` and `cam2` and the
   corpus has 99 cameras, none called that, so `roi.get` returned None for every frame.
   `scripts/derive_roi.py` measures an outline per camera from the footage and `roi.load_all`
   now reads them underneath the hand-drawn store. `run_slot` passes each camera its own.
2. **A motion gate.** If a frame is called ACTIVE_PLAY but differs from the previous frame of
   the same camera by less than **1.098** mean absolute greyscale difference at 160x90, the
   verdict becomes EMPTY. One direction only - stillness is evidence against a match,
   movement is not evidence for one.

**Why, and the evidence is not from the corpus.** On a 234-second clip from a venue with no
labelled frames in the dataset, hand-labelled at 15-second samples:

| | false-play on 13 empty frames |
|---|---|
| whole frame, no boundary | 0.74 |
| boundary only | 0.38 |
| boundary + motion gate | **0.15** |

The threshold was fitted on venue_01's recorded frames and applied to that clip unchanged, so
this is transfer rather than a fit.

**Both were measured as useless first, and that is the finding worth recording.** On venue_01
camera B - the only split in the corpus with a real class mix - ROI pooling scored -0.0019 and
the motion rule -0.0159, and both were written up as negative results. That split cannot show
either effect: the model scores 0.9386 on it, so there is no failure for either to repair.
**The corpus contains no test set on which an intervention aimed at cross-venue failure can be
seen to work**, which is item 1 of `thesis/data_requests.md` arriving from a fourth direction.

**What this does not fix.** A person walking across an empty pitch moves, so the gate says
PLAY, and they are inside the boundary. That is `3_people_not_playing`, six recorded frames in
the whole corpus, and no cue computed from two frames supplies it.

**Reported as an operational change, not a result.** Every headline figure in this thesis is
the probe on frozen features and is unaffected. The boundary and the gate change what the
deployed system answers, and their evidence is one clip at one venue - enough to adopt them in
the product, not enough to claim a measured improvement to the model.

---

### 2026-09-16 — A15: the training side is pruned to one frame per scene

**What changes.** `splits.distinct_rows` keeps one frame per distinct scene, and the training
side of every experiment that fits a probe may use it. **Test sides are never pruned**, because
deduplicating a test set changes what its number means.

**Why.** Measured with a perceptual hash within (venue, class): the corpus is **1,881 frames
and 290 scenes**. The EMPTY class is **525 frames and 28 scenes**, and the five recorded ones
are one venue, two cameras, two days. Fitting on all of them tells the probe that those five
backgrounds *are* what an empty pitch looks like, with the confidence a hundred observations
would justify and five do not. Class weighting does not help: it balances EMPTY against PLAY by
**count**, not by scene.

**Every number that depends on the fit, full against pruned:**

| test | metric | full | pruned |
|---|---|---|---|
| H3 cross-venue, 7 folds | play-recall | 0.9444 | **1.0000** |
| H3 cross-venue, 7 folds | false-play on 243 recorded EMPTY | 0.7684 | **0.6173** |
| H3 cross-venue, 7 folds | balanced | 0.1761 | **0.3827** |
| venue_01 camera B | macro-F1 | **0.9386** | 0.8420 |
| venue_01 camera B | false-play | **0.0000** | 0.0288 |
| unseen clip, boundary + gate | false-play | 0.31 | **0.00** |

**The one number that gets worse is the one measured on the repeated scenes.** venue_01 camera
B is drawn from the same five EMPTY backgrounds the duplicates come from, so 0.9386 says how
well the probe reproduces what it memorised. Cross-venue recall goes to 1.0000, cross-venue
false-play falls by 0.15, and on unseen footage false-play reaches zero - while the pruned
probe still says ACTIVE_PLAY twice in sixteen minutes, both times with a person on the pitch.
It is not conservative, it is correct.

**What this costs the thesis, said plainly.** Every figure quoted as "1,692 frames" is a
sampling rate, not a sample. The honest description of this dataset is **290 scenes**, and the
EMPTY class is **28**, of which 23 are generated. `effective_sample_audit` already reported
this for evaluation; A15 is the same correction applied to the fit.

**Scene ids are a sidecar**, `data/processed/scene_ids.csv`, not a manifest column:
`build_manifest` regenerates the manifest from filenames on disk and would drop the 189
generated rows that `ingest_synthetic.py` wrote into it.

---

### 2026-09-17 — A16: a person count inside the boundary enters the prediction path

**What changes.** `vision/people.PersonGate` overrules ACTIVE_PLAY with EMPTY when a detector
finds **nobody standing inside the camera's boundary**. One direction, like A14's motion gate.
The frozen backbone and the linear probe are untouched, so every headline figure stands.

**The evidence.** 521 recorded frames, venue_01 camera B, counted inside the boundary:

| class | n | median count | zero |
|---|---|---|---|
| EMPTY | 243 | 0 | **89%** |
| ACTIVE_PLAY | 278 | 6 | **0.4%** |

A count threshold alone beats the fitted probe on the split the probe was tuned on:

| | recall | false-play | balanced |
|---|---|---|---|
| person count, `PLAY if >= 2` | 0.9604 | 0.0453 | **+0.9152** |
| probe, full training set | 0.8849 | 0.0000 | +0.8849 |
| probe, pruned to distinct scenes | 0.7302 | 0.0288 | +0.7014 |

**End to end on an unseen clip**, through `run_slot` with all three additions:

| configuration | false-play on 13 empty minutes |
|---|---|
| no boundary, no gates | 0.74 |
| boundary only | 0.38 |
| boundary + motion gate | 0.15 |
| **boundary + motion + person gate** | **0.00** |

The person gate fixes precisely the two the motion gate cannot: the first frame of a camera,
which has no predecessor and so no motion cue, and a frame where something outside the pitch
moved enough to clear the motion threshold.

**Why it is a gate and not the classifier.** It is measured at one venue. A rule that beats a
trained probe on one split is a promising rule, not a replacement for the thing the thesis is
about - and the project's own history says a number measured on venue_01 camera B may not mean
what it appears to.

**What was measured and deliberately not adopted.** The three-class version of the rule fails:
32% of genuine ACTIVE_PLAY frames show four or fewer people inside the boundary, because a
camera sees part of a pitch and a detector misses distant players. "One to four people means
not playing" would be wrong on 88 real matches out of 278. The count is evidence toward C3 and
never a verdict of it.

**`detect_people` was running at 640 pixels.** The parameter was not exposed, so every caller
got the default, which downscales a 1080p frame until a distant player is a few pixels across.
That is most of why `synthetic_data_protocol.md` §3a recorded the detector as finding nobody on
frames with people in them. At 1280 the counts separate the classes. §3a's measurement stands
for what it examined - people in a dugout, small, occluded, behind a barrier - which remains a
harder problem than a person standing on a pitch.

**Cost.** About a second per frame on CPU, and the detector runs only when the verdict is
ACTIVE_PLAY, since that is the only verdict this gate can change. At one frame per camera per
minute that is affordable.

---

### 2026-09-17 — A17: the ball is detected and recorded, and is not allowed to decide

**What changes.** `vision/people.detect_inside` returns people *and* ball from a single
detector pass; `PersonGate.inspect` returns both; `SlotRun` carries `people_counts` and
`ball_minutes` to the caller. `explain.detect_objects` generalises `detect_people` so the two
classes come out of one forward pass, which is why the ball costs nothing. **No verdict
changes.** The gate's rule is unchanged - nobody inside the boundary overrules ACTIVE_PLAY -
and a frame with a ball and no people is still overruled.

**Why it was measured.** It is the third part of the rule the supervisor asked for, and the
missing half of the second: the person count cannot carry C3 because 32% of genuine
ACTIVE_PLAY frames show four or fewer people (A16), and *four people with a ball* is a
kickabout where *four without* is not.

**The venue_01 evidence, which is the best in the project and is not the answer.** 521
recorded frames, camera B, COCO `sports ball` at confidence 0.10 inside the boundary:

| | recall | false-play | balanced |
|---|---|---|---|
| **ball found inside the boundary** | **1.000** | **0.033** | **+0.967** |
| person count, `PLAY if >= 2` | 0.9604 | 0.0453 | +0.9152 |
| probe, full training set | 0.8849 | 0.0000 | +0.8849 |

**The cross-venue evidence, which is.** All 396 recorded clip-venue frames, every one genuine
ACTIVE_PLAY: a ball is found in **40%**, by venue from 6% to 89%. The person count finds people
in **100%** of the same frames. On 60% of real play at an unseen site there is no ball to find,
so **the absence of a ball is not evidence of the absence of play** - exactly the direction a
C3 rule would have to lean on.

**Why the 1.000 is not believed.** 278 frames carrying **13 distinct scenes**, against 3 for
the EMPTY side; a sample of 70 landed 69 times in one scene. Perfect separation over 13 scenes
at one site is what this corpus produces for almost everything it is asked.

**The detections themselves were checked, not assumed.** Within the dominant scene the ball's
centre has sd 0.10 of frame width and 0.06 of frame height and ranges across most of the
pitch, so it is not a fixture; median box area is 404 px (p10-p90 347-536) at venue_01 and 435
px at the clip venues, which is a football at that distance and not a head or a bag.

**Threshold.** `BALL_CONFIDENCE = 0.10`, below the person threshold of 0.25 and deliberately
so: a football is ~400 px and the detector is unsure of it. That is affordable *only because
the ball decides nothing* - a false ball is a wrong note in the record where a false person
would be a wrong verdict. A test asserts `inspect` never reads the ball when deciding, so if
the ball is ever promoted, the promotion has to argue with the threshold first.

**Risk this amendment accepts.** A recorded signal that looks decisive at one venue invites a
later reader - or a later me - to promote it. The log, this entry and the test each say the
same thing in a different place, which is the only protection available against a number that
reads as +0.967.

---

### 2026-09-17 — A18: a small group with no ball is a C3 verdict

**What changes.** `PersonGate` gains a second overrule: ACTIVE_PLAY becomes
C3_MAINTENANCE_NON_SPORTING when the detector finds **1 to `small_group_max` people inside the
boundary and no ball**. Still one direction - a play verdict can be weakened and never
manufactured - and a test checks that property over every combination of count and ball. The
frozen backbone and the linear probe are untouched, so every headline figure stands.
`PersonGate(small_group_max=0)` restores A16 exactly.

**Why this reverses A16's refusal.** A16 refused "1-4 people means not playing" on the strength
of the 88 real matches out of 278 it would have mislabelled. That tested half the rule. The
rule as stated had a second clause - and a ball - and with it:

| frames | n | count alone | with the ball clause |
|---|---|---|---|
| venue_01 camera B, ACTIVE_PLAY | 278 | 88 (31.7%) | **0 (0.0%)** |
| nine clip venues, all ACTIVE_PLAY | 396 | 9 (2.3%) | **3 (0.8%)** |

Both rows are entirely genuine play, so both columns are errors introduced. **0.8% across 396
frames at nine unseen venues is the best-evidenced cross-venue cost in this project.**

**What protects the rule is the count, not the ball.** A found ball vetoing C3 is sound in any
venue. The rule also requires the ball to be absent, which is *not* sound - 60% of real play at
an unseen venue shows no detectable ball (A17) - and the count is what bounds it: median 10
people inside the boundary at those venues, so the unsound clause is consulted on 9 frames out
of 396. A camera framing less of its pitch would break this, and that is the condition under
which the rule should be withdrawn.

**What is not evidenced.** The corpus holds **6 recorded C3 frames**, one slot at one camera,
and the rule identifies **1**. A cost measured on 396 frames against a benefit measured on 6 is
not a balanced case, and this amendment claims only that the rule is safe.

**Why it was adopted on that basis.** Before it, the deployed path could not return C3 at all:
two gates both pointing at EMPTY, and a probe with 6 real frames of the class. An unmeasurable
recall at a measured cost below one percent is preferable to a system structurally incapable of
the answer - but the preference is a judgement, not a result, and it is the supervisor's to
overturn with one argument.

**End to end on the unseen clip**, through `run_slot` with boundary and all three gates:
false-play stays 0/13, minutes 9 and 15 - one person walking, no ball - become C3, and the slot
verdict moves from REVIEW to **NOTUSED**, "empty in 88% of samples with only 0% active play".
Minute 12 remains wrong: a person is visible and the detector finds nobody inside the boundary,
so A16 overrules it to EMPTY before this rule is consulted. That is a detector limit, not a
rule limit, and three of the six recorded C3 frames fail the same way.

**Risk this amendment accepts.** C3 is now reachable in production with essentially no recorded
evidence behind its precision. The class was previously unreachable, which was a different and
quieter error - the system reported no maintenance because it could not, not because there was
none.

---

### 2026-09-17 — A20: the cross-venue numbers are re-reported for the system, not the probe

**Corrected the same day, and the correction is larger than the amendment.** False-play does not count C3 verdicts on an empty pitch, and the gate produces them: 24 of the 243 control frames, on a camera whose boundary reaches past the goal line into the car park, so people on tarmac behind the fence are counted as on the pitch. Counting every wrong verdict, the probe answers EMPTY on **none** of the 243 frames in any of the seven folds - 0.6173 measured which *kind* of wrong it was, not whether it was wrong - and the gate takes total error from **1.0000 to 0.4844**, not to 0.0123. It only weakens ACTIVE_PLAY, so the frames the probe calls C3 pass through untouched. The thesis must report two numbers: false-play 0.6173 to 0.0123, total error 1.0000 to 0.4844.

**What changes.** No code. An evaluation that was missing: every cross-venue figure in this
project describes the probe, and the deployed path has not been the probe alone since A14.
`experiments/h3_with_gates.py` re-runs H3's seven leave-one-venue-out folds with the person
gate and its A18 extension applied to the verdicts, which is the order `run_slot` uses.

| arm | play-recall | false-play | balanced |
|---|---|---|---|
| pruned, probe | 1.0000 | 0.6173 | 0.3827 |
| pruned, **gated** | 0.9991 | **0.0123** | **0.9868** |

**Why the result is arithmetic rather than luck.** Of the 243 control frames, 216 have nobody
inside the boundary and become EMPTY, 24 have one to four people and no ball and become C3, and
3 survive - the frames with a bystander *and* a ball, where the ball clause vetoes C3 by design.

**The half of this that is not a transfer result.** Recall is measured at held-out venues and
transfers. False-play is measured on the same 243 frames that every gate constant was read
from - detector size 1280, person confidence 0.25, ball confidence 0.10, `small_group_max = 4`.
The gate is out of sample with respect to the probe's training and in sample with respect to
its own thresholds, so **0.0123 is a floor**. The only out-of-sample check is the unseen clip:
0 false-play on 13 empty minutes, no wrong C3 calls, thresholds not derived from it. Thirteen
minutes is not a rate.

**What this obliges the write-up to say.** Not "the cross-venue failure is fixed". The probe's
cross-venue precision is unchanged and remains unusable; what changed is that the reported
system is no longer the probe. The thesis should report the gated figure as the system's and the
0.6173 as the probe's, together, because dropping either one misrepresents which component is
doing the work.

**Risk this amendment accepts.** A reader who takes 0.0123 as a measured cross-venue rate will
overestimate the system at a new site. The floor-not-estimate wording is load-bearing and must
survive into the thesis text.

---

### 2026-09-17 — A21: the detector is cached, and the cycle cost is re-measured for the gates

**What changes.** `explain.detect_objects` keeps one loaded detector per thread instead of
calling `YOLO(model_name)` on every frame. No output changes; only the clock does. Per thread
rather than globally, because `evaluation/latency.py` drives the pipeline from several threads
and an ultralytics model is not documented as safe to predict on from more than one.

**Why it was worth doing.** Eight 1080p frames at imgsz=1280: **292 ms per frame constructing,
152 ms reusing**. Building the model cost more than running it, and across 20 cameras that was
2.8 s of every 60 s cycle spent loading the same weights twenty times. Two tests pin it, since
nothing else can see the difference.

**The evaluation that was missing.** `efficiency_latency.csv` measures the probe, and RQ1 and
RQ2 both quote it; the deployed path has been probe plus two gates since A14.
`experiments/gate_latency.py` reports the round by play rate, because the detector only runs on
verdicts it could change and an empty site pays nothing:

| cameras in play | round | of the 60 s cycle |
|---|---|---|
| 0 of 20 | 6.7 s | 11% |
| **20 of 20** | **10.6 s** | **18%** |

**What is claimed and what is not.** This is a laptop, and so is every other timing in this
project; WP7-T1 on the target Mini-PC settles the deployment claim. What holds independent of
the machine is the shape: the gates add a term proportional to the play rate rather than a
constant, and that term is smaller than the backbone's own.

**Risk this amendment accepts.** A cached model is process state, and a caller that expected a
fresh model per call - swapping weights on disk mid-run, say - would now get the old one until
the process restarts. Nothing in this system does that, and the per-name cache means changing
`model_name` still loads a new model.

---

### 2026-09-17 — A22: the empty-pitch rule applies to a C3 verdict as well as a play verdict

**What changes.** `PersonGate` runs for any verdict except EMPTY, instead of only for
ACTIVE_PLAY. Finding nobody inside the boundary now overrules C3 as well as ACTIVE_PLAY. The
A18 small-group rule still applies to ACTIVE_PLAY only, since C3 is already that verdict.

**Why the restriction was wrong.** It was inherited from A16, where the gate was introduced as
a brake on false *play*. The evidence behind the rule - 89% of recorded empty frames show
nobody inside the boundary, 0.4% of recorded play frames do - says nothing about what the probe
guessed first. And the probe guesses C3 constantly: on the 243 recorded empty control frames it
answers ACTIVE_PLAY or C3 and **never EMPTY**, C3 on 38% of them on average.

| arm | play-recall | false-play | any wrong verdict on an empty pitch |
|---|---|---|---|
| pruned, probe | 1.0000 | 0.6173 | 1.0000 |
| pruned, gated before A22 | 0.9991 | 0.0123 | 0.4844 |
| pruned, gated **after A22** | 0.9991 | 0.0123 | **0.1111** |

The largest single improvement in this project, produced by deleting a condition rather than
adding an idea.

**What it costs.** Three of the six recorded C3 frames now read EMPTY - the ones where the
person is beside the pitch rather than on it, which the gate cannot distinguish from nobody
being there. Half the recorded C3 corpus, and that corpus is six frames from one slot.

**A labelling question the supervisor should settle.** For a system asking *was this pitch
used*, EMPTY may be the right verdict for a frame showing someone beside the pitch, and the
label the loose one. §2.5's "people present, not playing" does not say whether "present" means
present *on the pitch*. The boundary already assumes it does.

**The invariant this makes explicit.** ACTIVE_PLAY above C3 above EMPTY is a ladder of how much
activity is claimed, and every overrule the gate makes is a step down it - A16 play to empty,
A18 play to C3, A22 C3 to empty. A test checks it over every combination of count and ball, so
detector noise can never manufacture a busier pitch than the probe reported.

**Risk this amendment accepts.** C3 recall is now bounded by the boundary's accuracy, on a
class with six recorded frames. If the labelling question is settled the other way - present
means present anywhere in view - this amendment should be withdrawn rather than adjusted.

---

### 2026-09-17 — A25: the lighting labels were wrong, and a headline claim reverses (WP3-T5)

**What changes.** The `lighting` column for **216 of 396** recorded clip frames, from `day` to
`night`. No code in the prediction path, no model, no split definition.

**The bug.** `data/extract.py` assigns clip lighting by mean greyscale brightness, below 80 is
night, "calibrated against the known day/night recordings in raw/venue_01". A floodlit
five-a-side pitch fills its frame with intensely lit turf and reads *brighter* than an overcast
afternoon at venue_01 - 120 against 69 - so the rule files night football as daylight. It
measures how bright the picture is, not whether it is daytime.

**The audit.** One rendered frame per venue, evidence recorded per venue in
`scripts/relabel_clip_lighting.py`: black skies, lit floodlight fixtures, a burned-in timestamp
reading 21:02, indoor halls with no sky in frame. **Not one of the nine clip venues shows
daylight.** Every daylight frame in this corpus is now venue_01's.

**What reverses.**

| model | cross-venue play-recall | false-play | recall was |
|---|---|---|---|
| **clock_rule** | **1.0000** | **0.0206** | 0.219 |
| dinov2 | 0.9297 | 0.3086 | 0.930 |
| convnextv2 | 0.9105 | 0.9918 | 0.910 |
| vit | 0.8690 | 0.8354 | 0.869 |

A rule that never looks at the image beats all three frozen backbones on recall *and* on
false-play. `rq_matrix.md`'s "a lighting-only rule collapses across venues while the backbones
hold above 0.86" is **refuted**, and `defence_deck.md` slide 7 - which called the collapse "the
strongest single piece of evidence that the backbones learn something transferable" - is
rewritten to say the opposite and to say that it used to say the opposite.

H2 moves with it: on the grouped split the clock rule goes from 0.4907 to **0.4975**, which is
*exactly* ConvNeXtV2's and ViT's score, and DINOv2's lead over it goes from p_holm 0.0037
("differs") to **p_holm 0.375 ("indistinguishable")**.

**Why the rule wins, which matters more than the table.** The confound is not a labelling
mistake, it is how five-a-side pitches are used - people play in the evening:

| | EMPTY | ACTIVE_PLAY | C3 |
|---|---|---|---|
| day | 485 | 6 | 6 |
| night | 9 | 1186 | 0 |

"Night means play, day means not-play" is right on **1,677 of 1,692** recorded frames, 99.1%,
and on 98.8% within venue_01 alone. No experiment in this project can separate *recognises an
empty pitch* from *recognises daylight*.

**What this obliges.** The data request, stated precisely for the first time: the corpus needs
**empty pitches at night** and **play in daylight**, the two cells holding 9 and 6 frames.
"Empty pitches at another venue" was never the whole gap.

**Risk this amendment accepts.** A reader may take "the clock rule beats the backbones" as a
result about backbones. It is a result about this dataset, and every place the figure appears
now carries that sentence.

---

### 2026-09-17 — A27: what the locked final test set can and cannot answer, disclosed before it is opened

**What changes.** Nothing in code or data. A disclosure, made while the set is still sealed,
because the alternative is making it after a number exists.

**Composition**, readable from the manifest without opening the set: **114 frames, two venues,
19 clips, and one class** - C2_ACTIVE_PLAY throughout, night lighting throughout, no EMPTY, no
C3, no daylight.

**It can measure play-recall at two unseen venues, and nothing else.** Not false-play, not
precision, not macro-F1, not the EMPTY or C3 classes, not any lighting comparison, not RQ6's
operating point. **A model answering ACTIVE_PLAY unconditionally scores 1.000 on it**, and A25
showed that is not hypothetical - the clock rule does exactly that.

**The commitment this amendment makes.** When the set is opened, its result is reported as a
held-out confirmation of play-recall at two venues and never as an accuracy or a headline. A
single-class test set flatters every model, and the disclaimer is registered here, before the
figure exists, so it cannot be read as a response to whatever the figure turns out to be.

**What would make it a real test.** Recorded EMPTY frames from
`clipvenue_b_floodlit_track` or `clipvenue_c_teal_boards` specifically. Any other venue is
development data and cannot join a locked set afterwards.

**Risk this amendment accepts.** Spending the set as described buys little. The alternative -
unlocking it to rebalance - would destroy the only untouched evaluation this project has, and
is worse.

---

### 2026-09-19 — A36: the prediction path is rebuilt detector-first; the probe becomes a comparator

**What changes.** The deployed classifier stops being a frozen backbone with a linear head
that gates may weaken, and becomes a **detector with an explicit rule** that the backbone no
longer sits in front of. A person-and-ball detector counts what stands inside the camera's
boundary; a written decision table turns the count into one of the four folder classes or an
abstention; the DINOv2 probe, the clock rule and the gated probe stay in the repository as
the things the new path is measured against. Nothing already published is edited; the tables
that change are re-issued beside their predecessors.

**Why now, and why this far.** Three findings, none new to this document, add up to a change
of method rather than another gate:

1. The probe never answers EMPTY at a camera it has not seen - 0 of 243 held-out empty frames
   across seven folds (A20), and ACTIVE_PLAY at confidence 0.98 on an empty floodlit pitch
   (A26). One labelled empty frame from the target camera repairs it (`empty_recognition.csv`),
   which says the probe recognises cameras, not occupancy.
2. Its accuracy survives an eight-pixel blur and dies under greyscale-plus-crop
   (`input_ablation.csv`): the surviving signal is scene and lighting, which is the confound
   A25 and A26 measured, and the corpus cannot supply the frames that would break it.
3. The rule the gates already apply - nobody inside means empty, a count inside means
   present - beats the probe on the only split with a class mix, without training (A16), and
   is what got 13 of 13 empty minutes right on the real clip while the probe alone got 8 (A26).
   The gates were allowed to weaken a verdict and never to make one. This amendment lets them
   make it.

Two wiring defects are closed on the same branch, recorded here because they mean the deployed
path had been *worse* than every measured one: `scheduler.run_due` never passed the boundary or
the gates to `worker.run_slot`, so `worker --source video|live` ran the bare probe; and no key
in `configs/roi.json` or `roi_derived.json` matched a production camera id, so the boundary
was `None` wherever it mattered. A35 found the same defect on the review page; this is it one
layer up.

**The decision table**, per pitch and per minute, first matching row wins. `n` is the number
of people whose feet stand inside the boundary, persisted over a burst (below) and **summed
across the pitch's cameras**; `ball` is a ball seen inside the boundary in any burst frame;
`m` is the motion cue inside the boundary across the burst; `spread` is the mean pairwise
distance between people relative to the boundary's diagonal.

| # | condition | state | note |
|---|---|---|---|
| 1 | detector unavailable | UNCERTAIN | a missing detector is not an empty pitch |
| 2 | no boundary for this camera | UNCERTAIN | the boundary is now mandatory |
| 3 | `n = 0`, motion low or unmeasured | **EMPTY** | |
| 4 | `n = 0`, motion high | UNCERTAIN | something moved and nobody was found |
| 5 | vehicle inside, or hi-vis person with `n ≤ 4` | MAINTENANCE | best effort; unevaluable, see below |
| 6 | `1 ≤ n ≤ 4` | PEOPLE_NOT_PLAYING | with or without a ball |
| 7 | `n ≥ 5`, ball seen | **PLAYING** | highest confidence |
| 8 | `n ≥ 5`, no ball, motion very low, people clustered | PEOPLE_NOT_PLAYING | kept only if it costs < 1% of real play minutes; otherwise disabled in config, not deleted |
| 9 | `n ≥ 5`, no ball | **PLAYING** | lower confidence; ball absence is not evidence |

UNCERTAIN is a first-class per-minute state. `aggregate_slot` treats an abstained minute as a
missed one - it counts against the capture floor and can push a slot toward REVIEW and nowhere
else - so the REVIEW-never-accuses guarantee of `slots/reconcile.py` is unchanged.

**Pitch-level, not camera-level.** A16 measured that "one to four people means not playing"
applied per camera is wrong on **88 of 278** real matches at venue_01, because a camera sees
half a pitch and a detector misses far-side players. The count that the rule thresholds is
therefore the sum over the pitch's cameras for the same minute; per-camera counts are
evidence and are recorded, not classes. On a single-camera pitch the two coincide, and the
under-count risk is stated wherever a single-camera number appears.

**What the ball may and may not do.** A17 stands. Ball recall on genuine play is 0.06-0.89 by
venue (`ball_detection_rule.csv`); its presence raises the PLAYING confidence and its absence
lowers it, but absence alone never changes the class when five or more people stand on the
pitch. Row 8 is the one exception and it demands two independent cues besides the missing
ball; its cost is measured before it is enabled.

**Burst sampling.** Each camera is read as a short burst - three frames about a second apart -
once per minute, instead of one frame. A ball seen in any of the three counts; the burst
supplies the motion cue without waiting for the previous minute; a person must persist in two
of the three frames to be counted, which is the filter against a goalpost or a bag read as a
person (11% of camera B's recorded empty frames have one, A16). This changes the sampling
protocol stated in `thesis/protocol.md` and is the reason this is an amendment rather than a
code change.

**The truth changes, and this is the risk that matters most.** `labelling_protocol.md` §2.2
labelled a match "at any number of players" and §2.6 rule 1 repeated "however few". This
amendment adopts the rule the client stated on 2026-09-13 and again on 2026-09-19: **PLAYING
is more than four people engaged in play; four or fewer, with or without a ball, is
PEOPLE_NOT_PLAYING.** The threshold is a requirement from the facility, not a fitted number,
and is recorded as such so it cannot be read as an after-the-fact cutoff. Consequences:
`labelling_protocol.md` is amended with the old text struck through; the 11 held frames in
`_pending_4d/` are filed as `3_people_not_playing`; a hand-count audit over the 1,192
ACTIVE_PLAY frames identifies those with four or fewer real people, which are relabelled and
listed; every published table that counts ACTIVE_PLAY is re-run and issued beside the old one.

**What is tuned, on what, and what is not tuned.**

| parameter | status |
|---|---|
| `play_min = 5`, `small_group_max = 4` | requirement; not tuned |
| detector model, input size, tiling | chosen on a hand-counted sample of 100 development frames plus measured latency; the rule for choosing is written before the numbers exist |
| person confidence, minimum box height, ball confidence, burst-gap motion thresholds, cluster threshold | fitted on **venue_01 camera A only**, then frozen in `configs/rules.json` with the commit hash |
| burst length and spacing | fixed by design; sensitivity reported |

Never touched by tuning: venue_01 camera B (the 243 empties and 278 play frames that every
false-play number in this document rests on), the nine clip venues, the unseen floodlit clip,
any public footage admitted for evaluation, and the locked final venues.

**Evaluation commitments.** The rule has no training set, so every recorded frame that is not
in the tuning camera is evaluation data. Every reported recall is paired with the false-play
rate and EMPTY accuracy on camera B; ranking is on `recall − false_play`; the clock rule and
the DINOv2 probe (bare and gated) appear on every table the detector-first path appears on;
every number carries a bootstrap interval; the abstention (UNCERTAIN) rate is reported as a
first-class figure beside recall, because a path that abstains its way to a low false-play
rate has not earned it. The 4-class confusion is reported with `3 ↔ 4` marked as the accepted
confusion, since the client accepts it and the data cannot measure it.

**What remains unevaluable, and is said so.** MAINTENANCE has zero recorded frames; row 5 is
smoke-tested on generated frames only and is labelled best-effort wherever it appears. Motion
thresholds are fitted at one camera and their transfer is unmeasured, which is why motion is
allowed to abstain and never to promote. The far-side recall of a COCO person detector on
fisheye CCTV bounds PLAYING recall, and that bound is reported rather than argued around.

**Risk this amendment accepts.** A venue where the detector misses far-side players will
under-count, report PEOPLE_NOT_PLAYING or abstain, and send the slot to REVIEW instead of
USED - a recall cost, taken deliberately, in exchange for EMPTY becoming answerable at a camera
the system has never seen. The relabelling under the new threshold changes every published
PLAY count; those tables are re-issued, not edited. And the method now rests on a detector
whose training data this project did not choose, which is stated as a threat to validity in the
same words the backbones already carry.

---

### 2026-09-20 — A40: the rule answers three classes, and ACTIVE_PLAY must be shown

**What changes.** Two things, both at the client's instruction, both narrowing what the
system is allowed to claim.

**1. The prediction path answers three classes, not four.** `vision/rules.MinuteState` now
carries `C1_EMPTY`, `C2_ACTIVE_PLAY`, `C3_MAINTENANCE_NON_SPORTING` and `UNCERTAIN`. A36 gave
it the four *folder* values so that row 5 could answer `4_maintenance` separately from
`3_people_not_playing`. That distinction was never measurable: the corpus holds **6 real
`3_people_not_playing` frames and 0 real `4_maintenance` frames**, which is why A36 itself had
to write "the 4-class confusion is reported with `3 ↔ 4` marked as the accepted confusion,
since the client accepts it and the data cannot measure it". A cell that is forgiven in
advance is not a measurement, and a branch the data cannot evaluate does not belong in the
deployed path. The labelling folders are untouched - `data/taxonomy.to_class3` still collapses
them, `from_class4` still maps either onto C3 - so if maintenance footage ever arrives the
finer label is still on disk and the split can be re-opened as its own amendment.

Row 5 of A36 (vehicle inside, or hi-vis person) therefore stops producing a class of its own.
The cue is not deleted: a vehicle or a hi-vis person inside the boundary is recorded in the
verdict's trace and in `PitchCount`, so it is available to a later amendment and visible to an
operator reading a verdict. It simply no longer decides anything, because there is nothing to
check it against.

**2. ACTIVE_PLAY requires more than four people *and* a ball *and* movement.** A36's rows 7-9
made play the default above the head count: five or more people were PLAYING, a ball only
raised the confidence, and row 8 (still and clustered, no ball) was the sole escape and
disabled by default. The client's rule is the opposite and was stated plainly: a match has a
ball in it and people moving; a crowd standing on a pitch is not a booking being used. So the
burden of proof moves onto play. The table is now seven rows:

| # | condition | state | note |
|---|---|---|---|
| 1 | detector unavailable | UNCERTAIN | a missing detector is not an empty pitch |
| 2 | no boundary for this camera | UNCERTAIN | mandatory on the deployed path |
| 3 | `n = 0`, motion low or unmeasured | **C1 EMPTY** | nobody, and nothing moving |
| 4 | `n = 0`, motion high | UNCERTAIN | something moved and nobody was found |
| 5 | `1 ≤ n ≤ 4` | **C3** | too few for a game, ball or no ball |
| 6 | `n > 4` **and** a ball **and** motion | **C2 ACTIVE_PLAY** | the only way into play |
| 7 | `n > 4`, otherwise | **C3** | a crowd that is not playing |

`n` is still the count summed across the pitch's cameras (A36), and the ball still ORs across
them and across the burst - which matters more now than it did, because a pitch whose ball is
only ever visible to one camera would otherwise be scored C3 on both halves.

**A required cue that cannot be checked is reported, never assumed.** `require_motion` is on,
but `motion_play_min` is null until the WP9-T5 fit. Rather than let a requirement silently
never fire, `decide` skips the clause and writes *which* clause it skipped into the verdict's
trace, on every frame. The same holds for a still image, which has no burst and therefore no
motion cue. This repo has now found three guards that were not guarding (`scheduler.run_due`
passing no gates, no ROI key matching a production camera, the walkthrough page running the
probe alone); a fourth that announces itself is the cheapest available insurance.

**Both requirements are switches in `configs/rules.json`,** because turning them off is a
decision somebody should be able to take and to see taken.

**Risk this amendment accepts.** A17 measured cross-venue ball recall at **0.40, ranging
0.06-0.89 by venue** - 0.13 at `outdoor_trees`, 0.17 at `outdoor_bldg`. `require_ball` turns
every one of those misses into a genuine match reported as C3. On the venues where the
detector cannot see the ball this will cost the majority of real play minutes, and those slots
will read NOTUSED or REVIEW rather than USED. That is a much larger recall cost than A36's,
and it is taken at the facility's instruction rather than because a measurement supports it.
What this amendment commits to is measuring it rather than arguing about it: `rule_frame_eval`
is re-run with `require_ball` on and off, both arms published in `results/rule_frame_eval.csv`
with per-venue play recall and bootstrap intervals, so the cost of the facility's rule is a
number in the thesis and not a footnote. If that number is unacceptable the switch is where to
turn, and the switch is in the config with its reasoning beside it.

Two smaller risks come with it. Dropping the fourth class means the system can no longer even
*claim* to distinguish groundskeeping from a group standing about - previously it claimed to
and could not be checked, so this is a loss of a claim rather than of a capability, but it is
a narrowing of scope and is stated as one. And `results/rule_confusion_4class.csv` becomes
`rule_confusion_3class.csv` with a `truth_folder` column; the earlier log entry naming the old
filename is left as written, with a dated correction appended beside it.
