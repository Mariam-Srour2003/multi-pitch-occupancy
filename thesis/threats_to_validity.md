# Threats to validity (WP8-T3)

**Read this as a draft you have to make your own.** Every number below is in
`thesis/claims.md` and re-derived from its artefact on each run, so the facts are checked. The
*arguments* are not checked by anything, and an argument you have not made yourself falls
apart on the first follow-up. Rewrite each in your own words before the mock examination.

This chapter is organised by the four kinds of validity rather than by work package, because
an examiner asks "how do you know that measures what you say it does" and not "what did WP4
find". Within each kind the threats are sorted by how much damage they do to the thesis's
claims, worst first.

Three labels are used throughout and the distinction matters more than the taxonomy:

| | meaning |
|---|---|
| **Mitigated** | the design removes it, and something in the repository fails if the mitigation is removed |
| **Quantified** | it is still present, and its size is measured and reported |
| **Unquantifiable** | it is present, its size is not knowable from this data, and no amount of analysis will change that |

**The third category is the honest contribution of this project.** A thesis that reports only
mitigated threats has either solved a very easy problem or stopped looking. Several of the
findings below began as an attempt to mitigate something and ended as a measurement of how
large it is.

---

## 1 · Construct validity — does the measurement measure the thing?

This is the section where this project has the most to say, and most of it is uncomfortable.

### 1.1 The false-play rate is not a specificity measure *(quantified — and a correction)*

The strongest single threat, because it affects a number quoted across the write-up.

`false_play_rate` counts how often a model calls a held-out empty pitch a match. It has been
read throughout this project as "how well does the model recognise an empty pitch", with
DINOv2's **0.309** contrasted against ConvNeXtV2's **0.9918** as a headline strength. Asking
what the models answer *instead* of ACTIVE_PLAY shows that reading is wrong:

- DINOv2 answers ACTIVE_PLAY 75 times and MAINTENANCE 168 times on the 243 frames. **It is
  correct zero times** (`false-play-is-not-accuracy`).
- The best model on that set is ViT, at 40 of 243 — no model exceeds 0.165.

So the rate ranks models by *where they put their errors*, not by whether they are right. Both
published numbers remain correct as stated (`false-play-convnextv2`, `false-play-clock-rule`);
what does not follow is the inference. **Quote the rate as a rate.**

The mechanism is worse than a labelling accident. The protocol trains on camera A and tests on
camera B, so it is a **camera-transfer test**: no model has ever seen the camera it is scored
on. Give it one labelled empty frame of that camera and every backbone reaches **0.9793**
(`camera-transfer-one-frame`). The intermediate condition — camera B present in training
through its *playing* frames only — is itself confounded, because "camera B implies play"
becomes available as a shortcut, and the three backbones respond in opposite directions:
DINOv2 largely resists it (0.7325), ConvNeXtV2 partly takes it (0.1975), and **ViT takes it
completely and falls from 0.1646 to 0.0041**. On that axis the model ranking is partly a
ranking of shortcut resistance.

*What would fix it:* the fix is already known and cheap — a handful of labelled frames per
camera. What cannot be fixed by analysis is that the published column conflates two things,
and the write-up must say which.

### 1.2 Single-class test sets make each axis gameable *(mitigated)*

The cross-venue folds are **100% ACTIVE_PLAY**, so recall rises by answering "playing" more
often; the false-play set is 100% EMPTY, so its rate falls by answering "empty" more often.
Either axis alone rewards a bias rather than an ability. Under leave-one-venue-out a
**constant predictor scores macro-F1 1.000**, ahead of every backbone
(`constant-predictor-wins-cross-venue`).

*Mitigation:* both axes are reported together in every table that uses them, and the joint
summary is built from EMPTY **accuracy** rather than from `1 - false_play` — built the other
way it ranked the gated fusion head first in the table on the strength of 243 wrong answers.

### 1.3 The composed slot benchmark is saturated *(quantified)*

STAN scores **1.0000** on 200 held-out composed slots, beating a tuned HMM by 0.105 at
p < 0.0001. That is not a result about slot classification. The composed label is a
deterministic function of five templates; the templates stay separable after their boundaries
are jittered; so a model that reads *contiguity* — one long block of play versus scattered
short runs, exactly what a play ratio discards — recovers the generating process exactly.
Reaching the ceiling shows the architecture can do the thing. It says nothing about real slots,
where the verdict is not a function of five shapes.

*Unfixable by more synthesis:* composing more slots cannot break the tie. Only labelled real
slots can (WP2-T8), and there are two.

### 1.4 A rule that never looks at the image competes with the backbones *(quantified)*

The clock rule — majority class per lighting condition, no pixels — scores **0.4907** on the
honest grouped split, within **0.0068** of ConvNeXtV2 (`floor-clock-rule-grouped`,
`h2-clock-rule-grouped`). A 16-bin colour histogram scores **0.9616** on the leaky split
(`floor-histogram-random`). Where a trivial baseline is that close, the benchmark is measuring
scene recognition rather than occupancy.

*Mitigation:* four trivial baselines run in every protocol and their gap to the backbones is a
reported column, so the reader sees when a deep model is not earning its place. The clock rule
does collapse across venues — to **0.219** (`h3-cross-venue-clock-rule`) — which is the
strongest evidence in the thesis that the backbones learn something transferable.

### 1.5 The XAI evidence suggests one model is not reading players *(quantified, n = 9)*

Share of a frame's positive evidence falling inside detected person boxes, against those boxes'
share of the frame: ConvNeXtV2 **2.18×**, DINOv2 **1.61×**, **ViT 1.02×**. ViT classifies
ACTIVE_PLAY correctly while placing no more evidence on the players than on the turf — 1.02 is
the null exactly, and it is consistent with ViT being the weakest cross-venue model and with it
taking the camera shortcut in §1.1.

*Caveats, which are large:* nine frames; YOLO boxes are an imperfect proxy for where players
are; and a 16×16 patch grid upsampled to frame size cannot resolve a distant player. **A
direction, not a rate.** (Not in the claims ledger — it is a mean over nine frames rather than
a published summary row; see `results/xai_evidence_focus.csv`.)

---

## 2 · Internal validity — is the effect what causes the result?

### 2.1 Near-duplicate leakage across the split boundary *(mitigated, and the mitigation is measured)*

Frames are sampled every fifteen seconds from fixed cameras, and **1,667 of 1,692 frames
(98.5%) have a direct near-duplicate**. A random split puts **37.1%** of near-duplicate pairs
across the train/test boundary; the grouped split puts **0.8%** — a 49× reduction.

The effect on scores is not subtle: moving from a random split to a grouped one costs
ConvNeXtV2 **0.4904** macro-F1 (`h1-drop-convnextv2`) — the score roughly halves. And the
leakage is *attributable*: on the leaky split **every one of the 12 errors** is a frame the
model had seen a near-duplicate of (`leaky-errors-had-a-near-duplicate`), while the 32
grouped-split errors across three backbones all come from a **single slot** — one failure
counted many times rather than a taxonomy (`grouped-errors-are-one-slot`). **0.5158** of
DINOv2's random-split score is attributable to leakage (`leakage-attributable-dinov2`).

*Mitigation:* every headline number uses a grouped or leave-one-venue-out protocol; the random
split is reported only as the leaky comparison it is.

### 2.2 The day/night confound, found six times *(quantified — recurring)*

At `venue_01` lighting is close to a class indicator, and this confound has now surfaced in six
distinct places: the clock rule's 98.4% on labelled frames; the class × lighting cross-tab; the
zero-shot prompt sensitivity; the CLAHE gate; the fusion gate's weights, which correlate
**−0.653** with a night indicator across all seven venue folds; and the camera shortcut in
§1.1.

*What this means:* it is not a bug that keeps recurring, it is a **property of the dataset**
that re-expresses itself in each new component. The mitigation is the same each time — measure
the component against an input that carries *only* lighting, and report the comparison either
way. For the fusion gate that ablation showed the gate is reading lighting heavily but not
exclusively (a lighting-only gate is 0.1015 worse on recall).

*Irreducible without data:* empty-pitch footage at a second venue, and daytime match footage.

### 2.3 Selection optimism in the preprocessing search *(quantified)*

Two searches in this project select a winner and then report its margin on the folds it was
selected from, and both were re-scored rather than trusted.

The **preprocessing** search ran 88 evaluations. Its own resolution floor is **0.0119** — one
frame in the smallest venue fold moves the headline by that much (`search-resolution-floor`) —
against a bootstrap interval **0.0917** wide at the median configuration
(`search-interval-width`). Several adopted configurations were separated by less than the
search could resolve, and re-weighting the folds changes three of six winners.

The **prompt** search is the sharper case: its winner leads a pre-declared prompt set by
**0.4474** balanced score *on the folds that chose it* (`search-selection-optimism`). That is
amendment A6's "optimistically biased" expressed as a number.

*Mitigation:* A6 pre-declares a fixed prompt set for every hypothesis test rather than reusing
the search winner, so a selected number is never used to test the selection. `DECLARED_
PROMPT_SET` lives in the library rather than in an experiment, because two copies of a
pre-declared constant are two that can drift apart undetectably.

### 2.4 Guards that did not guard *(mitigated — a recurring defect class)*

Repeatedly, a protection existed in code and did not operate: a final-test-set lock resolved
relative to the working directory; a processor-geometry fingerprint that never changed; a
confidence threshold at 0.0 that can never fire; a gate criterion that grepped for a word
instead of running the verifier; a live camera source whose `n_minutes` raised, so it could
never run; and evidence selection that recorded a path of `None` for every frame it chose.

*Mitigation, and the methodological point of the thesis:* each new guard is now verified by
**deliberately breaking the thing it guards** and confirming the guard fires. That is the
practice the write-up should argue for, and it is what turned up half the findings here.

---

## 3 · External validity — does it generalise?

This is the weakest section and no analysis improves it.

### 3.1 EMPTY exists at exactly one venue *(unquantifiable)*

All 494 EMPTY frames and all 6 MAINTENANCE frames come from `venue_01`. The other 396 frames
are highlight clips and every one is ACTIVE_PLAY. Consequences, stated in
`preregistration.md` §"Not answerable":

- **Cross-venue three-class evaluation is impossible.** There is no empty pitch outside
  `venue_01` to hold out.
- **No frame-disjoint "train on composed, test on real slots" split exists** — the two recorded
  slots contain 1,296 of the 1,692 frames, including every EMPTY and every MAINTENANCE frame.
  The split does not exist; it was not done carelessly.

### 3.2 The effective sample is far smaller than the frame count *(quantified)*

243 held-out empty frames are **three distinct scenes**. The whole corpus of 1,692 frames is
roughly 150 distinct scenes. Every frame-level confidence interval and significance test in
this project inherits that.

*Mitigation:* the effective-sample audit reports distinct-scene counts per test set, and
`sign_flip_test` reports its own **resolution floor** — the smallest *p* the design could ever
return — so a non-significant result can be distinguished from a design that could not have
produced a significant one.

### 3.3 One facility, one camera vendor, no adverse weather *(unquantifiable)*

Two cameras, one pitch geometry, one installation height, one vendor's colour processing, no
rain, no fog, no floodlight failure. Nothing here estimates performance at a different site.
The nine clip venues add pitch-appearance variety to ACTIVE_PLAY only.

### 3.4 Deployment hardware is unmeasured *(unquantifiable, and cheap to fix)*

Every latency number is from the development laptop. The target Mini-PC has never run this
code, so the 10–24× headroom in the 60-second cycle, the 20-camera scaling claim and the
conditional-compute variant's justification are all extrapolations (WP7-T1).

---

## 4 · Conclusion validity — are the statistical claims sound?

### 4.1 Tests that could not have reached significance *(mitigated)*

Several comparisons run over seven venue folds. With seven informative pairs the smallest
two-sided *p* an exact sign-flip test can return is 2/2⁷ = **0.0156**; with four it is 0.125;
with one it is 1.000. So "p = 0.5, not significant" sometimes describes the sample size rather
than the effect.

*Mitigation:* `sign_flip_test` returns `min_achievable_p` alongside the p-value and
`can_reach(alpha)`, and every reported comparison prints it. The WP5-T2 routing test has one
informative fold and a floor of 1.000 — reported as such rather than as a null result.

### 4.2 Multiple comparisons *(mitigated)*

Holm–Bonferroni correction is applied within each declared family, in three reports.

### 4.3 Significance that did not survive a proper denominator *(quantified)*

Six pairwise comparisons on the held-out empty frames were significant at *p* down to 8e-53
under a frame-level test. Recomputed on distinct scenes, **not one survived**. The original
p-values were arithmetic on 243 correlated observations.

### 4.4 A degenerate calibration test set *(quantified)*

DINOv2 produces **890 of 907** identical calibrated confidences (`rq6-tied-confidences`), so
the reliability diagram has one populated bin and the risk–coverage curve's ordering of tied
points is arbitrary. RQ6's operating-point question is **blocked by the class mix**, not
answered.

*Mitigation:* `risk_coverage_band` reports the band across tie orderings rather than one
arbitrary curve.

### 4.5 The intervals cover the sample, not the construction *(quantified — and it cost a finding)*

Every interval in this project is a bootstrap over *observations*: resample the test rows,
recompute the metric. None of them resample the **construction** — which split was drawn,
which augmentation views were drawn, which synthetic sequences were composed. Those are a
separate source of variance, and one of them turned out to be the larger one.

Where the draw is the object of study, this project does replicate: the benchmark takes five
split replicates per protocol, the label-efficiency curve five seeds per training size, the
onboarding curve reports `n_seeds` with a min and a max. Exactly two results consumed a
random draw as a *means to an end* and took it once.

**The first was WP3-T6, and it was wrong.** `light` augmentation was reported at 0.8550
macro-F1 with empty-pitch recall 0.687. Re-drawn four times with the same rows, the same
preset, the same probe seed and the same test set, it scores 0.3479, 0.3501, 0.3510 and
0.4136 — sd **0.2206**, median **0.3510**, and four of the five draws below the 0.4406 the
probe reaches with no augmentation at all (`augmentation-light-is-draw-dependent`). A
bootstrap interval on any one of those five rows would have been narrow and would have told
the reader nothing about the other four.

**The second is STAN**, whose composed train and test sequences come from a single seeded
draw (`stan_preliminary.py`, `SEED` and `SEED + 1`). It is already reported as preliminary
for a different reason — two real slots — and this is a second, independent reason to hold
it there until the draws are replicated. The stage runs in three minutes.

*Mitigation:* `augmentation_transfer_spread.csv` reports mean, sd and range per preset, and
`augmentation_transfer.py` takes `--seeds`. The STAN replication is open (WP5-T1).

*What it costs to fix elsewhere:* nothing, where the replication already exists. The general
lesson is the cheap one — **a result that depends on a draw needs more than one draw, and a
confidence interval is not a substitute for it**, because the two quantify different things
and only the reported one is small.

---

## 5 · What would actually change these

Ordered by how much they buy, which is not the order of effort:

1. **≥30 real labelled slots** (WP2-T8) — lifts STAN out of preliminary, gives reconciliation
   a ground truth, and restores M4's original criterion. One conversation with the facility.
2. **Empty-pitch and maintenance footage at a second venue** — the single fix for §2.2, §3.1
   and the C3 scope reduction, all at once.
3. **A handful of labelled frames per camera** — §1.1 shows this is worth more than a larger
   dataset elsewhere, and it is nearly free.
4. **The Mini-PC** (WP7-T1) — turns §3.4 from an extrapolation into a measurement.
5. **A second annotator** (WP2-T5) — no inter-annotator agreement figure exists, so label
   quality is currently asserted rather than measured.

**Items 1–3 are cheap and would change what this thesis can claim.** That is worth saying
plainly in the defence: the limitations are mostly not methodological, they are a data-access
problem with a known and inexpensive solution.
