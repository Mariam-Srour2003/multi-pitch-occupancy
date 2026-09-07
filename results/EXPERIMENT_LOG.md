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

## 2026-09-06 - H1 / H2 - baseline floor and split leakage

`uv run python experiments/h1_h2_baseline_floor.py` | seed 42 | 2,000 CI resamples
-> `results/h1_h2_baseline_floor.csv`

> **Correction.** An earlier entry recorded H2 as "confirmed, emphatically" from a partial
> run in which only the ConvNeXtV2 cache existed. With ViT and DINOv2 included the
> conclusion reverses: the gap is measured against the *best* probe, and the best probe is
> no longer ConvNeXtV2. The partial result was not wrong about ConvNeXtV2; it was wrong to
> be called H2. Recorded rather than edited away.

| split | model | accuracy | macro-F1 [95% CI] |
|---|---|---|---|
| random | vit | 0.9949 | **0.9960** [0.987, 1.000] |
| random | cheap_histogram | 0.9594 | 0.6855 [0.665, 0.974] |
| random | convnextv2 | 0.9873 | 0.6573 [0.649, 0.997] |
| random | dinov2 | 0.9873 | 0.6573 [0.649, 0.997] |
| random | clock_rule | 0.9162 | 0.6050 [0.587, 0.931] |
| random | majority | 0.6954 | 0.2735 |
| grouped | dinov2 | 0.9846 | **0.5794** [0.494, 0.721] |
| grouped | convnextv2 | 0.9901 | 0.4975 [0.496, 0.499] |
| grouped | vit | 0.9901 | 0.4975 [0.496, 0.499] |
| grouped | clock_rule | 0.9636 | 0.4907 [0.488, 0.494] |
| grouped | majority | 0.0099 | 0.0098 |

**H2 - refuted as pre-registered, but only just, and only by one model.** The registered
form was "within 2 macro-F1 points of the best frozen-backbone probe". On the grouped
split the clock rule sits 8.9 points behind DINOv2 (p_holm 0.0037, g 0.31), so H2 fails.

What survives is narrower and still uncomfortable:

- The clock rule is **0.007 macro-F1** behind ConvNeXtV2 *and* ViT on the grouped split,
  using no image data at all. Only DINOv2 clears the trivial floor by a meaningful margin,
  and ConvNeXtV2 vs DINOv2 is itself not significant (p_holm 0.25).
- A colour histogram beats ConvNeXtV2 and DINOv2 on macro-F1 under the random split
  (0.686 vs 0.657).
- The clock rule reaches **96.4% accuracy** on the grouped split. Accuracy on this data
  remains uninformative regardless of which model wins.

**ViT's 0.996 on the random split is an artifact, not a result.** C3 contributes roughly
one frame to that test set, so classifying it correctly hands ViT a perfect F1 on a
three-frame class and lifts the macro average far above the others. Macro-F1 is unstable
to the point of meaninglessness at that support. Do not quote this number without the
caveat, and prefer the per-class table.

**H1 - directionally confirmed, not cleanly attributable.** Macro-F1 falls from random to
grouped for every model. Part is the leakage H1 predicts; part is that the grouped test
set is 99% single-class. The two cannot be separated on this data.

`majority` scoring 0.0099 on the grouped split illustrates the degeneracy: training rows
are mostly EMPTY, test rows almost entirely ACTIVE_PLAY.

---

## 2026-09-06 - H3: cross-venue active-play recall

`uv run python experiments/h3_cross_venue_recall.py` | seed 42 | 7 held-out venue folds
-> `results/h3_cross_venue_recall.csv`

| model | mean play-recall [95% CI] | worst fold | target 0.90 |
|---|---|---|---|
| dinov2 | **0.930** [0.834, 0.993] | 0.667 | meets |
| convnextv2 | **0.910** [0.821, 0.986] | 0.722 | meets |
| vit | 0.869 [0.714, 1.000] | 0.500 | below |
| clock_rule | **0.219** [0.029, 0.505] | 0.000 | fails |

**H3 confirmed for DINOv2 and ConvNeXtV2, not for ViT.**

The clock rule collapses to 0.219 - zero recall on four of seven venues - while the frozen
backbones hold above 0.86. Whatever the trivial baselines are exploiting inside venue_01,
it does not survive a change of venue, and the deep features do.

**The headline finding: the evaluation protocol inverts the model ranking.**

| protocol | 1st | 2nd | 3rd |
|---|---|---|---|
| random split (leaky) | **vit** 0.996 | convnextv2 / dinov2 0.657 | - |
| grouped split | **dinov2** 0.579 | convnextv2 / vit 0.498 | - |
| cross-venue recall | **dinov2** 0.930 | convnextv2 0.910 | vit 0.869 |

ViT is first under the leaky protocol and last under the honest one. This is the textbook
signature of a supervised ImageNet backbone fitting scene appearance, against a
self-supervised backbone (DINOv2) whose features transfer; and the pilot's protocol ranked
them backwards. It is a far stronger argument for leakage-free evaluation than a bare
accuracy drop, because a reader can see a decision being made wrongly, not just a number
moving.

Caveats for the write-up:

- Folds are small: five of seven venues contribute 12-30 frames, so CIs are wide and the
  fold mean is unweighted, giving tiny venues equal weight. Report the per-fold spread.
- Recall on one class only. Nothing here says whether an empty pitch is recognised at an
  unseen venue - there is no data for that.
- Both leading models are weakest on `clipvenue_h_teal_pitch` (0.72 / 0.67), one of the two
  venue groups still flagged `confidence=low`. Resolve that grouping before quoting the
  worst-fold figures.

---

## 2026-09-06 - Efficiency: per-frame latency and 20-camera throughput

`uv run python experiments/efficiency_latency.py` -> `results/efficiency_latency.csv`
Development laptop (AMD Zen 3, 4 torch threads), **not** the target Mini-PC.

| backbone | single median | single p95 | 20 cameras, measured | naive 20x | cycle headroom |
|---|---|---|---|---|---|
| convnextv2 | 150.9 ms | 165.8 ms | **2.5 s** | 3.0 s | 24.3x |
| vit | 303.3 ms | 345.7 ms | **4.5 s** | 6.1 s | 13.4x |
| dinov2 | 418.3 ms | 448.6 ms | **5.7 s** | 8.4 s | 10.6x |

**All three fit the 60-second sampling cycle with room to spare.** Even DINOv2 - the most
accurate on the honest protocols and the slowest here - uses under 10% of the cycle for 20
cameras. The CPU-only, one-frame-per-minute design is not latency-constrained on this
hardware, and the model choice can be made on accuracy rather than speed.

**Concurrency helped rather than hurt, contrary to the expectation this harness was built
to test.** Every backbone ran 20 streams *faster* than 20x its single-frame median
(ConvNeXtV2 2.5 s against 3.0 s extrapolated). At 4 threads, overlapping the Python and
I/O portions of each call outweighs memory-bandwidth contention. The prediction that
naive multiplication would flatter the system was wrong in direction, which is exactly why
it was measured rather than assumed - and on a many-core Mini-PC the balance may tip the
other way, so WP7-T1 must repeat this rather than reuse these numbers.

**The pilot's pooling bug appeared in the log, live.** Loading ViT prints:

```
pooler.dense.bias   | MISSING |
pooler.dense.weight | MISSING |
- MISSING: those params were newly initialized because missing from the checkpoint.
```

That is the randomly-initialised pooler which produced the pilot's false 38%. The current
code never reads `pooler_output` - it mean-pools `last_hidden_state` and stamps every cache
with `pooling="mean"` - so the warning is harmless here. Worth quoting in the thesis: the
hazard is not hypothetical, the library announces it on every load, and it is still easy
to walk past.

Caveat: this is a 4-thread AMD laptop. The deployment claim is only settled by WP7-T1 on
the actual Intel Mini-PC.

- 2026-09-06 | efficiency | `python experiments/efficiency_latency.py` | `efficiency_latency.csv` | dev laptop, 4 threads

---

## 2026-09-06 - Label efficiency: more labels made it worse

`uv run python experiments/label_efficiency.py` | grouped split | 5 seeds per budget
-> `results/label_efficiency.csv`

Macro-F1 against training-set size, stratified subsampling, clock rule (0 labels) as the
horizontal reference at **0.4907**.

| labels | convnextv2 | dinov2 | vit |
|---|---|---|---|
| 10 | 0.514 +/- 0.047 | 0.376 +/- 0.075 | 0.517 +/- 0.071 |
| 25 | 0.631 +/- 0.030 | 0.526 +/- 0.055 | 0.495 +/- 0.003 |
| 50 | 0.610 +/- 0.083 | 0.593 +/- 0.020 | 0.516 +/- 0.041 |
| 100 | **0.679** +/- 0.028 | 0.623 +/- 0.026 | 0.497 +/- 0.001 |
| 300 | 0.554 +/- 0.071 | **0.691** +/- 0.032 | 0.496 +/- 0.001 |
| 671 (all) | 0.498 | 0.579 | 0.498 |

**The curves are not monotone - they peak and then fall.** ConvNeXtV2 is best at 100
labels (0.679) and *worse with all 671* (0.498). DINOv2 peaks at 300 (0.691) and drops to
0.579 on the full set. Using every available label is the worst option for both.

This is not overfitting in the usual sense; a logistic head on frozen features does not
overfit 671 examples. The explanation is the confound. The subsampler stratifies by class,
so a 100-frame budget draws roughly balanced EMPTY and ACTIVE_PLAY. The full training pool
does not: under the grouped split it is dominated by the morning recording, which is
almost entirely EMPTY from one scene. Training on all of it teaches "this scene is empty";
training on a balanced subset teaches something closer to the actual task.

**Stratified subsampling is accidentally acting as a de-confounding intervention**, and it
buys 0.18 macro-F1 over using everything. That is a usable finding for WP3-T7 (class
balancing) - and a warning that "collect more of the same" would not have helped. More
frames from the same two recordings would have deepened the confound, not diluted it.

Practically, for onboarding a new site: **10-25 labels already beat the zero-label rule**
for ConvNeXtV2, and 100 is near its best. The labelling cost of a new venue is tens of
frames, not thousands - which is the answer RQ1/RQ2 wanted, even though the curve got
there by an unexpected route.

**ViT is flat at ~0.50 across every budget** - it never learns anything the clock rule did
not already have. Consistent with H3, where it was the weakest generaliser, and with the
ranking-inversion finding: its strong random-split number came from scene memorisation
that no amount of balanced labelling repairs.

Caveat: the test side of this grouped split is 99% single-class, so these macro-F1 values
are compressed and should be read as relative, not absolute.

- 2026-09-06 | label efficiency | `python experiments/label_efficiency.py` | `label_efficiency.csv` | 7 sizes x 5 seeds


---

## 2026-09-06 - RQ6: calibration and risk-coverage - MACHINERY BUILT, ANSWER BLOCKED

`uv run python experiments/rq6_calibration_riskcoverage.py` | seed 42
-> `results/rq6_calibration.csv`, `results/rq6_risk_coverage.csv`

| model | accuracy | T | ECE raw | ECE calibrated | 99% precision at |
|---|---|---|---|---|---|
| convnextv2 | 0.9901 | 0.95 | 0.0096 | 0.0097 | 100% coverage, 0% review |
| dinov2 | 0.9857 | **0.05** | 0.0064 | 0.0148 | 97.9% coverage, 2.1% review |
| vit | 0.9901 | 1.30 | 0.0087 | 0.0075 | 100% coverage, 0% review |

**These numbers are not usable, and the experiment is reported for the reason it fails.**

The headline reads "99% precision at zero human review", which would be a remarkable
operational result. It is an artifact. Four tells, three of them caught by the tooling:

1. **The test set is 99% single-class.** Predicting ACTIVE_PLAY constantly scores ~0.99, so
   a 99% precision target is met before confidence is consulted at all. The risk-coverage
   curve is flat because there is nothing for it to trade against.
2. **DINOv2's temperature pinned to the grid floor at 0.05**, and `fit_temperature` raised
   the boundary warning added for exactly this case. A temperature at the edge means the
   optimum lies outside the grid, which here means the calibration slice does not resemble
   the evaluation data - it is drawn from the morning recording while the test side is
   night footage and clips.
3. **Calibration made two of three models worse.** Temperature scaling fitted on one
   distribution and applied to another is not calibration, it is noise.
4. **ConvNeXtV2 puts all 907 test frames in a single reliability bin** (0.9-1.0). ECE over
   one bin measures almost nothing.

**Conclusion: RQ6 is unanswerable on this data**, for the same reason as the three-class
questions - the only honest split is degenerate. The `calibration.py` module is built,
tested (16 tests) and ready; it needs a test set with a real class mix, which needs empty
pitches from more than one venue.

Worth keeping as a methods contribution regardless: the boundary warning turned a
plausible-looking temperature of 0.05 into a visible failure. Without it, a fitted
parameter would have been reported as though it meant something.

- 2026-09-06 | RQ6 | `python experiments/rq6_calibration_riskcoverage.py` | seed 42 | `rq6_calibration.csv`, `rq6_risk_coverage.csv`

---

## 2026-09-06 · Venue-grouping audit (closes the `confidence=low` flags)

Visual audit of the two low-confidence groups in `configs/clip_venues.csv`, one middle frame
per clip plus cross-comparison against every other venue group. Comparison sheets saved to
`results/figs/venue_check/` (group_cg, group_ch, reference_all, zoom_pairs, cg_vs_ch_day).

- **`clipvenue_g_netting` — confirmed one facility (raised low → high).** The pitch-4 camera
  (`cg_1788518144607`) sees the neighbouring pitch's "5" sign in-frame, and
  `cg_1788518055948` is that pitch 5. Identical numbered blue tarps, yellow-padded corner
  poles, white diamond netting, floodlights, hillside backdrop. Multiple pitches, one venue —
  grouping is correct and conservative for leave-one-venue-out.
- **`clipvenue_h_teal_pitch` — plausibly one venue (raised low → medium).** Night pair is one
  camera/one match; the day clip matches on distinctive furniture (blue tarp with teal top
  cap, fine teal mesh) and the same shanty-hillside backdrop. Kept grouped.
- **Cross-group check: cg ≠ ch.** Direct high-res comparison of the ambiguous day members:
  different fencing systems (heavy white netting + posters + chain-link vs tarp + fine mesh),
  different skylines (roof-deck building vs shanty hill + crane), different floodlight styles.
  No two of the nine venue groups appear to be the same facility, so leave-one-venue-out is
  not optimistic on this account.
- **Residual caveat / cheap insurance:** `ch` is the worst H3 fold for both leading models
  (0.72 / 0.67 play-recall). A sensitivity re-run of H3 with cg+ch merged into a single fold
  would make the worst-fold claim unchallengeable; recommended before quoting worst-fold
  numbers in the thesis.

- 2026-09-06 | venue audit | visual, sheets in `results/figs/venue_check/` | `configs/clip_venues.csv` | cg low→high, ch low→medium

- 2026-09-06 | RQ6 | `python experiments/rq6_calibration_riskcoverage.py` | seed 42 | `rq6_calibration.csv`, `rq6_risk_coverage.csv`

---

## 2026-09-06 - End-to-end decision layer on the real slots

`uv run python experiments/end_to_end_slots.py` -> `results/end_to_end_slots.csv`

Per-minute states rebuilt from the labelled frames (each carries a camera tag and an
offset in seconds), fused across the two cameras, aggregated, then reconciled against
synthetic bookings.

| slot | fused minutes | play | empty | verdict | expected |
|---|---|---|---|---|---|
| 2026-07-11 10:00 | 59 | 0.02 | 0.93 | **NOTUSED** | NOTUSED |
| 2026-07-12 20:30 | 60 | 1.00 | 0.00 | **USED** | USED |

Both correct. Reconciliation reaches the intended anomaly on each fixture: a morning slot
recorded as used raises `NO_SHOW_OR_OVERRECORDED`; an unbooked evening slot with observed
play raises `UNBOOKED_USAGE` at serious severity.

**Run on ground-truth labels, not model output, deliberately.** The question is whether
the decision layer is correct *given* correct perception. If fusion or aggregation were
wrong, no classifier would rescue it, and folding model error into this run would hide
that. Model-driven runs arrive with the scheduler (WP6-T2).

The reconciliation result proves reachability, not precision. Measuring anomaly precision
and recall needs adjudicated slots - which ones really were no-shows - and that is WP6-T11,
still outstanding.

Note the sample count: 59 and 60 minutes. **These two slots are the entire real-slot
corpus**, and STAN needs roughly thirty.

- 2026-09-06 | end-to-end slots | `python experiments/end_to_end_slots.py` | `end_to_end_slots.csv` | 2 real slots, ground-truth labels

---

## 2026-09-06 - Input ablation: what is the model actually reading?

`uv run python experiments/input_ablation.py` | DINOv2 | 7 held-out venue folds
-> `results/input_ablation.csv`

Information is removed from the input and the cross-venue play-recall is re-measured.

| variant | what it removes | play-recall | delta | worst fold |
|---|---|---|---|---|
| crop50 | outer 50% border | **0.998** | +0.039 | 0.988 |
| grayscale | all colour | **0.982** | +0.023 | 0.917 |
| full | nothing | 0.960 | - | 0.800 |
| blur4 | fine detail (sigma 4px) | 0.929 | -0.030 | 0.589 |
| blur8 | almost all object detail | **0.840** | -0.120 | 0.583 |

**The bias control, run first.** Every clip-venue test set is 100% ACTIVE_PLAY, so recall
alone cannot separate "better at seeing play" from "more willing to say play" - any variant
that shifts the decision boundary toward PLAY scores higher for free. Checking how often
each variant calls a genuine EMPTY frame PLAY:

| variant | play-recall | false-play on EMPTY |
|---|---|---|
| full | 0.960 | 0.000 |
| grayscale | 0.982 | 0.000 |
| crop50 | 0.998 | 0.000 |
| blur4 | 0.929 | 0.002 |
| blur8 | 0.840 | 0.004 |

No variant becomes more willing to say PLAY, so the gains are not a boundary shift.
**Caveat: those EMPTY frames sit in each fold's training set** - venue_01 is always in
train for a clip-venue fold - so this is an in-sample control. It rules out gross bias,
not a subtle one. A clean version needs held-out empties, which needs empties from a
second venue.

### What it says

**The model is reading people, not scenery.** Blurring costs 0.12 recall, and at sigma 8
on a 224px input no individual is discernible. If the prediction rested on scene
composition - turf, floodlights, stand geometry - blur would leave it intact. It does not.
This is the direct answer to the worry H2 raised and the strongest evidence yet that the
frozen features do the actual task.

**Colour and the background periphery are net distractions.** Discarding all colour
*improves* cross-venue recall, and so does throwing away the outer half of the frame. Both
carry venue identity - turf hue, floodlight cast, stands, sky, adjacent pitches - and none
of it transfers. The model is better off without them.

Two consequences:

1. **ROI masking (WP3-T1) is now predicted to help, not merely to be tidy.** `crop50` is a
   crude proxy for it and gains 0.04 recall with the worst fold rising 0.80 -> 0.99.
   Hand-drawn polygons should do better than a blind centre crop.
2. **Grayscale is worth testing as a production setting**, not just a diagnostic. It costs
   nothing and removes a shortcut the model would otherwise lean on.

### An incidental finding

`full` scores 0.960 here against DINOv2's 0.930 in H3. The difference is preprocessing:
this run letterboxes to 224 before the HuggingFace processor, while the H3 cache passed the
raw image straight to it. **Aspect-preserving letterboxing is worth ~0.03 recall** on this
heavily fisheye footage - the processor's default resize distorts, and distortion is
already the hard part here.

- 2026-09-06 | input ablation | `python experiments/input_ablation.py` | `input_ablation.csv` | 5 variants x 7 folds, dinov2

- 2026-09-06 | H3 sensitivity (cg+ch merged) | `python experiments/h3_sensitivity_merged_venues.py` | seed 42 | `h3_sensitivity_merged_venues.csv` | 6 folds x 4 models

---

## 2026-09-06 · H3 sensitivity: cg+ch merged into one fold (venue-audit insurance)

`uv run python experiments/h3_sensitivity_merged_venues.py` | seed 42 | 6 folds
-> `results/h3_sensitivity_merged_venues.csv`

Pessimistic variant of H3: the two venues from the grouping audit
(`clipvenue_g_netting`, `clipvenue_h_teal_pitch`) relabelled as ONE venue, so holding the
pair out removes both from training. If they secretly were one facility, this is the
honest fold; if they are two (the audit's conclusion), this is strictly harder than needed.

| model | mean recall [95% CI] | worst fold | merged fold | target 0.90 |
|---|---|---|---|---|
| dinov2 | **0.967** [0.932, 0.993] | 0.889 | 0.958 | meets |
| convnextv2 | **0.927** [0.843, 0.989] | 0.750 | 0.875 | meets |
| vit | 0.903 [0.764, 1.000] | 0.583 | 0.833 | nominal only — CI spans the target |
| clock_rule | 0.243 [0.021, 0.556] | 0.000 | 0.125 | fails |

**H3 survives the pessimistic merge — the venue-grouping doubt cannot change the
conclusion.** DINOv2 and ConvNeXtV2 meet the 0.90 target under both groupings; the clock
rule collapses under both. The worst-fold caveat can now be quoted with confidence.

Two honest notes for the write-up:

- Means are higher than the main run (e.g. DINOv2 0.930 -> 0.967) mostly because merging
  turns two small weak folds into one medium fold in an unweighted mean of fewer folds —
  a fold-arithmetic effect, not a model improvement. Quote the main H3 numbers as primary
  and this run as the robustness check, not the other way around.
- The `ch`-alone fold recall (0.667 for DINOv2 in the main run) vs `ch`-within-merged
  (~0.875) differs by ~4 frames on an 18-frame fold — small-n noise territory. Another
  reason the per-fold table, not the worst single fold, should carry the claim.

---

## 2026-09-06 - Camera identity: the `(1)` suffix flips between days (confirmed)

`src/pitch_occupancy/vision/camera_id.py` | measured on the four venue_01 recordings

The review-session notes warned that the `(1)` filename suffix does not map to a stable
physical camera. Measured with a lighting-invariant view descriptor - median background,
CLAHE, gradient magnitude, downscaled and L2-normalised - it is confirmed:

| pair | similarity | |
|---|---|---|
| `day_base` <-> `night_(1)` | **0.883** | same physical view |
| `day_(1)` <-> `night_base` | **0.878** | same physical view |
| `day_(1)` <-> `day_base` | 0.758 | two halves, same day |
| `day_base` <-> `night_base` | 0.700 | two halves, different days |
| `day_(1)` <-> `night_(1)` | 0.578 | two halves, different days |
| `night_(1)` <-> `night_base` | 0.537 | two halves, same night |

Same-view-across-days (0.878-0.883) separates cleanly from different-view (0.537-0.758),
so the descriptor survives the day/night change that would defeat raw pixel matching.

**This was a live latent bug.** `discover_slots()` assigned `camB` from the suffix alone,
which is correct on one recording day and wrong on the other. Nothing would have raised:
swapped halves still fuse into a plausible verdict, and the maximum-activity fusion rule is
symmetric, so the error only surfaces once per-camera behaviour matters - ROI polygons,
camera-health baselines, or any claim about which half play occurred on.

Fixed by refusing to claim what the filename cannot support: `discover_slots()` now returns
`file0`/`file1`, and physical identity comes from matching view descriptors against
references. `match_cameras()` assigns greedily over the full similarity matrix so each
reference is used once - a per-recording argmax could hand both halves of a pitch to the
same camera - and leaves a poor match unassigned rather than forcing it, because an obvious
gap beats a silently swapped half.

- 2026-09-06 | input ablation | `python experiments/input_ablation.py` | `input_ablation.csv` | 6 variants x 7 folds, dinov2

---

## 2026-09-06 - Combined preprocessing: the removals do not compound

`uv run python experiments/input_ablation.py` (variant `gray+crop50` added) | DINOv2

| variant | play-recall | worst fold | delta | false-play on EMPTY |
|---|---|---|---|---|
| crop50 | **0.998** | 0.988 | +0.038 | 0.000 |
| grayscale | **0.982** | 0.917 | +0.022 | 0.000 |
| full | 0.960 | 0.800 | - | 0.000 |
| **gray+crop50** | **0.899** | 0.667 | **-0.061** | 0.000 |

**Removing both helps less than removing either - and is worse than removing neither.**
Each on its own beats the baseline; together they fall 0.06 below it, and the worst fold
drops from 0.99 (crop50) to 0.67.

The false-play rate stays at zero throughout, so this is not the model turning
conservative and refusing to say PLAY. It has genuinely lost the ability to recognise play
at unseen venues.

**Reading.** Colour and the frame border each carry two things: venue identity, which does
not transfer, and part of the evidence for play. Removing one strips a shortcut and leaves
enough signal. Removing both crosses an information floor - what remains is central,
greyscale structure, and at 224px that is not enough to distinguish a pitch with players
from one without at a venue never seen.

So "remove more distractors" is not monotone, which is the useful result here and one that
would not have been visible from the two single-variant runs. It also revises the
production recommendation: **not grayscale plus ROI, but ROI alone.** `crop50` is a blind
centre crop and is the best variant tested; hand-drawn polygons should beat it, and
stacking grayscale on top should not be assumed to help.

Caveat unchanged from the single-variant run: the false-play control uses EMPTY frames
that sit in each fold's training set, since venue_01 is always in train for a clip-venue
fold. It rules out gross bias, not a subtle one.

- 2026-09-06 | zero-shot prompt search | `python experiments/prompt_search.py` | `prompt_search.csv` | 375 prompt sets

---

## 2026-09-06 - Zero-shot prompt search: no labels beats every trained probe

`uv run python experiments/prompt_search.py --limit 600` | CLIP ViT-B/32 (LAION-2B)
-> `results/prompt_search.csv`, `results/prompt_search_best.json` | 375 prompt sets

Exhaustive over 125 descriptor combinations x 3 template-ensemble sizes. Affordable
because images are embedded once and a prompt set is a few short strings - the opposite
cost profile to the preprocessing search.

| approach | labels needed | cross-venue play recall | false-play |
|---|---|---|---|
| **CLIP zero-shot, searched prompt** | **0** | **0.988** | **0.000** |
| DINOv2 + trained probe | ~1,500 | 0.930 | 0.000 |
| ConvNeXtV2 + trained probe | ~1,500 | 0.910 | 0.000 |
| ViT + trained probe | ~1,500 | 0.869 | 0.000 |
| CLIP zero-shot, pilot's single prompt | 0 | (75.8% in-venue accuracy) | - |

Best prompt: **"people playing football on a pitch"**, one template ("a photo of {}").
Elaborate template ensembles did not help - the top set uses a single template, and the
descriptor carries the discrimination.

**This is a genuinely large result for the onboarding question.** RQ1 asks what a new
client site costs. For active-play detection the answer is now *no labels at all*: a
well-chosen prompt transfers to seven unseen venues better than a probe trained on 1,500
labelled frames. The label-efficiency curve said 10-25 labels beat the zero-label clock
rule; this says the right zero-label method beats the fully-labelled probe.

### The caveat, which is not small

**The prompt was selected on the same venues it is scored on.** 375 candidates were
evaluated against these seven development folds and the best was reported - that is
selection on the evaluation data, and 375 tries is a great deal of freedom. The number is
a *development* result and must not be quoted as clean generalisation.

The locked final venues (`clipvenue_b_floodlit_track`, `clipvenue_c_teal_boards`) were
excluded throughout, so the honest confirmation is available: evaluate this one prompt on
them, once, at the end. Until then the claim is "a searchable prompt space contains sets
this good", not "this prompt achieves 0.988 on unseen venues".

### The false-play control earned its place again

Ranking on recall alone would have chosen a different prompt: 1.000 recall with 0.0256
false-play, against the balanced winner's 0.988 with 0.000. The higher-recall prompt is
simply readier to say PLAY, and on a test set that is 100% ACTIVE_PLAY that is free. The
`balanced` score (recall minus false-play) is what ranks the table.

### Also worth recording

`get_image_features` in transformers 5.x returns a `BaseModelOutputWithPooling`, and the
embedding is its `pooler_output` - the same attribute name that produced the pilot's false
38%. It is **not** the same thing: there it was a randomly initialised head on a plain ViT,
here it is CLIP's trained projection (512-d = `projection_dim`, against a 768-d hidden
state). Verified functionally before use rather than by name: a real ACTIVE_PLAY frame
scores 0.296 against "people playing football", 0.062 against "a plate of spaghetti" and
-0.011 against "a cat on a sofa".

---

## Preprocessing search (WP3-T8) — 2026-09-07

`experiments/preprocess_search.py`, greedy forward selection over ten preprocessing
switches, scored on cross-venue ACTIVE_PLAY recall with the false-play control.
**45 evaluations, 740 minutes of compute, four rounds**, all on the full 1,578-frame
development set. Output: `results/preprocess_search.json`.

### What it found

| configuration | recall | vs baseline | worst fold | false-play |
|---|---|---|---|---|
| `gamma=0.7 sharpen=0.6` | **1.0000** | +0.0159 | 1.000 | 0.0000 |
| `saturation=0.5 sharpen=0.6` | **1.0000** | +0.0159 | 1.000 | 0.0000 |
| `gamma=0.7 denoise=25 sharpen=0.6` | **1.0000** | +0.0159 | 1.000 | 0.0000 |
| `sharpen=0.6` | 0.9991 | +0.0150 | 0.994 | 0.0000 |
| `saturation=0.5` | 0.9983 | +0.0142 | 0.988 | 0.0000 |
| baseline (no preprocessing) | 0.9841 | — | 0.889 | 0.0000 |
| `centre_crop=0.5 sharpen=0.6` | 0.7841 | **&minus;0.2000** | — | 0.0000 |

**Sharpening is the one switch that matters.** It is the largest single-switch gain
(+0.0150) and appears in every configuration at the top of the table. Nothing else comes
close on its own.

### Three caveats, and the first one is fatal to quoting the number

**1. The metric saturated.** Three configurations tie at exactly 1.0000 with a worst fold
of 1.000. That is not "preprocessing is solved" — it is the measurement running out of
resolution. There is no way to rank those three, and any of them may be better than the
others on data this development set cannot see. A ceiling result means the next experiment
needs a harder test set, not a finer search.

**2. It was selected on the folds it reports.** Same structure as the prompt search: 45
configurations were scored against these development folds and the winner reported.
That is selection on the evaluation data. This is a **development** result. The locked
final venues were excluded throughout, so an honest confirmation is available — evaluate
one chosen configuration on them, once, at the end.

**3. The false-play control never fired.** It reads 0.0000 for all 45 configurations, so
it did nothing to discriminate here. It is not evidence the control is unnecessary — it
did real work in the prompt search — but on this axis it carries no information, and the
ranking rests on recall alone.

### The interaction trap, confirmed a second time

`sharpen=0.6` alone is +0.0150. `centre_crop=0.5` alone is &minus;0.0289. Together they are
**&minus;0.2000** — far worse than the sum, and the worst result in the search. This is the
same non-additivity the input ablation found with grayscale and crop50, now reproduced on a
different pair of switches. **One-at-a-time preprocessing tables cannot be trusted to
compose**, which is the argument for searching the space rather than tabulating it.

### The finding that matters most, and it was nearly missed

The search ran entirely on **ConvNeXtV2** (the script's default), while the project's
configured default model is **DINOv2** (`config.default_model_key`). That would be a minor
bookkeeping note if preprocessing effects transferred between backbones. They do not:

| removing colour | DINOv2 (input ablation) | ConvNeXtV2 (this search) |
|---|---|---|
| full desaturation | **+0.022** | **&minus;0.119** |
| worst fold | 0.917 | 0.500 |

Both numbers come from the *same operation* — `cv2.COLOR_BGR2GRAY` then back to three
channels; `_apply_saturation(0.0)` and the ablation's `grayscale=True` are the same code
path.

They also come from the same aggregation, which was worth checking rather than assuming.
When the DINOv2 search started, its baseline scored **0.9595** — identical to the input
ablation's `full` variant, to four decimal places. Same folds, same averaging, same
untouched input. So the comparison is exact: **nothing differs between these two numbers
except the backbone.**

**So preprocessing choices are model-specific, and this search's recommendation must not be
applied to the default model.** Doing so would have taken a step that helps DINOv2 and
recorded it as harmful, or applied ConvNeXtV2's optimum to a backbone it was never measured
on. The search supports `--models dinov2 convnextv2`; the DINOv2 run is what the
recommendation actually needs.

This also answers a question the search was built to ask: whether the best preprocessing
depends on the model. It does, and the dependence is large enough to reverse the sign.

---

## Frame quality and camera health (WP3-T5) — 2026-09-07

`src/pitch_occupancy/vision/quality.py`, `experiments/camera_health.py`, outputs
`results/camera_health.csv` and `results/frame_quality.csv`.

### The result: a global threshold would have flagged one venue, not bad frames

The obvious design is a constant — Laplacian variance below *x* is blurry. Measured over all
1,692 frames it fails outright:

| camera group | median RMS contrast | median blur |
|---|---|---|
| `venue_01` camera A, night | 20.5 | 111 |
| `venue_01` camera A, day | 16.9 | 253 |
| `clipvenue_i_outdoor_trees`, day | 47.1 | 2398 |

Blur spans **88 to 2755, a thirty-fold range**, and it tracks *which camera took the frame* —
optics, resolution, compression — not whether the frame is usable. A cutoff at the 10th
percentile flags 169 frames, and **100% of them come from `venue_01`** — the primary venue,
and the only one with full-length recordings. That filter would have discarded the venue the
system actually runs on, while reporting itself as a quality improvement.

This is the project's central confound arriving a third time. It is the same mistake as
day-versus-night: a measurement that looks like it is about the target is actually about the
scene. Every threshold is therefore relative to **the camera's own history under the same
lighting**, and the design note in `quality.py` states the reason with the numbers.

### Two false-positive modes, both found by looking at the frames

1. **Statistically unusual is not unusable.** A 3-MAD rule alone flagged four `venue_01`
   frames at blur 221–235 against a median of 253. Rendered beside a typical frame they are
   indistinguishable — same pitch, same light, equally readable. `venue_01`'s camera is so
   consistent that its MAD is tiny, so the outlier floor sat within 9% of the median. A frame
   must now be **materially** worse as well as unusual: half the detail, or two-thirds the
   contrast.
2. **A night frame is not a degraded day frame.** Baselines keyed on camera alone judged six
   readable floodlit `clipvenue_g_netting` frames against a median built from that venue's
   daylight frames, and called them low-contrast. Lighting is now part of the baseline key.

After both fixes the filter flags **6 of 1,692 frames (0.4%)**, all from
`clipvenue_b_floodlit_track`, where "camera" is really several unrelated clips — so those
reflect clip-to-clip variation, not a degrading camera.

### The honest statement about validation

**This dataset contains no known-bad frames.** Nothing is clipped beyond 1.5% against a 10%
limit, and every frame the filter flagged proved readable when rendered. So the filter is
validated against *deliberately degraded* frames in `tests/test_quality.py` — blurred,
blown out, crushed, flattened — and the absence of real positives is recorded here rather
than hidden by tuning the thresholds until something is flagged.

---

## Correction: the `lighting` column does not mean time of day outside `venue_01`

Found while building the quality baselines, and it changes how one reported number should be
read.

`lighting` has two different provenances, and only one of them is a clock:

* **`venue_01` (1,296 frames, 77%)** — derived from the recording's real slot time
  (`_lighting_for` in `data/manifest.py`); the 10:00 slot is day, the 20:30 slot is night.
  **Reliable.**
* **The nine clip venues (396 frames)** — derived from *mean frame brightness*
  (`NIGHT_BRIGHTNESS_BELOW = 80.0` in `data/extract.py`). **Not a clock at all.**

Floodlit and indoor pitches are bright. Rendering one frame per venue and lighting group makes
the failure plain: **`clipvenue_b_floodlit_track` (78 frames) and `clipvenue_f_outdoor_bldg`
(12 frames) are unambiguously night football — black sky, lit floodlight poles, distant city
lights — and both are labelled `day`.** Several others are indoor, where time of day and illumination are genuinely unrelated.

**One venue settles it beyond visual judgement.** `clipvenue_c_teal_boards` carries a
burned-in camera clock, and at full resolution it reads **`08-23-2026 Sun 20:55:46`**. Ten
to nine at night, 36 frames, labelled `day`. That is not an interpretation of a dark sky —
it is the camera stating the time on the frame while the manifest says otherwise. No other
clip venue exposes a legible clock in the frame, so this is the one case that can be proved
rather than argued, and it proves the mechanism: brightness was measured, time was not.

`c_teal_boards` is one of the two **locked final-test venues**, so it does not touch the H3
development folds below. It does mean the final evaluation would inherit the same error, and
it shows the mislabelling is systematic rather than one bad clip.

### Why this matters: it is exactly what H3's clock rule reads

`ClockRule.predict` uses `r.lighting` and nothing else, so its cross-venue recall is a direct
readout of these labels. The per-fold numbers are precisely the fraction of each venue's
frames labelled `night`:

| fold | recall | labelled night |
|---|---|---|
| `a_blue_barrier` | 1.000 | 168 / 168 |
| `d_indoor_dome` | 0.000 | 0 / 18 |
| `e_pink_boards` | 0.000 | 0 / 18 |
| `f_outdoor_bldg` | 0.000 | 0 / 12 |
| `g_netting` | 0.200 | 6 / 30 |
| `h_teal_pitch` | 0.000 | 0 / 18 |
| `i_outdoor_trees` | 0.333 | 6 / 18 |
| **mean** | **0.219** | |

`f_outdoor_bldg` is night football scored as though a clock would have said "day". Correcting
that single fold — twelve frames — moves the clock rule from **0.219 to 0.362**, a 65%
relative change from one twelfth of the test data.

### What survives and what does not

**The venue_01 confound finding is untouched.** It rests on real slot times: EMPTY is 98%
daytime, ACTIVE_PLAY 99% night, and a clock rule scores 98.4%. That is the headline result and
it stands.

**H3's qualitative conclusion also survives.** Even corrected, the clock rule is somewhere
around 0.36 while the frozen backbones hold above 0.86 — still a collapse, still the point.

**The specific figure 0.219 should not be quoted as the clock rule's cross-venue recall.** It
is a lower bound produced partly by label error. The honest claim is *"a lighting-only rule
collapses across venues"*, and the thesis should say the exact value depends on lighting
labels that are a brightness proxy for the clip venues.

**What would fix it.** The clips carry burned-in timestamps in at least some venues, and the
venue names themselves record the answer for others. Relabelling those 396 frames by hand is
a small job — under an hour — and it is the only way to get a defensible number. Until then
the column would be more honestly named `illumination` than `lighting`.

---

## Normalisation audit (WP3-T3) — 2026-09-07: the processor crops after preprocessing runs

Full protocol table in `thesis/protocol.md`. Tests in `tests/test_normalisation_audit.py`.

**The photometric half was fine.** Each model is normalised with its own statistics — ViT on
0.5/0.5, the other two on ImageNet — because each model's own processor is called.

**The geometric half was not.** The Hugging Face processor does not only normalise. Given a
frame that `preprocess.py` has already resized to 224×224 with letterbox padding, two of the
three models resize it *again* to 256 and centre-crop back to 224:

| model | processor geometry on a 224×224 frame | frame kept | cosine vs. geometry-off |
|---|---|---|---|
| **ConvNeXtV2** | `crop_pct=0.875` → resize 256, crop 224 | **76.6%** | **0.886** |
| ViT | resize to exactly 224×224 | 100% | 1.000 |
| **DINOv2** | resize shortest edge 256, crop 224 | **76.6%** | **0.961** |

**23.4% of every frame is discarded after preprocessing has run**, and the discarded border
is exactly where WP3-T2's letterbox padding sits. The two affected models are the default
model and the model the 740-minute preprocessing search ran on. ViT is the only backbone that
sees what preprocessing produced.

### The audit's own first pass got this wrong, which is the point

ConvNeXtV2's config reads `size: {'shortest_edge': 224}` with no `do_center_crop` flag. On a
224×224 input that reads as a no-op, and this audit **recorded it as one** before the
embeddings disagreed. `crop_pct: 0.875` is the field that matters: it resizes to
`224 / 0.875 = 256` first. Reading `size` — the field one naturally reads — gives the wrong
answer. DINOv2 declares its crop openly, which is why it was found first and ConvNeXtV2
second, by measurement rather than by reading.

### It explains a result already logged as a puzzle

The preprocessing search recorded `centre_crop=0.5` at −0.029 alone and **−0.200** combined
with `sharpen`, filed as non-additivity with no mechanism. There is now a mechanism: a hidden
crop already removes 23.4%, so an explicit half-crop is **a crop applied to a crop**, leaving
the model a small central patch of pitch. The input ablation's `crop50` gain (+0.038 on
DINOv2) is the same compound, and its "information floor" reading should be re-read in that
light — the floor may be a crop-on-crop artefact rather than a property of information
removal.

### What it does not overturn

Every model was measured under the same pipeline, so **model-to-model comparisons stand**,
and the confound findings rest on labels rather than pixels. What is affected is the
*interpretation of the geometric preprocessing switches*: they are applied and then partly
overwritten, so they do not mean exactly what their names say.

### The decision is deferred deliberately, not forgotten

`embed_batch(..., processor_geometry=False)` keeps rescale and normalisation but disables the
processor's resize and crop. It is not the default, because 256-then-crop is the canonical
inference recipe for both models and departing from it is a choice to be **measured**, not
applied quietly as a bug fix — and because flipping it invalidates every cached feature. The
flag is part of the cache fingerprint, so the conventions cannot mix.

**What settles it:** one cross-venue run per model under each convention, once the DINOv2
preprocessing search finishes and the CPU is free.

---

## Class balancing (WP3-T7) — 2026-09-07: kept, for a reason the task did not anticipate

`experiments/class_balancing.py` → `results/class_balancing.csv`. DINOv2 cached features.

### The acceptance criterion cannot be met

WP3-T7 accepts on *"C3 recall improves on validation vs unweighted"*. **C3 has six frames and
they are all the same moment** — `venue_01`, camera A, the 2026-07-11 10:00 slot, daylight,
four of them seconds apart. So it never appears on both sides of a leakage-free split:

| protocol | C3 in train | C3 in test |
|---|---|---|
| grouped split | 6 | **0** |
| leave-one-venue-out, all seven clip folds | 6 | **0** |
| leave-one-venue-out, `venue_01` fold | **0** | 6 |

Folds with C3 on both sides: **zero**. A number could be produced by shuffling frames
randomly, but four of the six are near-duplicates, so it would measure memorisation of one
30-second window. The criterion is unanswerable, and reporting a random-split figure instead
would be reporting an artifact.

### What is answerable reverses the obvious conclusion

`balanced` weights those six frames about **88×**, so the natural worry is that it distorts
the boundary for the classes that matter. The first measurement appeared to confirm it:

| weighting | cross-venue play recall | worst fold |
|---|---|---|
| balanced | 0.9595 | 0.800 |
| unweighted | **0.9881** | **0.9167** |

Turning balancing off looked worth +0.0286. **It is not.** Every cross-venue fold is 100%
ACTIVE_PLAY, so recall can be bought by predicting PLAY more often — and removing the weight
on a minority class is precisely the change that would do that. On held-out EMPTY frames:

| weighting | false-play rate on 243 held-out EMPTY frames |
|---|---|
| balanced | **0.2305** |
| unweighted | **0.4650** |

The unweighted probe calls **nearly half of all empty pitches a match**. The recall was free
and the cost was real. `balanced=True` stays the default — earned by halving the false-play
rate, not by the C3 recall the task asked for.

### The control nearly missed it, which is the transferable lesson

The first version of the control used a grouped split inside `venue_01`. That splits on
`slot_id`, and `venue_01` has **two** slots, so it is nearly all-or-nothing: it left **nine**
EMPTY frames, both settings scored exactly 0.000, and the recall gain looked clean. Nine
frames cannot distinguish a false-play rate of 0.0 from one of 0.5.

Splitting on the **physical camera** instead — two genuinely different views, spanning both
slots — leaves 243 EMPTY frames and inverts the answer. **A control too small to separate the
hypotheses is not a control**, and it fails silently, by agreeing with whatever it is asked.

### An honest residual

Even balanced, 23% of held-out empty frames are called ACTIVE_PLAY across cameras. That is a
weak number and it is not hidden here: it is measured across two different camera views,
which is harder than the deployed case, but it is the clearest signal yet that EMPTY-vs-PLAY
is not solved once the evaluation stops being degenerate.

---

## Correction: the search's false-play control was measuring training data — 2026-09-07

`experiments/rescore_false_play.py` → `results/false_play_rescored.csv`.

### The bug

```python
empties = [r for r in fold.train if r.class3 == EMPTY]   # the probe's own fitting data
```

The control scored each probe on **frames it had just been fitted on**, so it read exactly
`0.0000` for all 52 evaluations across both models and separated none of them. It went
unnoticed for a structural reason: every clip-venue fold has **494 EMPTY frames in train and
zero in test**, so there was nothing on the held-out side to measure and the code reached
for the training side instead.

The entry above records this search's metric as *saturated* — three configurations tied at
1.0000 and "there is no way to rank those three". **That conclusion was wrong.** The metric
was not saturated; the second dimension was broken, so ranking had only one axis.

### What the repaired control shows

Trained on `venue_01` camera A, scored on camera B's 243 held-out EMPTY frames:

| configuration | recall | worst fold | false-play |
|---|---|---|---|
| `saturation=0.5 sharpen=0.6` | 1.0000 | 1.000 | **0.021** |
| `gamma=0.7 sharpen=0.6` | 1.0000 | 1.000 | **0.663** |
| `gamma=0.7 denoise=25 sharpen=0.6` | 1.0000 | 1.000 | **0.979** |
| `gamma=0.7 saturation=0.5 sharpen=0.6` | 0.9983 | 0.988 | 0.021 |
| `sharpen=0.6` | 0.9991 | 0.994 | 0.276 |

The three "unrankable" configurations are decisively ranked. Two of them would call **66%
and 98% of empty pitches a match** — unusable for a billing audit, and indistinguishable
from the winner on recall alone.

**The recommendation is `saturation=0.5 sharpen=0.6`**: perfect cross-venue play recall with
a 2% false-play rate. The search now ranks on `balanced = recall − false_play`, as the prompt
search already did.

The correlation between recall and false-play is weak (+0.112 for ConvNeXtV2, −0.154 for
DINOv2), so the search was not *systematically* buying recall — but individual top
configurations plainly were, which is precisely what a control exists to catch.

### The larger finding: training on the clip venues destroys EMPTY detection

While repairing the control, the first version trained on everything except the held-out
camera — which includes the clip venues. It returned **1.0000 for every DINOv2
configuration**. Isolating the cause:

| training set | n | fraction ACTIVE_PLAY | false-play on 243 held-out EMPTY frames |
|---|---|---|---|
| `venue_01` camera A only | 775 | 0.668 | **0.231** |
| camera A **+ clip venues** | 1,057 | 0.757 | **1.000** |

All 282 clip-venue development frames are ACTIVE_PLAY and **none are EMPTY**. Adding them
takes the probe from calling 23% of unseen empty pitches a match to calling **all of them**.

Class balancing is already on, so this is not simply a shifted prior — those frames appear to
widen the PLAY region of feature space until it swallows an unseen camera's empty frames.

**This is the most serious consequence of the dataset gap found so far.** Every cross-venue
protocol in this project trains on venue_01 plus six clip venues, and H3 measured only
ACTIVE_PLAY recall on all-play test sets — so a probe that says PLAY to everything scores
perfectly and looks excellent. The high cross-venue recall numbers are real, but they were
never evidence that the model can recognise an empty pitch, and under that training set it
cannot.

**What it does not mean.** It is not evidence the approach fails: with training restricted to
a venue that has both classes, false-play is 0.021 for the best configuration. It means the
*clip venues cannot supply training data for the EMPTY class*, which is a labelled-data
problem, not a modelling one. Full-length recordings from any second venue would close it,
which is the same conclusion the blocked-questions diagram already draws.

---

## Near-duplicate audit and a label review (WP2-T4) — 2026-09-07

`src/pitch_occupancy/data/dedup.py`, `experiments/near_duplicate_audit.py` →
`results/near_duplicates.csv`. Perceptual difference hash (dHash, 64 bits), compared by
Hamming distance.

### The threshold is justified by the data, not carried in from elsewhere

| pairs | p5 | median | p95 |
|---|---|---|---|
| within one slot | 1 | 4 | 35 |
| across different slots | 18 | 30 | 38 |

Frames from different slots fall under the threshold of 6 only **0.42%** of the time, so the
threshold separates scenes rather than merging them.

### H1's leakage, finally quantified

There are 139,865 near-duplicate pairs among the development frames. The question is how many
each split puts on opposite sides — because a straddling pair is one frame being scored
against a copy of itself:

| split | straddling pairs | share |
|---|---|---|
| random split (H1's leaky control) | **51,946** | **37.1%** |
| grouped split (what replaced it) | 1,067 | **0.8%** |

**A 49× reduction.** H1 reported that a 16-bin colour histogram beat a deep probe under a
random split, and concluded the evaluation was measuring the dataset rather than the models.
This is the mechanism behind that result, counted.

### A mistake worth recording: single-link chaining is not a duplicate count

The first version grouped frames transitively and announced **96.9% redundant, "52 distinct
scenes"**. That number was an artifact, and the same output refuted it: the largest group
held 512 frames whose **maximum internal distance was 19** against a threshold of 6, and
contained both EMPTY and ACTIVE_PLAY frames. A fixed camera drifts slowly, so every frame is
near its neighbour in time and an entire slot chains into one blob. An empty pitch and a
match in progress are not duplicates of each other.

Duplication is now reported **pairwise**: 1,667 of 1,692 frames (98.5%) have at least one
direct near-duplicate. Chaining is kept in the code, documented as informative about slot
continuity and useless as a count.

### Two verified labelling errors, found by the cross-label signal

Pairs that are near-identical yet carry different labels are label-review candidates. At
distance ≤ 1 the ranking is led by five frames, and rendering them settles each:

| frame | label | verdict on inspection |
|---|---|---|
| `2_playing/…_t000021_m.jpg` | ACTIVE_PLAY | **wrong — the pitch is empty** |
| `2_playing/…_t000027_m.jpg` | ACTIVE_PLAY | **wrong — the pitch is empty** |
| `3_people_not_playing/…_t000269.jpg` | MAINTENANCE | correct — one person on the pitch |
| `3_people_not_playing/…_t000494.jpg` | MAINTENANCE | correct — one person on the pitch |
| `3_people_not_playing/…_t000509.jpg` | MAINTENANCE | correct — one person on the pitch |

Both errors are `labeled_by=human`, so this is annotation noise rather than a bulk-filing
bug.

**dHash is a candidate generator, not an oracle**, and the C3 rows are why. Measured on this
camera, frames with three or four players plus a cone sit **5–6 bits** from an empty frame,
but a **single** person near the frame edge sits at **0–1** — an 8×8 downscale erases one
figure. So it surfaces frames worth a human look and cannot decide any of them. Three of five
top candidates were correctly labelled.

### Why two frames matter more than two frames

Those two are in `venue_01`'s **morning** slot, which holds 247 EMPTY, 6 ACTIVE_PLAY and 6
MAINTENANCE frames on camera A, and 238 EMPTY with **zero** ACTIVE_PLAY on camera B. They are
therefore among the only **daytime ACTIVE_PLAY frames in the entire venue_01 recording**.

Removing them takes daytime ACTIVE_PLAY at `venue_01` from 6 frames to 4 — so the day/night
confound is *more* absolute than reported, not less, and part of the clock rule's 1.6% error
was it being right where the label was wrong.

**Not corrected in the data yet, deliberately.** Moving two files invalidates every feature
cache and shifts every number in this log, and a preprocessing search is running. The
evidence is recorded here and the correction is queued in `TODO.md`.

---

## H3 revisited with a false-play control — 2026-09-07

`experiments/h3_with_false_play.py` → `results/h3_with_false_play.csv`. Runs on the existing
feature caches, so it costs seconds rather than an embedding pass.

**The recall column reproduces the published H3 table exactly** — 0.219, 0.910, 0.930, 0.869
— before anything was added to it. A harness that cannot reproduce the number it is extending
has no business adding a second one.

| model | cross-venue play recall | worst fold | **false-play on 243 held-out EMPTY frames** |
|---|---|---|---|
| `clock_rule` | 0.219 | 0.000 | **0.021** |
| `convnextv2` | 0.910 | 0.722 | **0.992** |
| `dinov2` | **0.930** | 0.667 | **0.309** |
| `vit` | 0.869 | 0.500 | **0.835** |

### What this changes

H3's headline is *"across unseen venues the frozen backbones hold above 0.86 while a clock
rule collapses to 0.219"*. On recall that is exactly right. With the second column it reads
very differently:

- **ConvNeXtV2 calls 99.2% of held-out empty pitches a match.** Its 0.910 recall is not
  evidence of generalisation; it is close to what a model scores by answering "playing" to
  everything, on test folds that are 100% ACTIVE_PLAY.
- **ViT is nearly as bad at 0.835.**
- **The clock rule, the designated straw man, has the *lowest* false-play rate of any model**
  — 0.021 — because in daylight it says EMPTY and in this venue daylight means empty.
- **DINOv2 is the only model with a defensible balance**: the best recall *and* by far the
  best false-play of the three backbones.

Ranking on `recall − false-play` inverts the table: DINOv2 **0.621**, clock rule 0.198, ViT
0.034, **ConvNeXtV2 −0.082**. On a balanced view the pilot's production lead scores *below a
rule that reads the clock and never looks at the image*.

This is the third time the same lesson has arrived: RQ2 already moved the production pick
from ConvNeXtV2 to DINOv2 on accuracy under honest evaluation. This is a second, independent
reason, and a much starker one.

### The caveat, stated because the two columns are not the same fit

They cannot be. Every cross-venue fold has **zero EMPTY frames in its test set**, so
false-play is unmeasurable on the fold that produced the recall. Recall is cross-*venue*
(trained on six venues, tested on a seventh); false-play is cross-*camera* within `venue_01`
(trained on camera A, tested on camera B's empty frames). They measure different
generalisation from different fits, so `recall − false-play` is **indicative, not a single
coherent metric**, and the two numbers should be quoted side by side rather than combined
into one.

What makes the comparison fair is that every model faces the identical pair of tasks.

### Also recorded: why H3 has seven folds and not eight

The `venue_01` fold trains on the clip venues alone, and all 282 of their development frames
are ACTIVE_PLAY. A single class cannot be fitted, so the fold is dropped. That is the dataset
gap, not a code limitation, and it is the same gap behind every finding in this section.
