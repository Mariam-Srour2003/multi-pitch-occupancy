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
| `centre_crop=0.5 sharpen=0.6` | 0.7841 | **−0.2000** | — | 0.0000 |

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

`sharpen=0.6` alone is +0.0150. `centre_crop=0.5` alone is −0.0289. Together they are
**−0.2000** — far worse than the sum, and the worst result in the search. This is the
same non-additivity the input ablation found with grayscale and crop50, now reproduced on a
different pair of switches. **One-at-a-time preprocessing tables cannot be trusted to
compose**, which is the argument for searching the space rather than tabulating it.

### The finding that matters most, and it was nearly missed

The search ran entirely on **ConvNeXtV2** (the script's default), while the project's
configured default model is **DINOv2** (`config.default_model_key`). That would be a minor
bookkeeping note if preprocessing effects transferred between backbones. They do not:

| removing colour | DINOv2 (input ablation) | ConvNeXtV2 (this search) |
|---|---|---|
| full desaturation | **+0.022** | **−0.119** |
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

> **Corrected 2026-09-08 — the last sentence was false when written.** The flag was not in
> the fingerprint and `build_cache` had no parameter for it, so the alternative convention
> could not be cached at all; had it been, both conventions would have collided on one cache
> key *and* one filename. Only `embed_batch` knew about the flag. Fixed: `build_cache`
> threads it through, it is in the fingerprint, and the non-default convention gets its own
> `<backbone>_nogeom.npz`. This is the second claim in this project to assert a guard that
> did not exist — the first was the final-test-set lock, which was cwd-relative — and both
> were written in good faith about code that read as though it did the thing.

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

> **Read this with the significance entry below.** The 243 held-out empty frames amount to
> three to ten *distinct scenes*, and no pairwise comparison survives Holm correction once
> that is accounted for. The rates above are observed behaviour on one pitch, not an
> established ranking — and the failure is a lack of power, not evidence of equivalence.

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

---

## Significance testing on the false-play finding (WP4-T4) — 2026-09-07

`experiments/false_play_significance.py` → `results/false_play_significance.csv`.

**This qualifies the entry above, which stated the model reordering more confidently than
the evidence supports.**

### Taken at face value, everything is significant

| model | false-play | 95% bootstrap CI |
|---|---|---|
| `clock_rule` | 0.0206 | [0.0041, 0.0412] |
| `dinov2` | 0.3086 | [0.2510, 0.3663] |
| `vit` | 0.8354 | [0.7860, 0.8807] |
| `convnextv2` | 0.9918 | [0.9794, 1.0000] |

Non-overlapping intervals, and all **six** pairwise McNemar tests significant after
Holm–Bonferroni, with p-values down to 8e-53.

### Those numbers are arithmetic, not evidence

All 243 frames are consecutive samples of **one camera watching one empty pitch**. The
near-duplicate audit found their median pairwise dHash distance is **2 bits**. Requiring each
retained frame to differ from every other retained frame leaves:

| distinctness threshold | distinct scenes | comparisons still significant |
|---|---|---|
| 2 bits (very strict) | 10 | **0 of 6** |
| 6 bits (the audit's threshold) | 3 | **0 of 6** |

The count is stable across shuffles of the input order, so it is a property of the frames
rather than of the greedy selection. **Not one of the six comparisons survives.** The
p-values above are what you get by counting 243 views of the same empty pitch as 243
independent observations.

### What stands and what does not

**Stands — descriptive.** On this data ConvNeXtV2 called 99.2% of held-out empty frames a
match and DINOv2 30.9%. That is a fact about how these models behave here, and it is why the
model page now shows the column.

**Does not stand — inferential.** *"ConvNeXtV2 is significantly worse than DINOv2 at avoiding
false play"* is **not supported**. The effective sample is three to ten distinct scenes.

**And the distinction that stops this becoming an over-correction:** a non-significant result
here is *underpowered*, not null. The observed gap is enormous — 0.99 against 0.31 — and with
three to ten independent scenes no test could detect even a real difference of that size.
Absence of significance is not evidence of no difference. The honest statement is **"we do
not have the data to establish this"**, not "the models are equivalent".

### What would settle it

Empty-pitch footage from **more than one scene**. A second venue with genuine downtime, or
`venue_01` on other days, would turn three effective observations into dozens. This is the
same missing data behind every blocked question in this project, and it now also blocks the
statistical backing for its most striking model comparison.

Until then the false-play column should be presented as **observed behaviour on one pitch**,
with the effective sample stated beside it.

---

## Effective sample size across the reported tests (WP4-T4b) — 2026-09-07

`experiments/effective_sample_audit.py` → `results/effective_sample_audit.csv`. All six
published accuracies reproduce exactly before anything is qualified, and the script now
refuses to continue if they ever stop.

The false-play work showed 243 frames were three to ten distinct scenes. This asks the same
question of the H1/H2 test sets.

| split | nominal test frames | distinct scenes | share |
|---|---|---|---|
| random (leaky) | 394 | **62** | 15.7% |
| grouped | 907 | **94** | 10.4% |

### The one comparison that does not survive

| split | comparison | p (all frames) | p (distinct scenes) | survives |
|---|---|---|---|---|
| random | **DINOv2 vs colour histogram** | 7.4e-03 | **1.00** | **no** |
| random | DINOv2 vs clock rule | 8.2e-07 | 6.1e-05 | yes |
| random | colour histogram vs clock rule | 1.5e-02 | 6.1e-05 | yes |
| grouped | DINOv2 vs colour histogram | 1.9e-171 | 1.2e-15 | yes |
| grouped | DINOv2 vs clock rule | 1.2e-03 | **2.2e-02** | yes, marginally |
| grouped | colour histogram vs clock rule | 2.8e-157 | 4.2e-11 | yes |

**On the 62 distinct scenes of the leaky split, DINOv2 and a 16-bin colour histogram both
score 1.0000.** Not "close" — identical. The 0.9873 against 0.9594 difference, and its
p = 0.0074, come entirely from near-duplicate frames on which one model happened to slip.

### This strengthens H1 rather than undermining it

H1's claim is that the random split measures the dataset rather than the models. The
significance test attached to it was inflated, but removing the inflation makes the point
harder, not softer: under the leaky protocol a **colour histogram is statistically
indistinguishable from a deep self-supervised backbone**. The original phrasing — that the
histogram *beat* the deep probe on macro-F1 — should become *"the protocol cannot tell them
apart"*, which is a cleaner statement of the same finding and no longer depends on a
difference that is not there.

### H2 survives, and it is worth saying it is now marginal

Under the grouped split DINOv2 (0.9846) still beats the clock rule (0.9636) significantly,
but the p-value moves from 1.2e-03 to **2.2e-02** — from comfortable to just inside the
threshold, on 94 scenes. H2's point was always that the gap is *small*; it is now small and
weakly evidenced, which is the honest version.

### H3 is sound, and for a specific reason

H3 is deliberately out of scope here. Its intervals bootstrap over the **seven venue folds**,
not over frames, so seven different facilities count as seven observations and the width
already reflects that — the clock rule's interval is [0.029, 0.505], which is not the
interval of an over-confident test. **Choosing the resampling unit to match the thing being
generalised over is what made H3 robust**, and it is the practice the frame-level tests
lacked.

### The rule this settles for the thesis

Every frame-level statistic in this project should be reported with its effective sample
size beside it. The dataset is 1,692 frames and roughly **150 distinct scenes**; the second
number is the one that governs how confident any frame-level claim may be.

---

## 2026-09-07 · WP2-T10: camera-view fingerprinting

`uv run python experiments/camera_fingerprint_audit.py`
-> `results/camera_fingerprint.csv` | module `vision/fingerprint.py`, 25 tests

A camera view is reduced to the per-pixel **median background** of its frames (players move,
the pitch does not) and described by a coarse grid of gradient-orientation histograms plus
rg-chromaticity histograms, watermark corner masked. 70 views: 66 clip cameras + venue_01's
four camera-slots.

**Correction to how this was first written up: the camera-identity half was already
done.** `vision/camera_id.py` (built earlier, same WP2-T10) had already measured the suffix
swap by gradient-magnitude similarity - 0.88 cross-day against 0.70 own-suffix - and
`db/seed.py` already encodes it. This module was written without noticing that, so its
first claim to have "measured Gotcha 2.7" was wrong: it *replicated* the measurement. What
follows is corrected to say so, and the duplication was turned into a cross-check rather
than deleted (see point 4).

**1. The suffix swap, independently replicated.** The `(1).mp4` suffix does swap which
physical camera it means between the two venue_01 recording days:

| view | mutual nearest neighbour | distance |
|---|---|---|
| `slot_20260711_1000_camA` | `slot_20260712_2030_camB` | **0.111** |
| `slot_20260711_1000_camB` | `slot_20260712_2030_camA` | **0.277** |

Both pairings are *mutual* nearest neighbours across days with opposite suffixes, and both
distances sit below the same-venue mean (0.256), while same-day camA-vs-camB sit at 0.46 and
0.57 — far apart, as two cameras on opposite halves should be. A hand-maintained metadata
convention was wrong and the pixels caught it. Anything keyed on the suffix (fusion pairing,
per-camera ROIs) must key on the fingerprint instead.

**2. Venue assignment from pixels works; venue *discovery* does not.**

| descriptor block | 1-NN venue assignment | separation AUC |
|---|---|---|
| structure + 0.5x chroma | **67/70 = 0.957** | **0.880** |
| structure only | **67/70 = 0.957** | 0.844 |
| chromaticity only | 59/70 = 0.843 | 0.800 |

> **Corrected 2026-09-07 — the AUC column was wrong and the conclusion drawn from it was
> backwards.** This table originally read 0.799 for structure-only and **0.970** for
> chromaticity-only, and argued from those: *"the two blocks are good at different things —
> structure identifies, chromaticity separates."* Neither figure was reproducible, because
> `suggest_threshold` was only ever called on the combined descriptor, so **just one of the
> three AUCs came from the artefact** and the other two came from somewhere unrecorded. The
> audit script now computes the AUC per block (`assignment_*` rows in
> `results/camera_fingerprint.csv` carry it), and measured: structure-only **0.844**,
> chromaticity-only **0.800**.
>
> **Chromaticity does not separate better — it is worse on both axes.** The claimed ordering
> (chroma 0.970 > combined 0.880 > structure 0.799) is in fact the reverse
> (combined 0.880 > structure 0.844 > chroma 0.800). The "different things" story does not
> survive.
>
> **The decision it was supporting survives anyway, on a plainer argument.** The combination
> beats *both* of its components on separation (0.880 against 0.844 and 0.800) while matching
> the better one on assignment (67/70). That is ordinary complementarity and it is sufficient
> reason to keep both blocks — it simply is not the reason the entry gave. The lesson is the
> one this project keeps relearning: a number quoted in a write-up but not emitted by the
> script is a number nobody has checked.

Both blocks are kept, at the weight that costs nothing on assignment (measured over 0.3–1.5).
Same-venue and different-venue distances **overlap** (0.194 min vs 0.644 max), so no
threshold settles venue identity on its own. Unsupervised clustering peaks at ARI 0.714 with
**4 clusters still mixing two facilities each** — enough to break a leave-one-venue-out split
if it were trusted. The coarsest grouping with zero mixing needs threshold 0.23 and 20
clusters for 70 views.

**Accept criterion partly met, and recorded as such.** WP2-T10 asked for "all existing
footage auto-assigned correctly": 67/70, not 70/70. The three misses are `cb_...680610`,
`cg_...155619` and `ci_...712737` — all day or dusk clips whose nearest neighbour is a
different facility. Use the module to *check* a hand grouping and to assign new footage
against known references, with a human confirming; not to invent a grouping.

**3. It confirms the visual audit's load-bearing claim.** `clipvenue_g_netting` and
`clipvenue_h_teal_pitch` share no cluster at any threshold — they are different facilities,
as the visual audit concluded, so the leave-one-venue-out folds are not leaking between them.
The audit's other claim (that `g` is one facility) is **not testable this way and must not be
read as contradicted**: `g` splits into 2 view-clusters because it *is* two pitches at one
site, which is what a view fingerprint should do — `venue_01` splits the same way, into its
two cameras. The audit established `g` from a pitch number visible across the fence, and a
descriptor cannot read a number board.

Two metrics were tried and discarded before this: ARI alone punishes the correct
multi-pitch splitting, and homogeneity alone is gamed by splitting into singletons (it
maximises at 30 clusters for 70 views). The reported pair — coarsest pure threshold and best
agreement — is what survives both objections.

**4. The duplication became a cross-check.** `camera_id` and `fingerprint` share only the
median-background step: one keeps gradient *magnitude* on a 48x27 grid with no colour, the
other keeps gradient *orientation* histograms plus rg-chromaticity with the watermark
masked. `tests/test_camera_id_agreement.py` (6 tests) asserts that both reach the same
pairing and that `db/seed.py`'s `PHYSICAL_CAMERA` still encodes it, and one test guards the
premise by failing if the two descriptors ever converge into the same measurement. Two
independent descriptors agreeing is stronger evidence for the write-up than one measured
twice — which is the only reason keeping both modules is defensible. If they are ever
merged, that test must go with them.

**What each module is for, so the next reader does not pick the wrong one:** `camera_id`
answers *which physical camera is this recording* (video files, references, greedy
matching) and is the one wired into the system. `fingerprint` answers *how do these 70
views group into venues* (extracted frames, distance matrices, clustering, threshold
search) and is an audit tool, not a runtime component.

- 2026-09-07 | WP2-T10 camera fingerprinting | `python experiments/camera_fingerprint_audit.py` | `camera_fingerprint.csv` | 70 views, 10 venues

---

## H1/H2 re-reported on a stable estimand (WP4-T4b) — 2026-09-07

`experiments/h1_h2_baseline_floor.py` -> `results/h1_h2_baseline_floor.csv`, figures
regenerated. **This changes the random split's numbers substantially and leaves the grouped
split's untouched.** H1 gets stronger; the ranking-inversion finding survives but one of its
components turns out to have been an artefact.

### The defect

`bootstrap_metric_ci` was called as `lambda t, p: evaluate(t, p).macro_f1`. `evaluate`
excludes zero-support classes from the macro average - which is correct - but it was being
asked to do so **once per resample**, so the class set was re-derived 10,000 times and the
estimand moved with it.

C3 has support **1** in the random split's test set (n=394: C2 274, C1 119, C3 1). A
bootstrap resample of that size contains that single frame **63.4% of the time**. So roughly
two thirds of the resamples averaged three classes and one third averaged two, and the
resulting interval covered two different quantities. The grouped split is unaffected: both
its classes appear in essentially every resample (C1 support 9 of 907).

Support 1 is not evaluable in any case - one frame yields an F1 of 0 or 1 with nothing in
between, and no resampling scheme manufactures the missing information. The class set is now
fixed **once** from the full test set, with the point estimate and the interval computed over
the same restricted data, and the retained classes plus anything dropped are written into
every CSV row (`macro_over_classes`, `excluded_low_support`, `n_test_evaluable`).

`preregistration.md` already promised this: *"metrics are reported 2-class where C3 support is
zero, and this is stated in every table."* Support 1 slipped through a rule written for
support 0.

### What moved

| split | model | before | after | delta | CI width before | after |
|---|---|---|---|---|---|---|
| random | vit | 0.9960 | 0.9940 | -0.002 | 0.013 | 0.016 |
| random | convnextv2 | 0.6573 | **0.9879** | **+0.331** | 0.348 | **0.023** |
| random | dinov2 | 0.6573 | **0.9879** | **+0.331** | 0.348 | **0.023** |
| random | cheap_histogram | 0.6855 | **0.9616** | +0.276 | 0.309 | 0.037 |
| random | clock_rule | 0.6050 | **0.9091** | +0.304 | 0.344 | 0.059 |
| grouped | *every model* | - | - | **0.0000** | - | - |

The intervals were not merely wrong, they were **fifteen times too wide** on the affected
rows - the mixed estimand was most of the apparent uncertainty.

### H1 is understated by a factor of three, not overstated

H1 claims same-scene evaluation inflates apparent performance. Measured on a consistent
metric, the drop from the leaky to the honest protocol is:

| model | random | grouped | drop |
|---|---|---|---|
| vit | 0.9940 | 0.4975 | **-0.497** |
| convnextv2 | 0.9879 | 0.4975 | **-0.490** |
| dinov2 | 0.9879 | 0.5794 | **-0.409** |

Previously reported as -0.159 for ConvNeXtV2. The honest figure is **-0.490, about 3.1x
larger**. The claim was not too strong; it was far too weak. Leakage does not shave points off
this benchmark, it **halves the score** - and that is a much cleaner statement of the thesis's
central methodological result.

It also sharpens H2 on the leaky side, where a fair comparison now exists: under the leaky
protocol a 16-bin colour histogram reaches **0.9616** against the best deep probe's 0.9940,
and a clock rule that never looks at the image reaches **0.9091**. Everything is at ceiling,
which is what "the protocol is measuring the dataset" looks like when every model is scored on
the same classes.

### One component of the ranking inversion was an artefact

The inversion **stands**: ViT ranks 1st under the leaky protocol (0.9940) and is tied 2nd/3rd
under the honest one (0.4975, against DINOv2's 0.5794). A reader following the pilot's
protocol still selects the weakest generaliser.

But its *magnitude* on the leaky side was not real. ViT's old 0.9960 against the field's 0.6573
was a gap of 0.339, and almost all of it came from ViT happening to classify the single C3
frame correctly while the others did not - a three-class average against a two-class one. On
equal footing the leaky-protocol gap is **0.006**. The inversion is a fact about ranking, not
about margins, and it should be quoted that way.

### The figure did not read the table it illustrates

Regenerating `figs/ranking_inversion.png` changed nothing, which is how a second defect
surfaced: `fig_ranking_inversion()` **hardcoded both the ranks and the printed scores**. It
never read `h1_h2_baseline_floor.csv`, so it kept displaying DINOv2 and ConvNeXtV2 at the
random split's old 0.657 after the corrected value became 0.988 - the most quoted figure in the
thesis contradicting its own source table, while the `figures` reproduction stage reported
success and its own docstring said "regenerated from the CSVs".

Both columns are derived from the CSVs now, ties are detected from the data rather than
described in a comment, and the subtitle is computed too: it asserted ViT was "last under the
honest one", which was true of the old numbers and is not true of the corrected ones - ViT is
**tied 2nd**. A caption is a claim, so it is now checked on every regeneration rather than
remembered. Score labels also carry a background halo, because two tied scores sit a fraction
apart and a third series' line ran straight through one of them.

This is the same defect class as the fingerprint AUC column found earlier the same day: a
number that appears in an output but is not computed from the data it describes. Two instances
in one review is enough to call it the pattern to watch for here.

### Accuracy stays on the full test set, and another experiment's guard is why

The first version of this fix restricted **accuracy** to the evaluable subset too. That was
wrong, and `experiments/effective_sample_audit.py` caught it: that script reproduces the
published H1/H2 accuracies to 5e-4 and refuses to run if they have moved. Dropping the single
C3 frame shifted them by up to **0.0025** - four to five times its tolerance - so the audit
exited rather than report anything.

It was right to. The estimand problem belongs specifically to the macro *average over
classes*, where an undefined per-class F1 has to be either excluded or invented. Accuracy has
no such difficulty: every frame has a right answer whatever its class's support, so dropping a
real observation from it buys nothing. Accuracy and balanced accuracy are computed on all 394
frames; only the macro-F1 and its interval use the restricted set, and the row records both
sizes.

**The guard was worth more than the number it protects.** It is a reproduce-before-extending
check written for a different purpose, and it caught a change two files away that no test
covered.

### What does not change

Every grouped-split number, every H2 conclusion drawn on the grouped split, and all of H3 -
which never used this code path, and whose intervals bootstrap over venue folds rather than
frames. The effective-sample audit's conclusions are also untouched: they concern
independence between frames, not the class set.

- 2026-09-07 | WP4-T4b H1/H2 stable estimand | `python experiments/h1_h2_baseline_floor.py` | `h1_h2_baseline_floor.csv` | random split +0.33 macro-F1, grouped unchanged; H1's effect 3.1x larger

- 2026-09-07 · H1/H2 · `python experiments/h1_h2_baseline_floor.py` · seed 42 · `h1_h2_baseline_floor.csv` · 14 rows over 2 splits

---

## WP5-T9: the naive ensemble takes 74% of the fusion headroom, and fails on the other axis — 2026-09-07

`experiments/logit_average_baseline.py` -> `results/logit_average_baseline.csv`, 10 tests.
The sanity baseline the plan asks for **before** building the gated-fusion module (WP5-T2).
All three single-backbone means reproduce the published H3 table exactly before anything is
added.

The case for a gate rests on complementarity: DINOv2 leads the fold mean at 0.9297 but is
strictly beaten on three of seven venue folds, and a per-fold oracle reaches 0.9603 -
**+0.031** of headroom. The question this answers is whether a *learned* gate is needed to
collect it.

| model | play recall | worst fold | false-play | vs best single |
|---|---|---|---|---|
| convnextv2 | 0.9105 | 0.722 | 0.9918 | -0.019 |
| **dinov2** | **0.9297** | 0.667 | **0.3086** | - |
| vit | 0.8690 | 0.500 | 0.8354 | -0.061 |
| mean-probability, ConvNeXtV2+DINOv2 | 0.9473 | 0.667 | **1.0000** | +0.018 |
| mean-log-probability, ConvNeXtV2+DINOv2 | 0.9524 | 0.667 | **1.0000** | +0.023 |
| mean-probability, all three | 0.9524 | 0.667 | **1.0000** | +0.023 |
| mean-log-probability, all three | 0.9524 | 0.667 | **1.0000** | +0.023 |
| *oracle (per-fold best single)* | *0.9603* | - | - | *+0.031* |

### Two findings, and the second one decides it

**1. Averaging captures 74% of the headroom for free.** The best naive ensemble reaches
0.9524 against a 0.9603 ceiling that requires knowing which venue a frame came from. A
parameter-free mean gets three quarters of the way there, leaving **0.008** for a learned
gate to fight over - and a gate cannot reach the oracle anyway, because it does not have
venue identity. On recall alone, WP5-T9 has already answered the question the plan warned
about: *if a naive ensemble matches the learned gate, the ensemble is the contribution.*

**2. Every ensemble scores 1.0000 false-play, which is worse than every backbone alone.**
This is the finding that matters. DINOv2's one genuinely valuable property is that it does
*not* call empty pitches a match (0.3086, against ConvNeXtV2's 0.9918). Blend ConvNeXtV2
into the decision and that property is gone completely: the ensembles call **100% of 243
held-out empty frames** a match. The +0.023 of recall is bought at +0.69 of false-play.

On the balanced view the naive ensemble is **strictly worse than DINOv2 alone**, and this is
the third time on this dataset that a configuration has looked better on a single-class test
set while being worse at the thing that actually matters. Recall on 100%-ACTIVE_PLAY folds
rises by answering "playing" more often, and an ensemble is a very efficient way to answer
"playing" more often.

> ### ⚠ Finding 2 was REFUTED on 2026-09-08 — see the re-run entry at the end of this log
>
> It rests on ConvNeXtV2's false-play of **0.9918**, and the geometry probe run the next day
> shows that figure is largely an artefact of the **input path** rather than a property of
> the model. Every cache this experiment used was built from *raw frames* handed to the HF
> processor, which resizes a 1920x1080 frame shortest-edge to 256 and centre-crops 224 -
> keeping roughly the middle half of the pitch horizontally. Apply `preprocess.py`'s
> aspect-preserving letterbox instead and ConvNeXtV2's false-play measures **0.0206**, the
> lowest of any arm tried, with recall *rising* to 0.9841.
>
> If that holds, the premise of finding 2 inverts: under preprocessing ConvNeXtV2 is better
> than DINOv2 on **both** axes, so "blending ConvNeXtV2 in destroys DINOv2's one valuable
> property" describes the raw path only. Finding 1 - that a parameter-free average captures
> 74% of the oracle headroom - is unaffected, since it is a recall-side result.
>
> **Settled by the re-run.** On letterboxed caches the two-model ensemble scores **0.0288**
> false-play at **1.0000** recall - the best balanced score measured on this dataset, better
> than any single backbone. Blending was never disqualified; blending *a model that had been
> shown a central strip of the pitch* was. Finding 1 survives and strengthens: the naive
> average now captures **100%** of the oracle headroom, so the gate has no room - and that
> conclusion no longer depends on the false-play axis at all. WP5-T2's answer is unchanged
> and its reasoning is now the opposite.

### What this does to WP5-T2

It removes the version of the module that was easiest to build. **Blending is disqualified**:
any fusion that mixes ConvNeXtV2's logits into the verdict inherits its empty-pitch
blindness, whatever the weights. So a gate here cannot be a soft blend - it has to be a
**hard router with DINOv2 as the default**, invoking another backbone only where it is
confident the scene is not empty.

Which puts the module back against the confound it was always going to face: deciding "this
scene might be empty" from cheap image statistics, on a venue where brightness is nearly a
day/night indicator and day/night is nearly the class label. That is the clock rule wearing a
different hat, and it is now the *whole* module rather than a caveat on it.

**The honest read: WP5-T2's expected outcome is a negative result**, and it should be planned
as one - built to be reported either way, against these numbers as the baseline, with the
lighting-only gate ablation run first. That is a legitimate contribution under the WP5 rules,
and it is a much better position than discovering it in week 18.

### What is not established

The **+0.023 recall gain is not tested**. It is an unweighted mean over seven folds, three of
which have 12-18 play frames, and no paired test was run over the fold distribution. It should
not be quoted as a real improvement - and it barely matters, because the false-play column
makes the direction of the overall comparison negative regardless of whether that gain is
real. A paired bootstrap over folds belongs with WP4-T4b.

- 2026-09-07 | WP5-T9 logit-average baseline | `python experiments/logit_average_baseline.py` | `logit_average_baseline.csv` | naive ensemble takes 74% of oracle headroom; false-play 1.0000, worse than every single backbone

- 2026-09-07 | WP5-T9 logit-average baseline | `python experiments/logit_average_baseline.py` | `logit_average_baseline.csv` | best single dinov2 0.9297, oracle 0.9603

---

---

## WP3-T3: the convention question is not runnable yet, and finding out why matters more — 2026-09-08

Setting up *"one cross-venue run per model under each convention"* - the measurement
`protocol.md` says will settle whether to disable the HF processor's geometry - turned up
three things in ascending order of importance.

### 1. The flag could not be cached, and the fingerprint claim was false

`embed_batch` accepted `processor_geometry`; nothing above it did. `build_cache` had no such
parameter, so **the alternative convention could not be produced at all**. And the flag was
not in the fingerprint either, though this log and `thesis/protocol.md` both asserted that it
was - so the two conventions would have collided on one cache key *and* one filename, the
second build silently overwriting the first.

Fixed: the parameter threads through, it is in the fingerprint, and the non-default
convention gets `<backbone>_nogeom.npz`. Six tests.

This is the **second** claim in this project to assert a guard that did not exist - the
first was the final-test-set lock, which resolved a cwd-relative path. Both were written in
good faith about code that read as though it did the thing.

### 2. The existing caches were restamped, not re-embedded, and that was checked first

Adding the flag to the fingerprint changed the hash of the *default* convention, so the
three main caches carried a stale one. The features are unchanged - the flag defaults to the
behaviour they were built with - so restamping is the faithful action and re-embedding would
be waste. But "faithful" is a premise, so it was tested rather than assumed: 24 frames per
backbone re-embedded at `processor_geometry=True` and compared to what was stored.

| backbone | identical | max abs diff | fingerprint |
|---|---|---|---|
| convnextv2 | yes | 0.000e+00 | `1a9884ba1a9f536f` -> `1b0463ed87f74f7d` |
| dinov2 | yes | 0.000e+00 | `b1e06cdd436e8522` -> `fb1f12b5d77d212d` |
| vit | yes | 0.000e+00 | `52bd9716fba57b92` -> `9ff3ae02598ffd4e` |

Bit-for-bit on all three, and the migration script refuses to restamp anything if a single
check fails. `load_cache(..., expect_preproc_hash=...)` now passes for all three, which it
could not before. **No published number moves.**

### 3. `build_cache` does not apply `preprocess.py` — and that is the finding

Turning the processor's geometry off crashed:

```
ValueError: Input image size (1080*1920) doesn't match model (224*224)
```

Because `build_cache` opens **raw frames** and hands them to the HF processor, whose resize
is the only thing making them model-sized. Its `preproc` argument is a dict of *labels for
the fingerprint* - it has never driven a transform, and `preprocess` is not imported there.

So the letterbox of WP3-T2, ROI masking, CLAHE and the rest of the ten searched switches
**are not in the path that produced any cached feature the headline experiments read.** The
preprocessing search and the input ablation are the exception: those scripts call
`preprocess()` themselves, which is exactly why they keep a separate cache family
(`data/cache/search/`, `ablate_*`).

`protocol.md` calls `preprocess.py` "the single preprocessing path". For the main caches it
is not a path at all. The pipeline diagram there is the *intended* one and the one the search
operates in - not the one behind the cross-venue numbers.

### What this does to the experiment

It stops being a one-line run. Comparing the conventions honestly means putting
`preprocess.py` into the main cache path, which changes the input to **every** published
number rather than just the nogeom arm. That is a decision, not a patch, so the options are
recorded in TODO WP3-T3 rather than chosen here: **(a)** a self-contained side experiment
supplying `preprocess_fn` to both arms, touching no existing cache; **(b)** adopt
`preprocess.py` in `build_cache` and regenerate everything - the coherent end state, and it
invalidates every cached feature; **(c)** leave the default path alone and say plainly that
the searched switches apply only to the search and ablation caches.

**(a) first**, because it is cheap and it tells you whether (b) is worth its cost.

Meanwhile `build_cache` requires an explicit `preprocess_fn` when the flag is off, so the
dependency is legible instead of a `ValueError` five frames deep in transformers.

- 2026-09-08 | WP3-T3 processor-geometry setup | `pytest tests/test_feature_cache.py` | `data/cache/*.npz` restamped | flag now cacheable and fingerprinted; build_cache found not to apply preprocess.py at all

## H4: confirmed arithmetically, and it means nothing — 2026-09-07

`experiments/h4_model_equivalence.py` -> `results/h4_model_equivalence.csv`. The three
grouped-split macro-F1 values reproduce the published table exactly before anything is added.
First of the three pre-registered hypotheses that had never been reported (A5).

H4: *ConvNeXtV2-Tiny is statistically indistinguishable from ViT-Base on macro-F1 under
grouped splitting, while being >= 2x faster per frame.* Its decision rule reads *a null
result confirms H4* - and this run is a concrete demonstration of why that rule is unsafe.

### The equivalence is real, and it is degenerate

| pair | ΔmacroF1 | 95% CI | verdict at ±0.02 | Holm p (accuracy) |
|---|---|---|---|---|
| ConvNeXtV2 vs ViT | **+0.0000** | **[+0.0000, +0.0000]** | equivalent | 1.000 |
| ConvNeXtV2 vs DINOv2 | -0.0819 | [-0.2308, +0.0028] | inconclusive | 0.375 |
| DINOv2 vs ViT | +0.0819 | [-0.0028, +0.2308] | inconclusive | 0.375 |

The interval for H4's pair has **zero width**, which is the tell. These two models are not
similar - they are *identical*. Both predict ACTIVE_PLAY for all 907 test frames and EMPTY
**zero times**:

| | precision | recall | F1 | support |
|---|---|---|---|---|
| C1_EMPTY | 0.000 | 0.000 | **0.000** | 9 |
| C2_ACTIVE_PLAY | 0.990 | 1.000 | 0.995 | 898 |

0.4975 is `(0.995 + 0.000) / 2`. It is the score of a model that never gets an empty pitch
right, and both backbones sit on exactly that point. **They are equivalent to each other and
equally equivalent to a constant predictor.** DINOv2, at 0.5794, is the only one of the three
that ever predicts EMPTY - consistent with its false-play rate of 0.309 against ConvNeXtV2's
0.992.

So H4's accuracy clause is *confirmed* and carries no information about whether the two
models are interchangeable in production. It measures a test set that cannot distinguish
anything, which is H1's finding arriving from a third direction.

### The pre-registered decision rule would have got this wrong three times

*A null result confirms H4* is the absence-of-evidence error, and on a 99%-single-class test
set a null result is close to guaranteed. Under that rule all three pairs "confirm"
equivalence - including the two where DINOv2 is **8.2 macro-F1 points** better and the
interval runs out to -0.23. Read as intervals against a margin declared in advance
(±0.02, borrowed from H2's own threshold rather than invented here), those two are
**inconclusive**, which is the honest third answer the rule collapses away.

The pair that is genuinely equivalent is equivalent for a reason that disqualifies the
conclusion. Reporting *equivalent* and *inconclusive* separately is what makes that visible.

### The speed clause fails on its own stated condition

| measurement | ViT | ConvNeXtV2 | ratio | >= 2x? |
|---|---|---|---|---|
| single-frame median | 303.3 ms | 150.9 ms | **2.01x** | yes, barely |
| **20-camera concurrent median** | 4361.5 ms | 2388.1 ms | **1.83x** | **no** |

H4 says ">= 2x faster per frame", and its own decision rule says latency is *"reported as
median and p95 under a 20-camera concurrent load"*. Under that condition the ratio is
**1.83x** and the clause is not met. It is met only on the single-frame median, which is the
measurement the pre-registration explicitly declined to rely on. Memory-bandwidth contention
compresses the gap, exactly as WP0-T10 predicted it might.

### Verdict

**H4 is refuted as a whole, and its accuracy half is confirmed but uninformative.** It cannot
support the production recommendation, which is what its decision rule was for: *"a null
result confirms H4; the production recommendation then rests on latency."* Latency does not
clear the bar it set, and the equivalence it relies on is an artefact of a degenerate test
set. **RQ2's recommendation of DINOv2 stands on the cross-venue evidence instead**, where the
three models are not equivalent at all (0.930 / 0.910 / 0.869 play recall, and 0.309 / 0.992 /
0.835 false-play).

### Two caveats on this run

The distinct-scene rows are reported but should not be read as a second opinion: 95 scenes
contain **2 EMPTY frames**, so a macro average over them rests on two observations. They are
in the CSV for completeness and because the standing rule requires the effective sample
beside every frame-level number, not because they settle anything.

The scene count is 95 here against the 94 in `effective_sample_audit.csv`. The rows are
filtered to frames present in **all three** caches rather than DINOv2's alone, which shifts
the split by one frame. Recorded rather than reconciled - it changes no conclusion, and a
silent one-frame discrepancy between two scripts is worth more written down than explained
away.

- 2026-09-07 | H4 model equivalence | `python experiments/h4_model_equivalence.py` | `h4_model_equivalence.csv` | ConvNeXtV2 vs ViT: equivalent at margin 0.02

- 2026-09-07 | preprocessing search | `python experiments/preprocess_search.py` | `preprocess_search.json` | 88 evaluations

---

## WP3-T3 answered, and it moves more than the convention — 2026-09-08

`experiments/geometry_convention_probe.py` -> `results/geometry_convention_probe.csv`.
Three arms x three backbones, both axes, in a cache directory of its own
(`data/cache/geom_probe/`) so nothing published could be overwritten - the main caches were
md5-checked before and after and are byte-identical.

| backbone | arm | recall | worst fold | false-play | balanced |
|---|---|---|---|---|---|
| convnextv2 | raw+geom *(published)* | 0.9105 | 0.722 | **0.9918** | −0.0813 |
| convnextv2 | **preproc+geom** | **0.9841** | **0.889** | **0.0206** | **+0.9635** |
| convnextv2 | preproc+nogeom | 0.9711 | 0.833 | 0.4815 | +0.4896 |
| dinov2 | raw+geom *(published)* | 0.9297 | 0.667 | 0.3086 | +0.6211 |
| dinov2 | **preproc+geom** | **0.9595** | **0.800** | **0.2305** | **+0.7290** |
| dinov2 | preproc+nogeom | 0.8690 | 0.417 | 0.1687 | +0.7003 |
| vit | raw+geom *(published)* | 0.8690 | 0.500 | 0.8354 | +0.0336 |
| vit | preproc+geom | 0.9118 | 0.611 | 0.9712 | −0.0594 |
| vit | preproc+nogeom | 0.9118 | 0.611 | 0.9712 | −0.0594 |

### 1. The convention question: keep the processor's geometry

`protocol.md` leaned the other way - disabling it "is the coherent choice for a project where
preprocessing is a *searched* variable". Measured, that is wrong on this data:

* **ConvNeXtV2**: disabling costs 0.47 of balanced score (0.964 -> 0.490), almost all of it
  false-play (0.021 -> 0.482).
* **DINOv2**: a genuine trade - false-play improves (0.231 -> 0.169) but recall falls further
  (0.960 -> 0.869) and the worst fold collapses from 0.800 to **0.417**. Balanced favours
  keeping it, narrowly.
* **ViT**: no difference whatsoever, and that is a *confirmation*. Its two caches are
  **bit-identical** (max abs diff 0.0), while ConvNeXtV2's differ by 7.38 and DINOv2's by
  1.22. `protocol.md` predicted exactly this - ViT's processor resizes to exactly 224x224 and
  crops nothing - so the probe reproduces the audit's own finding from the other direction.

**Recommendation: never disable it.** Why it works is worth stating, because it is not
obvious: the letterbox produces a 224x224 frame with grey bars, and the processor then
resizes to 256 and crops 224 back out - which trims most of the padding. The pair is
aspect-preserved content with the padding removed. Each step alone is worse than both.

### 2. The finding that matters: ConvNeXtV2's false-play was mostly the input path

`preproc+geom` against the published `raw+geom`:

| backbone | Δrecall | Δfalse-play | Δbalanced |
|---|---|---|---|
| convnextv2 | +0.0736 | **−0.9712** | **+1.0448** |
| dinov2 | +0.0298 | −0.0781 | +0.1079 |
| vit | +0.0428 | +0.1358 | −0.0930 |

**ConvNeXtV2's 0.9918 false-play - the number behind "ConvNeXtV2 says PLAY to almost
everything", and one of the reasons the production pick moved to DINOv2 - drops to 0.0206
when the frame is letterboxed instead of handed raw to the processor.** The mechanism is
plain: a 1920x1080 frame resized shortest-edge to 256 and centre-cropped to 224 keeps about
the middle half of the pitch horizontally. The model was being shown a narrow central strip
and asked whether the pitch was empty.

So under preprocessing the ranking inverts. ConvNeXtV2 leads on **both** axes (0.9841 /
0.0206 against DINOv2's 0.9595 / 0.2305) *and* is the fastest of the three. That would
reverse RQ2's reversal.

**Preprocessing is not universally good, though.** It helps ConvNeXtV2 enormously, DINOv2
modestly, and **hurts ViT** - recall up but false-play up more. "Adopt `preprocess.py`
globally" is therefore not the clean conclusion the first backbone suggested; what to adopt
depends on which backbone is deployed.

### What is not established

Everything above is a **strong signal on thin evidence**, and the thin part is the same as
always: the false-play column is 243 frames that the effective-sample audit found to be
**three to ten distinct scenes**. A swing from 0.99 to 0.02 on ten effective observations is
worth acting on and is not a measured quantity. Nor are there CIs, multiple seeds, or a
paired test here; the recall side is an unweighted mean over seven folds, three of them 12-18
frames. Only the default letterbox was used - none of the ten searched switches.

**So this probe decides what to do next, not what to claim.** It says the convention question
is settled (keep the geometry), and it says the *input path* deserves the full protocol -
CIs, paired tests, effective sample - because a result this large sitting outside the
published pipeline is not something to leave in a side experiment.

### Consequences to work through

1. **RQ2's production recommendation is in question again.** Under preprocessing ConvNeXtV2 is
   better on both axes and 2x faster. Do not change the recommendation on this evidence -
   settle it under the full protocol first.
2. **WP5-T9's second finding is undercut**, as already flagged: "blending is disqualified"
   rests on ConvNeXtV2's 0.9918. Re-run it on these caches - they exist now.
3. **WP3-T3's third option is the live one.** Putting `preprocess.py` into `build_cache`
   changes the input to every published number. This probe is the cheap evidence that the
   question is worth the expense, which is what it was for.

- 2026-09-08 | WP3-T3 geometry probe | `python experiments/geometry_convention_probe.py` | `geometry_convention_probe.csv` | keep processor geometry; ConvNeXtV2 false-play 0.9918 -> 0.0206 under letterboxing

- 2026-09-07 | WP5-T9 logit-average baseline | `python experiments/logit_average_baseline.py` | `logit_average_baseline_preproc.csv` | best single convnextv2 0.9841, oracle 1.0000

---

## WP5-T9 re-run on preprocessed caches: finding 2 refuted, finding 1 strengthened — 2026-09-08

`experiments/logit_average_baseline.py --cache-dir data/cache/geom_probe --suffix _preproc`
-> `results/logit_average_baseline_preproc.csv`. Same experiment, same seed, same folds; the
only change is that the features come from letterboxed frames rather than raw ones. Both
outputs are kept side by side, and the reproduction check against the published H3 table is
skipped rather than failed, because on a different cache family those numbers *should* differ.

| model | raw recall | raw false-play | raw balanced | | preproc recall | preproc false-play | preproc balanced |
|---|---|---|---|---|---|---|---|
| convnextv2 | 0.9105 | 0.9918 | −0.081 | | **0.9841** | **0.0206** | **+0.9635** |
| dinov2 | 0.9297 | 0.3086 | +0.621 | | 0.9595 | 0.2305 | +0.7290 |
| vit | 0.8690 | 0.8354 | +0.034 | | 0.9118 | 0.9712 | −0.0594 |
| ens ConvNeXtV2+DINOv2 | 0.9473 | **1.0000** | −0.053 | | **1.0000** | **0.0288** | **+0.9712** |
| ens all three (log) | 0.9524 | **1.0000** | −0.048 | | 1.0000 | 0.1399 | +0.8601 |
| *oracle* | *0.9603* | – | – | | *1.0000* | – | – |

### Finding 2 is refuted, and it was an artefact of the input path

The claim was: *every ensemble scores 1.0000 false-play, worse than every backbone alone, so
blending is disqualified whatever the weights.* Under letterboxing the two-model ensemble
scores **0.0288** false-play at **1.0000** recall - the best balanced score of anything
measured on this dataset, better than any single backbone.

The 1.0000 came from ConvNeXtV2's 0.9918, and that came from the processor cropping a
1920x1080 frame down to roughly its middle half. Blending was never disqualified; blending
*a model that had been shown a central strip of the pitch* was.

### Finding 1 survives and gets stronger, which settles WP5-T2 anyway

A parameter-free average now captures **100%** of the oracle headroom, against 74% on the raw
caches. The oracle is 1.0000 and the ensemble reaches 1.0000, so **there is nothing left for a
learned gate to learn** - and this time the conclusion does not depend on the false-play axis
at all.

So WP5-T2's answer is unchanged and its reasoning is now the opposite. Not *"blending is
disqualified, so a gate must be a hard router"* but **"the naive ensemble is already at the
ceiling, so a gate has no room"**. That is a cleaner negative result and it needs no argument
about routing or confounds.

### The recall axis is saturated, and that limits what the headroom claim means

"Captures 100% of the headroom" sounds stronger than it is: the headroom was **+0.0159** to
begin with, because every fold already has some backbone at 1.0. Perfect recall on
100%-ACTIVE_PLAY folds means only "never said EMPTY on a play frame" - it is not a hard
target. The informative axis is false-play, where the ensemble (0.0288) is marginally *worse*
than ConvNeXtV2 alone (0.0206) and far better than DINOv2 (0.2305).

On balanced score the ensemble leads ConvNeXtV2 by **0.008**. On 243 empty frames amounting to
three to ten distinct scenes that is not a difference. **The honest reading: under
preprocessing, ConvNeXtV2 alone and the two-model ensemble are indistinguishable and both are
strong; DINOv2 is clearly behind; ViT is poor.** Adding ViT to the ensemble hurts it
(false-play 0.0288 -> 0.1399), consistent with ViT being the one backbone preprocessing harms.

### What this does and does not settle

**Settles:** the gate is not worth building, on the strongest available version of the
argument. Report WP5-T2 as a negative result against the naive ensemble.

**Does not settle:** which single model to deploy. That is RQ2, it now turns on the input-path
decision, and it needs the full protocol - CIs, paired tests, effective sample - not this
probe. The revision flags placed on WP5-T9 in `TODO.md` and `rq_matrix.md` are resolved by
this entry; the RQ2 flag stays.

- 2026-09-08 | WP5-T9 on preprocessed caches | `python experiments/logit_average_baseline.py --cache-dir data/cache/geom_probe --suffix _preproc` | `logit_average_baseline_preproc.csv` | finding 2 refuted (ensemble false-play 1.0000 -> 0.0288); finding 1 strengthened to 100% of headroom

---

## The `clahe='auto'` gate is vacuous, and what that reveals about the search's resolution — 2026-09-08

Checked because the geometry probe made preprocessing consequential: if a *searched* switch
fires selectively on some venues, a leave-one-venue-out protocol would be applying different
transforms to train and test. The audit that raised it suspected the gate was "close to a
venue indicator". **It is not** - and what is actually wrong is more useful.

### The gate fires on 99.81% of frames

`clahe='auto'` applies CLAHE when RMS contrast falls below `clahe_contrast_below = 40.0`.
Measured over all 1,578 development frames: **1,575 fire (0.9981)**.

| slice | fire rate |
|---|---|
| every venue except one | 1.0000 |
| `clipvenue_f_outdoor_bldg` | 0.7500 (n=12) |
| day / night | 0.9950 / 1.0000 |
| EMPTY / ACTIVE_PLAY / C3 | 1.0000 / 0.9972 / 1.0000 |

So it is *not* a venue or lighting discriminator - it fires nearly everywhere, near-uniformly.
The threshold is simply **far too high for this footage**: median RMS contrast is **20.46** on
the raw frame and 23.12 after letterboxing, against a cut-off of 40.0. (My first guess was
that the letterbox's grey padding was depressing the contrast; the data says otherwise - the
raw frames are already at 20, so the padding is not the cause. Recorded because the wrong
hypothesis is worth one line to save the next person testing it.)

**Consequence: `auto` and `on` are the same transform here**, differing on 3 frames of 1,578.
Two of the search's switch values are one switch, and every evaluation spent separating them
was spent on nothing.

> **RETRACTED 2026-09-08** - everything in this section that rests on the gate being
> vacuous is wrong. Contrast was measured on the letterboxed 224x224 *output*; the gate runs
> before the resize and tests the full-resolution frame. Counted rather than derived, it
> fires on **78.33%** of frames and `auto` differs from `on` on **342**. The resolution-floor
> conclusion survives on independent evidence - see the WP3-T8 entry below - but the
> "0.19% of the input moves the metric 0.02" argument does not.

### And the search still reports them 0.02 apart

Round 1 is the clean comparison - its base configuration is the default, exactly what was
measured above:

| model | `clahe=on` | `clahe=auto` | Δ |
|---|---|---|---|
| convnextv2 | 0.9348 | 0.9561 | **+0.0213** |
| dinov2 | 0.9752 | 0.9932 | **+0.0180** |

A **0.19% difference in the input** moves the searched metric by **0.02**. That is not noise in
the probe - the fits are seeded and deterministic - it is the fold structure amplifying three
frames:

| fold | n | one frame is worth, in the reported mean |
|---|---|---|
| `f_outdoor_bldg` | 12 | **0.0119** |
| four folds | 18 | 0.0079 |
| `g_netting` | 30 | 0.0048 |
| `a_blue_barrier` | 168 | 0.0009 |

The headline is an *unweighted* mean over seven folds, so a single frame in the smallest fold
is worth 1.2 points of it. Three frames in small folds reach **0.036**.

### What this means for reading the preprocessing search

**The search's resolution floor is about 0.02, and it has been adopting switches on margins of
that order.** Round 2 adopted `sharpen=0.6` for ConvNeXtV2 and round 3 `gamma=0.7`; the
non-additivity finding already recorded here turns on differences of similar size. None of
that is wrong, but none of it is separable from a three-frame perturbation either.

Two things follow, and neither is "re-run the search":

1. **Weight the fold mean by fold size, or report both.** An unweighted mean over folds of 12
   and 168 frames hands the 12-frame fold fourteen times the leverage per frame. H3 uses the
   same unweighted mean deliberately - it is bootstrapping over *venues*, where equal weight
   is the point - but the search is using it to rank *configurations*, which is a different
   question and does not want that property.
2. **Quote a resolution floor beside searched results.** A configuration that beats the
   baseline by less than ~0.02 has not been shown to beat it.

`clahe_contrast_below = 40.0` also joins the list of hand-picked constants that were never
calibrated against this footage, alongside the two confidence thresholds that default to 0.0.
Setting it from the measured distribution (median 20.5) would make `auto` mean something; the
right value is a decision, not a number to guess, and it belongs with the WP3 ablation.

- 2026-09-08 | clahe auto-gate diagnostic | `python -c` over the manifest | *(no CSV - diagnostic)* | **RETRACTED** - measured contrast on the resized output, not the frame the gate tests. Gate fires on 78.33%, not 99.81%; auto and on differ on 342 frames, not 3. See the WP3-T8 entry below

---

## 2026-09-08 — WP3-T3 option (b): the input-path finding under the full protocol

`experiments/input_path_protocol.py` → `results/input_path_protocol.csv`

The geometry probe left a note listing exactly what its own headline lacked: *"no CIs, one
seed, no paired test"*, on a false-play column of 243 frames amounting to three to ten
distinct scenes. This supplies all of it. Both arms already existed — `data/cache/` is the
published raw arm, `data/cache/geom_probe/` the letterboxed one — so nothing was re-embedded.

**Answer: the finding is observed, not established, and option (b) is not carried by it.**

### The camera swap breaks it

The published false-play control trains on venue_01 camera A and scores camera B's 243 empty
frames — **one measurement, one direction**. Swapping the cameras gives a second measurement
on 251 different frames from a different training set. It is the only replication this corpus
can offer, because every EMPTY frame in it is venue_01.

| backbone | train A → test B | train B → test A | |
|---|---|---|---|
| convnextv2 | 0.9918 → 0.0206 (**−0.9712**) | 0.0279 → 0.0159 (−0.0120) | replicates |
| dinov2 | 0.3086 → 0.2305 (−0.0782) | 0.0000 → 0.9761 (**+0.9761**) | **REVERSES** |
| vit | 0.8354 → 0.9712 (+0.1358) | 0.0000 → 0.0000 (0.0000) | no effect one side |

Two things fall out of the second column, and both matter more than the first.

**ConvNeXtV2's catastrophic raw false-play is a property of one training camera, not of the
input path.** Trained on camera B it is 0.0279 raw — there is no 0.99 to rescue. The
letterbox cannot be credited with fixing a defect that the other direction does not have.

**DINOv2 reverses completely.** Raw is *perfect* (0.0000) trained on camera B, and the
letterbox destroys it (0.9761). So "preprocessing helps ConvNeXtV2 hugely, DINOv2 modestly,
hurts ViT" — day two's summary — does not survive the swap either: DINOv2's modest help
becomes near-total harm in the other direction.

The swap is a replication, not a mirror, and the asymmetry is recorded rather than smoothed
over: all six C3 frames sit on camera A, so training on camera B is a two-class fit on 521
frames against camera A's three-class fit on 775.

### The recall axis could not have reached significance

Seven venue folds, paired, with an exact sign-flip test:

| backbone | raw | preproc | Δ | 95% CI over venues | p | floor |
|---|---|---|---|---|---|---|
| convnextv2 | 0.9105 | 0.9841 | +0.0736 | [−0.0176, +0.1735] | 0.3125 | 0.0625 |
| dinov2 | 0.9297 | 0.9595 | +0.0298 | [−0.0789, +0.1514] | 0.6875 | 0.0625 |
| vit | 0.8690 | 0.9118 | +0.0428 | [−0.1389, +0.2412] | 0.7500 | 0.0625 |

Every interval covers zero. But the number worth keeping is the **floor**: two folds tie for
every backbone, leaving five informative pairs, so the smallest two-sided p this design can
return is 2/2⁵ = **0.0625**. *No result of any size could have been significant here.*
Reporting "p = 0.31, not significant" without that would describe the sample and not the
effect — the same error as H4's "a null result confirms it", one level down. Six venues
pointing the same way is the minimum that can clear 0.05.

`sign_flip_test` was added to `evaluation/stats.py` for this, and reports its own
`min_achievable_p` beside every p-value so the two readings cannot be confused again.

### And the frame-level significance dissolves when the frames are counted

The 243 (and 251) empty frames come from **two (camera × slot) cells** — the coarsest count,
needing no hash threshold. Recomputed on one frame per distinct scene:

| direction | scenes @2 bits | surviving after Holm | scenes @6 bits | surviving |
|---|---|---|---|---|
| train A | 10 | convnextv2 only, p = 0.047 | 3 | none |
| train B | 4 | none | 2 | too few to test |

At the project's default threshold nothing survives in either direction. The one survivor at
the strict threshold is the ConvNeXtV2 comparison the swap has already shown to be
direction-specific.

### The seed axis was a phantom

The probe's note listed "one seed" among the gaps. `LinearProbe` takes a `seed` and passes it
to `LogisticRegression`, which solves with lbfgs — **deterministic**. Verified: across five
seeds not one prediction changes. Running more seeds would have produced identical numbers
and presented a fixed quantity as a robustness check. The parameter is harmless, but it is
another case of something that reads like a knob and turns nothing.

### What this does not say

It does not say the letterbox is bad, or that the middle-half crop is fine. The mechanism
found by the probe is real and visible: handing a 1920×1080 frame to a processor that resizes
shortest-edge to 256 and crops 224 keeps roughly the central half of the pitch. What it says
is that **this dataset cannot tell you what that costs**, because the only footage of an empty
pitch is one venue, two cameras, two slots. Option (b) — putting `preprocess.py` into
`build_cache` and regenerating everything — would change the input to every published number
on evidence that reverses when the two available cameras are swapped.

The blocker is data, not method, and it is the same blocker as C3, RQ6's calibration and the
confidence thresholds: **empty-pitch footage from more than one venue**. That is a line in
`thesis/data_requests.md`, not an experiment.

- 2026-09-08 | WP3-T3(b) input path under protocol | `python experiments/input_path_protocol.py` | `input_path_protocol.csv` | paired over 7 venues + both camera directions, with the effective sample

- 2026-09-08 | H4 model equivalence | `python experiments/h4_model_equivalence.py` | `h4_model_equivalence.csv` | ConvNeXtV2 vs ViT: equivalent at margin 0.02

- 2026-09-08 | label efficiency | `python experiments/label_efficiency.py` | `label_efficiency.csv` | 7 sizes x 5 seeds

- 2026-09-08 | RQ6 | `python experiments/rq6_calibration_riskcoverage.py` | seed 42 | `rq6_calibration.csv`, `rq6_risk_coverage.csv`

---

## 2026-09-08 — a seeded split that was not reproducible, and what it was hiding

Found while refactoring `evaluable_subset` out of `h1_h2_baseline_floor.py` into the library.
The refactor was checked the way this project checks refactors — regenerate the CSV and
require it byte-identical — and it was not. Point estimates matched exactly; **the confidence
intervals beside them did not.**

Then the control run: the *unmodified* code did not reproduce its own committed CSV either,
and two consecutive runs of it disagreed with each other. So the refactor was innocent and
something older was wrong.

### `grouped_split` returned the same frames in a different order every process

```python
test_names: set[str] = set()          # membership
...
test = [r for g in test_names for r in groups[g]]   # ...used as an ordering
```

`PYTHONHASHSEED` is randomised per process, so iterating a set of strings gives a different
order each run. The *membership* was identical, which is exactly why it survived:

- every accuracy, macro-F1 and balanced accuracy reproduced exactly — they do not depend on
  row order;
- every **bootstrap interval** did not, because `bootstrap_metric_ci` resamples *positions*,
  and a reordered vector is a different resample.

`test_grouped_split_is_deterministic` had existed all along and passed throughout, because it
called the function twice **in one process**, where set order is fixed for the process's
lifetime. The replacement spawns interpreters under three `PYTHONHASHSEED` values and
compares hashes of the emitted sequence; verified both ways — it fails on the old line and
passes on the new one. A static sweep of the package for sets iterated into ordered
structures found this to be the only site.

**What moved on regeneration** (point estimates: nothing, anywhere):

| file | column | largest change |
|---|---|---|
| `h1_h2_baseline_floor.csv` | `macro_f1_lo` / `macro_f1_hi` | 0.0073 |
| `effective_sample_audit.csv` | `n_distinct` (grouped) | 94 → 95 |
| `rq6_risk_coverage.csv` | `accuracy` | **0.125**, on 57 of 180 rows |

The first two are what a reordering defect is expected to cost. The third was not, and it
turned out to be a second defect underneath.

### The risk–coverage curve was ordering frames the model had not ordered

A risk–coverage curve answers "how accurate is the most confident *k*?" — which is only a
question when the *k*-th and (*k*+1)-th confidences differ. Measured:

| model | fitted T | distinct calibrated confidences | largest exact tie |
|---|---|---|---|
| convnextv2 | 0.95 | 907 / 907 | 1 |
| **dinov2** | **0.05** | **18 / 907** | **890** |
| vit | 1.30 | 907 / 907 | 1 |

DINOv2's temperature hits the search grid's lower boundary — the code **already warns** about
this at runtime — and sharpens the probabilities until 890 of 907 calibrated confidences are
exactly 1.0. Its whole curve was a single arbitrary ordering of one tied block, which is why
all 57 moved rows were DINOv2's and why fixing the split order changed them so much.

Fixing the ordering makes that curve *reproducible*. It does not make it *identified*, and
publishing a reproducible arbitrary number would have been the worse outcome of the two.

So `risk_coverage_band` replaces it: at each coverage, the best and worst accuracy the
confidences permit — order the correct members of a straddled tie group first, or last.
Where confidences are distinct the two bounds coincide and it is the old curve exactly.
`coverage_for_target_accuracy` now reads the **lower** bound, since an operating point that
holds only for a favourable ordering of indistinguishable frames is not one the facility can
be held to.

The band is wide where it matters:

| DINOv2, coverage | accuracy |
|---|---|
| 0.20 | 0.957 – 1.000 |
| 0.41 | 0.978 – 1.000 |
| 0.81 | 0.989 – 1.000 |
| 1.00 | 0.9857 (identified) |

The reported operating points are unchanged — 99% precision at 97.9% coverage — because the
tie block is large enough that the worst case lands on the same coverage. They are now
*identified* rather than coincidental, and the tie counts ship in both CSVs so the next
reader can see the curve is a band before quoting a point on it.

### What this says about the rest

RQ6 was already "blocked by data", and this sharpens why: with a 99% single-class test set
the probe saturates, temperature scaling runs to its boundary trying to correct it, and
selective prediction has nothing left to select on. The band is the honest picture of a
measurement this dataset cannot support — not a new problem, a visible one.

- 2026-09-08 | reproducibility | set-order leak in `grouped_split` | `h1_h2_baseline_floor.csv`, `effective_sample_audit.csv`, `rq6_risk_coverage.csv` | seeded split was not reproducible across processes; every bootstrap CI on a grouped split was a different draw
- 2026-09-08 | RQ6 | risk-coverage as a band | `rq6_risk_coverage.csv` | 890/907 of DINOv2's calibrated confidences are exactly tied; the curve is a band and the operating point is its lower bound

- 2026-09-08 | H1/H2 | `python experiments/h1_h2_baseline_floor.py` | seed 42 | `h1_h2_baseline_floor.csv` | 14 rows over 2 splits

- 2026-09-08 | H3 | `python experiments/h3_cross_venue_recall.py` | seed 42 | `h3_cross_venue_recall.csv` | 7 folds x 4 models

- 2026-09-08 | H3 sensitivity (cg+ch merged) | `python experiments/h3_sensitivity_merged_venues.py` | seed 42 | `h3_sensitivity_merged_venues.csv` | 6 folds x 4 models

- 2026-09-08 | end-to-end slots | `python experiments/end_to_end_slots.py` | `end_to_end_slots.csv` | 2 real slots, ground-truth labels

- 2026-09-08 | efficiency | `python experiments/efficiency_latency.py` | `efficiency_latency.csv` | dev laptop, 4 threads

- 2026-09-08 | WP2-T10 camera fingerprinting | `python experiments/camera_fingerprint_audit.py` | `camera_fingerprint.csv` | 70 views, 10 venues

- 2026-09-08 | WP5-T9 logit-average baseline | `python experiments/logit_average_baseline.py` | `logit_average_baseline.csv` | best single dinov2 0.9297, oracle 0.9603

- 2026-09-08 | WP4-T1 benchmark v2 | `python experiments/benchmark_v2.py` | `benchmark_v2.csv` | 8 models x 4 protocols; every protocol degenerate, each differently

---

## 2026-09-08 — WP4-T1: four split protocols, and a constant predictor that wins one

`experiments/benchmark_v2.py` → `results/benchmark_v2.csv`, `benchmark_v2_protocols.json`

Eight predictors under four protocols. The plan asks for the random-vs-grouped delta as
"the thesis's first key figure"; running all four side by side gives that and two things it
could not give alone.

### Macro-F1 by protocol (mean ± sd over replicates)

| model | random (leaky) | grouped_slot | lo_venue_out | temporal |
|---|---|---|---|---|
| majority | 0.4059 ±0.004 | 0.0166 ±0.015 | **1.0000 ±0.000** | 0.0111 |
| clock_rule | 0.9147 ±0.011 | 0.4005 ±0.200 | 0.2619 ±0.383 | 0.0111 |
| cheap_intensity | 0.3487 ±0.048 | 0.1558 ±0.084 | 0.0000 ±0.000 | 0.3645 |
| cheap_histogram | 0.9695 ±0.008 | 0.1269 ±0.092 | 0.8535 ±0.262 | 0.0000 |
| convnextv2 | 0.9849 ±0.004 | 0.4064 ±0.203 | 0.9493 ±0.071 | 0.4965 |
| dinov2 | 0.9873 ±0.002 | 0.4715 ±0.240 | 0.9595 ±0.074 | 0.5463 |
| vit | 0.9926 ±0.004 | 0.4569 ±0.090 | 0.9148 ±0.147 | 0.4936 |
| clip_zeroshot *(control)* | 0.7612 ±0.026 | 0.5778 ±0.108 | 0.9907 ±0.016 | 0.6332 |

### 1. A model that always answers "playing" scores a perfect cross-venue macro-F1

`majority` predicts the most frequent training class and reads no pixels. Every held-out
clip venue is **100% ACTIVE_PLAY**, so it is right on every frame, and the macro average is
over the one class present: **1.0000**, ahead of DINOv2's 0.9595.

H3 and the false-play control had already shown cross-venue *recall* is free on single-class
folds. This is the same defect reaching the headline metric: on this protocol macro-F1 does
not distinguish a backbone from a constant, and the ordering it produces is not about
occupancy. `clip_zeroshot` scoring 0.9907 — above every trained backbone — is the same
artefact from the other side.

The floor is quantified per protocol now rather than asserted:

| protocol | best trivial | best backbone | gap |
|---|---|---|---|
| random | cheap_histogram 0.9695 | vit 0.9926 | +0.0231 |
| grouped_slot | clock_rule 0.4005 | dinov2 0.4715 | +0.0710 |
| **lo_venue_out** | **majority 1.0000** | dinov2 0.9595 | **−0.0405** |
| temporal | cheap_intensity 0.3645 | dinov2 0.5463 | +0.1818 |

### 2. A third of H1's drop is test-set composition, not leakage

The random→grouped drop is 0.52–0.58 for the backbones. But **a model that never trains
cannot leak**, and OpenCLIP scored zero-shot on the identical test sets drops **0.1834** on
the same change of protocol. That part of the fall is what any model would show from the
change in what is being tested.

| model | raw drop | composition (control) | leakage-attributable |
|---|---|---|---|
| convnextv2 | 0.5784 | 0.1834 | **0.3950** |
| vit | 0.5357 | 0.1834 | **0.3522** |
| dinov2 | 0.5158 | 0.1834 | **0.3323** |
| cheap_intensity | 0.1929 | 0.1834 | 0.0094 |
| cheap_histogram | 0.8426 | 0.1834 | 0.6592 |

H1's effect survives and is still large — a third of the score, not the two-thirds the raw
delta suggests. `cheap_intensity`'s attributable drop of 0.0094 is a sanity check pointing
the right way: a model with almost nothing to memorise loses almost nothing to leakage.

**This is a control, not a proof.** It assumes the composition effect is additive and
similar across models. CLIP's errors are not a probe's, and the subtraction is stated as an
assumption in the script and here. Its prompt set was fixed in advance — the first
descriptor of each class, all five templates — and is deliberately *not* the prompt search's
winner, which was selected on the folds it reports.

### 3. One grouped split is a sample of size one

Across five seeds the grouped protocol's spread is **±0.240** for DINOv2 and ±0.203 for
ConvNeXtV2 — on a mean of 0.47. Every published grouped-split number in this project comes
from `seed=42` alone. They reproduce exactly (the guard checks 14 of them before anything
else runs), but the seed-to-seed variation is of the same order as the differences between
models, so **no ranking on the grouped split is supported by one split**. The plan's ★ item
asked for ≥5 splits; this is why.

### 4. The temporal protocol does not measure drift

venue_01's day one is 97.6% EMPTY and entirely daylight; day two is 98.9% ACTIVE_PLAY and
entirely floodlit. Training before the cut and testing after it is a class-and-lighting
flip, and every model lands between 0.49 and 0.55 — including a `cheap_histogram` at exactly
**0.0000**, which predicts C3 for everything after six upweighted maintenance frames in
training. Report it as a demonstration that the corpus cannot measure drift, not as drift.

Setting it up found that `temporal_split` placed undated frames in *train* by string
comparison (`"" < "2026-07-12"`), and on this corpus all 282 undated frames are the seven
clip venues — none of which appear on the test side. The split called "temporal" was a
venue-and-time split and nothing in its output said so. It now takes an explicit `undated`
policy, defaulting to excluding them.

### What this protocol table is for

Not to pick a winner. Read together the four columns say the same thing from four sides:
**this corpus can measure whether a pitch has people on it only inside one venue, and can
measure generalisation only in the one direction where the answer is always yes.** Every
number above is quotable only with its diagnostics — test size, class count, majority
share — which is why they ship in the same CSV rows.

- 2026-09-08 | WP4-T1 benchmark v2 | `python -m experiments.benchmark_v2` | `benchmark_v2.csv` | 8 models x 4 protocols; a constant predictor scores macro-F1 1.000 cross-venue; a third of H1's drop is composition

- 2026-09-08 | H6 zero-shot vs trained probes | `python -m experiments.h6_zero_shot_gap` | `h6_zero_shot_gap.csv` | declared prompt set; realised family 3; H6 refuted

- 2026-09-08 | H6 zero-shot vs trained probes | `python -m experiments.h6_zero_shot_gap` | `h6_zero_shot_gap.csv` | declared prompt set; realised family 3; H6 inconclusive; prompt choice spans 0.021-0.747 macro-F1

---

## 2026-09-08 — H6 reported: the prompt matters more than the model

`experiments/h6_zero_shot_gap.py` → `results/h6_zero_shot_gap.csv`

The last of the six pre-registered hypotheses never to have been reported. It was blocked on
two things and both had just been removed for other reasons: OpenCLIP's image cache now
covers all 1,578 development frames (it held 600), and `benchmark_v2.py` needed a prompt set
declared in advance, which is now `vision.zeroshot.DECLARED_PROMPT_SET` — first descriptor
per class, all five templates, fixed by position and deliberately not the search's winner.

> **H6.** OpenCLIP zero-shot is significantly worse than every trained probe on macro-F1.
> **Family:** zero-shot vs trained. **Realised family: 3**, not the 4 anticipated.

### The pre-registered comparison, grouped split, seed 42

| comparison | zero-shot | trained | Δ | 95% CI | p (Holm) | g |
|---|---|---|---|---|---|---|
| vs ConvNeXtV2 | 0.6291 | 0.4975 | **+0.1316** | [+0.0538, +0.2059] | 1.4e−06 | 0.342 |
| vs DINOv2 | 0.6291 | 0.5794 | **+0.0497** | [−0.1158, +0.1767] | 1.5e−05 | 0.293 |
| vs ViT | 0.6291 | 0.4975 | **+0.1316** | [+0.0538, +0.2059] | 1.4e−06 | 0.342 |

Zero-shot is significantly worse than **0 of 3** and significantly **better than 3**. Across
five grouped splits it wins **5 of 5** against each probe. Read on the declared set alone the
verdict is "refuted", and stopping there would have been the confident answer.

### Two checks say stop there is wrong

**The declared set is a lucky one.** Scoring *every* prompt set in the declared space — 375 of
them, nothing selected, the distribution is the point:

| | macro-F1 |
|---|---|
| best trained probe | 0.5794 |
| **declared prompt set** | **0.6291** (above 85% of the space) |
| prompt space, worst / median / best | 0.0207 / **0.4967** / 0.7472 |
| 10th–90th percentile | 0.3524 – 0.6403 |

**Only 22.9% of prompt sets beat the best trained probe, and the median one loses to it.**
So the refutation depends on which prompt was declared, which means it is not a refutation.
Declaring in advance was still the right procedure — it buys an unbiased point estimate — but
an unbiased estimate of a quantity this variable does not settle a comparison.

**And the significance is frame-level.** The 907-frame grouped test set is **95 distinct
scenes**; recounted on one frame per scene, no comparison is significant (p = 0.69, 1.00,
0.69). Same story as the false-play family.

**Verdict: H6 is inconclusive.** Not confirmed — the direction is against it on every split
and every seed. Not refuted — the margin belongs to a prompt at the 85th percentile.

### The finding worth keeping is the spread, not the ranking

Prompt choice moves macro-F1 by **0.726** (0.021 → 0.747). The three trained backbones span
**0.082**. On this benchmark **the prompt matters roughly nine times more than the model** —
which reframes what "the cost of a no-label deployment" means: it is not a fixed penalty for
skipping labels, it is a wide distribution whose position nobody can know at a new site
without labels to check against. That is a sharper answer to RQ1's onboarding question than
H6 asked for, and a worse one operationally.

### A6, quantified

The search's winner and the declared set, scored on the winner's own protocol — the folds it
was selected from:

| prompt set | play recall | false play | balanced |
|---|---|---|---|
| declared | 0.9820 | 0.4595 | 0.5225 |
| selected | 0.9739 | 0.0040 | **0.9699** |

**+0.4474.** A6 said a searched configuration is "selected, not tested" and its margin
"optimistically biased"; that is the number. It is not an unbiased estimate of the bias — no
held-out prompt data exists — but it bounds what the search's headline can be worth as a
model claim, and it is why H6 does not use it.

---

## 2026-09-08 — the standalone site had stopped covering the thesis, silently

`experiments/make_site.py` → `results/project_site.html`

Noticed while committing H6: the log grew by 79 lines, the site regenerated, and the output
was **byte-identical**. `make_site.py` reads a fixed list of eight result files, so the
eleven experiments added since it was written — the false-play control, H4, the geometry
probe, the input-path protocol, WP4-T1, H6 and the rest — were absent from a page whose
docstring said it covered *"the whole thesis"*. Regenerating did nothing and reported
nothing. Same shape as the branch reference that named a stale tip and the search viewer that
served a withdrawn run: a generated artefact whose claim outran what it did.

The served front end was never affected — `api/thesis_site.py` renders `EXPERIMENT_LOG.md`
directly, so a finding appears there as soon as it is logged. It is the standalone export
that had fallen behind.

**Two changes, and deliberately not "render everything".**

The page's own headline was *"the protocol reverses the ranking"*, and WP4-T1 had just
overtaken it, so that result goes on: the four-protocol floor table showing **a constant
predictor winning leave-one-venue-out**, and the zero-shot subtraction separating leakage
from composition. Both read from `benchmark_v2.csv`; the composition figure quoted in the
prose is pulled from the same rows rather than typed, since a hardcoded number in a
generated page is the defect this project has already fixed twice.

And `tests/test_site_coverage.py` makes omission a decision instead of a default: every
committed result file must be either read by the exporter or named in an `OUT_OF_SCOPE` list
with the reason. Twenty-one are out of scope and say why — most because their value is the
argument rather than the row, and the served site carries the argument. Adding an experiment
now fails the test until someone chooses. A companion test asserts the new sections are in
the **generated HTML**, not merely producible by the builder, which is the distinction the
search-viewer safeguard failed on.

The docstring is corrected too. It now says what the page is — a curated export — and the
summary line is what the test checks, because the body quotes the old wording to record why
it changed.

- 2026-09-08 | site coverage | `python experiments/make_site.py` | `project_site.html` | export had silently omitted 11 experiments; WP4-T1 added, coverage inventory now tested

- 2026-09-08 | WP3-T8 search resolution and ranking | `python -m experiments.search_resolution` | `search_resolution.csv` | floor 0.0119 per frame; weighting changes 3 adopted configuration(s); clahe gate fires on 78.33% of frames

---

## 2026-09-08 — WP3-T8: the search re-ranked, and a retraction

`experiments/search_resolution.py` → `results/search_resolution.csv`,
`search_resolution_gate.json`

Two fixes were named for the preprocessing search, and the note attached said the important
thing: **neither is "re-run it"**. All 88 evaluations still have their feature caches, so
this is re-scoring — minutes against the twelve hours the search cost. Both published tables
are reproduced first: all 88 unweighted recalls, and all 52 repaired false-play rates.

### Retraction: the CLAHE gate is weak, not vacuous

The 2026-09-08 diagnostic recorded above reported that `clahe='auto'` fires on **99.81%** of
frames, so `auto` and `on` were "the same transform, differing on 3 frames of 1,578". **That
is wrong.** It measured RMS contrast on the letterboxed 224×224 *output*, but the gate runs
in the photometric stage — **before** the resize — so it tests the full-resolution frame,
whose contrast distribution is a different one.

Settled by counting rather than deriving: run `preprocess` with `clahe='auto'` and with
`clahe='on'` and compare the outputs. Wherever they differ, the gate did not fire. No
assumption about which image is measured survives that.

| | claimed | measured |
|---|---|---|
| gate fires on | 99.81% | **78.33%** |
| `auto` vs `on` differ on | 3 frames | **342 frames** |
| median RMS contrast of the gate's input | 20.46 | **30.9** (range 18.9–56.8) |

So `auto` is a **weak** switch, not a vacuous one, and the sub-claim it supported — that a
0.19% difference in input moved the searched metric by 0.02 — collapses with it. A 21.67%
difference in input moving the metric by 0.02 is unremarkable.

`preprocess.py`'s docstring carried the wrong figures too and now carries the corrected
ones plus the retraction. The calibration is counted in code from now on, and the
experiment prints the derived rate beside the counted one so a future disagreement is
visible instead of silent.

### The resolution floor survives, on better evidence

The floor was the point of the diagnostic, and it does not depend on the CLAHE argument. It
is derivable directly from the fold structure:

| fold | play frames | one frame moves the unweighted mean by |
|---|---|---|
| `f_outdoor_bldg` | 12 | **0.0119** |
| four folds | 18 | 0.0079 |
| `g_netting` | 30 | 0.0048 |
| `a_blue_barrier` | 168 | 0.0009 |

**Floor: 0.0119** — a margin below that is one frame in the smallest fold. And the bootstrap
interval over the seven folds is **0.092** wide at the median configuration, which is the
more honest floor: most of what separates these configurations is inside it.

### Frame-weighting changes what the search adopted

The unweighted mean is right for H3, which bootstraps over *venues*. Ranking configurations
is a different question and does not want a 12-frame fold carrying fourteen times the
per-frame leverage of a 168-frame one. Re-ranked on pooled recall over held-out play frames:

| model | round | winner changes? | winning margin |
|---|---|---|---|
| convnextv2 | 1 | same | 0.0145 |
| convnextv2 | 2 | **different** | 0.0231 |
| convnextv2 | 3 | **different** | 0.0094 *(below the floor)* |
| dinov2 | 1 | same | 0.0110 *(below the floor)* |
| dinov2 | 2 | same | 0.0371 |
| dinov2 | 3 | **different** | 0.0141 |

**Three of six rounds adopt a different configuration**, and two rounds were decided on
margins smaller than one frame in the smallest fold. The greedy search compounds this: a
round-2 choice conditions everything after it.

Nothing here says the search was run wrongly. It says its output is a set of candidates, not
a ranking — and that a configuration is adopted on evidence only when its margin clears the
floor and its false-play control agrees.

### The gate threshold, for the ablation

`clahe_contrast_below = 40.0` sits above the 90th percentile (43.2) of the distribution it
tests, so it applies CLAHE almost everywhere while claiming to be selective. Calibrated
candidates from the footage: the median **30.9** makes `auto` mean "the darker half", the
10th percentile **20.4** makes it "the worst tenth". Which one is a decision for the
ablation; 40.0 is not among them.

- 2026-09-08 | WP3-T8 search resolution and ranking | `python -m experiments.search_resolution` | `search_resolution.csv` | floor 0.0119 per frame; weighting changes 3 of 6 adopted configurations; CLAHE gate fires on 78.33% (retracting 99.81%)

---

## 2026-09-08 — WP4-T9: the risk–coverage figure, drawn as a band

`experiments/make_figures.py` → `results/figs/risk_coverage_band.{png,pdf}`

The machinery existed; the figure did not, and it is the one plot in this project where the
conventional form would actively mislead. Two things had to survive it.

**DINOv2's calibrated confidences are 890 of 907 identical**, so "the most confident *k*" is
undefined across most of the range and its accuracy there is an interval. A line through the
middle of 0.981–1.000 would be a claim the data cannot make. The upper panel shades every
accuracy the confidences permit and draws the **worst case** as the solid edge — the same
bound `coverage_for_target_accuracy` reads, so what a reader traces is what an operator can
be held to.

**ConvNeXtV2 and ViT trace a near-perfect curve while never predicting EMPTY at all.** They
are right on all 898 ACTIVE_PLAY frames and wrong on all 9 empty ones, so confidence ranks
the nine last and the curve sits at 1.000 until 88% coverage. *The two best-looking lines
belong to the two models that cannot do the job* — and a figure that showed only the curves
would recommend them.

A second panel carries **the width the confidences leave**, because one axis cannot hold both
"what is the accuracy" and "how determined is that answer". ConvNeXtV2 and ViT sit flat at
zero there; DINOv2 starts at 1.00 and never reaches zero. That panel is the reason the upper
one is shaded rather than drawn.

Design notes worth keeping: the y-axis is clipped to 0.95 so the operational region is
legible, with the excursion below it labelled rather than hidden; names sit *on* their own
lines rather than a fixed distance below, because below DINOv2's line is the band and a name
floating inside it reads as a curve that is not there; and the two identified curves are
noted as coinciding until 88% coverage, since otherwise one of them looks missing.

`tests/test_figures.py` checks the property rather than the source text: it renders the
figure, finds the heavy line, and asserts its y-values are the worst-case column.
Verified both ways — swap the heavy line to the best-case edge and the test fails.

It is on the standalone site too, in the Findings view, and a test asserts it reached the
page. `rq6_risk_coverage.csv` is listed there as "shown as a figure rather than a table",
and a reason like that is a fiction unless the figure is actually present.

- 2026-09-08 | WP4-T9 risk-coverage figure | `python -m experiments.make_figures` | `figs/risk_coverage_band.png` | drawn as a band; the two best-looking curves belong to the models that never predict EMPTY

---

## 2026-09-08 — the README said the implementation had not started

`README.md`, `scripts/branch_report.py`

The first thing a supervisor or an examiner reads said **"Planning complete. Implementation
starting."** It said that throughout the period in which the implementation was written, all
six pre-registered hypotheses were reported, and four defects were found in the evaluation
itself. Nobody edits a status line while doing the work it describes.

It is the same failure as the branch reference that named a stale tip and the export that
stopped covering the thesis — a claim with nothing checking it — and it had the widest
audience of the three.

**Fixed in two halves, because a status has two kinds of content.** The countable part is now
generated: `scripts/branch_report.py` maintains a `<!-- status:start -->` block in `README.md`
alongside the one it already maintained in `docs/CODEBASE.md`, so module, experiment, test and
result counts come from git. `--check` now reports either file stale, and `--write` rewrites
both.

The judgement — *what state is this project in* — cannot be generated, so it is written by
hand and guarded instead. Three tests: the block is current, the counts in it match git, and
the README does not claim the work has not started while results exist. Verified by breaking
each: reinstate the old phrase and the guard fires; change a count and it fires.

**And the README had no results section at all.** A reader got the pilot's superseded
accuracy table and nothing about what the project has actually found. It now leads with the
findings — the leakage decomposition, the constant predictor that wins the cross-venue
protocol, the prompt variance, the risk–coverage band — each linked to its artefact, with the
pilot table demoted to history. The section naming the defects is deliberately included: they
are a result of this project, not an embarrassment to be kept out of the front page.

- 2026-09-08 | README status | `python scripts/branch_report.py --write` | `README.md` | status block generated from git; findings section added; three staleness guards
