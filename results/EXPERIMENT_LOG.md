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
