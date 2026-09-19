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

> **Superseded by A25 (2026-09-17).** The clock rule's numbers here were computed with the
> `lighting` column that A25 found wrong for 216 of 396 clip frames. On the corrected labels it
> equals ConvNeXtV2 and ViT exactly, and DINOv2's lead over it is no longer significant. See
> *"the lighting labels were wrong"* at the end of this file.

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

> **CORRECTION 2026-09-09.** The measurement above stands; the clause after the dash does not.
> On the 243 held-out empty frames that produce the 0.309, DINOv2 predicts EMPTY **zero**
> times - it answers ACTIVE_PLAY 75 times and MAINTENANCE 168. A low false-play rate is
> therefore not evidence that a model predicts EMPTY, and the two facts are not "consistent"
> in the way this sentence claims; they are unrelated. See the WP5-T2 entry below.

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

- 2026-09-08 | H5 preprocessing switches | `python -m experiments.h5_preprocessing_switches` | `h5_preprocessing_switches.csv` | ROI clause unrunnable (no polygon); CLAHE clause 4/4 significant

---

## 2026-09-08 — H5 reported: one clause unrunnable, one refuted

`experiments/h5_preprocessing_switches.py` → `results/h5_preprocessing_switches.csv`

The sixth and last pre-registered hypothesis. Its two clauses fail in different ways, and
keeping them apart is the whole point: collapsing them into one verdict would either invent a
null result or bury a real one. Every arm's published cross-venue recall reproduces before
anything new is computed.

### Clause 1 — ROI masking: **unrunnable**, not refuted

ROI masking has never been evaluated and cannot be. `configs/cameras.json` does not exist,
WP3-T1 needs a human to draw one polygon per camera, and `roi_mask` returns the frame
untouched when the polygon is absent. All 88 search evaluations carry `roi: False`, and
`SWITCHES` **deliberately excludes** `roi` for exactly this reason — a switch that cannot
change the image must never enter a search, or the search records "ROI masking does not help"
from a transform that never ran.

Reported unrunnable. A null result here would be manufactured, and the project already has a
test (`test_preprocess_switches.py`) whose stated purpose is preventing that.

### Clause 2 — CLAHE: **refuted**, and in the opposite direction

`clahe='on'` and `clahe='auto'` differ from the baseline in that switch alone and their caches
survive, so this is re-scoring. Grouped split, seed 42, macro-F1, paired throughout:

| model | switch | n | baseline | arm | Δ | 95% CI | p (Holm) | g |
|---|---|---|---|---|---|---|---|---|
| convnextv2 | auto | 799 | 0.4972 | 0.4888 | **−0.0084** | [−0.0117, −0.0054] | 9.4e−07 | 0.500 |
| convnextv2 | on | 799 | 0.4972 | 0.4878 | **−0.0093** | [−0.0129, −0.0061] | 4.0e−07 | 0.500 |
| dinov2 | auto | 799 | 0.7617 | 0.4946 | **−0.2671** | [−0.4012, −0.0500] | 1.3e−24 | 0.468 |
| dinov2 | on | 799 | 0.7617 | 0.4904 | **−0.2712** | [−0.4054, −0.0532] | 1.5e−21 | 0.456 |

**4 of 4 significant after Holm, and CLAHE is worse in all four.** The clause predicted an
improvement; every interval lies entirely below zero and Cohen's g of ~0.5 means the
disagreements are wholly one-sided. On DINOv2 it costs 0.27 macro-F1 — it collapses the model
onto roughly the score of one that never recognises an empty pitch. Across five splits it
improves in **0 of 4** night test sets for either model.

Realised family **4**, not the 5 switches the hypothesis anticipated: only CLAHE has arms.

### And the clause's "specifically" cannot be tested at all

The clause says CLAHE helps *on the night subset specifically*, which needs a day column to
compare against. This corpus cannot supply one. venue_01 has exactly two recording days — a
daylight morning and a floodlit night — and holding whole slots out fills the test side from
one of them:

| seed | 42 | 43 | 44 | 45 | 46 |
|---|---|---|---|---|---|
| test set | night 799 | night 799 | night 799 | night 799 | **day 497** |

**No grouped split has both.** So the day figures come from a different partition and are
reported as a second observation, never as a contrast. (Read that way: on the one day split,
CLAHE *helps* ConvNeXtV2 by ~+0.10 and hurts DINOv2 by ~−0.15 — which is a reason to want the
contrast, not a substitute for it.)

The lighting stratification is restricted to venue_01 deliberately: `lighting` elsewhere is a
brightness proxy WP3-T5 found wrong for at least three clip venues, while within venue_01 it
is the recording day. The grouped split's test set is venue_01 anyway, so the restriction
costs nothing and removes the label error.

### The effective sample, as always

799 night frames are **71 distinct scenes**. The intervals above are frame-level and narrower
than the truth.

- 2026-09-08 | H5 preprocessing switches | `python -m experiments.h5_preprocessing_switches` | `h5_preprocessing_switches.csv` | ROI clause unrunnable (no polygon); CLAHE clause refuted - 4/4 significant and all negative

- 2026-09-08 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 11 claims verified against their artefacts, 2 recorded as unsupported

- 2026-09-08 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 15 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-08 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 18 claims verified against their artefacts, 0 recorded as unsupported

---

## 2026-09-08 — WP8-T5: the claims ledger, and it earned its keep before it finished

`thesis/claims.toml` (the ledger) · `experiments/verify_claims.py` (the verifier) ·
`thesis/claims.md` (generated)

Every quantitative claim the write-up makes, with the artefact that produced it and enough
of a locator to re-derive it. The verifier recomputes each value from its source and checks
that the same number appears in every live document the claim says states it.

**A ledger has to be executable.** A hand-maintained one is a document like any other, and
every hand-maintained claim in this project has drifted at least once — a figure with its
ranks hardcoded contradicting its own CSV, a README describing a project that had not
started, an export that stopped covering the thesis, a diagnostic quoted for a day before a
second measurement retracted it. None was caught by a test. This is the generalisation of
the guards added one at a time for each.

A claim fails in three distinguishable ways, and the output separates them because they call
for different work: **stale result** (the source moved, the prose did not), **stale prose**
(the prose moved, the ledger did not), and **unsupported** (nothing checks it at all).

### What it found on its first run

**Two claims were quoted in live documents but could not be re-derived from any committed
artefact.** The preprocessing search's resolution floor (0.0119) lived only in a script's
printed output, and H6's prompt-space span (0.021–0.747) was absent from the CSV, which
stored the median and the 10th–90th percentiles but not the extremes. Both experiments now
store them, and both claims are checked.

**And two `where` fields were wrong.** The ledger said the README states H1's 0.4904 drop —
it states it approximately on purpose ("falls from ~0.99 to ~0.50"), so the precise figure
belongs to the RQ matrix alone. It also said the README states H6's 22.9%; the README quotes
the span instead. Neither was a drifted number; both were the ledger mis-recording where a
claim lives, which is the second thing it is for.

**18 claims, all verified, none unsupported.**

### One correction to my own reasoning

I first checked the prose with a plain substring test, then hardened it to a numeric-boundary
match and wrote that the loose form had matched an unrelated number. **That was wrong** — the
`0.930` it matched was the same claim stated at three decimals, which is a legitimate match.
The hardening is still right for a real reason: `"0.93" in "the value was 0.9302"` is true,
so the loose renderings are prefixes of other numbers and the check could pass on a document
that states a different number and never states this one. The docstring now says that
instead, and a test pins both halves — the prefix is rejected, and rounding is still allowed.

### The design decisions worth keeping

**`where` lists live documents, never `EXPERIMENT_LOG.md`.** The log is append-only history:
entries record what was true when written and superseded ones stay with a retraction beside
them. Requiring it to match current values would forbid keeping that history.

**An ambiguous selector is an error, not a first match.** "The ConvNeXtV2 row" quietly
becoming "the first of five ConvNeXtV2 rows" is how a claim starts describing something other
than what it says, so a multi-row select without an explicit `aggregate` fails loudly.

**The pipeline's dependency on the ledger is derived from the ledger.** `reproduce_all.py`
reads `claims.toml` for the stage's requirements rather than listing them, because a
hand-written copy would be the copy that goes stale — which is what the ledger exists to
catch.

- 2026-09-08 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 18 claims verified against their artefacts; two unsupported claims closed by storing the values, two `where` fields corrected

- 2026-09-08 | WP4-T6 error taxonomy | `python -m experiments.error_taxonomy` | `error_taxonomy.csv` | 221 misclassifications across four protocols; the grouped split's errors are one slot

---

## 2026-09-08 — WP4-T6: the error taxonomy, and a direct measure of what leaked

`experiments/error_taxonomy.py` → `results/error_taxonomy.csv`, `error_taxonomy_summary.csv`

The task asks for every grouped-split misclassification categorised by condition, confusion
pair and camera. Done on that split alone the table is nearly empty, and the emptiness is the
first finding: **all 32 grouped-split errors across the three backbones come from one slot,
at night, and for two of the three every one is the same confusion** — an empty pitch called
active play. Nine errors from one scene is a single failure counted nine times, not a
taxonomy.

So the categorisation runs across all four protocols, and every row carries a column the
others cannot supply.

### 100% of leaky-split errors had a near-duplicate in training; 0% of honest ones did

| protocol | errors with a near-duplicate on the training side |
|---|---|
| random (leaky) | **12 of 12 — 100%** |
| grouped by slot | **0 of 32 — 0%** |

This is a direct measurement of what the split protocols do, taken through the error set
rather than over the corpus. On the leaky split even the frames the models get *wrong* are
frames they had seen a copy of; on the honest split not one is. The near-duplicate audit
established that 37.1% of duplicate pairs straddle a random split; this says what that means
for the evaluation, and it is the cleanest single number for H1 yet.

On an honest split the count should be zero by construction, so it is also a check on the
split — a non-zero value would be a defect in the partition rather than a property of the
model. It is zero, and a test pins both halves.

### Only one protocol has an error set worth categorising

| protocol | errors | distinct slots | verdict |
|---|---|---|---|
| random | 12 | 2 | a spread worth categorising |
| grouped_slot | 32 | **1** | a single failure counted many times |
| **lo_venue_out** | **50** | **16** | a spread worth categorising |
| temporal | 127 | **1** | a single failure counted many times |

Cross-venue is the only protocol whose errors span venues, slots and both lighting
conditions. `clipvenue_h_teal_pitch` is the hardest venue on this data — it takes 25–43% of
each model's cross-venue errors.

### Two model-specific findings

**DINOv2 collapses under the day→night shift in a specific way.** On the temporal protocol it
makes 92 errors of 799 against ConvNeXtV2's 11, and **88% of them are C2→C3**: it calls
active play *maintenance* once trained on daylight and tested on floodlight. ViT does the
same thing more mildly (62% of 24). ConvNeXtV2's errors go the other way — 82% C1→C2.

**The grouped split cannot separate the models.** Two of the three make exactly nine errors,
all the same confusion, all in the same slot. Their error sets are not merely similar sizes,
they are the same failure.

### The explainability half is not done, and not because it was forgotten

Attention rollout and Grad-CAM would write frame images to `results/figs/xai/`. WP1-T5
records that nine images are already permanent in git history including five sheets of
unblurred players, and publishing more frames is the open data-release decision. That is not
mine to take, so the overlays wait for it. Recorded here rather than left as a silent gap in
the acceptance criterion.

- 2026-09-08 | WP4-T6 error taxonomy | `python -m experiments.error_taxonomy` | `error_taxonomy.csv` | 100% of leaky-split errors had a near-duplicate in training against 0% of honest ones; only cross-venue has an error set worth categorising

- 2026-09-08 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 21 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-08 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 23 claims verified against their artefacts, 0 recorded as unsupported

---

## 2026-09-08 — the trivial-baseline floor chart, and a claim A8 had quietly invalidated

`experiments/make_figures.py` → `results/figs/baseline_floor.{png,pdf}`

WP4-T10's floor is the frame this project reads every benchmark through, and one picture of
it answers the question an examiner asks first: *would something trivial have done this?*
Drawing it found that the answer recorded in three documents was out of date.

### The correction (amendment A12)

WP4-T10's headline read *"under the random split a 16-bin colour histogram beat ConvNeXtV2 on
macro-F1 (0.686 vs 0.657)"*. **Both numbers were superseded by A8**, which fixed
`bootstrap_metric_ci` re-deriving the class set inside the bootstrap and moved every
random-split macro-F1. On the corrected estimand the comparison **reverses**:

| random split, macro-F1 | as quoted | corrected (A8) |
|---|---|---|
| ConvNeXtV2 | 0.657 | **0.9879** |
| colour histogram | 0.686 | **0.9616** |

A8 was written up for H1 and correctly reported the grouped-split numbers as unchanged. But
H2's floor claim reads the same column, and nobody re-read it — so a sentence false on its own
data survived in `rq_matrix.md`, `TODO.md` and `docs/CODEBASE.md` for a day. **An amendment
that corrects a number has to be followed to every claim resting on it.**

What survives is most of it, and the better half:

* the **grouped-split** comparison is untouched — the clock rule scores 0.4907 against
  ConvNeXtV2's 0.4975, within **0.0068**, using no pixels at all;
* **A2's scene-level restatement** is now the only true form of the random-split claim: on
  the 62 distinct scenes in that test set the histogram and DINOv2 both score 1.0000, so *the
  leaky protocol cannot tell them apart*. That was already the cleaner sentence;
* **H2's pre-registered verdict is unchanged** — refuted at the 2-point threshold.

Both figures are in the claims ledger now. That is exactly why this went unnoticed: the
ledger checks what is in it, and these two were not. WP8-T5's standing instruction — add a
claim when you write the sentence — applies to amendments too.

### The chart

Four protocols share an axis, each showing the best trivial baseline against the best
backbone:

| protocol | best trivial | best backbone | |
|---|---|---|---|
| random (leaky) | colour histogram 0.970 | ViT 0.993 | |
| grouped by slot | clock rule 0.400 | DINOv2 0.471 | |
| **cross-venue** | **majority class 1.000** | DINOv2 0.960 | **the floor is not cleared** |
| temporal | mean intensity 0.364 | DINOv2 0.546 | |

The cross-venue column is why the chart is worth drawing: the trivial bar is *above* the
backbone bar, because a constant predictor scores a perfect macro-F1 on folds containing one
class. A chart of any single protocol would answer the examiner's question with a number;
four answer it with *it depends entirely on which protocol you ask*, which is the finding.

- 2026-09-08 | WP4-T10 floor chart | `python -m experiments.make_figures` | `figs/baseline_floor.png` | four protocols; the cross-venue floor is not cleared. Drawing it surfaced A12: A8 reversed the histogram-vs-ConvNeXtV2 comparison and three documents still quoted the old one

- 2026-09-08 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 24 claims verified against their artefacts, 0 recorded as unsupported

---

## 2026-09-09 — the four documents the plan kept pointing at, and one constraint made real

Four items were referenced repeatedly by other tasks and did not exist. Writing them cost less
than continuing to route around them.

**WP6-T12 — "the system must never bill", enforced rather than stated.**
`src/pitch_occupancy/slots/authority.py`, 13 tests. Reconciliation can implicate a named member
of staff, so the constraint matters, and a sentence in a chapter is worth very little at an
examination.

`Advisory` is the only output a discrepancy can produce. It carries no monetary field, and
`requires_human_confirmation` is a **property rather than a parameter** — there is no way to
construct one that does not need a person. The permitted vocabulary is three actions
(`flag_for_review`, `record_observation`, `aggregate_by_field`) and `FORBIDDEN` names what was
deliberately left out, so the omission is legible rather than an oversight waiting to be
filled in.

The boundary is **walked, not asserted**: the tests parse every FastAPI decorator and fail if a
financial route, a booking write, or an unexpected mutating route appears, and scan the
database layer for a monetary column. Verified by adding a `POST /slots/{id}/charge` route —
two guards fire.

It is a statement about what the software *offers*, kept true by tests. It is not a safeguard
against a determined operator, and the docstring says so; claiming otherwise would be the
weaker position at an examination.

**WP0-T8 — `docs/backup.md`.** Three copies, two media, one off-site; `data/cache/` excluded as
derived, `data/raw/` and `data/processed/` never; encrypted at rest because the footage shows
identifiable people. Deliberately **manual rather than a sync** — a two-way sync propagates a
deletion as faithfully as it propagates a file, and the failure mode this exists to prevent is
losing footage. The restore procedure is checkable rather than hopeful: a restore has succeeded
when `reproduce_all --check` reports nothing missing and the committed CSVs reproduce.

*The policy is written. The backup is still not made.*

**WP2-T2 — `thesis/data_requests.md`.** Seven items in priority order with what each unblocks.
Empty pitches at any second venue leads, because it is twenty minutes of a camera pointing at
nothing and it unblocks four separate questions. Then 30–40 labelled slots, because frames are
not the bottleneck and the slot-level test set is n=2. There is a §7 saying what is *not* being
asked for — no names, no payment data, no staff identifiers, no continuous recording — since an
over-broad request is harder to approve than a narrow one.

**WP1-T6 — `thesis/alternatives.md`.** PIR, turnstile, floodlight draw, app check-in and manual
logging, compared on cost, accuracy, failure modes and retrofit. Four arguments that survive
scrutiny; four concessions where the alternatives genuinely win; and a closing section naming
where the argument is weakest — the comparison is *argued rather than measured*, no PIR was
ever deployed alongside the cameras, and "the cameras are already there" is a property of this
client rather than of the problem.

**WP8-T6 — `thesis/defence_redteam.md`.** Ten questions with the evidence and a pointer for
each, four more worth having ready, and a rehearsal note. Every figure in it is in the claims
ledger, so the document cannot drift from the results — and three claims gained
`thesis/defence_redteam.md` in their `where` list, which is what makes that true rather than
intended. One number was removed while writing it: a count of the ledger's own entries, which
would have drifted the moment a claim was added.

The answers in it are arguments, and they are the candidate's to own. An argument in someone
else's phrasing collapses on the first follow-up, and the document says so at the top.

- 2026-09-09 | WP6-T12, WP0-T8, WP2-T2, WP1-T6, WP8-T6 | `pytest tests/test_authority.py` | `slots/authority.py`, `docs/backup.md`, `thesis/{data_requests,alternatives,defence_redteam}.md` | the never-bill constraint enforced at the HTTP and database boundary; four referenced-but-absent documents written

---

## 2026-09-09 — the RQ2 figure, and a degraded mode that was reported but never enforced

**WP8-T2b — `results/figs/accuracy_vs_latency.{png,pdf}`.** The plan calls this "one picture
that answers RQ2 completely", and it can because the answer turns on a *negative*: 20 cameras
take 2.5–5.7 s of a 60 s sampling cycle, so every candidate sits in the leftmost tenth of the
axis and latency cannot discriminate between them. Drawing the budget line is what makes that
visible — a scatter without it invites the reader to compare 2.5 s against 5.7 s as though the
difference mattered.

The vertical axis is **recall minus false play**, not recall, and each model is drawn twice:
hollow at its cross-venue recall, filled at the same model once the control is subtracted, the
two joined. The gap is the argument. Held-out venues contain no empty pitch, so recall alone is
earned by answering "playing" more often — and the model with the **best recall has the worst
balanced score**, by 0.70. A plot of recall would recommend it.

**WP7-T5 — the runbook, and one row of it made real.** `docs/runbook.md` is an eight-row
degraded-mode ladder, each row marked *enforced* or *not implemented*, because a runbook that
does not distinguish those is a wish list.

Writing it found row 3 was a gap. **`capture_rate` was computed, reported and flagged to the
operator — and never gated the verdict.** A slot that lost 48 of its 60 minutes produced a
confident `NOTUSED` from the fragment that survived, which is a claim about an hour made from a
fifth of it. That is precisely the failure WP7-T5 names: a camera dies twenty minutes in and
the system reports the pitch empty from the twenty minutes before it filled up.

`Thresholds.review_below_capture` (0.5) now downgrades it to REVIEW, naming how many minutes of
how many were captured. `worker.py` passes the slot's real length through — **without which the
ratios are taken over whatever arrived and the check never fires**, which is the vacuous form
this project has met three times and is covered by its own test.

**It ships on, and the runbook explains why it differs from `review_below_confidence`**, which
ships at 0.0 and is inert. Capture rate is not a model quantity: *"we saw eleven minutes of this
hour"* is not a statement about a pitch whatever the classifier says about those eleven minutes,
so a floor follows from first principles. A confidence threshold is a hyper-parameter, picking
one by hand is the mistake `aggregate.py`'s header warns about, and it is blocked on RQ6. **0.5
is a judgement and is recorded as one** — the weakest defensible reading, not a value calibrated
against real degraded slots, of which the corpus contains none.

The two real slots captured 59–60 of 60 minutes, so no published verdict moves.

One implementation note worth keeping: the new field went in at the **end** of `Thresholds`,
not the front. Inserting it before the three positional fields silently rebound
`Thresholds(used_min, notused_max, notused_empty)` in `tune_thresholds` — caught immediately by
an existing test, and a good argument for the ones that assert an invariant rather than a value.

- 2026-09-09 | WP8-T2b RQ2 figure + WP7-T5 runbook | `python -m experiments.make_figures` | `figs/accuracy_vs_latency.png`, `docs/runbook.md` | capture rate was reported but never gated the verdict; a slot losing 48 of 60 minutes now goes to REVIEW instead of asserting NOTUSED

---

## 2026-09-09 — WP6-T8: retention, and the refusals that are the actual work

`src/pitch_occupancy/retention.py`, `pitch retention`, 12 tests.

`thesis/ethics.md` committed to *"raw frames purged after 7 days in production; evidence images
retained 365 days for audit"* and added **"Code enforces this (WP6-T8)"** — a claim about code,
in the ethics chapter, with nothing behind it. That is the fourth guard in this project found
to describe something that did not operate, and the first in a document about obligations
rather than results.

### The dangerous half

**A retention worker in this repository can delete the one irreplaceable thing.** `data/raw/`
is 4.2 GB from a client facility with a single copy; `data/processed/` is the hand-labelled
corpus. So the interesting question is not whether the worker deletes old files — it is whether
it can be persuaded to delete the wrong ones.

- `PROTECTED_ROOTS` names them, and planning against one **raises** rather than scanning it.
- `apply()` **aborts** on a plan containing a protected path rather than skipping that one: a
  plan with one in it was built wrongly, and the rest of it is not to be trusted either.
- A test writes a file into `data/raw/`, hands `apply()` a plan naming it, and asserts the file
  survives the abort.

**"Raw frames" in the commitment means frames sampled by the running system** — `data/interim/`
— not the research corpus. Conflating the two would be the most expensive bug this project
could ship, so the distinction is written down rather than understood.

### The cautious half

Dry run is the default and the only thing `plan()` does; deleting needs `apply(confirm=True)`
or `--apply`. The acceptance criterion is *"a dry run prints the correct purge set"*, so **the
plan is the artefact** and deletion operates on it.

A file whose age cannot be read is **kept and reported**, never swept up in a wildcard —
deleting on a failed `stat()` is how a retention worker becomes a data-loss incident. Writing
that test found a real bug: `is_file()` stats too, and it sat *outside* the guard, so the
OSError escaped one call before the handler. Same shape as the failure the handler exists for.

The clock is injectable, which is the only way a 365-day rule gets a test before the year is
up. The two periods are checked **against `thesis/ethics.md` itself**, so the document and the
code cannot drift; and that document now says the constraint is enforced and notes that it
previously said so untruthfully.

The runbook's disk-full row moves from *not implemented* to *partly*: growth is bounded now,
but nothing reacts to a full disk and nothing runs retention on a schedule. Both belong with
WP7-T2's service units, and the row says so rather than implying the problem is solved.

- 2026-09-09 | WP6-T8 retention worker | `pitch retention` | `src/pitch_occupancy/retention.py` | the ethics doc claimed code enforced retention and none existed; the corpus is protected by explicit refusal, tested by trying to delete from it

- 2026-09-08 | WP6-T10 reconciliation value | `python -m experiments.reconciliation_value` | `reconciliation_value.csv` | break-even flag precision 95.0% on stated assumptions; precision itself is unmeasurable here (WP6-T11)

- 2026-09-08 | WP6-T10 reconciliation value | `python -m experiments.reconciliation_value` | `reconciliation_value.csv` | break-even flag precision 94.7% on stated assumptions; precision itself is unmeasurable here (WP6-T11)

---

## 2026-09-09 — WP6-T10: what reconciliation is worth, and how good it has to be

`experiments/reconciliation_value.py` → `results/reconciliation_value.csv`

The task asks for *"expected € recovered per 1,000 slots at the chosen operating point, with
the assumptions stated"*. Two of the three inputs that figure needs are assumptions, and the
third cannot be measured on this corpus at all — so a single euro figure would be a number
invented out of a parameter nobody has measured. **The honest form of the answer is a
break-even.**

| | status |
|---|---|
| **what the system flags** | **known** — `reconcile.py` is deterministic given (booking, record, verdict), so the flag rate follows from the rule table; only the assumed case mix enters |
| **prices and costs** | **assumed** — every one is a parameter with a default and a sensitivity sweep |
| **flag precision** | **unknown, and unknowable here** — WP6-T11: it needs adjudicated slots, and this corpus has two |

### The result

On the stated defaults (€40 slot, 10 staff-minutes to investigate, €120 for a wrong
accusation, 5% unbooked usage), the system produces **200 flags per 1,000 slots** — of which
only the 50 `UNBOOKED_USAGE` ones recover money.

**Break-even precision: 94.7%.** Below that, flagging costs more than it recovers.

That is a high bar, and it is the finding. The arithmetic is not subtle: 200 investigations
cost €600 whether or not they are right, and every wrong flag costs €120 more, against at most
€2,000 of recoverable revenue. **The viability of the reconciliation feature hinges almost
entirely on how often a flag is right**, and that is exactly the quantity WP6-T11 says cannot
be estimated without adjudicated slots.

It also gives the human-in-the-loop design a number rather than a principle: at a break-even
of 95%, no automated action could ever be justified on this cost structure.

### Sensitivity

| assumption | half | default | double |
|---|---|---|---|
| slot price | 98.5% | **94.7%** | 87.9% |
| minutes to investigate | 93.5% | **94.7%** | 97.0% |
| cost of a wrong accusation | 90.0% | **94.7%** | 97.2% |

And below roughly a €20 slot price there is no bar to clear at all: the feature cannot pay at
*any* precision, because the investigations cost more than the recoverable revenue is worth
even when every flag is right. The sign of the result depends on a number the facility
supplies, which is the clearest argument for reporting a break-even rather than a euro figure.

### A modelling error worth recording

The first version multiplied by precision **twice** — once to count the true flags, and again
against the total recoverable — which halved the recovery at p=0.5 and moved the break-even by
several points. Caught by writing the test that says doubling precision must double recovery.
Recovery is now per anomaly *type*, since an unbooked slot is revenue never invoiced while a
no-show is a correction to a record, and flattening them credited the system with money it did
not find.

**This is an evaluation, not an action.** Attaching money to the confusion matrix sizes the
feature; the system takes no financial action of any kind and cannot be made to
(`slots/authority.py`, WP6-T12). The two are easy to conflate and are opposites.

- 2026-09-09 | WP6-T10 reconciliation value | `python -m experiments.reconciliation_value` | `reconciliation_value.csv` | break-even flag precision 94.7% on stated assumptions; below a EUR20 slot price the feature never pays at any precision

- 2026-09-08 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 26 claims verified against their artefacts, 0 recorded as unsupported

---

## 2026-09-09 — WP8-T2b: the leakage decomposition, and a reliability diagram that cannot exist

**`results/figs/leakage_decomposition.{png,pdf}`.** The thesis's stated first key figure, and
the raw drop on its own overstates the case. Each model's fall from the leaky split to the
honest one is drawn as **one stacked bar**, because the question is how a single quantity
divides rather than how two compare: the pale segment is the 0.183 that a model which *never
trains* shows on the same change of test set, the solid segment is what is left for leakage.
OpenCLIP is drawn as a fourth bar so the subtraction is visible rather than taken on trust.

About **two thirds** of the penalty is attributable (0.332–0.395 of 0.516–0.578), and the
title derives that fraction from the data rather than stating it — a ratio in a title is a
claim, and a claim typed into a generated figure is exactly what let an earlier plot
contradict its own source table.

**And a reliability diagram that is not worth drawing.** `rq6_calibration_riskcoverage.py` now
persists its bins to `rq6_reliability.csv` — previously only the worst bin was printed, so the
figure would have had to recompute the probes and could have silently disagreed with the
table. With the bins in hand the answer is that there is no diagram:

| model | bins occupied | largest |
|---|---|---|
| convnextv2 | **1** | 907 of 907 in [0.9, 1.0] |
| dinov2 | 2 | 905 of 907 in [0.9, 1.0] |
| vit | 1 | 907 of 907 |

A ten-bin reliability plot would be a single dot on the right-hand edge. That is the same fact
the risk–coverage band already reports from the other side — 890 of 907 confidences identical
— so the figure is **deliberately not drawn**, the bins are committed as the artefact behind
the "worst bin" claim, and the reason is recorded here rather than left as a gap in WP8-T2b's
list.

- 2026-09-09 | WP8-T2b leakage figure | `python -m experiments.make_figures` | `figs/leakage_decomposition.png`, `rq6_reliability.csv` | the leakage penalty split into composition and attributable; a reliability diagram would be one dot, so it is not drawn and the bins are committed instead

---

## 2026-09-09 — WP1-T1: the related-work scaffolding, and one rule about citations

`thesis/ch2_related_work.md`. Scaffolding, not a draft: each strand states *what it has to
establish* and *why this project needs it*, with the project's own findings placed against it,
so the literature search has a target rather than a topic.

**Nine strands, not the seven the plan listed.** Two were added because the work produced
results that need them:

- **§2.8 prior art.** The gap statement is exposed until someone has actually looked for a
  commercial or academic system that already does facility occupancy audit. If one exists the
  gap narrows to CPU-only on existing CCTV, leakage-free evaluation, and reconciliation with a
  human in the loop — which is still a gap, and much better stated deliberately than heard at
  the defence. Flagged as the strand to do first if time is short, because it is the only one
  that can change what the thesis claims.
- **§2.9 selective prediction.** The REVIEW band *is* selective prediction with a human
  fallback. Citing that literature upgrades it from an engineering convenience to a principled
  design, and supplies the vocabulary for the risk–coverage work — including why the curve
  here has to be a band.

**One rule, in bold at the top of the document: no references are invented.** A fabricated or
half-remembered citation is the single error in a thesis that cannot be defended, and it is
the specific failure an LLM-assisted draft is most likely to introduce. Every `[CITE]` is a
placeholder for a paper the author has opened. The same discipline as the claims ledger,
applied to the claims that have authors attached.

Where the strands are most load-bearing, the project's own numbers are placed against them so
the chapter can *narrow* its claims rather than imply novelty by omission — §2.3 carries the
100%-vs-0% near-duplicate result, §2.4 the constant predictor scoring 1.000, §2.6 the 905-of-907
single reliability bin, §2.7 the 95% break-even.

- 2026-09-09 | WP1-T1 related-work skeleton | *(document)* | `thesis/ch2_related_work.md` | nine strands with what each must establish; two added from this project's own results; no citations invented

- 2026-09-09 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 2 gate(s) met on artefacts, 3 waiting on a person

---

## 2026-09-09 — WP2-T7 / WP4-T8: the gate tracker, checked instead of ticked

`experiments/gate_check.py` → `results/gate_status.md`, `gate_status.json`

Every row of the milestone tracker in `TODO.md` read `[ ]`, and two gates had been met for
some time. Same drift as the README that said the implementation had not started and the
export that stopped covering the thesis — a hand-maintained status that nobody updates while
doing the work it describes.

So the criteria are predicates over the repository. **Three outcomes, not two**, because the
difference matters: *met*, *not met*, and **needs a person** — supervisor sign-off, a client
conversation, a live deployment. Reporting the third as unmet would say the work was not done
when nothing in the repository could do it.

| gate | verdict |
|---|---|
| M1 | waiting on a person — protocol, labelling protocol, taxonomy and pre-registration all present; supervisor and DPIA sign-off are not artefacts |
| **M2** | **passed** — coverage matrix, EMPTY in exactly one venue (494 frames), near-duplicate rate, per-split leakage 100% vs 0%, effective sample, "not answerable" list |
| **M3** | **passed** — 8 models × 4 protocols, intervals on every row, Holm-corrected families in three reports, four trivial baselines, figures exported, 26 claims re-deriving |
| M4 | not passed — the logit-average baseline answers WP5-T2 negatively, but the fusion head and STAN are not built |
| M5 | not passed — reconciliation, end-to-end, retention and degraded mode are all in; the scheduler service is not |
| M6 | waiting on a person — runbook written, live validation needs a deployment |
| M7 | waiting on a person — ledger, red-team and Chapter 2 present; submission is not an artefact |

### Two checks that were wrong in a plausible way

Both found by testing the property rather than trusting the output, and both worth recording
because a check that reports a *believable* wrong number is worse than one that fails — nobody
looks twice at it.

**The EMPTY concentration check read the wrong table.** It took the first row beginning
`| EMPTY`, which is the class-by-*lighting* table earlier in `coverage.md`, and reported
"EMPTY appears in 2 venues" — meaning day and night. It now scopes to the class-by-venue
section and reports *exactly one venue*, which is the fact M2 exists to quantify.

**The claims criterion grepped the generated page for "stale".** That passes or fails on prose,
including the prose explaining what staleness is. It runs the verifier now. Its test had the
same bug one level up: an earlier version asserted the string was absent from the function and
failed on the docstring saying why grepping is wrong, so the test checks the behaviour —
append "stale" to the ledger page and the criterion must still pass.

And one criterion is deliberately strict: **the fusion gate does not pass on the baseline
alone.** The logit-average result answers WP5-T2's question negatively, which is a finding, but
it is not the module M4 asks to have ablated. Conflating them would pass a gate on work that
was deliberately not done.

The tracker in `TODO.md` now points here, and a test asserts the gate names and weeks in the
two agree — so re-cutting a gate in one place and not the other fails rather than diverging.

### And a third, found by the guard it had just been given

The first committed `gate_status.md` said M3 was **not passed**. It was generated by running
`experiments/gate_check.py` by path rather than with `-m`, which puts `experiments/` on the
path instead of the repository root — so the claims criterion's import failed and it degraded
quietly to *"the claims verifier could not be loaded"*, flipping a gate.

**A check whose answer depends on how it was invoked is not a check.** The path is repaired
rather than the failure swallowed, and a test now runs the module both ways and requires the
same verdict. Caught by the currency test written twenty minutes earlier, which is the
argument for writing it.

- 2026-09-09 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | M2 and M3 met on artefacts; M1/M6/M7 wait on a person; three checks found wrong in a plausible way and fixed

- 2026-09-09 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 1 gate(s) met on artefacts, 2 waiting on a person

- 2026-09-09 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 3 gate(s) met on artefacts, 3 waiting on a person

---

## 2026-09-09 — WP6-T2: the scheduler, and a foreign key that named a real confusion

`src/pitch_occupancy/scheduler.py`, `configs/slots_schedule.json`, `pitch schedule`, 18 tests.
**M5 passes with it** — it was the single unmet criterion.

`worker.run_slot` had sampled, classified, fused and aggregated a slot for some time. What was
missing was the part that decides *when*: reading a schedule, working out what is running now,
and writing the result at slot end.

**Two decisions shape it.** The clock is a parameter everywhere — a scheduler whose behaviour
can only be observed by waiting an hour is a scheduler with no tests, and "runs continuously"
is the least interesting half of the acceptance criterion. And deciding is separated from
doing: `due()` is pure, so *what would run* is answerable without running anything, the same
shape as `pitch retention` and for the same reason — the part that touches the world should be
the smaller, later half.

### The foreign key was right and the model was wrong

The first version wrote samples straight to the database and hit
`sqlite3.IntegrityError: FOREIGN KEY constraint failed`. The temptation is to disable the
constraint; the schema was correct and the model was not.

`ScheduledSlot` had one `field_id` doing two jobs. The manifest and the database key a slot
instance as **`<venue>_<date>_<HHMM>`**, while `rental_slots.field_id` references a **pitch** —
and a venue may have several. So the venue names the slot and the field owns the cameras, and
one attribute could not be both. Split, with the reason in the docstring.

`db/store.ensure_slot` now declares the venue, field, cameras and rental slot before any sample
is written — idempotent, so a slot that runs again after a restart duplicates nothing. **The
thing that knows a slot is starting is the thing that should declare it**, and that is the
scheduler, not the sample writer.

### What it deliberately is not

**Not a daemon.** No supervision, no restart policy, no systemd unit — those are WP7-T2 and are
not written. `run_forever` is a plain loop with an injectable sleep and an iteration bound,
enough to run under a service manager without pretending to be one.

**Not a deployment.** It has never run against a camera; the tests hand it recorded video. The
`--from-recordings` mode derives a schedule from the footage that exists, and its docstring says
why that is for exercising the path only: **a real deployment reads the config**, because the
schedule is what says a slot *should* have happened, and deriving it from what was recorded
would make a missing hour invisible — which is exactly what reconciliation exists to catch.

One dead camera does not stop the other pitches: a slot that raises is reported and skipped,
with a test. Schedule validation is strict — a missing field, a slot with no cameras, two slots
at the same time on one venue, an unknown weekday all raise, because a silently-dropped entry
is a slot with no footage and no explanation.

- 2026-09-09 | WP6-T2 scheduler | `pitch schedule` | `src/pitch_occupancy/scheduler.py` | M5 passes; a foreign-key failure named a real confusion between venue and field, fixed in the model rather than papered over

## 2026-09-09 — WP5-T2: the gate does nothing, and the axis that would have judged it measures something else

`src/pitch_occupancy/slots/fusion_head.py`, `experiments/fusion_head_ablation.py`, 21 tests.
The module M4 asks for, built and reported as the negative result it is — and it turned up a
larger problem in a number this repository has been quoting for a month.

**The gate does not earn its place.** The plan's architecture is cheap image statistics → tiny
MLP → per-backbone weights → shared linear head. Built as specified, it is worth **−0.0238**
cross-venue play recall against the same head with the gate switched off, on one informative
fold out of seven, p = 1.000. WP5-T2 is answered negatively.

**The ablation is a three-rung ladder, and that was not the first design.** The first version
had an on/off switch: gated versus not. It showed the gate reaching 0.000 false-play where the
ungated head reached 0.967, which reads as the gate working. It is not. The gate puts about
0.70 of its weight on DINOv2 in *every* fold and moves it by 0.086 between frames — almost all
of its effect is a learned constant, not routing. So the ladder is `uniform` → `constant` →
`mlp`, each rung adding exactly one capability with head, trainer, seed, epochs and
regularisation held identical, and the two differences are named separately: learned mixing,
then routing. Routing is the part the module exists for, and it is the part worth nothing.

The implementation can route — that is checked rather than assumed. On a fixture where the two
backbones carry opposite labels and which one is right flips with the gate statistic, `uniform`
and `constant` both score 0.671 and `mlp` scores 1.000; feeding the same gate a matrix of
zeros collapses it back to 0.663. The null on the real data is a fact about the data.

**The finding that matters is on the other axis.** A column was added to check whether a
false-play rate of 0.000 was real: what does the model answer *instead* of ACTIVE_PLAY? On the
243 held-out empty frames —

| model | ACTIVE_PLAY | MAINTENANCE | EMPTY (correct) |
|---|---|---|---|
| convnextv2 | 241 | 0 | **2** |
| dinov2 | 75 | 168 | **0** |
| vit | 203 | 0 | **40** |
| ens_convnextv2_dinov2 | 243 | 0 | **0** |
| fusion_gated_stats | 0 | 243 | **0** |

**The complement of the false-play rate is not correctness.** DINOv2's 0.309 — quoted in the
README, in `rq_matrix.md` and throughout this log as dominating ConvNeXtV2's 0.992 — is the
rate at which it makes one kind of mistake rather than another. It gets **none** of the 243
right. The gated head's perfect 0.000 is 243 wrong answers filed under MAINTENANCE. The best
model on the axis is ViT at 40 of 243, and no model exceeds 0.165.

Both published numbers remain correct as stated, and the two ledger claims re-derive
unchanged; what does not follow is the reading that a low rate means a model can recognise an
empty pitch. One sentence in the H4 entry above drew exactly that inference and is corrected
in place.

**The mechanism is the dataset, not the head.** This protocol trains on 518 ACTIVE_PLAY, 251
EMPTY and **6** MAINTENANCE frames, and `class_weight="balanced"` gives a six-frame class a
weight of 43. Out-of-distribution frames land in it. That is why an EMPTY-accuracy column now
sits beside the rate, and why the joint summary is built from accuracy rather than from
`1 - false_play`: built the other way it ranked the gated head first in the table.

**The confound test says "partly".** The gate's DINOv2 weight correlates **−0.653** with a
night indicator across all seven folds, so lighting is much of what those three statistics
carry — the fifth appearance of this project's recurring confound, in the gate's inputs rather
than in a classifier. But a gate fed *only* lighting is 0.1015 worse on recall, so it is not
all of it. One coincidence is worth recording: the lighting-only gate's false-play rate is
0.0206, which is the published clock rule's rate to four decimals, both being 5 of 243.

**Nothing here is significant, and the design could not have made it so.** The recall
comparisons run on seven venue folds with one to four informative pairs, so the sign-flip
floor is 0.125 at best and 1.000 for the routing test. The empty set is 243 frames but **three
distinct scenes**. Every number in that column, including the large ones, rests on three
independent observations. The table is reported with both counts rather than with the
flattering one.

- 2026-09-09 | WP5-T2 gated fusion head, ablated | `python experiments/fusion_head_ablation.py` | `fusion_head_ablation.csv` | routing worth -0.0238 recall, p=1.000 (floor 1.000); no model exceeds 0.1646 EMPTY accuracy on the held-out camera

- 2026-09-09 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 27 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-09 | WP5-T1 STAN, preliminary | `python experiments/stan_preliminary.py` | `stan_preliminary.csv` | composed test: stan 1.0000 vs best baseline summary_logistic 0.9300; 2 real slots, below the 30-slot gate

- 2026-09-09 | WP5-T1 STAN, preliminary | `python experiments/stan_preliminary.py` | `stan_preliminary.csv` | composed test: stan 1.0000 vs best baseline hmm 0.8950; 2 real slots, below the 30-slot gate

- 2026-09-09 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 4 gate(s) met on artefacts, 3 waiting on a person

- 2026-09-09 | WP4-T13 what the false-play control measures | `python experiments/empty_recognition.py` | `empty_recognition.csv` | DINOv2 EMPTY accuracy 0.0000 -> 0.7325 once camera B is in training at all; removing C3 does not change it - the control is a camera-transfer test

## 2026-09-09 — WP4-T13: a prediction of mine that was wrong, and the better answer underneath it

`experiments/empty_recognition.py`, `empty_recognition.csv`, 9 tests.

The WP5-T2 entry above blamed the zero EMPTY accuracy on the training mix — 6 MAINTENANCE
frames with `class_weight="balanced"` giving that class a weight of 43. The obvious next move
was to drop C3 and watch the problem go away. **It does not.**

| training set (train camera A → camera B's 243 empties) | EMPTY accuracy |
|---|---|
| 3-class, balanced *(the published configuration)* | 0.0000 |
| 3-class, unbalanced | 0.0000 |
| 2-class, C3 dropped entirely | 0.0000 |

Removing C3 stops the MAINTENANCE absorption and pushes all 243 frames into ACTIVE_PLAY
instead: false-play goes **0.309 → 1.000**. The class weighting decides *which* wrong answer
appears. It has nothing to do with whether the model can recognise an empty pitch. The earlier
entry's mechanism was right about the MAINTENANCE dumping and wrong about the zero.

**What actually decides it is whether the model has ever seen a labelled empty frame of that
camera.**

| training set | DINOv2 | ConvNeXtV2 | ViT |
|---|---|---|---|
| camera A only *(the published protocol)* | 0.0000 | 0.0082 | 0.1646 |
| + camera B's ACTIVE_PLAY frames | 0.7325 | 0.1975 | 0.0041 |
| + **one** labelled empty frame of camera B | **0.9793** | **0.9793** | **0.9793** |
| + 25 labelled empty frames | 0.9817 | 0.9817 | 0.9817 |

So **the false-play control is a camera-transfer test, not a specificity test.** Every
false-play number this project has published is a cross-camera transfer number. The numbers
stand; the caption does not.

The middle row shows how badly the two are conflated. There camera B appears in training in
*one class only*, so "camera B implies play" is available as a shortcut, and the three
backbones go in opposite directions — DINOv2 largely resists it, ConvNeXtV2 partly takes it,
and **ViT takes it completely and falls from 0.165 to 0.004**. On that axis the model ranking
is partly a ranking of shortcut resistance. The same confound, in a new place, for the sixth
time.

**The practical finding: one labelled empty frame of a new camera is worth more than 769
frames of a different one.** All three backbones reach 0.9793 at k=1 and stay flat to k=25 —
there is no curve to climb. Adding a camera does not need a bigger dataset; it needs a handful
of labels from that camera.

Read with its caveats, which are in the table itself rather than in a footnote. The 243 empty
frames are **three distinct scenes**, which is *why* one frame suffices — a fixed camera
pointed at an empty pitch sees about three views. Every k-shot row carries a count of
near-duplicate pairs crossing the train/test boundary, 1,400 already at k=1: the single
training frame is a near copy of much of the test set. That is the mechanism stated plainly,
and it is also the operational reality, since a deployed system really would be scoring frames
that look like the ones it was given. The within-camera 1.0000, with 13,406 crossing pairs, is
the leaky ceiling and not a score.

- 2026-09-09 | WP4-T13 what the false-play control measures | `python experiments/empty_recognition.py` | `empty_recognition.csv` | DINOv2 EMPTY accuracy 0.0000 -> 0.9793 with ONE labelled empty frame of the held-out camera; removing C3 changes nothing - the control is a camera-transfer test

## 2026-09-09 — WP4-T5: the XAI overlays, and a model that classifies play without looking at players

`src/pitch_occupancy/vision/explain.py`, `experiments/make_xai_figures.py`,
`results/figs/xai/` (27 sheets), `xai_evidence_focus.csv`, 16 tests. Unblocked by the author's
decision to proceed; the images are redacted regardless, for the reason below.

**The decomposition is exact, and that is not a rhetorical claim.** Every probe here is a
logistic regression on *mean-pooled* frozen features, which is linear end to end, so a class
score is exactly the mean over spatial positions of a per-position contribution. Grad-CAM
exists because a head is usually non-linear over its feature map; here it is not, so there is
no gradient to approximate and no smoothing parameter to pick. The map *is* the summands of
the score. Every figure is generated with the reconstruction printed beside it and the run
aborts if the two disagree — worst error over 27 explanations, **8.65e-07**.

**The figures were turned into a measurement, which is where the finding is.** Pictures invite
eyeballing, so the experiment also asks: what share of a frame's positive evidence falls
inside the YOLO person boxes, against what share of the frame those boxes occupy? A model
reading *players* scores far above 1; a model whose evidence ignores them scores 1.

| backbone | evidence on people | their area | ratio |
|---|---|---|---|
| ConvNeXtV2 | 0.133 | 0.054 | **2.18×** |
| DINOv2 | 0.082 | 0.054 | **1.61×** |
| ViT | 0.054 | 0.054 | **1.02×** |

**ViT classifies ACTIVE_PLAY correctly on all three frames while placing no more evidence on
the players than on the turf.** 1.02× is the null exactly. It is reading scene context rather
than people — consistent with it being the weakest cross-venue model (0.869) and with it
taking the "camera B implies play" shortcut completely in the WP4-T13 entry above. Nine frames,
so a direction rather than a rate, and the caveats are printed with it: YOLO boxes are a proxy
for where players are, and a 16×16 patch grid cannot resolve a distant one.

**Every frame is redacted before it is written, in two layers.** YOLO pixelates each detected
person, then a blur floor is applied to the whole frame so a missed detection is still not an
identifiable face. The heatmap is computed on the *original* frame and drawn over the redacted
copy, which has identical geometry — explaining the pixelated frame instead would produce an
honest picture of a model looking at pixelation. Redaction is not a flag and cannot be turned
off from the command line: these images enter git history, where they are permanent, and
WP1-T5 still governs what may be published from this dataset. `people detected: 0` is reported
as "the detector found none", never as "the frame is empty", and a detector that fails to load
reports −1 rather than silently writing an unredacted frame that the pipeline calls redacted.

One implementation note worth recording because it produced a *missing figure* rather than an
error: transformers now defaults to SDPA attention, whose fused kernel never materialises the
attention matrix, so `output_attentions=True` is silently ignored and rollout returns nothing.
`load_for_attention` loads with eager attention for this one purpose.

- 2026-09-09 | WP4-T5 XAI overlays | `python experiments/make_xai_figures.py` | `results/figs/xai/` (27 sheets) | exact linear decomposition, worst reconstruction error 8.6e-07; ViT puts 1.02x evidence on players (the null exactly); every frame person-pixelated and blurred before writing
- 2026-09-09 | WP6-T4 booking importer | `pitch bookings` | `configs/bookings_example.csv` | read-only by interface, not by flag; 8 example rows covering every status, keyed to the two really-recorded slots

- 2026-09-09 | WP4-T5 XAI overlays | `python experiments/make_xai_figures.py` | `results/figs/xai/` (27 sheets) | exact linear decomposition, worst reconstruction error 8.6e-07; every frame person-pixelated and blurred before writing

## 2026-09-09 — WP6-T3 and WP6-T7: a live path that could not run, and an override that must not label

`frame_source.RTSPSource` rewritten with `scheduler.live_sources` (17 tests),
`src/pitch_occupancy/retraining.py` with `pitch retraining` (13 tests).

**The RTSP source had been written and never exercised, and it could not run.** `n_minutes`
raised `NotImplementedError` with the comment "the scheduler decides" — but the scheduler
calls `worker.run_slot`, whose first statement is `for minute in range(source.n_minutes)`. So
the one class whose entire purpose is live operation was structurally incompatible with the
only code that would ever drive it. Nothing noticed because nothing tested it. The same shape
as every other defect in this log: not a wrong number, an unexercised path.

It also had no pacing. `run_slot` iterates minutes without waiting, which is correct for a
recording — minute *k* is a seek — and wrong for a stream, where it would have taken sixty
snapshots back to back in under a second and recorded that as an hour of football. `read` now
blocks until the minute has arrived, and *catches up* rather than stretching when a run has
fallen behind: missed minutes belong in the capture rate, where a degraded verdict is visible,
not hidden inside a slot that quietly ran long.

The clock, the sleep and the socket are all parameters, which is the only reason an hour-long
class can be tested in milliseconds. Two of those tests were wrong first, and both were the
test double's fault rather than the code's — worth recording because a bad fake reports a bug
that does not exist. A frozen clock made `wait_for` sleep a full minute on *every* call
including the second camera of the same minute, so the first run reported a pacing bug; a fake
that cannot represent time passing cannot test code whose job is waiting. And the fake stream
reset its failure count on each new connection, so a source that reconnects per retry — which
is the design — could never succeed.

**The override harvest departs from WP6-T7 as written, deliberately.** The task says copy
overridden slots' evidence frames into `data/dataset/_incoming/<corrected_class>/`. That
cannot be done soundly: **a slot override is not a frame label.** An operator overriding an
hour to USED is saying there was a match, not that each of the three evidence frames shows
active play — and under the tuned thresholds a slot is USED at 35% play, so a majority of its
minutes may show an empty pitch. Auto-filing would inject confidently wrong labels from the
one source this project treats as ground truth.

So frames land in `_incoming/_unfiled/<slot_status>/` with a sidecar recording exactly what is
known — slot, operator, timestamp, note, and both verdicts — and an empty `frame_label` for a
human to fill. The underscore prefix is load-bearing: `build_manifest` globs `[0-9]_*`, so
nothing staged can reach a training set by accident, and a test asserts that rather than
trusting it.

Same refusal discipline as `retention.py`: `plan` is pure, `apply(confirm=False)` is inert,
frames are copied rather than moved (the evidence still justifies a billing decision and the
audit trail points at it), and an existing destination is skipped so a re-run cannot undo a
human's annotation.

**The feedback loop is named rather than left implicit.** Operators override the verdicts they
notice, and they notice the ones that look wrong, so this harvests the model's own errors
preferentially. That is what makes it valuable for training and what makes it a biased sample:
a head re-fitted on it is fitted on a corrected error distribution, not the operating one, and
its accuracy on that data estimates nothing. Evaluation stays on the held-out sets. Written
into the module docstring because the temptation to report "accuracy after retraining" will be
strong and the number would be meaningless.

- 2026-09-09 | WP6-T3 live RTSP path | `scheduler.live_sources` | `frame_source.py` | the live source could not run at all - `n_minutes` raised and `run_slot` reads it on line one; now paced, injectable and tested through the real worker
- 2026-09-09 | WP6-T7 override harvest | `pitch retraining` | `data/dataset/_incoming/_unfiled/` | staged unfiled, never auto-labelled: a slot verdict is not a frame label

- 2026-09-09 | WP6-T1 simulator snapshot API | `GET /api/v1/cameras/{id}/snapshot` | `api/simulator.py` | 17 tests; an unset PITCH_SIMULATOR_TOKEN disables the endpoint (503) rather than opening it, mode=live refuses rather than serving the recording, and a gap is a 404 not a placeholder

- 2026-09-09 | WP6-T6 evidence images | `run_slot(evidence_dir=...)` + `GET /api/v1/slots/{id}/evidence/{i}` | dashboard inspector | 15 tests; `run_slot` had set every `EvidenceFrame.image_path` to None, so three good minutes were chosen and three paths to nothing recorded - the inspector's "no evidence images bound" was accurate, and the WP6-T7 harvest would have found every frame missing

- 2026-09-09 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 28 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-09 | WP8-T3 threats to validity | `thesis/threats_to_validity.md` | 28 claims re-derive | four kinds of validity, three confidence labels; every cited figure is a claim id and a test asserts each exists and has a source artefact. Two numbers were wrong in the first draft and caught by checking against the ledger rather than by reading: the preprocessing search ran 88 evaluations (740 was minutes), and 0.4474 is the *prompt* search's selection optimism, not the preprocessing search's

- 2026-09-09 | WP4-T3 onboarding cost, camera-level | `python experiments/onboarding_cost.py` | `onboarding_cost.csv` | leave-one-venue-out adaptation is not runnable (1 of 8 venues carry >1 class); at camera level DINOv2 goes 0.441 -> 0.990 macro-F1 with 5 labelled frames of the new camera

- 2026-09-09 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 29 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-09 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 30 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-09 | WP4-T3 onboarding cost | `python experiments/onboarding_cost.py` | `onboarding_cost.csv` | leave-one-venue-out adaptation is not runnable (1 of 8 venues carries >1 class); at camera level macro-F1 goes 0.441 -> 0.9895 with one labelled frame, and **from k=5 training on the target frames alone matches them plus 775 source frames** - the source set stops contributing, on all three backbones

- 2026-09-09 | WP8-T2b onboarding figure | `python experiments/make_figures.py` | `figs/onboarding_cost` + served site | two series per backbone so the dashed target-only line carries the finding; crossover and title numbers read from the CSV. Labels sit at k=0 because the right-hand end is where all three converge - the first version stacked three words on one point

- 2026-09-10 | WP8-T4 defence deck | `thesis/defence_deck.md` | 16 tests | a slide plan that opens by forcing the framing choice - apologise for the negative results, or claim them as the thesis. Tests pin the un-cuttable demo section, its fallback, and four sentences most likely to be softened in an edit. One test forbids hardcoding a claim count, which the first draft did (28, now 30)

- 2026-09-10 | WP6-T6 field matrix | `GET /api/v1/fields/day/{day}` + dashboard grid | 10 tests | a scheduled slot with no verdict is drawn hollow rather than as a neutral chip - an unobserved hour and an observed-empty hour are different claims. Found and fixed while building it: the evidence endpoint was sending absolute server paths into the page, and two JavaScript regexes were invalid Python escapes that happened to work

- 2026-09-10 | WP6-T6 schedule editor | `GET/PUT /api/v1/schedule` | `api/schedule_editor.py`, 22 tests | the API's only write path. Validation delegates to `load_schedule` itself rather than reimplementing its rules; writes go through `os.replace` from a staging file beside the target; the previous version is kept under `configs/schedule_history/`. A rejected edit leaves no trace, not even a backup

- 2026-09-09 | WP3-T6 augmentation across cameras | `python experiments/augmentation_transfer.py --views 4` | `augmentation_transfer.csv` | best preset light 0.8550 vs 0.4406 unaugmented and 0.3625 for the duplicate-rows control; closes 75% of the gap one labelled frame closes

- 2026-09-09 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 32 claims verified against their artefacts, 0 recorded as unsupported

## 2026-09-10 — WP3-T6: one augmentation works, and turning them all on erases it

> **RETRACTED 2026-09-10, later the same day** — the headline below is one augmentation
> draw, not an effect. Re-drawn four more times with the same rows, the same probe seed and
> the same test set, `light` scores 0.3479, 0.3501, 0.3510 and 0.4136 against the 0.8550
> reported here; **four of the five draws land below the 0.4406 the probe reaches with no
> augmentation at all**, and three of them are the trivial one-class predictor. The
> duplicate-rows control and the `full` observation still stand as reasoning; the number
> does not. See the retraction entry at the end of this log.

`experiments/augmentation_transfer.py`, `augmentation_transfer.csv`, 11 tests. The question
`onboarding_cost.py` left open: a probe trained on camera A scores 0.441 macro-F1 on camera B
and 0.9895 with one labelled frame of B — **can augmentation close that without any labels?**

| preset | macro-F1 | vs baseline | EMPTY recall |
|---|---|---|---|
| no augmentation | 0.4406 | — | 0.000 |
| `none` *(duplicate rows, the control)* | 0.3625 | −0.0781 | 0.000 |
| `colour` | 0.3479 | −0.0926 | 0.000 |
| **`light`** | **0.8550** | **+0.4144** | **0.687** |
| `weather` | 0.4486 | +0.0080 | 0.099 |
| `full` *(everything)* | 0.3479 | −0.0926 | 0.000 |

**One preset works and it works on the failure that mattered.** Brightness, gamma and sensor
noise take empty-pitch recall from 0.000 to 0.687 with no labels from the target camera,
closing 75% of the gap one labelled frame closes. The duplicate-rows control moves *down*, so
the gain is variety rather than row count — which is why that control was run.

**`full` is the finding to take away.** It contains every one of `light`'s effects and adds
colour jitter, fog, rain and flip, and it lands on 0.3479 — exactly `colour` alone, with empty
recall back at zero. More augmentation did not dilute the benefit, it erased it. Match the
augmentation to the shift being fought; adding the rest costs you the gain. That runs against
instinct, so a test pins it against being "corrected" later.

Neither clean story survives. 0.855 is not 0.9895, so augmentation is not a substitute for the
five-frame recipe — but the earlier claim that it does not help across cameras was wrong too.

**One seed per preset.** The draw is seeded and reproducible but only one was taken, so
nothing here bounds the variance of 0.855. Read the ordering, not the third decimal.

The first run of this died during the fourth preset after an hour and lost the three already
finished, because the CSV was written after the loop. It saves after every preset now: a long
experiment that keeps nothing until it finishes is one crash away from having done nothing.

- 2026-09-10 | WP3-T6 augmentation across cameras | `python experiments/augmentation_transfer.py --views 4` | `augmentation_transfer.csv` | `light` 0.8550 vs 0.4406 unaugmented and 0.3625 for the duplicate-rows control; closes 75% of the gap one labelled frame closes; `full` erases the gain entirely

- 2026-09-10 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 4 gate(s) met on artefacts, 3 waiting on a person

---

## 2026-09-10 — WP6-T2: the pipeline had no head on it

`src/pitch_occupancy/vision/classifier.py`, a runnable `python -m pitch_occupancy.worker`,
14 new tests - 10 on the classifier, 4 on the entry point.

`run_slot` has taken a `classify` callable since it was written, and `run_due` passes one
straight through. **Nothing outside a test ever supplied one.** Every scheduler test hands
the seam a stub, `end_to_end_slots.py` rebuilds each slot's per-minute sequence from the
*labels* rather than from a model, and `worker.main` raised `NotImplementedError` whose
message named exactly this. So sampling, two-camera fusion, aggregation, evidence selection
and reconciliation had all been exercised end to end without a single frame ever having been
classified by the system itself.

Nothing was broken and no test failed. The pipeline simply had no head on it, and the
missing piece was small enough to keep not noticing — which is the same shape as the RTSP
source that could not get past its first line, and the evidence images that recorded three
paths to nothing.

### The input path is the whole risk

A deployed classifier that embeds frames even slightly differently from the way the feature
cache was built is a model evaluated on one distribution and run on another, and this project
has already been bitten by that once: every published number comes from raw frames handed to
the HF processor, which keeps roughly the middle half of a 16:9 pitch, and letterboxing
instead moved ConvNeXtV2's false-play rate from 99.2% to 2.1%.

So the module implements no preprocessing of its own. It converts BGR to RGB, wraps the array
in a PIL image and calls the same `embed_batch` under the same processor geometry that
`feature_cache.py` uses. The check is against the cache itself rather than against a
description of it: a test embeds real frames through the deployment path — OpenCV decode, BGR
to RGB — and compares them with the stored vectors. The two decoders agree **exactly** (max
absolute difference 0.0 across the sampled frames), which is the assumption everything else
rests on and the one thing nothing else in the repository would notice breaking.

### Two decisions worth stating

**Fitted at construction, not loaded from a pickle.** A serialised sklearn pipeline is a
second artefact that can drift from the manifest it claims to come from, breaks on a library
upgrade, and that nothing here regenerates — three failure modes bought for the sake of
saving a second. Refitting from the cache means the deployed model is always the one the
current cache and manifest imply.

**Development rows only.** The probe is fitted on 1,578 frames and the locked venues are not
among them, so a deployment cannot quietly train on the held-out set and make the one honest
number in the thesis unquotable. There is a second explicit check behind `development_rows`
that fires only if that function ever changes.

The confidence is the winning class probability and is **not calibrated**. `fuse` uses it to
weigh two cameras against each other and `aggregate_slot` never reads it, so nothing turns it
into a verdict on its own; anything that wants a threshold on it has to go through
`evaluation/calibration.py` and say which temperature it fitted.

### Running it found three defects, which is why it was run

`run_due` hands `on_slot` **the exception** when a slot fails — one dead camera must not stop
the other pitches being observed. The first version of `main` printed `run.verdict`
unconditionally and so died inside the handler for the failure it was there to report. It
fired on the first real run, because the schedule derived from the recordings names four slot
instances and only two were ever exported. Two slots whose windows overlap are both reached
by the first `run_due` call, so the second call would have run the pair again and written one
slot's verdict twice.

The third is the one worth keeping. **`run_due` persists only when it is given a
connection**, and `main` did not give it one — so the first version classified both slots in
full, stored nothing, and printed *"2 slot(s) written to the database"*. It was caught by
looking in the database rather than by reading the output: the two rows there were dated
2026-09-06 and had come from `pitch seed`. A defaulted parameter whose absence silently
disables the write is the same shape as the evidence images that recorded three paths to
nothing, and the failure is not a crash but a confident sentence about something that did not
happen. All three are fixed and pinned by tests.

### The acceptance criterion, with a model in the loop

WP6-T2's original criterion was *"runs continuously; DB fills; verdicts correct on recorded
slots"*. Both exported slots now agree with the label-derived verdicts in
`end_to_end_slots.csv`, and the rows in the database were written by this run rather than by
`pitch seed`:

| slot | model | labels |
|---|---|---|
| `venue_01_2026-07-11_1000` | NOTUSED, play 0.018 empty 0.947, 57/57 minutes | NOTUSED, play 0.017 empty 0.932 |
| `venue_01_2026-07-12_2030` | USED, play 1.000 empty 0.000, 59/59 minutes | USED, play 1.0 empty 0.0 |

The ratios differ slightly because the counts do: the video source yields 57 readable minutes
of the first slot where the manifest holds 59 labelled frames of it. The two remaining slot
instances are reported as SKIPPED — the schedule derived from the recordings names four and
only two were ever exported — which is the "one dead camera does not stop the other pitches"
guard doing its job on real input.

**That the model wrote them is checkable rather than asserted.** The seeded rows carried a
constant `mean_confidence` of 0.95; these carry 0.9932 and 0.9999, and the 469 frame samples
behind them hold 233 distinct confidences.

**This is a wiring check and not an accuracy result.** Both slots' frames are in the probe's
training set, so what it demonstrates is that the pipeline is connected end to end — not
anything about generalisation, which is what WP4 measures honestly and on held-out venues. It
is still worth having: until today the only thing that had ever produced a per-minute class
for these slots was the label column.

### The comparison is now an artefact, not a paragraph

`experiments/end_to_end_model.py` -> `results/end_to_end_model_slots.csv`, a `reproduce_all`
stage, 8 tests. M5's *"end-to-end run on real slots"* has been satisfied by
`end_to_end_slots.csv`, whose per-minute sequences are rebuilt from the **label column** — a
criterion that reads as though a model were involved and was true of nothing in the
repository. The database rows this run writes are `.gitignore`d, so without a committed table
the distinction would stay a claim in a log entry.

| slot | model | labels | minutes agreeing |
|---|---|---|---|
| `venue_01_2026-07-11_1000` | NOTUSED, play 0.018 empty 0.947 | NOTUSED, play 0.017 empty 0.932 | 53/53 |
| `venue_01_2026-07-12_2030` | USED, play 1.000 empty 0.000 | USED, play 1.000 empty 0.000 | 53/53 |

**106 of 106 comparable minutes agree, and that number needs its two caveats attached.** Both
slots' frames are in the probe's training set, so this is in-sample and not an accuracy
measurement — WP4 measures that honestly, on held-out venues, and gets very different numbers.
And the minutes are not frame-aligned: `VideoSlotSource` reads the frame at exactly minute x
60 s while the labelled frames of that minute sit at irregular instants, so a prediction is
compared against **the labels of its minute**.

**Ten minutes were set aside because their own labels disagree** — four in the first slot, six
in the second. Two frames seconds apart caught a change, or the two cameras were labelled
differently. Such a minute cannot make a single prediction right or wrong, and folding it into
either column would be a choice dressed as a measurement, so it is counted and reported
separately. It is also a small, real observation about label granularity: the taxonomy is
per-frame and the decision layer is per-minute, and those disagree about 8% of the time here.

Still not a deployment. The live path has never been pointed at a camera, and there is no
supervision or restart policy (WP7-T2, WP7-T3).

- 2026-09-10 | WP6-T2 production classifier | `python -m pitch_occupancy.worker --source video` | `vision/classifier.py` | 14 new tests; the `classify` seam had never been given a real classifier, so the whole pipeline had been run end to end without a frame ever being classified. Both recorded slots now agree with the label-derived verdicts; the deployment input path reproduces the cached features exactly

- 2026-09-10 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 32 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-10 | WP3-T6 augmentation across cameras | `python experiments/augmentation_transfer.py --views 4 --seeds 13,99,123` | `augmentation_transfer.csv` | best preset light 0.4635 (5 draws, 0.3479-0.8550) vs 0.4406 unaugmented and 0.3625 for the duplicate-rows control; closes 4% of the gap one labelled frame closes

---

## 2026-09-10 — WP3-T6 RETRACTED: the augmentation result was a draw, not an effect

`experiments/augmentation_transfer.py --seeds 42,7,13,99,123 --presets light`,
`augmentation_transfer_spread.csv`. **This retracts the headline of the WP3-T6 entry above,
written earlier the same day.**

That entry reported `light` — brightness, gamma and sensor noise — taking empty-pitch recall
on an unseen camera from 0.000 to **0.687** with no target labels, at 0.8550 macro-F1 against
0.4406 unaugmented, and called it "the one preset that works". It also said, in its own last
paragraph, that only one draw had been taken and nothing bounded the variance. That paragraph
was the whole finding, and it was filed as a caveat.

| draw (seed) | macro-F1 | EMPTY recall | frames called EMPTY (of 521) |
|---|---|---|---|
| 99 | 0.3479 | 0.000 | 0 |
| 7 | 0.3501 | 0.000 | 0 |
| 123 | 0.3510 | 0.000 | 0 |
| 13 | 0.4136 | 0.062 | 15 |
| **42** *(the published draw)* | **0.8550** | **0.687** | — |
| *no augmentation* | *0.4406* | *0.000* | *0* |

Mean 0.4635, sd **0.2206**, median **0.3510**. The published number is the **maximum of
five**, and four of the five are below the unaugmented baseline.

**Nothing differs between those rows but the random draw.** Same 775 source frames, same
preset, same probe seed, same test set, same 3,875 training rows. `light` applies each of its
three effects with probability 0.5, so a draw is 3,100 views' worth of coin flips whose
aggregate distribution is near-identical between seeds. The metric is not: the range is 0.3479-0.8550 with a standard deviation of 0.2206, on a metric bounded in [0, 1].

**The mechanism is a boundary that either reaches camera B or does not.** `pred_empty` counts
how many of camera B's 521 frames the probe called EMPTY: three of the five draws call **none** of them empty, one calls 15, and the published draw found the boundary. Camera A's
EMPTY frames are one morning at one venue, so the fitted boundary for EMPTY sits close to
camera B's empty pitch without being anchored by anything from it; which side it lands on is
decided by where the random brightness and gamma shifts happened to fall. That is why the
result does not degrade gracefully across draws.

**And it explains the numbers that repeat exactly.** `colour` scored 0.3479 on seed 42 and
0.3479 again on seed 7 — a suspicious agreement that turns out to be the tell. A preset
scoring 0.3479 has predicted ACTIVE_PLAY for all 521 frames; 0.3479 is the macro-F1 of the
trivial one-class predictor on this test set, not a measurement of colour jitter. The
`pred_empty` column was added for this run precisely because the repeated value looked wrong,
and it is what made the collapse legible.

### What survives

- **The duplicate-rows control still did its job.** It moved *down* from the baseline, so the
  seed-42 gain was never attributable to row count. That reasoning was sound; it was
  answering a question about the wrong thing.
- **Augmentation can do this, on this boundary.** One draw reached 0.687 empty recall from no
  target labels at all, and that is not noise — the four remaining draws are the reason it
  cannot be quoted as a method.
- **The five-frame recipe is unaffected**, and its case is now stronger: 0.9895 every time,
  against a recipe that works occasionally and gives no warning which run you are in.

### What this says about the rest of the project

The check that found this is the cheapest one available — run it again with a different seed
— and it was not run because the draw was a *means to an end* rather than the object of
study. Where the draw is the object, this project already replicates: the benchmark takes
five split replicates per protocol, the label-efficiency curve five seeds per training size,
the onboarding curve reports `n_seeds` and a min–max. Exactly two results consumed a random
draw once. This was one. **The other is STAN**, whose composed train and test sequences come
from a single seeded draw (`stan_preliminary.py`, `SEED` and `SEED + 1`) — already reported
as preliminary for a different reason, and now carrying this one too. That stage runs in
three minutes; five draws of it is a quarter of an hour and is on the list.

A bootstrap interval over a test set does not cover this. Every interval in this project
resamples the *observations*; none of them resample the construction — which split, which
augmentation draw, which synthetic sequence. Those are separate sources of variance and only
the first is reported.

**Two tests pinned the retracted claim** and neither could have caught it: both read the
seed-42 row, which is still exactly what it was. A test that fixes a value can only detect a
change in that value, and nothing had changed. They now pin the spread instead — including
one that fails if the range ever narrows enough to make the original claim quotable again,
so an un-retraction has to be a deliberate act.

- 2026-09-10 | WP3-T6 RETRACTION: augmentation draws | `python experiments/augmentation_transfer.py --seeds 42,7,13,99,123 --presets light` | `augmentation_transfer_spread.csv` | the 0.8550 headline is the maximum of five draws (median 0.3510, sd 0.2206); four of five fall below the 0.4406 unaugmented baseline and three are the trivial one-class predictor. The effect was a draw

- 2026-09-10 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 34 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-10 | WP6-T2 end-to-end with the model | `python -m experiments.end_to_end_model` | `end_to_end_model_slots.csv` | 2/2 slot verdicts agree with the label-derived ones and 106/106 comparable minutes agree; in-sample, so a wiring check rather than an accuracy result

- 2026-09-10 | WP5-T1 STAN, preliminary | `python experiments/stan_preliminary.py` | `stan_preliminary.csv` | composed test: stan 1.0000 vs best baseline hmm 0.8950; 2 real slots, below the 30-slot gate

---

## 2026-09-10 — WP5-T1: the same check on STAN, and this one mostly survives

`python experiments/stan_preliminary.py --seeds 42,7,13,99,123` → `stan_draw_spread.csv`,
4 new tests.

The augmentation retraction earlier today identified exactly two results in this project that
consumed a random draw as a means to an end and took it once. That was one. This is the
other: `stan_preliminary.py` splits the frame pool and composes both slot sets from a single
seed, and every number in the WP5-T1 entry rests on that one construction. Rather than note
the risk and move on — which is precisely what the augmentation entry did, in its last
paragraph, hours before it was retracted — the draws were run.

The seed moves the pool split and both compositions. The probe's own seed does not, so a
difference between draws is the construction and cannot be the fit.

| model | mean | sd | min | max |
|---|---|---|---|---|
| tuned_thresholds | 0.7990 | 0.0334 | 0.7500 | 0.8350 |
| median_smoothing | 0.8250 | 0.0354 | 0.7800 | 0.8700 |
| hmm | 0.8260 | 0.0446 | 0.7850 | 0.8950 |
| summary_logistic | 0.8420 | 0.0863 | 0.7350 | 0.9550 |
| **stan** | **0.9420** | 0.0884 | 0.8000 | 1.0000 |

**The ordering survives: STAN is first in five draws of five**, and no baseline matches it in
any of them. That is a stronger statement than the original entry could make, and it is the
part of WP5-T1 worth carrying forward — still preliminary by the WP5-T8 gate, which is about
the two real slots and is untouched by any of this.

**The ceiling does not survive.** STAN scores 1.0000 in three draws and 0.9100 and 0.8000 in
the other two. The published entry read that 1.0000 as *"this test set is exhausted"* — a
claim about the benchmark's design. It is a claim about particular compositions: the composed
set **can** be exhausted, often enough that one draw is likely to show it, which is a
different and weaker statement. `threats_to_validity.md` §1.3 and the defence deck's slide 13
both said the stronger one and now say this.

**And the baselines move more than the table suggested.** `summary_logistic` runs
0.7350–0.9550 (sd 0.0863) and is the best baseline in one draw and the worst in another. So
the published *"beats a tuned HMM by 0.105"* is one draw's margin between two quantities that
each move by more than a tenth. The mean gap over five draws is +0.116, which is the number
with a spread behind it.

**Two replications, two different outcomes, and that is the point.** The augmentation one
destroyed its headline; this one confirmed the direction and corrected the overstatement
built on top of it. Neither result could have been told from the other in advance, and
neither was visible from a single draw — which is the argument for the check rather than for
any particular expectation about what it will find.

- 2026-09-10 | WP5-T1 STAN construction draws | `python experiments/stan_preliminary.py --seeds 42,7,13,99,123` | `stan_draw_spread.csv` | stan first in 5 of 5 draws (mean 0.9420, sd 0.0884) - the ordering replicates; the 1.0000 does not, holding in 3 of 5, so "the benchmark is saturated" is a property of the draw

---

## 2026-09-10 — M5's end-to-end criterion was true of the decision layer, not of any model

`experiments/gate_check.py`, 1 new test.

M5 reads *"end-to-end run on real slots"*, and it passed on
`results/end_to_end_slots.csv` — a table whose per-minute sequences are rebuilt from the
manifest's **label column**. That was not a shortcut anyone took: for most of this project
nothing could produce them any other way, because `run_slot` accepted a `classify` callable
and nothing outside a test ever supplied one. The criterion described the decision layer and
read as though it described the system.

It now reads `end_to_end_model_slots.csv` — the same two slots through the deployed
classifier — and checks the verdicts **agree with the label-derived ones** rather than that a
file exists, because a run that classified both slots wrongly would satisfy the second
exactly as well as the first. It reports the minute agreement alongside, marked in-sample.

Both halves are tested by breaking them, which is what the M4 criteria were strengthened to
do in September: flip a verdict in the CSV and the criterion must fail while the file sits
there; hide the file and it must name the command to run rather than quietly falling back to
the label-derived table. A gate criterion that degrades to a weaker artefact when its own is
missing is worse than one that fails, because it goes on reporting *met*.

M5 still passes, on stronger evidence than before.

- 2026-09-10 | M5 criterion strengthened | `python -m experiments.gate_check` | `gate_status.md` | the end-to-end criterion read a label-derived table; it now reads the model's own and checks the verdicts agree, tested by breaking it both ways

---

## 2026-09-10 — WP0-T4: a split is identified by its rows, not by its seed

`src/pitch_occupancy/data/splits.py`, 8 new tests.

The module's first documented rule was *"splits are materialised, referenced by name, never
re-randomised"*. They are not. `write_split` and `read_split` have no callers outside the
tests, `results/splits/` holds only the lock file, and every experiment calls
`grouped_split(..., seed=42)` directly — which is deterministic *given the same rows*, and
the row list is not something the seed determines. It depends on which feature caches the
calling script filtered to. In September `effective_sample_audit.py` read the DINOv2 cache
and counted 94 distinct scenes while `h4_model_equivalence.py` read all three and counted 95.
Both were correct. The seed was identical.

**Run today, the two agree** — identity `f73f5a29b425` from both. The reproducibility repair
of 2026-09-08, which found `grouped_split` building its test side from a set, brought their
row lists back together. So the drift is historical rather than live, and that is the sharper
version of the problem: **nothing recorded that they had diverged, and nothing recorded that
they had converged either.** The defect was never a wrong number. It was that no artefact
could have told you which split a number came from.

**The decision, and it is a decision rather than a task.** The plan intended a retrofit:
materialise the canonical splits once and have every experiment `read_split` them. That was
not taken, and the reason belongs in the record rather than being left as an omission —
materialising the canonical splits *now* would change which rows several already-published
experiments were fitted on. It would invalidate results in order to protect them, in write-up
week. So the claim was dropped instead of being made true.

**What replaces it.** A split's identity is the strategy, the group key, the seed **and the
ordered row list**, and `split_identity()` returns a fingerprint of exactly that. It detects
disagreement; it does not prevent it, and the docstring says so. Prevention is the retrofit,
which stays available and works.

The tests pin the two cases a seed cannot: one row fewer at the same seed gives a different
identity, and so does the same membership in a different order — the second being the failure
that made every bootstrap interval on a grouped split a different draw in September, while
every point estimate reproduced exactly.

**And the materialisation path now verifies itself.** `write_split` records a partition digest
and `read_split` checks it. A materialised split is a CSV and a CSV is editable: flipping one
row from test to train produces a file that reads back cleanly and quietly describes a
different experiment. The digest deliberately excludes the *name*, so copying a split to a new
filename stays legal, and a file written before the column existed carries no digest and reads
as it always did — a missing digest is a real state, not a corruption.

**And it is quoted where it matters, because a fingerprint nobody writes down detects
nothing.** `effective_sample_audit.py` and `h4_model_equivalence.py` — the two scripts that
disagreed — now record their split identity in their CSVs, and a test compares the two files.
It is a cross-script check that neither script could make alone, and it fails the moment they
drift apart again, which is what the twenty-script retrofit would have prevented at twenty
times the cost.

Adding it exposed one more thing worth writing down: `h4_model_equivalence.py` writes with
`extrasaction="ignore"` and an explicit column list, so the new field was attached to every
record and reached the file on none of them. The test skipped rather than passing, which is
the only reason it was noticed — a column silently dropped by a writer is indistinguishable
from a column never added.

- 2026-09-10 | WP0-T4 split identity | `split_identity()`, `write_split`/`read_split` digest | `src/pitch_occupancy/data/splits.py` | the "materialised, never re-randomised" claim was not practised and the retrofit would have changed published rows; the claim was dropped, the row list made part of the split's identity, and the materialisation path now verifies its own partition on read

- 2026-09-10 | H4 model equivalence | `python experiments/h4_model_equivalence.py` | `h4_model_equivalence.csv` | ConvNeXtV2 vs ViT: equivalent at margin 0.02

---

## 2026-09-10 — WP2-T4: the two label errors, verified and costed rather than fixed

`thesis/threats_to_validity.md` §2.5, `thesis/labelling_protocol.md`, TODO WP2-T4.

Two human-labelled frames in `2_playing/` have been on the open list since September as
"verified label errors". They were re-verified by eye today, and the destination was checked
against the protocol rather than assumed: the playing surface is empty in both, and the only
people are off-pitch by the sideline shelter behind the barrier. §2.1 says people outside the
pitch do not count and §2.3 defines EMPTY as no people *within the ROI*, so the correct folder
is `1_empty` — C1, not `3_people_not_playing`. The error is real and the target is not
ambiguous.

**They stay as they are, and the reason is now measured rather than asserted.** The pixels are
not the expensive part: the images do not change, so the cached vectors could be re-keyed
rather than recomputed. It is the two *labels* that move everything downstream. Counted from
the pipeline's own stage table, 24 stages read a feature cache — **203 minutes** — and
**32 of the 34 ledger claims** derive from one and would move, each of them also quoted in
prose somewhere. Half a day, most of it re-checking sentences.

Against that: the two frames are 2 of only 6 daytime ACTIVE_PLAY frames at `venue_01`, so
correcting them makes the day/night confound **more** absolute, not less. The correction buys
no scientific gain and moves 32 numbers by amounts nobody would notice.

So the record is corrected instead of the data. §2.5 reports them as a **0.12% label-noise
floor** — and says plainly that it is a floor and not a rate, because it counts the errors that
were found and the search was not systematic: the 396 clip frames carry `labeled_by=bulk`,
verified with a detector plus review of the outliers rather than one by one, and that
spot-check is still open. Naming two known errors and how they were found is a better position
than quietly fixing them and being unable to say how many remain.

The option stays open and is now cheap to take, because the cost is written down.

- 2026-09-10 | WP2-T4 label errors | visual re-verification + `reproduce_all` stage costs | `threats_to_validity.md` §2.5 | both confirmed empty and `1_empty` confirmed as the destination; correcting them costs 203 min of recompute and moves 32 of 34 claims for no scientific gain, so they are reported as a 0.12% label-noise floor instead

- 2026-09-10 | efficiency | `python experiments/efficiency_latency.py` | `efficiency_latency.csv` | dev laptop, 4 threads

---

## 2026-09-10 — WP0-T10: the latency table was measured under load, and the bias has a sign

`python experiments/efficiency_latency.py` on an idle machine, `efficiency_latency.csv`,
`h4_model_equivalence.csv`, pre-registration amendment A13.

`efficiency_latency.py`'s docstring has said since September that ConvNeXtV2 read **150.9 ms**
in a batch against **101.2 ms** idle, that this moved H4's speed ratio, and that
`reproduce_all.py` therefore leaves the stage out of `--force`. All true, all written down —
and the stage was never re-run. `results/efficiency_latency.csv` has one commit in its
history, and the value in it is 150.9. **The number the warning describes is the number that
was published**, which is the same shape as every other finding in this project: not a missing
guard, a guard that was written and then not acted on.

Re-measured with nothing else running, on the same hardware and thread count:

| backbone | published | idle | concurrent x20, published | idle |
|---|---|---|---|---|
| ConvNeXtV2 | 150.9 ms | **100.5 ms** | 2388.1 ms | **1858.3 ms** |
| DINOv2 | 418.3 ms | **219.4 ms** | 5501.8 ms | **4021.5 ms** |
| ViT | 303.3 ms | **169.4 ms** | 4361.5 ms | **3023.1 ms** |

100.5 against the docstring's remembered 101.2 — the diagnosis was exactly right.

**The interesting part is the direction.** Contention costs the heavier model more, so load
does not merely add noise to a speed comparison: it **inflates the ratio between models of
different weight**. ViT slowed by 1.79x under load where ConvNeXtV2 slowed by 1.50x, so the
published ratio was biased *towards the compact model* — the direction that would have
supported shipping ConvNeXtV2 — and it was enough to carry H4's single-frame line over a 2x
bar it does not clear. Idle, ConvNeXtV2 is **1.69x** faster than ViT on single frames and
**1.63x** under a 20-camera load, against the >= 2x the hypothesis requires: refuted on both
readings, where A9 could refute it on only one.

**Nothing about the recommendation changes, and one thing gets stronger.** The 60-second-cycle
headroom improves from 10–24x to **14–31x**, so "latency is not the binding constraint" holds
with more room than was claimed, and RQ2's choice of DINOv2 continues to rest on cross-venue
accuracy. `README.md`, `rq_matrix.md`, `threats_to_validity.md` and `TODO.md` carry the
corrected figures; A9 stands as written with A13 beside it.

**Measured on an AMD development laptop, not the Mini-PC.** WP7-T1 is still open, and every
number in this table is the wrong machine — which is a separate problem from having been the
wrong conditions.

- 2026-09-10 | WP0-T10 latency re-measured idle | `python experiments/efficiency_latency.py` | `efficiency_latency.csv` | the published table was taken under load, as its own docstring warned: ConvNeXtV2 150.9 -> 100.5 ms, DINOv2 418.3 -> 219.4, ViT 303.3 -> 169.4. Contention inflates ratios between models of different weight, so H4's speed clause now fails on both readings (1.69x, 1.63x) and cycle headroom rises to 14-31x

---

## 2026-09-10 — WP2-T11: the starved class stays, and the recommendation was wrong

`thesis/protocol.md`, from `results/empty_recognition.csv` (measured 2026-09-09).

C3 holds six frames from one moment on one camera. The fallback tree written in week 3
offered three ways out, and the risk register had recommended **Option C** — report the
benchmark as 2-class and state the scope reduction plainly. The decision was overdue by
thirteen weeks, which is its own small finding: an item that says *"decide by week 6 — do not
drift past it"* drifted, and nothing in the repository could notice, because a decision is not
an artefact.

**Taken now, and the measurement refuses the recommended option.** On the 243 held-out empty
frames of the camera-transfer control:

| configuration | DINOv2 false-play rate | absorbed by C3 |
|---|---|---|
| 3-class, balanced | **0.309** | 0.691 |
| 3-class, unbalanced | 0.881 | 0.119 |
| **2-class (Option C)** | **1.000** | — |

Dropping C3 does not leave the false-play axis unchanged, which is how the earlier note put
it. It takes the production model from calling 31% of unfamiliar empty pitches *play* to
calling **all of them** play. Six frames are absorbing 69% of what the model cannot place.

**The reason is operational, and that is what makes it worth keeping.** `empty_accuracy` is
0.000 in every configuration — no arrangement of classes makes the model recognise an empty
pitch on this transfer set. What changes is **where the errors land**. C3 maps to
NOTUSED/REVIEW and ACTIVE_PLAY maps to USED, so a misplaced empty frame is either a slot sent
to a human or a slot billed as used. The starved class converts 69% of would-be **billing
errors** into correct-or-reviewable verdicts. It earns its place as a *none of the above*
sink — which is not the job it was defined to do, and is a better reason than the one it was
created for.

**So the taxonomy is unchanged and the claim is what shrinks.** Three classes stay; C3 is not
claimable — six frames from one moment support no per-class metric, and no leakage-free split
puts it on both sides. Every table reporting macro over three classes must name C3's support
beside it, and no sentence may say the system detects maintenance. It has a sink, and the sink
is load-bearing.

Option A (compositing hi-vis workers) stays rejected on the 2026-09-09 evidence. Option B (an
open-vocabulary detector as an explicit C3 branch) is now *better* motivated than when it was
written — the sink works, and a detector would make it deliberate rather than incidental — but
it is a new module in write-up week and nothing rests on it.

- 2026-09-10 | WP2-T11 C3 decision | `thesis/protocol.md` | `empty_recognition.csv` | the recommended Option C is refuted for the production model: 2-class takes DINOv2 from 0.309 false-play on held-out empty frames to 1.000. C3 stays as a load-bearing "none of the above" sink that converts 69% of would-be billing errors into NOTUSED/REVIEW; the class is kept and the claim dropped

- 2026-09-10 | WP5-T2 gated fusion head, ablated | `python experiments/fusion_head_ablation.py` | `fusion_head_ablation.csv` | routing worth -0.0238 recall, p=1.000 (floor 1.000); no model exceeds 0.1646 EMPTY accuracy on the held-out camera

---

## 2026-09-10 — WP4-T4b: the one report that quoted p-values without its own standing rule

`experiments/fusion_head_ablation.py` → `fusion_head_comparisons.csv`.

WP0-T6 says Holm-corrected families and an effect size beside every p-value, and the gate
check confirms "Holm-corrected families in 3 reports". Auditing every results CSV for the
four columns found the exception: **`fusion_head_comparisons.csv` shipped five paired
sign-flip tests with raw p-values, no correction and no effect size** — the WP5-T2 ablation,
which is an M4 gate artefact.

Now corrected, and the correction changes nothing, which is the honest headline:

| comparison | p | p_holm | d |
|---|---|---|---|
| routing: mlp vs constant gate | 1.0000 | 1.0000 | −0.378 |
| learned mixing: constant vs uniform | 0.5000 | 1.0000 | −0.483 |
| gate vs mean-probability ensemble | 0.5000 | 1.0000 | −0.387 |
| confound: stats gate vs lighting gate | 0.5000 | 1.0000 | +0.416 |
| trainer control: torch vs published probe | 1.0000 | 1.0000 | +0.330 |

Nothing was significant uncorrected, so nothing could become significant corrected. The run
says exactly that rather than leaving a reader to work it out — the reason to apply the
correction anyway is that *five comparisons, one of them at p = 0.125* is a family whether or
not it is declared one, and a reader cannot tell which unless the report says so. The family
and its size are now printed with the source they were declared from.

**The effect sizes earn their place on the first run.** The lighting-gate confound's mean
delta of **+0.1015** reads as a different order of thing from routing's **−0.0238** — one
looks like a finding and the other like nothing. Standardised over the seven folds they are
**+0.416** and **−0.378**: the same size of effect, and both are differences this design
cannot separate from fold-to-fold variation. A mean delta over seven folds this wide carries
no sense of how wide they were, which is the whole argument for the rule.

- 2026-09-10 | WP4-T4b Holm and effect sizes | `python -m experiments.fusion_head_ablation` | `fusion_head_comparisons.csv` | the one report quoting p-values without the project's own standing rule; Holm and Cohen's d added, no conclusion changes, and the +0.1015 confound delta turns out to be d=+0.416 against routing's d=-0.378

---

## 2026-09-10 — the derived artefacts the latency correction left behind

`results/figs/accuracy_vs_latency.png`, `results/project_site.html`, and a stale claim in
`thesis/mvt.md` and the risk register.

Re-measuring the latency table moved three numbers. Three things downstream had already been
built from the old ones and did not notice, because **`reproduce_all --check` tests whether an
output exists, not whether it is older than its input.** Every stage read "done".

- `figs/accuracy_vs_latency.png` plotted the contended concurrent times. Regenerated: DINOv2
  4.1 s, ViT 3.1 s, ConvNeXtV2 1.9 s, against the 5.5/4.4/2.5 it was drawn with. Its argument
  is unchanged and slightly stronger — the 60-second budget line is further away than the
  figure claimed. Only this PNG changed; the other seven are byte-identical, which is the
  check that says the latency table was the only input that moved.
- `results/project_site.html` carried 418/151/303 ms in its model table and now carries
  219/100/169.
- `thesis/mvt.md` still headed its labelling-protocol section *"The one thing the floor does
  require that is not yet done"* while its own body said the document was drafted on
  2026-09-07, and the risk register still listed *"the labelling protocol stays unwritten"* as
  live. It is 219 lines long and the M1 gate checks it. The residual risk is **sign-off**, not
  writing, and both now say so.

**The useful part is the shape, not the three files.** A freshness check is a different thing
from an existence check, and this pipeline only has the second. Nothing here was wrong when it
was written; it went stale silently, which is the same failure as the latency table itself —
a warning recorded and then not acted on — one level up.

- 2026-09-10 | derived artefacts refreshed | `make_figures`, `make_site` | `figs/accuracy_vs_latency.png`, `project_site.html` | the latency correction left three downstream artefacts stale and `reproduce_all --check` could not see it: it tests existence, not freshness

---

## 2026-09-10 — a freshness check for the reproduction pipeline, and what it found first

`experiments/reproduce_all.py` — `Stage.stale_inputs()` and a `--check` section.

Re-measuring the latency table left three artefacts built from the old numbers, and
`--check` reported every stage "done" throughout, because **it tests whether an output exists,
not whether it is still the output of its inputs.** That gap is now reported.

**It was written twice, and the first version was wrong in an instructive way.** The obvious
implementation compares modification times. Run against this repository it reported four stale
stages and **every one was a false positive**: `h3_cross_venue_recall.csv` had been rewritten
byte-for-byte by a rerun on 2026-09-08 and has not changed *content* since 2026-09-06, so
everything downstream of it looked stale. A check that cries wolf on its first contact with
the data is worse than no check, because the next person learns to skim past it.

The second version reads **git**, which commits only when content changed — a content check
this project already keeps. An uncommitted modification counts as "changed now"; anything git
does not track returns None and is skipped, so a stage reading only the gitignored `data/` is
never called stale. It is advisory, printed beside "done", and never used to decide what to
run: an output can also be legitimately unchanged because the input's change did not reach it,
and a build system that guessed would be worse than one that asks.

### What it found on its first run

**`false_play_rescored.csv` covered 52 of the search's 88 evaluations.** `preprocess_search.json`
was committed thirteen hours after it, so the rescoring had been run against an earlier search
and nothing said so. Re-run: **52 → 88 rows, and not one of the 52 moved** — identical
`play_recall`, `worst_fold` and `false_play_fixed` throughout. The table was incomplete, not
wrong, and the sentence claiming "all 52 repaired false-play rates" was describing 59% of the
search.

**And a caveat that fell out of it.** `false_play_old` is meant to hold what the *broken*
control returned — 0.0000 for everything — but it reads `false_play` from the search JSON, and
`preprocess_search.py` repairs cached entries **in place** when it encounters them. Seven
DINOv2 rows now carry the repaired value in the "old" column, so the before/after contrast the
table exists to show is eroding as the JSON is touched. The original zeros cannot be recovered;
the 2026-09-08 entry above is now the only record of what the broken control returned.

That is a second instance of the same shape as the latency table: an artefact whose meaning
depends on when it was built, with nothing recording when that was.

**And a third revision, from watching it run.** The first content-based version compared each
input against the stage's *oldest* output, which flagged `figures` permanently: it draws eight
PNGs, the latency correction moved exactly one, and the other seven are correct precisely
because they did not need to change. A stage writes all of its outputs in one run, so the
**newest** is the best available answer to "when did this last produce something". With that,
the report is clean — no standing false flag, and the one real finding fixed.

- 2026-09-10 | reproduction freshness check | `python experiments/reproduce_all.py --check` | `experiments/reproduce_all.py` | existence is not freshness; the mtime version gave 4/4 false positives so it reads git instead. First true positive: `false_play_rescored.csv` covered 52 of 88 search evaluations, now 88 with none of the 52 moved

---

## 2026-09-11 — the redaction that was mandatory in one script and absent in the other

`experiments/augmentation_grid.py`, `thesis/ethics.md`, `README.md`, 2 new tests.

`thesis/ethics.md` commits to blurring faces in any published figure, and
`vision/explain.py` states that its redaction "is not optional and not a flag". Both were
true of `make_xai_figures.py`, which pixelates every detected person before drawing and says
so. Neither was true of `augmentation_grid.py`, which reads two real frames, draws them into
a sheet with fifteen augmented copies, **commits it to git and serves it on the thesis site** —
and never called the redaction. The first tile is a night match with six players in it.

Now it redacts, and the run reports the count rather than the fact:

    night · active play          6 person box(es)
    day · empty                  0 person box(es)  <- none found; that is not the same as none present

The second line is the part worth keeping. `redact_people` returns how many boxes it found so
a caller can say *that*, instead of printing "redacted" and implying a frame was examined and
found empty. The script also **refuses to write the sheet** if the detector will not load —
`redact_people` reports `-1` for that case precisely so a missing weight cannot produce
unredacted output that looks checked.

The augmentation is applied to the already-redacted frame, so what the sheet shows is what a
reader can verify, and the figure still does its job: people remain visible as people in every
night draw, which is the thing the sheet exists to let you check.

**The guard is now a test rather than a habit.** `tests/test_explain.py` finds every
experiment that reads a frame and writes an image, and fails if any of them lacks a redaction
call. Nothing downstream can tell whether a committed JPEG was redacted or merely small
enough that nobody looked closely.

### What this does not fix, and the document now says so

`results/figs/venue_check/` holds five audit sheets of operator footage with visible players,
committed in September. `api/app.py` already refuses to serve that directory and explains why
— but a guard at the HTTP layer is the weaker one, because a repository is handed over whole.
Removing them means rewriting published history across a hundred branches. That is a decision
to be taken deliberately, not one to slip into a commit about figures, so `ethics.md` and the
README now name the exception instead of claiming a blanket that was not true.

- 2026-09-11 | figure redaction | `python -m experiments.augmentation_grid` | `figs/augmentation_grid.jpg` | the ethics commitment was enforced in one figure script and absent from the other, which had published a night match with six unpixelated players; now redacted, counted, refused if the detector is missing, and pinned by a test over every frame-publishing script

---

## 2026-09-11 — a stale booking export would have accused a facility of nothing it did

`src/pitch_occupancy/bookings.py`, `slots/reconcile.py`, runbook row 8, 9 new tests.

Row 8 of `docs/runbook.md` has read **"not implemented"** since the runbook was written: *the
booking export is stale or absent → reconciliation has nothing to compare against*. That
description is too kind. Reconciliation does not lose the comparison when the export stops
short — it makes the comparison anyway and gets it confidently wrong. Every observed slot past
the export's last day matches no booking, and a slot with no booking that shows play is
`UNBOOKED_USAGE`, which the matrix rates **SERIOUS**. A month-old export therefore hands an
operator a page of serious anomalies against a facility that did nothing wrong — and
`authority.py` requires a human to confirm every one of them, so the cost is somebody's
afternoon and somebody else's standing.

**Now:** `bookings.covers` answers whether the export reaches a date at all, and `reconcile`
takes `records_cover_this_day`. A day outside the span returns `NEEDS_REVIEW` at **info**, with
an explanation saying an absent booking cannot be told from an absent record. The check runs
**before** anything that can return SERIOUS, which is the ordering the guard consists of.

Two things it deliberately does not do. It tests the *span*, not the booked days — a facility
with no bookings on a Tuesday still has a Tuesday inside an export that covers the week, and
treating that as uncovered would suppress the genuine unbooked-usage finding, which is the
finding reconciliation exists to make. And an **empty** export covers nothing rather than
everything, because an importer that read zero rows — wrong path, wrong delimiter — must not
produce that same page of serious anomalies.

**The parameter has a caller, because a parameter without one is the next silent failure.**
`bookings.reconcile_slot` takes the records and a slot, computes coverage itself, and returns
the reconciliation. The safe call was longer than the unsafe one, and that is exactly the
shape of thing this project keeps finding: nothing errors when a caller forgets, it just
reports SERIOUS.

### And an anomaly that cannot happen

Writing the tests turned up a second thing. **`BLOCKED_SLOT_SOLD` is unreachable from any real
export.** It fires on `maintenance_window and booked`; both are derived from one `status`
column, `booked = status in {confirmed, no_show}` and `maintenance_window = status ==
"maintenance"`. One column cannot hold two values, so a pitch that was closed for maintenance
*and sold anyway* is inexpressible in the schema WP6-T4 asks the client for. Every instance of
it in this repository is a hand-written fixture.

That is a finding about the **request**, not a defect in the code: the matrix advertises a
SERIOUS anomaly that the available data can never produce. `data_requests.md` §3 now asks for
a `blocked` flag separate from the booking status — one more column of an export the facility
already runs — and a test fails if the schema gains it while the derivation stays as it is,
which is the moment the anomaly becomes real.

- 2026-09-11 | booking export coverage | `bookings.covers`, `reconcile_slot` | runbook row 8 | a stale export made every observed slot look unbooked and unbooked usage is SERIOUS; a day outside the export now returns NEEDS_REVIEW before anything serious can fire. Separately: BLOCKED_SLOT_SOLD cannot be produced from the requested schema, so the request gained a column

---

## 2026-09-11 — a full disk, and the difference between no evidence and some evidence

`src/pitch_occupancy/retention.py`, `worker.run_slot`, runbook row 7, 4 new tests.

Row 7 read *"nothing checks free space; retention bounds growth but does not react to a full
disk"* — **partly** implemented, because `pitch retention` exists and a space guard did not.
`retention.py` bounds what this system *keeps*; it says nothing about a disk filled by
something else, and nothing looked before writing.

The failure is specific and it is not "the write fails". `_write_evidence` already catches a
failed write and returns None, so a full disk mid-slot produces a verdict backed by a
**partial** set of evidence images — frames for minutes 0 to 18 and nothing after. That is
worse than a verdict with no images at all, because the inspector cannot tell it from a slot
that was never configured to save any, and the override harvest would treat the gap as a
missing frame rather than a full disk.

So the decision is made **once, before minute zero**: `has_room` checks a 500 MB floor — a
slot's peak is about 15 MB, so this is a floor for starting, not a quota — and the slot runs
with evidence disabled and a line saying why. The verdict is produced either way. A full disk
must not cost an hour of observation.

**A disk that cannot be measured is written to.** `free_bytes` returns None rather than 0 or
infinity, because both of those are answers and this is the absence of one, and `has_room`
treats it as permission. Refusing to record a verdict because a `statvfs` failed would turn a
diagnostic problem into lost data, which is the wrong direction for a system whose whole
output is observation.

That leaves one row of the ladder at "not implemented": facility-wide confidence collapse,
which is blocked on calibration and must not be fixed by choosing a number — the same
"hyper-parameters, not constants" mistake this project has already met three times.

- 2026-09-11 | disk-space guard | `retention.has_room` in `run_slot` | runbook row 7 | a full disk mid-slot left a verdict backed by a partial set of evidence images, which an inspector cannot tell from none; the check now runs once before the first write and the verdict is produced either way

- 2026-09-10 | input ablation | `python experiments/input_ablation.py` | `input_ablation.csv` | 6 variants x 7 folds, dinov2

---

## 2026-09-11 — WP3-T8: the input ablation could not run, and its best variant was an artefact

`experiments/input_ablation.py`, `input_ablation.csv`, and corrections in six places.

Two findings, and the first is the reason the second went unnoticed.

### The stage could not run at all

`input_ablation.py` raised `TypeError: vars() argument must have __dict__ attribute` on its
first variant. `PreprocessConfig` is `@dataclass(frozen=True, slots=True)` and a slotted class
has no `__dict__`, so `vars(cfg)` cannot work — the fingerprint line has been dead since the
config gained slots. `asdict` is the fix.

**This is the second time this script has been un-runnable while its numbers were cited.**
The first is recorded in its own source: `grayscale=True` became `saturation=0.0` when the
switch was generalised into a dial, and the module-level dict raised on import. Both times
`reproduce_all --check` reported the stage **done**, because the CSV it wrote before the
breakage still existed. Existence is not health, which is the same distinction the freshness
check drew yesterday, one level further down: that check asks whether an output is older than
its inputs, and this asks whether the thing that produced it still runs.

### And the best row in the table never identifies an empty pitch

The ablation was reported as ACTIVE_PLAY recall on held-out venues **with no false-play
control** — the axis this project has twice established is gameable, because every held-out
venue is 100% active play and a constant predictor scores 1.000 there. Added, using the
camera-transfer control WP4-T13 settled on:

| variant | recall | false-play | empty accuracy |
|---|---|---|---|
| `full` | 0.960 | 0.230 | 0.770 |
| **`grayscale`** | **0.982** | **0.021** | **0.979** |
| `blur4` | 0.929 | 1.000 | 0.000 |
| `blur8` | 0.840 | 0.926 | 0.074 |
| `crop50` | **0.998** | 0.313 | **0.000** |
| `gray+crop50` | 0.899 | 0.988 | 0.012 |

**`crop50` was the best row and it has empty accuracy 0.000.** Its 0.998 recall is the
artefact the axis cannot see: the folds hold nothing it could get wrong. Read on recall alone
it says *the border was redundant*; read with the control it says the crop moved the model
toward PLAY. `blur4` is the same story more starkly — "signal survives" at 0.929 recall while
calling **every** held-out empty frame a match.

**Grayscale is the real finding, and it was previously indistinguishable from the artefact.**
It is the only removal that improves both axes: recall +0.023 *and* false-play 0.230 → 0.021
at 0.979 empty accuracy. Turf hue is a venue cue that does not transfer, and discarding it
helps the model recognise an empty pitch rather than helping it say PLAY.

So the cited reading — *"removal has a floor: grayscale helped, the crop helped more, both
together fell below baseline"* — was two-thirds artefact. One removal helps; the second only
appeared to; the combination is worse than either and calls 98.8% of empty pitches a match.
Corrected in `augment.py`, `thesis_site.py`, `diagrams.py` (the drawn figure now carries the
false-play rate in each box), `make_site.py`, `IDEAS.md` and the ablation's own docstring. The
standalone export's table gained both columns and tells the reader to read them first.

Adding rows to the CSV also broke `make_site.py`, which assumed every row carried a
`play_recall`. Fixed, and it now renders the control rather than skipping it.

- 2026-09-11 | WP3-T8 input ablation | `python -m experiments.input_ablation` | `input_ablation.csv` | the stage had been un-runnable since `PreprocessConfig` gained slots (second time it could not run while cited); with the false-play control added, `crop50`'s table-leading 0.998 recall comes with **0.000 empty accuracy** and grayscale is the only removal that improves both axes

---

## 2026-09-11 — where else the control was missing, and where it was only printed

A sweep of every committed CSV for a `recall` column without a false-play or empty-accuracy
column beside it, prompted by the input ablation turning out to have led its table with a
variant that never identifies an empty pitch.

Five artefacts came back. Three were fine on inspection:

* `augmentation_transfer*.csv` report `empty_recall` directly — the test set is camera B,
  which has both classes, so the control *is* the measurement.
* `h3_cross_venue_recall.csv` has its control in a file of its own,
  `h3_with_false_play.csv`, which is where the 99.2% false-play finding came from.

Two needed work, and they needed different work.

**`class_balancing.csv` computed the control and did not write it down.**
`_false_play_control` has been in the script since it was written, and its result — the
unweighted probe calls **46.5%** of held-out empty pitches a match against balanced's
**23.1%** — was printed and recorded in the log, while the CSV carried `play_recall` alone.
The recommendation rests entirely on the control: unweighted looks **+0.0286 better** on
cross-venue recall, and a reader who opened the artefact rather than the log would draw the
opposite conclusion from the same run. Both rates are now rows in the file.

**`h3_sensitivity_merged_venues.csv` inherits H3's control rather than repeating it**, and
now says so. It reports play recall on folds that are 100% ACTIVE_PLAY, so nothing in it
separates a model that transfers from one that has shifted toward PLAY. It does not need to:
the question it asks is whether merging two venue groups changes what the main run concluded,
which is a comparison between folds rather than a claim about a model's sight. The docstring
now says to quote its absolute numbers only beside `h3_with_false_play.csv`.

The distinction is worth keeping: a missing control and an *unwritten* control fail the same
way for a reader, and only one of them is visible from the code.

- 2026-09-11 | control sweep | every results CSV with a recall column | `class_balancing.csv`, `h3_sensitivity_merged_venues.py` | class balancing computed its false-play control and printed it without writing it to the artefact, where the recommendation reverses (46.5% unweighted against 23.1% balanced); the H3 sensitivity check inherits H3's control and now says so

---

## 2026-09-11 — the remaining novel modules are blocked on measurement, not on effort

WP5-T6 (feed both camera halves to STAN), WP5-T3 (context head) and WP5-T4 (distillation)
are the three unbuilt modules in the plan, and WP5-T6 is described there as *"one of the
cheapest real novelty gains available"*. It is cheap to build. It cannot be evaluated.

The draw replication settles it: STAN scores **1.0000 in three draws of five** and 0.9100 and
0.8000 in the other two, mean 0.9420, sd 0.0884. A benchmark that reaches its ceiling in most
draws, and moves by nearly a tenth between them, cannot rank a new architecture against the
current one. A two-channel STAN scoring 1.0000 would be indistinguishable from this STAN; one
scoring 0.95 would sit inside the spread. Either way the ablation would be a number without a
comparison, which is the shape of result WP5-T8's gate exists to refuse.

This is worth writing down as a *blocker* rather than leaving the item open as though it were
waiting for time. The distinction matters for scope decisions in the last weeks: building
these would produce three more preliminary results, not three more findings.

**What unblocks them is what unblocks everything else here** — ≥30 real labelled slot
verdicts (WP2-T8). A test set whose label is not a deterministic function of five templates is
one on which two sequence models can actually differ. Composing more synthetic slots cannot
break the tie, and that was measured rather than assumed.

- 2026-09-11 | WP5-T3/T4/T6 blocked | `stan_draw_spread.csv` | TODO WP5-T6 | the composed benchmark hits its ceiling in 3 draws of 5 with sd 0.0884, so it cannot rank a new architecture against the current one; the three unbuilt modules are blocked on measurement rather than on effort, and ≥30 real slots is what changes that

- 2026-09-11 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 34 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-11 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 4 gate(s) met on artefacts, 3 waiting on a person

- 2026-09-12 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 4 gate(s) met on artefacts, 3 waiting on a person

- 2026-09-13 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 4 gate(s) met on artefacts, 3 waiting on a person

- 2026-09-14 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 4 gate(s) met on artefacts, 3 waiting on a person

- 2026-09-15 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 4 gate(s) met on artefacts, 3 waiting on a person

---

## 2026-09-15 — the boundary reached the picture but not the pooling

An operator drew a pitch boundary, ran the image walkthrough, and reported that the evidence
map still lit up outside the outline — so the model looked like it was still reading the
neighbouring pitch. The conclusion was wrong and the observation was right, which is the
combination worth recording.

`roi.apply` had replaced everything outside the outline before the frame reached the backbone,
so the neighbour was genuinely gone. But `embed_batch` pooled with a plain mean over **every**
patch position, filled ones included. So the *fill* was what reached the probe: a large uniform
region, of a kind no pretraining set contains, averaged into the scored vector and decomposed
onto the map being read. The boundary had removed one distraction and introduced another, and
the evidence map was showing that honestly.

`grid_weights` now maps the polygon onto the backbone's patch grid — through the processor's
own resize and centre crop, by pushing a rasterised mask through the same processor call, so
ConvNeXtV2's `crop_pct` and DINOv2's crop are applied to the boundary exactly as they are to
the frame rather than re-derived. Those weights drive both the pooling and the decomposition,
so positions outside the outline contribute exactly zero to the score and exactly zero to the
map. A full-frame polygon reproduces the plain mean bit for bit on all three backbones.

**The measurement says three things, and the third was a surprise.**

*The neighbouring pitch could already not reach the model* — with `black` or `mean` fill, two
frames identical inside the outline and completely different outside it embed to cosine 1.0,
before and after this change. **Except under `blur`**, which blurs the outside rather than
replacing it: a busy neighbouring pitch arrives as a blurred busy pitch, at cosine 0.896 (vit)
to 0.988 (dinov2). `blur` is the one fill that does not do the job the boundary exists for.

*The fill did reach the model*, and now does not — which is what the zero-outside evidence map
shows directly.

*And confining the pool moved the vector back toward the cache convention rather than away
from it.* This was the stated risk of the change: the feature cache pools over the whole frame,
so an ROI-pooled vector is a distribution the probe was not fitted on. Measured against the
unbounded embedding of the same frame, pooling the fill out **narrows** the gap on every
backbone and every fill — black fill goes 0.786→0.817 (vit), 0.923→0.946 (convnextv2),
0.935→0.967 (dinov2). Removing a large black region from the average is a smaller departure
from "what this frame looks like to the backbone" than leaving it in. The divergence is real
and still wants a development-split check, but it points the other way from the worry.

`mean` fill is the better default on this evidence — it blocks outside content as completely
as `black` and sits closer to the cache convention on all three backbones (0.961/0.835/0.971
against black's 0.946/0.817/0.967). `DEFAULT_FILL` is left at `black` pending that decision,
which `roi.py` has always said is empirical rather than settled.

- 2026-09-15 | ROI reached the picture but not the pooling | `roi_pooling_leak.csv` | `python -m experiments.roi_pooling_leak` | boundary positions are now pooled out rather than only filled in, so evidence outside the outline is exactly 0.0; `blur` is the only fill that leaks outside content (0.896–0.988 where black and mean are 1.0); confining the pool moves the vector *closer* to the unbounded cache convention on all 3 backbones

## A13 condition 3 — do the generated EMPTY frames repair the false-play collapse?

`uv run python experiments/a13_false_play_repair.py --backbone {dinov2,convnextv2,vit}`
-> `results/a13_false_play_repair.csv`

The collapse recorded above — adding the clip venues takes false-play on held-out EMPTY
frames from 0.231 to 1.000 — is the defect the generated EMPTY frames were made for: 31 of
them across six clip venues that had none. Test set is **243 recorded EMPTY frames** from
venue_01 camera B, plus **278 recorded PLAY frames** from the same camera. No generated frame
is ever on a test side.

| backbone | training set | false-play | play-recall | balanced |
|---|---|---|---|---|
| dinov2 | camera_A | 0.3086 | 1.0000 | +0.6914 |
| dinov2 | camera_A + clip | 0.9959 | 1.0000 | +0.0041 |
| dinov2 | **+ 31 generated EMPTY** | **0.0000** | 0.9245 | **+0.9245** |
| dinov2 | + all 79 generated | 0.0000 | 0.8525 | +0.8525 |
| convnextv2 | camera_A + clip | 1.0000 | 1.0000 | +0.0000 |
| convnextv2 | + 31 generated EMPTY | 0.9877 | 1.0000 | +0.0123 |
| vit | camera_A + clip | 1.0000 | 1.0000 | +0.0000 |
| vit | + 31 generated EMPTY | 0.9835 | 0.9892 | +0.0057 |

**On DINOv2 the repair is complete and it is not a shifted prior.** False-play goes to zero
while play-recall holds at 0.92, so `balanced` rises from +0.0041 to +0.9245 — past even the
camera_A-only baseline of +0.6914. Thirty-one generated frames undo a failure that 396 real
clip frames caused.

**On ConvNeXtV2 and ViT it does not happen, and the reason is not the augmentation.** Both
were already at 0.99 and 0.84 false-play *before* the clip venues were added. They never
distinguished an empty pitch at an unseen camera, so there was no working behaviour for the
clip venues to destroy and none for 31 frames to restore.

**Both halves had to be measured.** False-play alone is a single-class test, and a probe that
answers EMPTY to everything scores a perfect 0.0000 on it while being useless — H3's flaw
with the classes swapped. The play-recall column is what separates a repair from a moved
decision boundary, and it is why the DINOv2 row can be believed.

**More generated data is worse than the right generated data.** Adding all 79 instead of the
31 EMPTY ones costs DINOv2 seven points of `balanced` (+0.9245 -> +0.8525) and turns ViT
negative. The maintenance and people frames buy nothing here and cost play-recall.

**What this is not.** One camera, one venue, one held-out split: evidence that the frames
address the known failure, not that the model generalises. RQ1 still needs real empty pitches
at an unseen venue, because the test set has to be real. And the result is backbone-specific,
which strengthens RQ2's recommendation of DINOv2 rather than any claim about synthetic data
in general.

## H3's recall, with a false-play control beside it

`uv run python experiments/h3_with_generated_empty.py` -> `results/h3_with_generated_empty.csv`

H3 reports mean cross-venue ACTIVE_PLAY recall of 0.930 for DINOv2 and meets its 0.90 target.
Its own docstring explains that every held-out venue fold is 100% ACTIVE_PLAY, so a model
answering PLAY to everything scores 1.000. This runs H3's protocol with a control it could not
carry: the false-play rate of each fitted model on **243 recorded EMPTY frames from venue_01
camera B**, held out of every training set here. DINOv2 only.

| arm | mean play-recall | mean false-play | balanced |
|---|---|---|---|
| H3 as pre-registered | 0.944 | **0.768** | +0.176 |
| + 31 generated EMPTY | 0.959 | **0.024** | **+0.936** |

**H3's recall was bought at a price the protocol could not see.** The model that scores 0.944
across unseen venues calls **77% of unseen empty pitches a match**. Per fold the two move
together: `clipvenue_a` scores a perfect 1.000 recall at 0.992 false-play, while
`clipvenue_h` - the *worst* fold by recall at 0.778 - has the second-best false-play at 0.165.
The fold that looks worst under H3 is close to the only one behaving sensibly.

**Adding the generated EMPTY frames improves both axes at once.** Recall does not fall - it
rises slightly, 0.944 to 0.959 - while false-play drops from 0.768 to 0.024 and `balanced`
goes from +0.176 to +0.936. A recall cost was the expected shape of this result and it did not
appear; the 31 frames are not trading one error for another.

**Why 31 generated frames outweigh ~250 real ones already in training.** Every fold trains on
venue_01, which holds hundreds of real EMPTY frames, and false-play was still 0.768. The 396
clip-venue frames are all ACTIVE_PLAY and they push the boundary until it covers unseen empty
pitches; the generated empties come from *those same venues* and are the only thing in the
training set that opposes them there. It is not the count that matters, it is where in feature
space they sit.

**This does not amend H3.** H3 is pre-registered and its numbers stand as reported. What this
adds is the reading: H3 measured whether the model finds play at a new venue, and it does -
but it was never evidence that the model can tell a new venue's empty pitch from a match, and
under the pre-registered training set it cannot.

## Motion as a cue, and whether it survives the deployment sampling rate

`uv run python experiments/motion_cue_probe.py` -> `results/motion_cue_probe.csv`

One number per frame: mean absolute difference against an earlier frame of the same slot and
camera, on a 160x90 greyscale downscale. No model, no training. Computed at four target gaps,
each pairing a frame with the earlier one *closest to* that gap - not the nearest available,
which would have kept using the 8-second pairs and answered the wrong question.

| target gap | actual median | pairs | PLAY vs EMPTY AUC |
|---|---|---|---|
| 15s | 15s | 1366 | **0.930** |
| 30s | 45s | 1285 | 0.890 |
| **60s** | **90s** | **1274** | **0.864** |
| 120s | 180s | 1265 | 0.832 |

**The cue degrades gracefully rather than dying.** The concern was that at the production rate
of one frame per camera per minute, two frames of a live match would be independent scenes and
the difference would measure scenery. It does not: separation falls from 0.930 to 0.864 as the
gap goes from 15 to ~90 seconds, and is still 0.832 at three minutes. A pitch with a match on
it looks different from itself a minute later; an empty one does not.

The 60s row is measured at an **actual median of 90 seconds**, because the corpus holds no
frames exactly 60s apart and the closest match inside the search window sits further out. A
real 60-second sampler would land between the 30s and 60s rows, so 0.864 is the conservative
reading, not the optimistic one.

**It is worth wiring in, on this evidence.** A single scalar at AUC 0.864 on the class pair
that matters, orthogonal to the frozen backbone - which is handed one frame and cannot know
what moved - for the cost of one subtraction on a downscaled image.

**What it does not settle.** `PLAY vs C3` moves between 0.733 and 0.877 across gaps on the
**six** recorded C3 frames in existence; that column is noise and should not be quoted. And
the boundary this was proposed for - a small static kickabout - is not measured here at all,
because the frames that would test it are the ones in `_pending_4d/`, still unlabelled pending
the player-count decision.

**Adding it needs an amendment.** The model's input goes from a frame to a pair, which the
pre-registration does not cover.

## Motion as a model input (A14): separates on its own, contributes nothing

`uv run python experiments/motion_feature_ablation.py --gap {15,60} --train-set {camera_A,camera_A+clip}`
-> `results/motion_feature_ablation.csv`

The cue works (AUC 0.864 at deployment spacing). Whether it *adds* anything to 768 backbone
dimensions is a separate question, and the answer here is no - three ways.

**1 · As a 769th feature on the split that works.** Train venue_01 camera A, test camera B,
1,268 frames with a motion value at ~60s (98%; frames without one are dropped, not imputed).

| arm | macro-F1 | play-recall | false-play | balanced |
|---|---|---|---|---|
| without motion (768) | **0.9882** | 0.9964 | 0.0211 | +0.9753 |
| with motion (769) | 0.9823 | 1.0000 | 0.0380 | +0.9620 |

Slightly worse, and the reason is visible in the baseline: 0.9882 leaves no headroom. This
split is saturated and cannot show a gain in anything.

**2 · As a 769th feature on the split that fails.** Adding the clip venues reproduces the
collapse - macro-F1 0.3484, false-play 1.0000 - and motion changes it by **exactly 0.0000**.
One standardised scalar among 768 already-predictive dimensions is shrunk to nothing by the
regulariser. A cue that separates on its own is not a cue that contributes.

**3 · As an override rule, which is what was actually proposed.** "Nothing moved, so not a
match" is a rule, not a feature, and a rule sits outside the probe where it cannot be shrunk.
Threshold chosen on the training frames only, as the value maximising recall - false-play
there. Result: false-play 0.9834, balanced **-0.0159**. It fails too.

**Why the rule fails, and it is the same gap as everything else.** The motion value is not
calibrated across venues:

| | median motion (15s) |
|---|---|
| venue_01 EMPTY | 0.890 |
| venue_01 ACTIVE_PLAY | 1.575 |
| **clip venues ACTIVE_PLAY** | **2.930** |
| clip venues EMPTY | *(none exist)* |

A threshold is a single number on a scale that means different things per venue - different
cameras, compression, and genuinely busier footage. Worse, the optimiser sees clip frames
only as PLAY with high motion, so it pushes the threshold up until it starts calling venue_01
play empty. **There are no clip-venue EMPTY frames to calibrate against**, which is item 1 of
`thesis/data_requests.md` for the third time in this log.

**And the corpus cannot test the deployment case at all.** The clip venues are 66 clips of six
frames spanning ~10 seconds. A 60-second motion difference does not exist in them and cannot
be computed - at `--gap 60` the clip venues contribute **zero** frames. Motion at deployment
spacing is measurable only at venue_01.

**Conclusion.** The idea is sound and the signal is real; this corpus cannot show it helping,
and the specific reason is the missing EMPTY frames at other venues plus clips too short to
hold a one-minute pair. Recorded as a negative result rather than tuned until it turns
positive. No amendment is filed, because nothing is being adopted.

## Probe regularisation: the default is fine, and inner CV cannot tell you otherwise

`uv run python experiments/probe_regularisation.py --backbone dinov2`
-> `results/probe_regularisation.csv`

`LinearProbe` has always used scikit-learn's default `C=1.0` and nothing had varied it. Swept
over five orders of magnitude, training on venue_01 camera A + clip venues + 31 generated
EMPTY, testing on venue_01 camera B (real EMPTY and real PLAY).

| C | macro-F1 | play-recall | false-play | inner CV |
|---|---|---|---|---|
| 0.001 | 0.9290 | 0.8705 | 0.0041 | 0.9839 |
| 0.1 | 0.9290 | 0.8705 | 0.0041 | **0.9884** |
| **1.0 (default)** | **0.9386** | 0.8849 | 0.0000 | 0.9873 |
| 3.0 | 0.9539 | 0.9137 | 0.0000 | 0.9873 |
| 10.0 | 0.8532 | 0.7266 | 0.0000 | 0.9872 |

**Nested selection picks C=0.1 and makes things worse** - macro-F1 0.9290 against the
default's 0.9386. Tuning C is worth **-0.0096**, so the default stays.

**The interesting part is why the selection fails.** Inner 5-fold CV on the training set
scores between 0.9838 and 0.9884 across the *entire* grid - a spread of 0.005 over a
10,000-fold change in the penalty. The curve it is choosing from is flat, so its argmax is
noise.

That is the near-duplicate problem showing up in hyperparameter selection. Random CV folds
put frames from the same 15-second window on both sides, so every model looks near-perfect
inside the training set regardless of regularisation, and the one measurement that could
choose C honestly cannot distinguish the candidates. **Cross-validation on this corpus is not
a usable model-selection tool** unless the folds are grouped, which is the same finding the
leakage work reached for evaluation, arriving now for selection.

The sweep's own best is C=3.0 at 0.9539, but that value is read off the test set and is not
quotable. It is recorded only to show the curve has real structure that inner CV cannot see.

## RQ6 answered on a test set with a real class mix

`uv run python experiments/rq6_on_a_real_class_mix.py --backbone dinov2`
-> `results/rq6_real_class_mix.csv`

`rq_matrix.md` recorded RQ6 as blocked: "machinery built and tested ... with a test set that
is 99% one class, a 99% precision target is met before confidence is consulted, so the
risk-coverage curve has nothing to trade against. `evaluation/calibration.py` is ready. It
needs a test set with a real class mix."

**One existed the whole time.** Holding out venue_01's physical **camera B** leaves 243
recorded EMPTY and 278 recorded PLAY frames - a 47/53 mix. No experiment used it because the
project's splits are grouped by slot or by venue, and venue_01 has only two slots, so neither
protocol produces this split. `class_balancing._false_play_control` had already reached for
the same camera hold-out for the same reason.

Training is camera A + clip venues + the 31 generated EMPTY frames. Without them the probe
calls ~100% of unseen empty pitches a match (`a13_false_play_repair`), and a risk-coverage
curve over that model would describe nothing worth an operating point.

| coverage | accuracy | threshold | review |
|---|---|---|---|
| 50% | 1.0000 | 0.9949 | 50% |
| 70% | 0.9918 | 0.9846 | 30% |
| 80% | 0.9856 | 0.9693 | 20% |
| 90% | 0.9701 | 0.9141 | 10% |
| 100% | 0.9386 | - | 0% |

**The operating points a manager can act on:**

- **95% accuracy on automated verdicts -> review 3%** (confidence >= 0.7447)
- **99% accuracy on automated verdicts -> review 27%** (confidence >= 0.9824)

**The tie problem that forced the band does not arise here.** Under the old degenerate split
DINOv2's temperature pinned to the grid floor and 890 of 907 confidences were exactly 1.0,
which is why risk-coverage is reported as a band at all. On this split **15 of 521** are, so
the worst and best bounds coincide at every coverage above and the band is a line.

**Scope, and it is narrow.** One venue, two cameras, two days. This answers *what review buys
here*, not what review buys. A threshold fitted on venue_01 has not been shown to transfer,
and RQ1's blocker is untouched - the test frames are real, but they are venue_01's. RQ6 moves
from "blocked by data" to "answered at one venue, transfer unmeasured".


## ROI pooling: the gain was the inconsistency, not the boundary

`uv run python experiments/probe_regularisation.py --backbone dinov2 [--roi-pooled]`
-> `results/probe_regularisation.csv`

`build_cache` gained `roi_for` so training features could be pooled inside a boundary the way
`classifier.classify_batch` already pools them at serve time. Boundaries for all 99 cameras
come from `scripts/derive_roi.py`. Train venue_01 camera A + clip venues + 31 generated EMPTY,
test venue_01 camera B on recorded frames, C=1.0.

| features | macro-F1 | play-recall | false-play | balanced |
|---|---|---|---|---|
| whole frame | **0.9386** | 0.8849 | 0.0000 | +0.8849 |
| ROI-pooled, generated frames **unbounded** | 0.9750 | 0.9568 | 0.0041 | +0.9527 |
| ROI-pooled, **every frame bounded** | 0.9367 | 0.8813 | 0.0000 | +0.8813 |

**The middle row is wrong and it is the one that looked like a result.** In that cache the
189 generated frames had no boundary - their `camera` is `synthetic_<batch>`, which
`derive_roi` had skipped - so they alone were pooled over the whole image while every recorded
frame was pooled inside one. The +0.0364 macro-F1 was that inconsistency, not the masking. Two
pooling conventions inside one training set produced a **better-looking number than the
correct configuration**, which is the trap worth recording: a skew does not announce itself as
a defect, it announces itself as an improvement.

**Applied consistently, ROI pooling changes nothing here: -0.0019 macro-F1, inside noise.**

RQ6's operating point moves the same way and slightly for the worse: 99% accuracy costs 54%
review on ROI features against 27% on whole-frame ones. The whole-frame configuration stays.

**What this does not settle.** One venue, one split, DINOv2. And the boundaries are convex
hulls, so a dugout inside the outline's span is still inside the pooling - `roi.py` has always
said the hull is a floor rather than a ceiling. A boundary that excluded the dugout might do
something these do not; that is a different experiment, not a different reading of this one.

## The boundary on unseen footage: false-play 0.74 -> 0.32

`uv run python scripts/classify_video.py VIDEO --polygon {none,cam2,derive} --every 10`
-> `results/video_verdicts_{none,cam2,derived}.csv`

A 234-second clip of a pitch at a **venue the model has never seen**, sampled every 10s. Of
24 samples, 19 show an empty pitch and 5 show a single person walking, with no ball and no
game. Ground truth read off the rendered strip by hand.

| boundary | false-play on the 19 empty frames | the 5 person frames called PLAY |
|---|---|---|
| none - whole frame | **14/19 = 0.74** | 4/5 |
| `cam2`, a different camera's outline | 13/19 = 0.68 | 4/5 |
| **derived from this video** | **6/19 = 0.32** | 3/5 |

**A correct boundary halves the false-play rate on footage the model has never seen.** The
whole frame, and another camera's outline, are indistinguishable from each other - which is
the point: `cam2` is not a boundary for this camera, it is an arbitrary polygon that happens
to be the right shape somewhere else. It cuts across this pitch unrelated to its edges and
buys 0.06.

**This does not contradict `probe_regularisation`, where ROI pooling was worth -0.0019.** That
measurement is venue_01 camera B, where the model already scores 0.9386 and there is nothing
for a boundary to repair. Here the model is at 0.26 accuracy on empty frames before the
boundary and 0.68 after. **The boundary matters where the model is failing, not where it is
already right**, and the corpus has no test set of the first kind - which is why the effect
was invisible until a video from an unseen venue was run through it.

**The remaining third is not a boundary problem.** Six empty frames still read ACTIVE_PLAY,
and the five person-frames are a class the model cannot express: `3_people_not_playing` holds
six recorded frames in the whole corpus, so in practice the deployed probe is two-class and a
lone walker has nowhere to go but PLAY.

**Motion would have caught most of it and is not wired in.** These frames carry motion 0.88 to
2.02 against the previous sample, against a corpus median of 0.89 for EMPTY - consistent with
an empty pitch. `motion_feature_ablation` found the cue adds nothing *as a feature on
venue_01*; it has never been tried on footage like this, for the same reason the boundary
effect was invisible.

**And "people detected 0" is the detector, not the scene.** `explain.redact_people` returns
zero on frames with four visible people (`synthetic_data_protocol.md` §3a). The people count
on the viewer is that detector, so it says nothing about whether anyone is there.

- 2026-09-16 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 4 gate(s) met on artefacts, 3 waiting on a person

## The motion rule on unseen footage: false-play 0.38 -> 0.15

`uv run python experiments/motion_override_on_video.py VIDEO --truth 9,12,15`
-> `results/motion_override_on_video.csv`

`motion_feature_ablation` found the cue worthless, as a feature and as a rule, and measured
that on venue_01 - where the model already scores 0.9386 and there is nothing to repair. The
boundary looked equally worthless on that split and then halved false-play on unseen footage,
so the motion result earned the same retest.

**A transfer test, not a fit.** The threshold is chosen on venue_01's recorded EMPTY and PLAY
frames at a 15-second gap, as the value maximising (recall - false-play) there, and applied to
the video unchanged: **1.098**. Nothing about the video informs it. The video is sampled every
15 seconds to match the gap the threshold was fitted at, because a cue compared against a
threshold learned at another spacing is a different quantity.

16 samples, boundary derived from the video, three with a person on the pitch (#9, #12, #15)
by hand. The figures at #0, #13 and #14 are behind the goal, outside the boundary, and are not
people on the pitch.

| | false-play on the 13 empty frames |
|---|---|
| whole frame, no boundary | 0.74 (from the 10s run) |
| boundary only | **5/13 = 0.38** |
| **boundary + motion rule** | **2/13 = 0.15** |

**The rule fixes three of the five remaining errors** - #2, #11 and #14, all of which sit just
under the threshold at 1.02-1.07. The two it does not fix, #13 at 1.610 and #15 at 1.269, are
genuinely moving frames: #15 has a person on the pitch and is arguably not an error at all.

**So both cues work, and both were measured as useless.** Each was tested on venue_01 camera
B, the only split the corpus offers with a real class mix, and that split cannot show either
effect because the model does not fail on it. Two negative results in this log - ROI pooling
at -0.0019 and the motion rule at -0.0159 - are not wrong, but they are answers to a question
about a working model, and the operational question is about a failing one.

**The corpus cannot produce this test.** It took a video from a venue with no labelled frames
in the dataset. That is item 1 of `thesis/data_requests.md` from a fourth direction: without
empty pitches at an unseen venue there is no split on which an intervention aimed at
cross-venue failure can be seen to work.

## The corpus is 179 scenes, and training on all 1,692 frames is what breaks it

`uv run python experiments/dedup_training_set.py --video CLIP --truth 9,12,15`

Distinct scenes by perceptual hash, within (venue, class):

| | frames | distinct | ratio |
|---|---|---|---|
| **venue_01 / EMPTY** | **494** | **5** | **0.01** |
| venue_01 / ACTIVE_PLAY | 796 | 94 | 0.12 |
| venue_01 / C3 | 6 | 3 | 0.50 |
| the nine clip venues, all ACTIVE_PLAY | 396 | 77 | 0.19 |
| **recorded total** | **1,692** | **179** | **0.11** |
| generated total | 189 | 112 | **0.59** |

**The entire EMPTY class of this corpus is five pictures.** One venue, two cameras, two days,
each scene repeated about a hundred times. That is the whole explanation for a model that
cannot recognise an empty pitch anywhere else: it has seen five of them.

**Pruning to distinct scenes and refitting:**

| training set | venue_01 cam B macro-F1 | unseen clip: false-play | says PLAY |
|---|---|---|---|
| full, 1,196 frames | **0.9386** | 4/13 = 0.31 | 6/16, 2 on person minutes |
| **pruned, 183 frames (15%)** | 0.8420 | **0/13 = 0.00** | **2/16, both on person minutes** |

**Zero false-play on unseen footage, from 15% of the data.** And not by collapsing to EMPTY:
the pruned probe says ACTIVE_PLAY exactly twice in sixteen minutes and both times a person is
on the pitch. It is not conservative, it is correct.

**The trade is in-domain accuracy for cross-venue generalisation**, and the in-domain loss is
on a test set drawn from the same five scenes the duplicates come from. 0.9386 measures how
well the probe reproduces backgrounds it has memorised; 0.00 measures whether it can tell an
empty pitch it has never seen. The second is what the system is for.

**Why repetition hurts rather than being neutral.** Class weighting balances EMPTY against
PLAY by count, not by scene. Five backgrounds at a hundred frames each tell the fit that those
five *are* what EMPTY looks like, with the confidence a hundred observations would justify and
five do not - so the decision boundary tightens around them and anything else falls outside.

**The generated frames are five times more scene-diverse than the recorded ones** (0.59 against
0.11), which is why 31 of them repaired a failure 396 recorded frames caused.

**Caveat on the absolute numbers.** Both arms train on whole-frame features and are evaluated
on ROI-pooled ones, which is the train/serve skew recorded above. It applies identically to
both, so the comparison holds; the absolute false-play figures carry it.

## Matching training features to serving: the tests disagree

`uv run python experiments/rerun_on_distinct_scenes.py [--roi-pooled] --video CLIP`

The boundary is now in the serving path (A14), so `classify_batch` pools inside it while every
cached training feature is whole-frame. That skew is permanent in production unless the
training cache is rebuilt ROI-pooled. Both, on three tests:

| test | metric | whole-frame train | ROI-pooled train |
|---|---|---|---|
| H3 cross-venue, pruned | false-play on 243 recorded EMPTY | **0.6173** | 0.9994 |
| H3 cross-venue, full | false-play | **0.7684** | 0.9759 |
| venue_01 cam B, pruned | macro-F1 | 0.8420 | **0.9155** |
| venue_01 cam B, full | macro-F1 | **0.9386** | 0.9367 |
| unseen clip, pruned | false-play | **0.00** | **0.00** |
| unseen clip, full | false-play | 0.31 | **0.23** |

**In-domain and on the clip, matching helps or ties. On the cross-venue false-play control it
is catastrophic** - 0.62 to 0.9994, a probe that calls essentially every unseen empty pitch a
match.

**The mechanism is worth stating because it is not obvious.** ROI pooling removes the
surroundings, and the surroundings are how the probe recognised *venue_01's* empty pitch - the
barrier, the dugout, the buildings. That recognition is memorisation, and masking it away is
supposed to be the point. But the clip venues are all ACTIVE_PLAY and their turf looks like
venue_01's turf, so once the backgrounds are gone the only thing left is a pitch, and a pitch
in this training set is overwhelmingly a pitch in use. The boundary removes a shortcut the
probe was relying on and the corpus has nothing to replace it with.

**So the two measurements are not in conflict about the boundary; they disagree about what
they are measuring.** The clip is unseen footage where memorisation cannot help, and there the
boundary helps or ties. The H3 control is venue_01's second camera, where memorisation is
exactly what was carrying the number, and there removing it hurts.

**Left as it is, and recorded rather than resolved.** Production keeps whole-frame training and
boundary-pooled serving - a real skew, now measured, costing 0.08 on the clip's `full` arm and
nothing on its `pruned` arm. Rebuilding the cache would trade that for a cross-venue false-play
of 0.9994 on the only recorded control that exists. Neither is defensible as an improvement,
and choosing between them on this evidence would be choosing a test set.

**What would settle it** is a recorded EMPTY frame at a venue that is not venue_01 - then
false-play could be measured where memorisation is impossible *and* on recorded data. That is
item 1 of `thesis/data_requests.md`, for the fifth time in this log.

## Generated C3 frames are learnable and do not transfer: subtraction works, addition does not

`uv run python experiments/three_class_on_video.py --video CLIP --truth 9,12,15`
-> `results/three_class_on_video.csv`

The corpus holds **six recorded C3 frames**, three scenes, one venue, one moment - so the class
cannot be trained and the deployed probe is three-class in name and two-class in behaviour. 158
generated C3 frames now exist, 103 scenes across seven venues. Two questions, opposite answers.

**They are learnable.** Leave-one-venue-out over the venues holding generated C3, mean recall
of C3 at an unseen venue **0.742**, four of seven venues at 1.000.

**They do not fire on a real person.** On the unseen clip - three minutes with one person on
the pitch, no ball, hand-labelled - the three-class probe says C3 **0 times out of 3**, and the
verdicts are identical to the two-class probe's:

| C3 trained on | C3 scenes | C3 on the 3 person minutes | C3 on the 13 empty minutes |
|---|---|---|---|
| people-not-playing only | 20 | **0/3** | 0/13 |
| maintenance only | 83 | **0/3** | 0/13 |
| both | 91 | **0/3** | 0/13 |

**It is not a composition problem.** The class is 75% maintenance-with-tools, so the obvious
explanation was that a lone walker is under-represented. Training C3 on the people-only subset
changes nothing: 0/3. Nine predictions across three arms, none of them C3.

**The mechanism, and it explains why the EMPTY frames worked and these did not.**

- The generated EMPTY frames were made by **subtraction** - people removed from a real frame.
  What remains is real pixels of a real pitch, and they repaired the false-play collapse.
- The generated C3 frames were made by **addition** - a person drawn onto a real pitch. What
  the probe learns is what a *drawn* person looks like, and a real one is not that.

A13's separability gate passed at 0.5332 because it asked whether generated frames as a whole
are distinguishable from recorded ones. It did not ask, and could not have asked, whether a
generated *person* stands in for a real one. On the only real test available the answer is no.

**Three positives.** A class that fails on three frames has been shown to fail on three frames.
But the failure is unanimous across three training compositions, and the two-class and
three-class probes are indistinguishable on this footage, which is the operational statement:
**the deployed system still cannot say "someone is here and not playing"**, and generating more
C3 frames is not the route to making it.

**What is:** real footage of a person walking across a pitch, which is item 4 of
`thesis/data_requests.md` and ten minutes of someone's time.

## EMPTY frames by temporal median: sound method, cannot reach the venues that need it

`uv run python scripts/make_median_empties.py`

Following the subtraction/addition finding: a temporal median is subtraction without a
generative model. Players move, the pitch does not, so the per-pixel median over several
frames of one camera is that camera's own pitch with the people gone - real pixels, real
exposure, no drawn person.

**It works where the footage is long enough, and the corpus is mostly not.**

| venue | median EMPTY frames produced |
|---|---|
| clipvenue_a_blue_barrier | 8 |
| venue_01 | 2 |
| clipvenue_e_pink_boards | 1, rejected by eye - two people still standing in it |
| **clipvenue_d, _f, _g, _h, _i** | **0** |

The nine clip venues are 66 clips of six frames spanning about ten seconds. Six frames is
enough for a median only if people move across them, and over ten seconds at a five-a-side
match many do not. So the technique produced frames for the venue that already has ten
generated empties and for the venue that already has 494 recorded ones, and nothing for the
five that have neither.

**Measured anyway, and it changes nothing:**

| training set | venue_01 cam B macro-F1 | unseen clip false-play |
|---|---|---|
| pruned, 183 frames | 0.8420 | 0/13 |
| + 10 median empties | 0.8394 | 0/13 |

**Two guards failed on the same frame, which is the part worth keeping.** The first check asked
whether the median resembles its closest member - a smear test. It passed a
`clipvenue_e_pink_boards` frame with two people plainly standing in it, because when nobody
moves the median reproduces the crowd and matches every member closely. The second check asked
whether anything moved at all, and that frame cleared it at 3.509 against a 3.5 floor. Raising
the floor to exclude it would be tuning a guard to the example it failed on.

So the frame was rejected by looking at it, which is what `synthetic_data_protocol.md` §3a
already concluded about `redact_people`: on this footage, a detector's approval is not a gate.
**The honest procedure for this script is automatic checks followed by a human pass over a
dozen thumbnails**, and that is cheap enough to be the procedure rather than an apology for one.

Also fixed on the way: the first version built medians from `clipvenue_b_floodlit_track`,
which is **locked final test set**. `derive_roi.py` refuses locked venues by name and this did
not, until it had already produced two frames from one. It refuses them now.

## Counting people inside the boundary beats the probe

`uv run python experiments/person_count_rule.py --imgsz 1280 --conf 0.25`
-> `results/person_count_rule.csv`

yolov8n, counted **inside the camera's boundary** using each box's foot point rather than its
centre, on 521 recorded frames from venue_01 camera B.

| class | n | median | zero | 1-4 | >=5 |
|---|---|---|---|---|---|
| EMPTY | 243 | 0 | **89%** | 11% | 0% |
| ACTIVE_PLAY | 278 | 6 | **0.4%** | 32% | 68% |

**On the split the probe was tuned on, the rule wins:**

| | recall | false-play | balanced |
|---|---|---|---|
| **person count, PLAY if >= 2** | **0.9604** | 0.0453 | **+0.9152** |
| probe, full training set | 0.8849 | 0.0000 | +0.8849 |
| probe, pruned to distinct scenes | 0.7302 | 0.0288 | +0.7014 |

No training, no venue memorisation, nothing fitted on venue_01 - a pretrained detector and a
threshold. On the unseen clip the same detector reported **zero people inside the boundary on
all thirteen empty minutes** at every resolution tried.

**Two things make it work that the earlier attempt did not do.** Counting inside the boundary,
so spectators behind a fence are not people on the pitch; and taking the **foot** of each box
rather than its centre, since a person standing at the touchline has their centre over the
pitch and their feet outside it.

**Why `synthetic_data_protocol.md` §3a concluded the opposite.** That measurement was on
venue_01 frames where the people are in the *dugout* - small, partly occluded, behind a
barrier - and it is still right about those. A person standing on the pitch is a different
detection problem, and the same detector finds them.

**The three-class version does not hold, and the numbers say exactly where.**

| truth | -> EMPTY (0) | -> not playing (1-4) | -> PLAY (>=5) |
|---|---|---|---|
| EMPTY | **89%** | 11% | 0% |
| ACTIVE_PLAY | 0.4% | **32%** | 68% |

Zero-versus-nonzero is a strong signal. **1-4 is not**: a third of genuine ACTIVE_PLAY frames
show four or fewer people inside the boundary, because the camera sees part of a pitch and a
detector misses distant players. A rule that called those "not playing" would be wrong on 88
real matches out of 278.

So the defensible rule from this measurement is **two-class on the count** - nobody inside the
boundary means empty - and the count is *evidence toward* C3 rather than a decision. The class
that needs a lone walker distinguished from a match still needs recorded footage of one.

**Cost.** About one second per frame on CPU at imgsz=1280. The system samples one frame per
camera per minute, so this is affordable where a per-frame model would not be.

## The person count transfers across every venue; the probe does not

`results/person_count_clip_venues.csv`

The count rule was measured at venue_01 and beat the probe there. The question that matters
for this project is whether it holds anywhere else, since the probe's cross-venue false-play
is 0.6173 even pruned. All 396 recorded clip-venue frames, every one ACTIVE_PLAY, counted
inside each camera's derived boundary:

| venue | n | median count | zero | >= 2 |
|---|---|---|---|---|
| clipvenue_a_blue_barrier | 168 | 9.5 | 0% | 99% |
| clipvenue_b_floodlit_track | 78 | 9.0 | 0% | 100% |
| clipvenue_c_teal_boards | 36 | 14.0 | 0% | 100% |
| clipvenue_d_indoor_dome | 18 | 8.0 | 0% | 100% |
| clipvenue_e_pink_boards | 18 | 14.0 | 0% | 100% |
| clipvenue_f_outdoor_bldg | 12 | 8.0 | 0% | 100% |
| clipvenue_g_netting | 30 | 9.0 | 0% | 100% |
| clipvenue_h_teal_pitch | 18 | 11.0 | 0% | 100% |
| clipvenue_i_outdoor_trees | 18 | 9.0 | 0% | 100% |
| **all nine** | **396** | **10.0** | **0%** | **100%** |

**Not one frame of real play at any venue was missed.** Indoor domes, floodlit night, teal
boards, netting in front of the lens - the count holds through all of it, because a pretrained
person detector has seen far more people in far more conditions than 290 scenes can teach a
linear probe.

**The comparison that matters:**

| | cross-venue play-recall | false-play |
|---|---|---|
| probe, pruned to distinct scenes | 1.0000 (H3 folds) | **0.6173** |
| probe, full training set | 0.9444 | 0.7684 |
| **person count, `PLAY if >= 2`** | **1.0000** (9 venues) | **0.0453** at venue_01 |

The probe reaches perfect cross-venue recall by calling 62% of unseen empty pitches a match.
The count reaches the same recall while, on the only frames where false-play is measurable,
being wrong 4.5% of the time.

**What is still not measured, and it is the same gap.** False-play for the count rule can only
be checked where recorded EMPTY frames exist, which is venue_01 and the one unseen clip. Nine
venues confirm the rule does not *miss* play; none of them can confirm it does not *invent*
it, because none of them has an empty pitch on record.

**So the count is the strongest component in the system and the weakest evidenced.** It is
wired in as a gate that can only turn ACTIVE_PLAY into EMPTY, which is the direction 396
frames say is safe, and it is not promoted to the classifier on the strength of one venue's
empty pitches.

## The REVIEW band is unchanged by the person gate, and the reason is the useful part

`experiments/rq6_on_a_real_class_mix.py --gates`, `results/rq6_real_class_mix.csv`

RQ6's operating point was computed on the probe alone. The gates now run in the deployed path,
so the band was recomputed with the A16 person gate applied before confidence is read - the
order `run_slot` uses, since the gate's verdict is the one that reaches review. Every figure
came back identical:

| | accuracy at full coverage | 95% target | 99% target |
|---|---|---|---|
| probe alone | 0.9386 | answer 97%, review 3% | answer 73%, review 27% |
| probe + person gate | 0.9386 | answer 97%, review 3% | answer 73%, review 27% |

**The gate overruled 0 of 521 verdicts.** Not a wiring fault - it is the split. The probe makes
32 errors here and **all 32 are missed play**; false-play on venue_01 camera B is 0.0000 once
the generated EMPTY frames are in training. The gate only turns ACTIVE_PLAY into EMPTY, so it
can act only where the probe says PLAY wrongly, and of the 246 frames it does call PLAY the
person count is zero on none of them, median 7. There is nothing here for it to catch.

This is the sixth measurement to stall on the same wall, and it is worth naming plainly:
**venue_01 camera B cannot exhibit the failure these gates exist to fix.** It is the only
recorded split with both classes, which is why RQ6 uses it, and it is also the split the probe
has effectively memorised. Every cross-venue intervention will read as "no change" here.

**What the one-way choice costs, measured rather than assumed.** Of the 32 missed-play frames,
25 have two or more people inside the boundary - a count rule running in the PLAY direction
would recover them and take accuracy from 0.9386 to **0.9655**. It would also call 11 of the
243 recorded EMPTY frames a match (false-play 0.0453). That trade is favourable *on this
split*, and it is still not taken: the 0.0453 is evidenced at one venue, and the reason the
gate is one-way is that 396 clip-venue frames confirm the count never misses play while no
venue but this one can confirm it does not invent it. The number above is what a future
supervisor decision would be buying, stated so the decision can be made on evidence.

**Also fixed here.** The results file gained a `gates` column. Without it the two arms append
as twelve indistinguishable rows, and a results file you cannot attribute to a run records
nothing.

- 2026-09-17 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 4 gate(s) met on artefacts, 3 waiting on a person

## The ball is the strongest signal at venue_01 and the weakest anywhere else (A17)

`experiments/ball_detection_rule.py`, `results/ball_detection_rule.csv`

The third part of the requested rule - "detect if there is a ball, it may help to know" - and
the only part not yet measured. It matters because it is exactly the evidence the second part
lacked: the person count cannot carry C3, since a third of genuine ACTIVE_PLAY frames show
four or fewer people inside the boundary, and *four people with a ball* is a kickabout where
*four people without one* is not.

COCO's `sports ball`, yolov8n at imgsz=1280, confidence 0.10, counted inside each camera's
derived boundary by the box centre rather than the foot point - a ball spends much of its time
in the air and has no feet to stand on.

**At venue_01 camera B it is the best number this project has produced:**

| class | n | a ball was found |
|---|---|---|
| C1_EMPTY | 243 | **3%** |
| C2_ACTIVE_PLAY | 278 | **100%** |

Recall 1.000 against false-play 0.033, balanced **+0.967** - above the person count's +0.9152
and the probe's +0.8849. On the frames the count cannot separate, the 1-4 people band, it is
100% against 11%.

**It does not survive leaving the venue.** All 396 recorded clip-venue frames, every one
genuine ACTIVE_PLAY:

| venue | n | ball found |
|---|---|---|
| clipvenue_a_blue_barrier | 168 | 35% |
| clipvenue_b_floodlit_track | 78 | 46% |
| clipvenue_c_teal_boards | 36 | 86% |
| clipvenue_d_indoor_dome | 18 | 22% |
| clipvenue_e_pink_boards | 18 | 89% |
| clipvenue_f_outdoor_bldg | 12 | 17% |
| clipvenue_g_netting | 30 | 30% |
| clipvenue_h_teal_pitch | 18 | 17% |
| clipvenue_i_outdoor_trees | 18 | 6% |
| **all nine** | **396** | **40%** |

The person count found people in **100%** of those same frames. So on 60% of real play at an
unseen venue there is no ball to find, and **the absence of a ball is not evidence of the
absence of play** - which is precisely the direction the C3 rule would have needed.

**Why the venue_01 figure is not the answer.** 278 frames, **13 distinct scenes**, and a
sample of 70 of them landed 69 times in one. The 243 EMPTY frames are 3 scenes. A perfect
score over 13 scenes at one site is what this corpus produces for almost everything; the nine
venues are the only measurement in the pair that is not about venue_01.

**Two checks before believing the detections at all**, because a +0.967 in this project has
been a confound before:

- **It moves.** Within the dominant scene the detected ball's centre has a standard deviation
  of 0.10 of frame width and 0.06 of frame height, ranging across most of the pitch. A light
  fitting or a bin would sit still.
- **It is the right size.** Median box area 404 px on play frames, p10-p90 of 347-536, and 435
  px at the clip venues - about 20x20 pixels, which is what a football is on a 1080p frame at
  that distance. Nothing head-sized or bag-sized is being counted.

**Lowering the threshold is what buys the recall, and it is spent entirely on venue_01.**
Sweeping the ball confidence at venue_01: 1.000/0.033 at 0.10, 0.878/0.016 at 0.20, 0.788/0.004
at 0.25, 0.392/0.000 at 0.40. There is no setting that makes the cross-venue 40% respectable.

**What was wired in, and what was not.** `vision/people.py` now takes people and ball from a
single detector pass - `detect_objects` in `explain.py` generalises `detect_people`, so the
ball costs nothing, the detector was already running. `PersonGate.inspect` returns both, and
`SlotRun` carries `people_counts` and `ball_minutes` out to the caller. The ball **decides
nothing**: a frame with nobody and a ball inside the boundary is still overruled to EMPTY, and
a test asserts that `inspect` never reads the ball when deciding. A detection at 0.10
confidence is entitled to be recorded and not to be obeyed.

**On the unseen clip the system is unchanged** - 0/13 false-play, ACTIVE_PLAY only on person
minutes - and no ball was found on any minute, including the two minutes with a person walking
across the pitch. That is the correct answer on that footage and it is also the one the
cross-venue table says not to rely on.

**What would change this.** Not a lower threshold: a detector trained on footballs at CCTV
scale, or a tracker that accumulates a ball across the ten frames of a minute instead of
deciding from one. Both are outside a frozen-backbone CPU system, and both are the honest
recommendation for the class this corpus cannot separate.

## The rule as it was actually stated works, and the count-only version was the wrong test (A18)

`experiments/ball_detection_rule.py --clip-venues`, `results/ball_detection_rule.csv`

A16 refused "one to four people inside the boundary means not playing" because it is wrong on
**88 of 278** genuine ACTIVE_PLAY frames at venue_01: a camera sees part of a pitch and a
detector misses distant players, so a real match routinely shows four or fewer. That refusal
tested half a rule. The rule as given had a second clause - *and detect if there is a ball* -
and the ball is what separates four people having a kickabout from four people standing about.

**Both clauses, on frames that already have labels:**

| frames | n | count alone fires | with the ball clause |
|---|---|---|---|
| venue_01 camera B, ACTIVE_PLAY | 278 | 88 (31.7%) | **0 (0.0%)** |
| venue_01 camera B, EMPTY | 243 | 27 (11.1%) | 24 (9.9%) |
| nine clip venues, all ACTIVE_PLAY | 396 | 9 (2.3%) | **3 (0.8%)** |

Rows one and three are entirely genuine play, so every count in them is an error the rule would
introduce. The ball clause removes **all 88** at venue_01 and two thirds of the cross-venue
cost. **0.8% on 396 frames at nine venues the model has never seen is the best-evidenced
cross-venue number in this project** - better evidenced than any figure the probe has.

**Which clause is doing the protecting is not the obvious one.** A *found* ball vetoes the
not-playing call, and that direction is sound whatever the detector's recall, because a found
ball is a found ball. The rule also requires the ball to be *absent*, and that direction is
not sound - at an unseen venue 60% of real play shows no detectable ball. What keeps it from
mattering is the count: at those nine venues the median is 10 people inside the boundary, so
the unsound clause is only consulted on **9 frames out of 396** and gets 3 of them wrong. The
rule is protected by the count, not by the ball, and it would stop being safe the moment a
camera framed less of a pitch.

**The half that is not evidenced, stated as plainly as the half that is.** The corpus holds
**6 recorded C3 frames**, one slot at one camera, and the rule identifies **1**: three show
nobody inside the boundary at all, and two show a ball. A cost measured on 396 frames and a
benefit measured on 6 is not a balanced case. What is claimed here is that the rule is *safe*,
not that it is *shown to work*.

**It was adopted anyway, and the reason is not the numbers.** Before this the deployed path
could not return C3 under any circumstances - two gates, both pointing at EMPTY, and a probe
that has 6 real frames of the class. A rule with an unmeasurable recall and a measured cost
below one percent is better than a system that is structurally incapable of the answer. It can
be switched off with `PersonGate(small_group_max=0)`, which restores A16 exactly.

**On the unseen clip it gives the answer the footage deserves.** Through `run_slot`, boundary
plus all three gates:

| minute | verdict | what is there |
|---|---|---|
| 0-8, 10-14 | EMPTY | nobody |
| **9, 15** | **MAINTENANCE_NON_SPORTING** | **one person walking, no ball** |
| 12 | EMPTY | one person, not detected inside the boundary |

False-play stays **0/13**. The slot verdict moves from REVIEW - "intermittent activity, 12%
play, neither threshold met" - to **NOTUSED**, "empty in 88% of samples with only 0% active
play". A pitch nobody played on now reports as a pitch nobody played on, which is the question
the thesis exists to answer and the first time the deployed path has answered it correctly on
footage it had never seen.

**Minute 12 is the remaining failure and it is the person gate's, not this rule's.** A person
is visible and the detector finds nobody inside the boundary, so A16 overrules to EMPTY before
this rule is ever consulted. The 6 recorded C3 frames show the same thing - three of them have
a count of zero. Whatever fixes those minutes is a better detector or a wider boundary, not a
better rule on top of this count.

**RQ6 is unaffected**, rechecked rather than assumed: with the extended gate in the loop the
person gate still overrules 0 of 521 verdicts and every figure in the risk-coverage band is
identical. That follows from the table above - the combined rule fires on 0 of venue_01's 278
ACTIVE_PLAY frames - but a gate that gained a new output class is exactly the sort of change
that quietly moves a number nobody re-ran.

## The boundary clips a third of the pitch, both fixes are worse, and the tightness is the point (A19)

`experiments/roi_flat_field.py`, `results/roi_flat_field.csv`

Rendering the unseen clip's derived boundary with the person boxes drawn on it shows an
obvious defect: the outline keeps **56%** of the frame and stops a third of the way up the
pitch, with the far third and the goal outside it. A person standing at the far end is not
counted, and three of the six recorded C3 frames detect people in the frame and none inside
the boundary. The cause is not subtle either - `turf_mask` applies **one** Otsu threshold to
the whole frame, and a floodlit pitch is dimmer at the far end than at the near end, so the
threshold splits the pitch instead of separating pitch from not-pitch.

Two fixes were built and measured against the current mask on the same detections, so the
arms differ only in the outline.

**Flat-field correction** - divide the excess-green image by a heavily blurred copy of itself,
removing the smooth illumination surface. It does exactly what it was meant to on the clip, 56%
coverage to 86%. It is also unusable: a near-uniform image divided by a blur of itself is
noise, and Otsu splits the noise. **venue_01 camera A - where the only six recorded C3 frames
live - collapses from 49% coverage to 3%.**

**Hysteresis growth** - Otsu still decides what is certainly pitch, a lower threshold decides
what may join it, and only regions touching a certain one survive. It degrades gracefully by
construction: with nothing adjacent to grow into it reduces to the current mask exactly.

| arm | mean coverage | median | min | cameras under 20% |
|---|---|---|---|---|
| current | 60.1% | 64.0% | 26.1% | 0 |
| flat-field | 65.2% | 71.2% | **2.6%** | 1 |
| grown | 67.0% | 71.1% | 42.7% | 0 |

**On the corpus, growth looks like a mild win.** 923 labelled frames:

| | n | current | flat-field | grown |
|---|---|---|---|---|
| EMPTY frames still finding nobody | 243 | **88.9%** | 87.7% | 87.7% |
| venue_01 play, mean count | 278 | 6.37 | 6.44 | 6.44 |
| clip venues, mean count | 396 | 9.82 | 10.38 | 10.36 |
| clip venues, frames in the 1-4 band | 396 | 9 | 5 | **4** |
| recorded C3 frames with anybody inside | 6 | 3 | 0 | **4** |

Halving the 1-4 band across nine unseen venues is a direct improvement to A18: that band is
where the C3 rule's unsound clause is consulted, and it was the rule's stated weak point.
Three EMPTY frames of venue_01 safety for five fewer cross-venue danger frames is the sort of
trade this project has usually taken.

**End to end on the clip it is plainly worse, and that is the measurement that decides it.**
Through the boundary and all three gates:

| | coverage | wrong verdicts on the 13 empty minutes | minute 15, one person, no ball |
|---|---|---|---|
| current | 56% | **0** | C3, correct |
| grown | 85% | **2** (minutes 0 and 13 called C3) | **ACTIVE_PLAY, wrong** |

The recovered area is not only pitch. It reaches up to the barrier at the top of the frame,
where three people stand watching, and admitting them turns two empty minutes into C3 and
promotes the minute with one walker to a match. Minute 12 - the failure that started this - is
**not fixed by either arm**.

**The finding is not "the boundary is fine".** It genuinely does clip a third of that pitch.
The finding is that the clipped area is worth less than the off-pitch area that comes with it:
the boundary's value is in what it excludes, and a rule that recovers real pitch by relaxing a
threshold recovers the touchline and the barrier at the same rate. `turf_mask` is unchanged.
Both alternatives stay in `derive_roi.py` as the measured comparison behind that decision,
since a rejected path with a number attached is worth more than the same decision with nothing
behind it.

**What would actually fix it** is a boundary that is not a threshold on colour - the pitch's
line markings are a stronger geometric cue than its greenness, and four touchlines define the
surface exactly where a convex hull of green pixels only approximates it. That is a different
piece of work and it is the honest recommendation, not a tuned `GROW_FRACTION`.

**A correction to A18.** That entry attributed minute 12 to "a detector limit, not a rule
limit". More precisely: the detector finds the person, the boundary excludes them, and
widening the boundary enough to include them costs more than it returns. The limit is the
boundary's shape, and it is not repairable by loosening it.

## The headline cross-venue failure, measured on the system instead of the probe (A20)

> **Corrected below, and the correction is larger than the entry.** False-play
> is the right number and it is not the whole error: the gate also answers C3 on empty
> pitches, which false-play does not count. See *"false-play was the wrong denominator"*.

`experiments/h3_with_gates.py`, `results/h3_with_gates.csv`

Every cross-venue number this project reports describes **the probe**. The deployed path has
not been the probe alone since A14 - `run_slot` classifies, then two gates may overrule the
verdict, and it is the overruled verdict that reaches a user. So the failure the whole thesis
turns on, **false-play 0.6173 at held-out venues even after pruning to distinct scenes**, has
never been measured on the thing that actually runs.

It is worth measuring here and nowhere else. The gates were built for this error and every
other protocol is too clean to show it: venue_01 camera B has false-play 0.0000, which is why
RQ6 and the A16 recheck both came back "0 of 521 verdicts overruled". H3 is the one protocol
where a gate has anything to do.

Seven leave-one-venue-out folds, the same 243-frame false-play control, person gate and its
A18 extension applied to the probe's verdicts:

| arm | play-recall | false-play | balanced |
|---|---|---|---|
| full, probe | 0.9444 | 0.7684 | 0.1761 |
| full, **gated** | 0.9436 | **0.0123** | **0.9312** |
| pruned, probe | 1.0000 | 0.6173 | 0.3827 |
| pruned, **gated** | 0.9991 | **0.0123** | **0.9868** |

**0.6173 to 0.0123, for 0.0009 of recall.** The arithmetic is not mysterious and is worth
writing out, because it shows the number is not a fluke of one fold: of the 243 control frames,
216 (88.9%) have nobody inside the boundary and become EMPTY, 24 (9.9%) have one to four people
and no ball and become C3, and **3** survive as PLAY. Those three are the frames with a
bystander *and a ball* - the ball clause vetoing the C3 call, doing exactly what it is for.

**The caveat is not small and it is not the usual one.** The recall half of this table is a
transfer result: those are clip-venue frames at venues held out of training. The false-play
half is **not**. Every constant the gate uses - the 1280-pixel detector size, the 0.25 person
confidence, the 0.10 ball confidence, `small_group_max = 4` - was read off distributions
measured on these same 243 frames. The gate is out of sample with respect to the *probe's*
training, and in sample with respect to its own thresholds. **0.0123 is a floor, not an
estimate.**

**The one out-of-sample check that exists agrees, and it is thin.** On the unseen clip - a
venue never in any split, thresholds not derived from it - the same gates give **0 false-play
on 13 empty minutes and no wrong C3 calls**. Thirteen minutes is not a false-play rate. It is
the only recorded empty pitch outside venue_01 that this project has, which is the wall every
measurement here has hit for six weeks.

**What this changes in how the thesis reads.** The cross-venue result is no longer "the probe
transfers its recall and not its precision". It is: *the probe transfers its recall; its
precision is supplied by a detector that never saw this dataset, and the combination is
reportable while the probe alone is not.* That is a weaker claim about the linear probe and a
stronger one about the system, and it should be written that way rather than as a repaired
probe number.

**What would make it an estimate rather than a floor** is unchanged and is the same sentence
as always: recorded empty pitches at a venue that is not venue_01. Two hours of footage of one
unused pitch at one new site would convert the strongest claim in this project from a floor
into a measurement.

## The gates cost 18% of the cycle at worst, and the detector was being rebuilt every frame (A21)

`experiments/gate_latency.py`, `results/gate_latency.csv`

`efficiency_latency.csv` reports 20 cameras in 2.5-5.7 s against the 60 s cycle, and it
measures **the probe**. Since A14 the deployed path is probe plus motion gate plus person gate,
and the person gate runs a detector, so the reported throughput describes a system that stopped
existing three amendments ago. RQ1 and RQ2 both quote it.

**A defect found by measuring rather than by reading.** `detect_objects` called
`YOLO(model_name)` on every frame. On eight 1080p frames at imgsz=1280:

| | median per frame |
|---|---|
| constructing the model each call | **292 ms** |
| reusing a loaded model | **152 ms** |

The construction cost more than the inference. Across 20 cameras that is 2.8 s of every cycle
spent loading the same weights twenty times. The detector is now cached per thread - per
thread, not globally, because `evaluation/latency.py` runs the pipeline on several threads to
measure concurrency and an ultralytics model is not documented as safe to predict on from more
than one. Two tests pin it, because nothing else would notice: the outputs are identical either
way, only the clock changes.

**What a round costs now.** The detector runs only on cameras whose verdict survived as
ACTIVE_PLAY, so the cost depends on how busy the site is - an empty site pays nothing:

| stage | median | p95 |
|---|---|---|
| backbone embed (DINOv2, one frame) | 332 ms | 340 ms |
| motion cue (160x90 difference) | 4.1 ms | 4.3 ms |
| person gate detector (imgsz 1280) | 193 ms | 242 ms |

| cameras in play | round | of the 60 s cycle |
|---|---|---|
| 0 of 20 | 6.7 s | 11% |
| 5 of 20 | 7.7 s | 13% |
| 10 of 20 | 8.7 s | 14% |
| **20 of 20** | **10.6 s** | **18%** |

**The deployment claim survives, with room.** Worst case - every camera mid-match, every one
paying for a detector pass - is 10.6 s of a 60 s cycle. The motion gate is free at 4 ms and
sits first for that reason; it is the person gate that costs, and it costs only where it can
change the answer.

**This is a laptop and it is not the deployment claim**, the same caveat `efficiency_latency.py`
carries: WP7-T1's run on the target Mini-PC is what settles it. What this establishes is the
*shape* - that the gates add a term proportional to the play rate rather than a constant, and
that the term is smaller than the backbone's - which is machine-independent in a way the
seconds are not.

## False-play was the wrong denominator, and the probe is worse than 0.6173 said (A20, corrected)

`experiments/h3_with_gates.py`, `results/h3_with_gates.csv`

The A20 entry above reports false-play falling from 0.6173 to 0.0123 and is arithmetically
right. It is also the wrong question, and an audit of the 24 control frames the gate answers
C3 on is what showed it.

**Those 24 frames are labelled correctly and the gate is wrong on them.** Rendered with the
person boxes drawn, the pitch is plainly empty in every one; the people the detector finds are
standing behind the goal on the car-park tarmac, beyond the fence. The boundary for
`slot_20260711_1000_camB` reaches past the goal line, so people who are not on the pitch are
counted as being on it, and A18 turns that into "present but not playing". False-play does not
notice, because C3 is not a play verdict - so the metric credits the gate for every frame it
mislabels as maintenance.

**With every wrong verdict counted, not just the play-shaped ones:**

| arm | play-recall | false-play | **any wrong verdict on an empty pitch** |
|---|---|---|---|
| full, probe | 0.9444 | 0.7684 | 0.8342 |
| full, gated | 0.9436 | 0.0123 | **0.1617** |
| pruned, probe | 1.0000 | 0.6173 | **1.0000** |
| pruned, gated | 0.9991 | 0.0123 | **0.4844** |

**The probe never answers EMPTY. Not rarely - never.** In all seven pruned folds, across all
243 control frames, the predictions are only ever ACTIVE_PLAY or C3:

| held-out venue | says PLAY | says C3 | says EMPTY |
|---|---|---|---|
| clipvenue_a_blue_barrier | 75 | 168 | **0** |
| clipvenue_d_indoor_dome | 181 | 62 | **0** |
| clipvenue_e_pink_boards | 166 | 77 | **0** |
| clipvenue_f_outdoor_bldg | 198 | 45 | **0** |
| clipvenue_g_netting | 144 | 99 | **0** |
| clipvenue_h_teal_pitch | 108 | 135 | **0** |
| clipvenue_i_outdoor_trees | 178 | 65 | **0** |

0.6173 read as "the probe is wrong about 62% of unseen empty pitches". The truth is that it is
wrong about **all** of them, and 0.6173 was measuring only which *kind* of wrong. A probe whose
EMPTY training evidence is five scenes, most of them generated, does not have a concept of an
empty pitch to transfer.

**And the gate fixes half of that, not all of it.** It only ever weakens ACTIVE_PLAY, so a
frame the probe calls C3 passes through untouched - which is 38% of the control frames on
average and 69% in the worst fold. Wrong verdicts fall from 1.0000 to 0.4844, which is a real
improvement and is not the 0.0123 that false-play alone suggested.

**What A20 should have claimed, and what the thesis should say.** The gates remove the
*play-shaped* error at unseen venues almost entirely, and that is the error that matters for
pitch utilisation, since a false ACTIVE_PLAY inflates reported usage while a false C3 does not.
They leave a second error the probe makes just as often and that nothing currently addresses.
The honest headline is two numbers, not one: **false-play 0.6173 to 0.0123, total error 1.0000
to 0.4844.**

**Two repairs this points at, neither of them adopted here.** The boundary for venue_01
camera B should stop at the goal line - A19 measured *loosening* boundaries and found it
harmful, and this is the same finding from the other side, on a camera whose outline is already
too loose. And the gate's "nobody inside the boundary means EMPTY" rule is restricted to
ACTIVE_PLAY inputs for no reason stronger than caution; the evidence behind it - 89% of empty
frames show nobody, 0.4% of play frames - does not depend on what the probe said first.

## The empty-pitch rule applied to C3 verdicts too, which was the biggest single fix available (A22)

`experiments/h3_with_gates.py`, `results/h3_with_gates.csv`

The A20 correction ended with two repairs it had identified and not taken. This is the first.
`PersonGate` refused to act unless the probe had said ACTIVE_PLAY - a restriction inherited
from A16, where the gate was introduced as a brake on false play. Nothing in the evidence
behind the rule supports it. "Nobody inside the boundary means the pitch is empty" rests on 89%
of recorded empty frames showing nobody and 0.4% of recorded play frames doing so, and neither
number has anything to say about what the probe guessed first.

It mattered because the probe guesses C3 constantly. On the 243 recorded empty control frames
it answers ACTIVE_PLAY or C3 and **never EMPTY**, C3 on 38% on average and 69% in the worst
fold - all of which the gate was waving through.

| arm | play-recall | false-play | any wrong verdict on an empty pitch |
|---|---|---|---|
| pruned, probe | 1.0000 | 0.6173 | 1.0000 |
| pruned, gated *before A22* | 0.9991 | 0.0123 | 0.4844 |
| pruned, gated *after A22* | 0.9991 | 0.0123 | **0.1111** |
| full, gated *after A22* | 0.9436 | 0.0123 | **0.1011** |

**0.4844 to 0.1111 for no change in false-play and no change in recall.** It is the largest
single improvement any change in this project has produced, and it is a deleted condition
rather than a new idea - the rule was already right, it was being asked to justify itself
twice.

**What it costs, on the only evidence that can price it.** Three of the six recorded C3 frames
now read EMPTY: the ones where the person is outside the boundary, which the gate cannot
distinguish from nobody being there. That is half the recorded C3 corpus, and the corpus is six
frames from one slot at one camera. The trade is ~91 corrected frames per fold against 3 lost,
and it is taken on that arithmetic while stating plainly that the 3 are better evidenced per
frame than the 91 are per frame.

**A labelling question sits underneath and is not mine to settle.** Those three frames show
someone beside the pitch, not on it. For a system whose question is *was this pitch used*, a
verdict of EMPTY may be the right answer and the label the loose one. The protocol says
"people present, not playing"; whether "present" means present *on the pitch* is a supervisor
decision, and the boundary already assumes one answer.

**The ladder, made explicit.** The three classes order by how much activity they claim -
ACTIVE_PLAY above C3 above EMPTY - and every overrule the gate makes is a step down it: A16
play to empty, A18 play to C3, A22 C3 to empty. A test now checks that property over every
combination of count and ball rather than arguing it in a comment, so no detector noise can
manufacture a busier pitch than the probe reported.

**Unchanged on the unseen clip**: 0/13 false-play, minutes 9 and 15 still C3, slot verdict still
NOTUSED. Minute 12 still wrong, and A19 already established that it is the boundary's shape.

**Cost to the cycle, re-measured because the trigger changed.** The detector now runs on any
verdict except EMPTY rather than only on ACTIVE_PLAY, so A21's table was re-run: 5.5 s with
every camera already empty to **8.2 s with none of them**, 9% to 14% of the 60 s cycle. That
reads *lower* than A21's 6.7-10.6 s for the same work, which is machine load on a laptop and
is exactly what both entries warn about - the shape is what transfers, not the seconds. An
idle site still pays nothing, because an EMPTY verdict returns before the detector is reached.

## Tightening the boundary fails the same way loosening it did (A23)

`experiments/roi_flat_field.py`, `results/roi_flat_field.csv`

The second repair the A20 correction identified: venue_01 camera B's outline reaches past the
goal line into the car park, which is why 24 recorded empty frames read as C3. `derive_roi.py`
has always named the cause - the outline is a **convex hull**, and a hull cannot exclude
anything lying inside its span. Seen at an angle, the hull runs from the pitch's far corner to
its near one and swallows the tarmac between them.

So the obvious repair: read the same mask without the hull, following the region's own outline.
Nothing else changes - same median, same threshold, same morphology - so this isolates what the
hull costs.

| | n | current | contour |
|---|---|---|---|
| venue_01 EMPTY, finds nobody | 243 | 88.9% | **92.2%** |
| venue_01 EMPTY, in the 1-4 band (a wrong C3) | 243 | 27 | **19** |
| clip venues, finds nobody (a wrong EMPTY) | 396 | 0 | 2 |
| **clip venues, in the 1-4 band** | 396 | **9** | **55** |
| mean coverage across 69 cameras | | 60.1% | 54.4% |

**It fixes what it was aimed at and costs six times more elsewhere.** Eight fewer wrong C3
calls at venue_01, against **46 more** cross-venue play frames dropping into the 1-4 band -
the band where A18's unsound clause is consulted - plus the first two clip frames ever to
report an empty pitch during a match. The median count at the nine clip venues falls from 10 to
8: the contour clips players standing on the parts of the pitch the mask reads as slightly less
green, which the hull was bridging over.

**Both directions now fail, and that is the finding.** A19 loosened the boundary and lost
cross-venue accuracy by admitting spectators. A23 tightens it and loses cross-venue accuracy by
dropping players. The current hull is not a good boundary - it demonstrably contains a car park
- it is a **local optimum of a colour threshold**, and the two experiments bracket it from
either side.

**What the mask actually contains, looked at rather than inferred.** Rendered over the median
frame, the excess-green region at venue_01 camera B includes the trees beyond the fence at the
top left and the vegetation past the far touchline. Neither the hull nor the contour can help
with that: they are two ways of simplifying a mask that has already decided a tree is turf.

**So the recommendation is unchanged and is now measured from both sides.** A boundary from the
pitch's **line markings** - four touchlines bound the playing surface exactly, and they are
white on green under any lighting these cameras see - rather than from the colour of the
surface. Every remaining boundary error in this project is a colour error: a tree read as turf,
a dim far end read as not-turf, tarmac bridged by a hull. None of them is a geometry error.
`polygon_from_mask(hull=False)` stays as the measured alternative behind this entry.

## The line-marking boundary, attempted (A24)

`experiments/line_marking_boundary.py`, `results/line_marking_boundary.csv`

A19 and A23 bracket the colour-threshold boundary from both sides and both end with the same
recommendation: take the boundary from the pitch's **line markings**, which bound the playing
surface exactly and are white on green under any lighting these cameras see. Recommending it
three times without trying it is not a finding. This is the attempt, and it does not land.

**Markings are detectable.** A white top-hat - brighter than a neighbourhood wider than the
line - plus a low-saturation test, on the per-camera median. Rendered over venue_01 camera B it
picks out the halfway line, the penalty box and the far touchline unmistakably.

**And so is everything else bright and thin.** Fences, netting, window frames, building edges.
Across 14 cameras the hull of unrestricted "marking" pixels covers **81-94%** of the frame,
which is no boundary at all. Gating on the turf mask brings that to 44-66% - and inherits the
turf mask's defect, because `turf_mask` calls the trees beyond the fence turf, so tree
highlights arrive as markings.

**Requiring straightness removes the contamination and most of the markings with it.**

| stage | coverage across 14 cameras |
|---|---|
| current boundary | 55-79% |
| raw marking pixels | 81-94% |
| restricted to the turf mask | 44-66% |
| **only long straight segments** | **7-25%**, and 3 of 14 produce nothing at all |

**None of the 11 that produced a polygon lands within 10 percentage points of the current
boundary.** At venue_01 camera B, `HoughLinesP` at a minimum length of 12% of frame width finds
**7** segments and their hull covers 14% against the current 55%; every one of the 7 is in the
far half of the pitch.

**Why it fails is specific, and is the part worth keeping.** A single top-hat kernel matches a
line of one width. Under perspective a touchline is several pixels across at the near edge of
the frame and sub-pixel at the far edge, so one kernel is right for one horizontal band and
wrong above and below it. At camera B the kernel suits the far half, which is exactly where the
segments were found. Lowering the threshold to catch the near half brings the fence back in.

**What the next attempt should do differently.** Either scale the kernel width with image row -
perspective is a smooth function of height for a fixed camera, and the median frame is stable
enough to estimate it - or fit a homography from a pitch template to the detected segments,
which uses the fact that markings are not merely lines but *a known arrangement of lines*. The
second is the standard method in sports analytics and needs most of a pitch outline in frame,
which most of these cameras do not have.

**Nothing is wired in.** What changes is the status of the recommendation: it was an untested
idea appearing in three entries and is now a tested one with a named failure mode, which is
worth more even though the boundary is no better than it was this morning.

- 2026-09-17 | H3 | `python experiments/h3_cross_venue_recall.py` | seed 42 | `h3_cross_venue_recall.csv` | 7 folds x 4 models

- 2026-09-17 | H1/H2 | `python experiments/h1_h2_baseline_floor.py` | seed 42 | `h1_h2_baseline_floor.csv` | 14 rows over 2 splits

## The lighting labels were wrong, and correcting them reverses a headline claim (A25, WP3-T5)

`scripts/relabel_clip_lighting.py`, `results/h3_cross_venue_recall.csv`, `results/h3_with_false_play.csv`

`rq_matrix.md` has carried this warning since A12: *"Do not quote 0.219 as the clock rule's
cross-venue recall. It is a lower bound partly produced by label error ... the exact value
waits on the hand relabelling in TODO WP3-T5."* This is that relabelling. The exact value is
**1.000**, and it beats every backbone.

**The bug.** `data/extract.py` assigns a clip frame's lighting by mean greyscale brightness,
below 80 is night, "calibrated against the known day/night recordings in raw/venue_01". That
calibration does not survive leaving venue_01. A floodlit five-a-side pitch fills most of its
frame with intensely lit turf, so its mean brightness is **higher** than an overcast afternoon
at venue_01 - `clipvenue_b_floodlit_track` reads 120 against venue_01's daytime 69. The rule
measures how bright the picture is, not whether it is daytime, and files night football as
daylight.

**The audit.** One rendered frame per venue, judged by eye, evidence recorded per venue in the
script so the judgement can be disagreed with:

| venue | was | what the frame shows |
|---|---|---|
| a_blue_barrier | night | dark sky above the barrier; already correct |
| b_floodlit_track | **day** | black sky, floodlight fixtures visibly lit |
| c_teal_boards | **day** | burned-in timestamp reads `08-23-2026 Sun 21:02` |
| d_indoor_dome | **day** | air-dome interior, ceiling lights on, no daylight |
| e_pink_boards | **day** | indoor hall, ceiling lights, no sky in frame |
| f_outdoor_bldg | **day** | black sky, floodlight flare; the instance A12 already knew |
| g_netting | **mixed** | dark sky through the netting, floodlights above |
| h_teal_pitch | **day** | dark sky beyond the cage, floodlights along the top |
| i_outdoor_trees | **mixed** | night sky behind the trees |

**Not one of the nine clip venues shows daylight.** 216 of 396 recorded clip frames were
mislabelled. After correction, **every daylight frame in this corpus is venue_01's.**

**The reversal.** Re-running H3 unchanged except for the labels:

| model | cross-venue play-recall | false-play | was |
|---|---|---|---|
| **clock_rule** | **1.0000** | **0.0206** | 0.219 |
| dinov2 | 0.9297 | 0.3086 | 0.930 |
| convnextv2 | 0.9105 | 0.9918 | 0.910 |
| vit | 0.8690 | 0.8354 | 0.869 |

**A rule that never looks at the image beats all three frozen backbones on both axes at once** -
perfect recall, and a false-play rate fifteen times lower than the best of them. The previous
claim, in `rq_matrix.md` and twice in this log, was *"a lighting-only rule collapses across
venues while the backbones hold above 0.86"*. That claim was an artefact of the label error and
is **refuted**.

**Why it wins, and this is the part that matters more than the table.** The confound is not a
labelling mistake; it is how five-a-side pitches are used. People play in the evening and the
pitch is empty during the day:

| | C1_EMPTY | C2_ACTIVE_PLAY | C3 |
|---|---|---|---|
| day | **485** | 6 | 6 |
| night | 9 | **1186** | 0 |

**"Night means play, day means not-play" is correct on 1,677 of 1,692 recorded frames - 99.1%.**
Within venue_01 alone, which is the only venue with both classes, it is 98.8%. The corpus
cannot separate *recognises an empty pitch* from *recognises daylight*, and `confound_warnings`
now says so for every class:

> EMPTY: 98% of 494 frames come from a single lighting condition (day)
> ACTIVE_PLAY: 99% of 1192 frames come from a single lighting condition (night)

**What this does to the thesis, stated plainly.** It removes a claim about backbones and
strengthens the methodological contribution, which was always the better half. H3's folds are
100% ACTIVE_PLAY and now demonstrably 100% night, so H3 cannot distinguish a model from a
constant - the false-play control was added precisely because recall alone could not, and it is
the only reason the clock rule's 0.0206 is visible beside its 1.000. The protocol critique this
project is really about now has its sharpest example: *the strongest trivial baseline beat three
frozen backbones on the project's own cross-venue protocol, on both axes, and the only thing
that made that visible was a control the pre-registration did not originally require.*

**And it sharpens the data request, which has been vague for six weeks.** "Recorded empty
pitches at another venue" is not the whole gap. The corpus needs **empty pitches at night** and
**play in daylight** - the two cells that hold 9 and 6 frames. Without them no experiment here
can tell an occupancy model from a light meter, at any venue, including venue_01.

**Not corrected.** Generated frames keep `lighting = unknown`: the audit is of recorded footage,
and what light a generated image depicts is the generator's business. The binary field also
cannot distinguish floodlit-outdoor from indoor-artificial, which are two different conditions
now both filed as `night`; that is a real limitation and is not repaired here.

**H2 changes with it, and the direction is the same.** The grouped-split H1/H2 table was
regenerated - `benchmark_v2.py` refused to extend it first, because its reproduction guard
found four published numbers no longer matching, which is the guard working:

| grouped split, seed 42 | macro-F1 before | after |
|---|---|---|
| clock_rule | 0.4907 | **0.4975** |
| convnextv2 | 0.4975 | 0.4975 |
| vit | 0.4975 | 0.4975 |
| dinov2 | 0.5794 | 0.5794 |

The clock rule no longer sits 0.007 behind ConvNeXtV2 and ViT - it is now **exactly equal to
both**, to four decimals, because all three collapse to the same predictions. And across five
seeds DINOv2's lead over it moves from **+0.089, p_holm 0.0037, "differs"** to **+0.0819,
p_holm 0.375, "indistinguishable"**. The point estimate barely moved; the significance did.

So the sentence *"only DINOv2 clears the trivial floor by a meaningful margin"* no longer has a
significance test behind it. H2 as pre-registered - within 2 macro-F1 points - is still refuted
on the point estimate, and the claim that the refutation is meaningful is not.

- 2026-09-17 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 3 gate(s) met on artefacts, 2 waiting on a person

- 2026-09-17 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 33 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-17 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 34 claims verified against their artefacts, 0 recorded as unsupported

## The baseline that beats the backbones gets every minute of real footage wrong (A26)

`experiments/clock_rule_on_video.py`, `results/clock_rule_on_video.csv`

A25 left the project with an uncomfortable headline: on the corrected labels a rule that never
looks at the image beats all three frozen backbones cross-venue, on recall **and** on
false-play. Read at face value that says the deep half of this work is unnecessary and the
gates on top of it doubly so.

It should not be read at face value, and the reason is checkable rather than rhetorical. The
rule wins on a confound - night is 99% ACTIVE_PLAY, day is 98% EMPTY - and **the corpus holds
almost no frames where that shortcut fails**: 9 empty frames at night, 6 play frames in
daylight. The unseen clip is such a frame sixteen times over: floodlit night, nobody playing.

Three systems, the same 16 minutes, the same boundary:

| system | says PLAY | false-play | empty minutes correct |
|---|---|---|---|
| clock rule | **16 of 16** | **1.00** | **0 / 13** |
| probe alone (DINOv2) | 7 | 0.38 | 8 / 13 |
| **deployed path** (probe + boundary + motion + person gates) | **0** | **0.00** | **13 / 13** |

**The rule is wrong on every minute of the clip.** Not degraded - inverted. It cannot be
otherwise: it reads one variable, that variable says "night", and at night this corpus is 99%
football.

**And it was given the charitable setting.** It is told `lighting="night"`, which is the truth
by inspection - black sky, lit fixtures. Told `"day"` it would answer EMPTY to all sixteen and
score 13/13 on the empty minutes for exactly the wrong reason, which is worth stating because
it shows the rule has no way to be right *for a reason*.

**What this settles and what it does not.** It does not restore the backbone claim A25
refuted: on the corpus, DINOv2 really is indistinguishable from a light meter, and that
remains the honest reading of every corpus number. What it settles is which system to deploy,
and the ordering is unambiguous on the only footage in this project that contains the missing
cell: 0.00 false-play against 0.38 against 1.00.

**The methodological point, which is the thesis's contribution and now has its cleanest
statement.** A benchmark of 1,692 frames ranked the clock rule first and the deployed system
below it. **A single 234-second clip reversed that ranking completely.** The difference between
them is not size, sophistication or statistics - it is that the clip contains the class-lighting
combination the benchmark does not. No amount of care in the protocol compensates for a cell
the data never fills, and no amount of data compensates for never checking the model on
something it has not seen.

**It also prices the gates for the first time against a real alternative.** The probe alone is
0.38 false-play here; with the boundary and the two gates it is 0.00. Everything A14 through
A22 added is the difference between a system that is wrong on five of thirteen empty minutes
and one that is wrong on none.

- 2026-09-17 | WP4-T1 benchmark v2 | `python experiments/benchmark_v2.py` | `benchmark_v2.csv` | 8 models x 4 protocols; every protocol degenerate, each differently

**A25 addendum: the relabelling is contained to one model, and it moves that model everywhere.**
`benchmark_v2.py` was re-run once its published reference table was regenerated. Of its 153
rows, **only the clock rule's changed** - it is the only model that reads the `lighting` field,
so this is the containment check the change needed. Across five seeds on the random split its
macro-F1 goes from **0.9091 to 0.9865**, which moves it past ConvNeXtV2 (0.9849) and to within
0.0008 of DINOv2 (0.9873). On `lo_venue_out` it ties the majority-class baseline at a perfect
1.0000, which that protocol has always awarded to constants and is why it is reported as a
degenerate protocol rather than a result. The claims ledger re-derives 34 of 34.

**A25 follow-up: the rule that produced the bad labels is removed, not re-tuned.** Correcting
216 rows fixed the data and left the code that generated them in place, so a re-extraction
would have restored the defect. `data/extract.py` no longer infers lighting at all. It writes
`unknown`, which is what extraction actually knows, and keeps the `brightness` column, which was
a measurement and was never the problem - the inference from it was.

**No threshold replaces it, deliberately.** An indoor hall has no sky to be dark and a floodlit
pitch is brighter than an overcast one, so there is no cutoff that separates them; the earlier
attempt to find one is what filed 216 frames wrong. The judgement now lives in
`scripts/relabel_clip_lighting.py`, per venue, with the visual evidence written beside it.

`tests/test_extract.py::test_lighting_is_measured_not_assumed` asserted the old behaviour, and
its name was the mistake in miniature: the brightness *was* measured, and the conclusion drawn
from it was assumed. It is now `test_extraction_does_not_guess_the_lighting`, and it still
checks that the bright fixture reads brighter than the dark one - measuring was never what went
wrong.

## The locked final test set cannot answer the question it was locked for (A27)

`results/splits/FINAL_TESTSET_venues.csv`, `data/splits.final_test_rows`

The final test set has never been opened - `final_test_rows` still refuses without
`i_have_finished_all_development=True`, and no experiment has passed it. That guard has worked
all project. This entry is about what is *inside* it, which is readable from the manifest
without spending it, and which should be known before anyone does.

**Two venues, 114 frames, 19 clips - and one class.**

| | |
|---|---|
| frames | 114 |
| classes | **C2_ACTIVE_PLAY: 114.** No EMPTY. No C3. |
| lighting | **night: 114.** No daylight. |
| venues | clipvenue_b_floodlit_track (78), clipvenue_c_teal_boards (36) |

**What it can measure.** Play-recall at two unseen venues. That is the whole list.

**What it cannot measure, and what the thesis asks.** False-play, precision, macro-F1, accuracy,
the EMPTY class, the C3 class, any day-versus-night comparison, and the operating point RQ6
reports. **A model that answers ACTIVE_PLAY unconditionally scores 1.000 on it** - and A25
established that this is not hypothetical, since the clock rule does exactly that and would
score a perfect 1.000 here too.

**So spending it buys a confirmation, not an answer**, and the confirmation is of the one thing
already least in doubt: cross-venue play-recall is 0.93 for DINOv2 over seven development
venues, and the count rule finds people in 100% of 396 clip-venue frames. Nothing in this
project suggests two more venues would disagree.

**The recommendation, and it is a supervisor's decision.** Spend it once, at the end, reported
as what it is - a held-out confirmation of play-recall at two venues - and never as a headline
accuracy. The pre-registration should say so *before* the number exists, because a single-class
test set produces a flattering figure for any model and the time to disclaim it is now.

**What would make it a real test** is the same two cells A25 named: **an empty pitch at night**
and **play in daylight**. Two hours of a locked venue's footage with nobody on the pitch would
turn 114 single-class frames into a final test set that could refute something. It is the same
request as always, now with a specific address - it must be footage from
`clipvenue_b_floodlit_track` or `clipvenue_c_teal_boards`, because any other venue is
development data and cannot be added to a locked set after the fact.

## The probe recognises an empty pitch at night; the clock rule does not (A28)

> **Corrected by A29 (same day): 18 of 26, not 5 of 5.** These five frames came from
> six-frame medians at one venue and were the easy survivors of a weak filter. Whole-clip
> medians give 26 frames at three venues and the probe gets 69% of them, not 100%. The
> ordering against the clock rule stands; the reliability claim does not.

`scripts/make_median_empties.py`, `experiments/median_empty_night.py`, `results/median_empty_night.csv`

A25 left DINOv2 indistinguishable from a light meter on every corpus number, and A26 showed the
light meter fails on real footage while the deployed system does not. What neither could do is
test the probe *itself* on the missing cell, because the corpus holds 9 empty frames at night
and all nine are venue_01's.

That cell can be manufactured, by the one synthesis operation this project has shown works.
Generated EMPTY frames repaired the false-play collapse and generated C3 frames did nothing,
because the first were **subtraction** and the second addition. A temporal median is subtraction
with no model at all: players move, the pitch does not, so the per-pixel median of a clip is
that venue's pitch with the people gone, in its own pixels at its own exposure.

**A third guard, because the first two were proxies.** `make_median_empties.py` already refused
a median that resembles no frame (residual) and one that nothing moved in (spread). Both reason
about pixel differences and *infer* whether the people went. A detector asks directly, and on
the 11 frames the thresholds passed it found people inside the boundary on **four** - two people
on two of the `clipvenue_a` medians, one on a third, and **eight** on the `clipvenue_e` median,
which is the same venue whose survivors prompted the spread check in the first place. Tightening
a threshold was never going to fix that; it was the wrong instrument. **5 frames survive all
three**, from `clipvenue_a_blue_barrier`, floodlit night.

**What the frames can and cannot test, fixed before the numbers were looked at.** They cannot
test the person gate or anything built on the detector - the detector *selected* them, so it
scores perfectly by construction and the figure would mean nothing. It is not reported. They
can test a probe and the clock rule, neither of which had a say in the selection. The probe is
fitted with `clipvenue_a` held out, so the venue is unseen in the ordinary sense too.

| system | calls it EMPTY | calls it PLAY |
|---|---|---|
| **DINOv2 probe** | **5 / 5** | 0 / 5 |
| clock rule | 0 / 5 | **5 / 5** |

**This is the first evidence in the project that the probe reads the pitch and not the clock.**
Every corpus number is compatible with DINOv2 being a light meter, because in the corpus night
means football. Here is night that does not mean football, at a venue the probe never saw, and
it answers EMPTY five times out of five while the light meter answers PLAY five times out of
five.

**Held at its proper weight.** Five frames, one venue, and manufactured ones - every frame is an
average of frames that had people in them, and a median is smoother than a photograph in a way
that might itself read as emptiness. This does not substitute for a recorded empty pitch at
night and is not reported as one. What it does is move "the probe might be a light meter" from
*unexamined* to *examined once, and it was not*.

**With A26 it makes a pair.** The clock rule beats the backbones on 1,692 corpus frames, and
loses to them on every frame this project has that sits outside the corpus's confound: 16 clip
minutes and now 5 manufactured stills, at two different venues, 21 for 21.

## A better median set corrects A28 downward: 18 of 26, not 5 of 5 (A29)

`scripts/median_empties_from_clips.py`, `experiments/median_empty_night.py`

A28 reported DINOv2 calling **5 of 5** manufactured empty-night frames EMPTY and read it as the
first evidence that the probe reads the pitch rather than the clock. The direction survives.
The number does not.

**The medians were built from six frames each**, because that is the rate the dataset sampled
each clip at. Six frames of a ten-second highlight is a thin stack: a slow player, or one who
happens to stand where another stood, survives the median. The source clips hold **250-350
frames**, and a median over all of them is the same subtraction with fifty times the evidence -
a player would have to stand still for ten seconds to survive it.

| | frames | venues | rejected by the detector |
|---|---|---|---|
| A28, six-frame medians | 5 | 1 | 4 of 11 |
| **A29, whole-clip medians** | **26** | **3** | 10 of 66 |

**And the score falls.** Scored leave-one-venue-out, each venue's frames judged by a probe that
never saw that venue:

| | clipvenue_a | clipvenue_e | clipvenue_g | all |
|---|---|---|---|---|
| **DINOv2 probe, says EMPTY** | 17 / 21 | 0 / 1 | 1 / 4 | **18 / 26** |
| clock rule, says EMPTY | 0 / 21 | 0 / 1 | 0 / 4 | **0 / 26** |

**0.69, not 1.00.** A28's five frames were the survivors of a weaker filter at a single venue,
and they were the easy ones. The honest statement is that the probe gets **about two thirds** of
manufactured empty night pitches right at venues it has never seen, and still calls **8 of 26**
of them a match.

**The protocol mattered more than expected and is worth recording.** Holding out all three
venues at once - the first way this was run - gives **12 of 26**, because the fit drops from
1,410 frames over 7 venues to 1,362 over 5. Holding out one venue at a time, which is what H3
does and what this should have done from the start, gives 18. Six of the twenty-six frames
turned on nothing but how much training data the probe was left with.

**What stands from A28 and what does not.** The ordering stands, and it is not close: 18 against
0. The clock rule is wrong on every one of 26 frames at three venues, which is what it must be -
they are night, and in this corpus night means football. So the claim that the probe reads
something other than the clock survives. The claim that it reads it *reliably* does not, and
A28's phrasing - "it answers EMPTY five times out of five" - was a small sample flattering a
real effect.

**Still manufactured, still not a recorded empty pitch at night.** 26 averages of frames that
had people in them, at three venues, filtered by a detector that therefore cannot be evaluated
on them. The cell A25 named is still empty.

## A13's repair reproduces; the median empties do the opposite, and "subtraction" does not explain it (A30)

`experiments/median_empties_as_training.py`, `results/median_empties_as_training.csv`

Two questions on the same seven H3 folds. `development_rows` excludes synthetic frames
entirely, so H3's folds have never contained the generated EMPTY frames - A13's repair was
measured in a different train assembly and had never been checked under this protocol.

| arm | play-recall | false-play | balanced |
|---|---|---|---|
| full | 0.9444 | 0.7684 | 0.1761 |
| **full + generated empties (31)** | **0.9595** | **0.0235** | **0.9360** |
| full + generated, same 3 venues (19) | 0.9206 | 0.0553 | 0.8654 |
| full + median empties (26) | 0.8651 | **0.9618** | **-0.0967** |
| full + both | 0.8696 | 0.1070 | 0.7626 |
| pruned | 1.0000 | 0.6173 | 0.3827 |
| **pruned + generated empties (31)** | 0.9881 | **0.0770** | **0.9111** |
| pruned + generated, same 3 venues (19) | 0.9966 | 0.1699 | 0.8267 |
| pruned + median empties (26) | 0.9957 | 0.6537 | 0.3420 |

**A13 reproduces, and more strongly than it was reported.** Thirty-one generated EMPTY frames
take cross-venue false-play from 0.7684 to **0.0235** while *raising* play-recall, and from
0.6173 to 0.0770 on the pruned side. It is the largest single training-set effect in this
project and it now holds under the protocol the thesis reports cross-venue numbers from.

**The median frames do the opposite.** Twenty-six real-pixel empty night pitches, built by
exactly the subtraction A13 credited, push false-play from 0.7684 to **0.9618** - worse than
using no extra empty frames at all - and cost 8 points of recall. Adding them alongside the
generated frames drags 0.0235 to 0.1070. They are not neutral; they are harmful, and they are
**not** going into any training set.

**Two explanations tested and both wrong.**

*Venue coverage.* The generated set spans six venues and the medians three, so the comparison
might have been 31-from-6 against 26-from-3. Restricting the generated frames to the medians'
own three venues leaves **19** frames - fewer than the medians - and they still give 0.0553
against 0.9618. It is not how many venues the frames come from.

*Smoothness.* A median of 300 frames ought to be blurrier than a photograph, and a probe
learning "smooth means empty" would not transfer to venue_01's sharp recorded empties. Measured
by variance of the Laplacian, the medians are the **sharpest** set of the four - 1978 against
906 for recorded EMPTY - so the explanation is not merely unproven, it points the wrong way.
That comparison is itself confounded: the medians are produced at 960x540 while recorded frames
are downscaled to it from 1080p, and downscaling low-passes. It is reported as a failed
explanation rather than a finding.

**So the lesson this project has been repeating is too simple.** "Subtraction works, addition
does not" was drawn from generated EMPTY frames repairing false-play while generated C3 frames
did nothing. A whole-clip median is subtraction in its purest form - no model, real pixels, the
camera's own exposure - and it fails at the same task that inpainted frames succeed at, at the
same venues, in greater number. Whatever the generated EMPTY frames supply, it is not
*subtraction*, and this project does not currently know what it is.

**What that costs and what it does not.** It does not disturb the deployed system: the
generated frames stay, on evidence that is now stronger than when they were adopted. It does
disturb the explanation attached to them in `synthetic_data_protocol.md` and in three log
entries, which should be read as *an observation about these 31 frames* rather than a principle
about synthesis. A17's account of why ball detection fails, and A13's of why C3 frames failed,
both lean on that principle and are weaker than they read.

## Three explanations for A30 tested and eliminated; the mechanism is open (A31)

`experiments/median_empties_as_training.py`, `results/median_empties_as_training.csv`

A30 left one question: 31 inpainted EMPTY frames take cross-venue false-play from 0.7684 to
0.0235, and 26 median EMPTY frames - real pixels, same venues, the same subtraction the project
credits - take it to 0.9618. Three candidate mechanisms, each testable, each tested.

**1. The generated frames resemble the test set.** If the repair worked by putting training
frames near the control rather than by teaching a class, it would be a leakage-shaped effect and
A13 would need retracting. Cosine similarity in DINOv2 feature space to the 243 control frames:

| set | mean | nearest |
|---|---|---|
| generated EMPTY (31) | +0.739 | +0.777 |
| median EMPTY (26) | +0.725 | +0.742 |
| **recorded clip PLAY (396)** | **+0.723** | +0.752 |
| the control itself | +0.980 | +1.000 |

The generated frames are 0.016 closer to the control than *play frames* are, against a control
that is +0.980 similar to itself. They are not near it in any useful sense. **Rejected**, and
A13 is not a leakage effect.

**2. The medians are too smooth.** A median of 300 frames should be blurrier than a photograph,
and a probe learning "smooth means empty" would not transfer. By variance of the Laplacian the
medians are the **sharpest** of the four sets, 1978 against 906 for recorded EMPTY. The
explanation points the wrong way. **Rejected**, with the caveat that the comparison is
confounded by the medians being produced at 960x540 while recorded frames are downscaled to it.

**3. A median has a near-identical twin in training labelled ACTIVE_PLAY.** This one looked
decisive. A median is built *from* a clip's frames, so it lands beside them: the nearest
development frame to every one of the 26 is an ACTIVE_PLAY frame at cosine **0.939**, and for
**13 of 26** it is a frame of the *same clip*. For the generated frames the nearest neighbour is
also ACTIVE_PLAY at the same 0.937, but the same clip for **0 of 31**. A direct label conflict
between near-identical vectors is exactly the kind of thing a 2,307-parameter probe should
break on.

It makes a prediction, so it was tested: drop the 13 conflicted medians and the harm should go.

| arm | play-recall | false-play |
|---|---|---|
| full + median empties (26) | 0.8651 | 0.9618 |
| full + medians, no twin in train (13) | 0.8810 | **0.9306** |
| pruned + median empties (26) | 0.9957 | 0.6537 |
| pruned + medians, no twin in train (13) | 0.9983 | **0.6267** |

**It does not.** 0.9618 to 0.9306 against the generated frames' 0.0235. The conflict is a real
correlate - 13 of 26 against 0 of 31 - and removing it recovers almost nothing. **Rejected.**

**So the question is open, and the elimination is the contribution.** Whatever the 31 generated
frames supply, it is not proximity to the test set, not photographic texture, and not the
absence of a duplicate-label conflict. The medians also cost play-recall in the full arm, 0.9444
to 0.8651, which a pure addition to the EMPTY class has no obvious reason to do, and both median
arms cost it equally - so the property is of median frames as a kind, not of the conflicted half.

**What is settled, and it is the part that matters for the model.** The generated EMPTY frames
stay, on evidence stronger than when they were adopted. The median frames do not enter training
under any subsetting tried. And the principle written in `synthetic_data_protocol.md` -
"subtraction works, addition does not" - is a description of 31 frames and not a mechanism; four
entries lean on it and should be read that way until something replaces it.

- 2026-09-17 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 39 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-17 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 37 claims verified against their artefacts, 0 recorded as unsupported

- 2026-09-17 | WP8-T5 claims ledger | `python -m experiments.verify_claims` | `thesis/claims.md` | 38 claims verified against their artefacts, 0 recorded as unsupported

## A consolidation pass, and a guard that was not guarding (A32)

`thesis/claims.toml`, `experiments/verify_claims.py`, `experiments/reproduce_all.py`, `README.md`

Today's entries changed what the project's headline claims are, and the documents a supervisor
opens first still described the previous ones. Three things were brought into line, and the
second found a defect.

**The README's status section**, which is hand-written and was last true this morning. It now
states the confound in its sharpest form - *"night means play, day means not-play" is correct on
99.1% of the corpus* - and the pair that follows from it: a rule that never looks at the image
beats all three backbones on the cross-venue protocol and is wrong on every minute of the one
clip that contains the missing cell.

**Five claims added to the ledger**, so today's figures are re-derived from their artefacts
rather than retyped: the gated cross-venue false-play (0.0123) and total error (0.1111), the
clock rule's 1.00 false-play on the clip, the deployed path's 0.00 on the same clip, and the
generated frames' 0.0235 under the H3 folds. The ledger re-derives 39 of 39.

**And two of those five passed for the wrong reason.** `verify_claims.states` checks that a
document actually *states* a claimed value, and its list of legitimate renderings included the
bare `%g` form - which turns 1.0 into `"1"` and 0.0 into `"0"`. The boundary check then matches
any document containing a standalone 1 or 0: "1 person", "0 of 13", a section number. Both new
claims verified against a README that never stated their value.

The fix keeps the bare form only above `BARE_INTEGER_FLOOR = 10`, because a bare integer *is*
how prose states a count - `rq6-tied-confidences` is 890 and "890" is correct for it - while
every rate in this ledger is at or below 1. Tightening it immediately failed the two new claims
and nothing else, which is the behaviour wanted: the README now writes "false-play **1.00**"
and "**0.00**" rather than leaving the reader to infer them.

**The reproduction pipeline objected too, and was right.** `reproduce_all.py` takes the
claims-ledger stage's requirements from the ledger itself, so adding a claim sourced from
`clock_rule_on_video.csv` made the pipeline require a file no stage produced. Five stages were
added - ball detection, H3 with gates, the whole-clip medians, the medians as training data, and
the clock-rule-on-video comparison - with the two that need footage not in the repository
naming the paths they need at the top of the file. A claim whose artefact nothing reproduces is
a claim that cannot be checked by anyone but its author.

- 2026-09-17 | H6 zero-shot vs trained probes | `python -m experiments.h6_zero_shot_gap` | `h6_zero_shot_gap.csv` | declared prompt set; realised family 3; H6 inconclusive; prompt choice spans 0.021-0.747 macro-F1

- 2026-09-17 | H4 model equivalence | `python experiments/h4_model_equivalence.py` | `h4_model_equivalence.csv` | ConvNeXtV2 vs ViT: equivalent at margin 0.02

- 2026-09-17 | WP5-T9 logit-average baseline | `python experiments/logit_average_baseline.py` | `logit_average_baseline.csv` | best single dinov2 0.9297, oracle 0.9603

- 2026-09-17 | WP4-T13 what the false-play control measures | `python experiments/empty_recognition.py` | `empty_recognition.csv` | DINOv2 EMPTY accuracy 0.0000 -> 0.9793 with ONE labelled empty frame of the held-out camera; removing C3 changes nothing - the control is a camera-transfer test

## The staleness check found six stale stages, and one moves a significance result (A33)

`experiments/reproduce_all.py --check`, `results/effective_sample_audit.csv`

A25 rewrote 216 rows of the manifest and four experiments were re-run by hand. That is not a
method for finding out what else moved. `reproduce_all.py --check` compares each stage's output
against the git content-timestamps of its inputs and named **six** stages built before an input
they depend on last changed:

> false-play-significance, effective-sample-audit, h6-zero-shot, h4-equivalence, logit-average,
> empty-recognition

All six were re-run. Five produced byte-identical output, which is the answer the check is
designed to permit - "re-run to be sure" rather than "this is wrong". One did not.

**The effective-sample audit is the project's most careful significance test** - it compares
models on *distinct scenes* rather than frames, because 394 nominal test frames are 62 scenes and
907 are 95. On the corrected labels:

| split | comparison | p (frames) | p (distinct scenes) | was significant | now |
|---|---|---|---|---|---|
| random | DINOv2 vs **clock rule** | 1.000 | 1.000 | yes (8.2e-07, 6.1e-05) | **no** |
| grouped | DINOv2 vs **clock rule** | 0.125 | 1.000 | yes (1.2e-03, 2.1e-02) | **no** |
| grouped | colour histogram vs clock rule | 4.2e-172 | 7.0e-16 | yes | yes |

**DINOv2 is no longer distinguishable from the clock rule on either split, at either grain.**
Before the relabelling it beat the rule significantly on both, four tests out of four; it now
loses all four. The colour histogram is still distinguishable from the rule, and in the opposite
direction - the rule beats *it* decisively on the grouped split.

**This is the third independent confirmation of the same reversal**, and they are not the same
test: H2's five-seed comparison moved from p_holm 0.0037 to 0.375 (A25), H3's cross-venue table
moved from the rule collapsing to the rule winning (A25), and this is the scene-level audit
agreeing with both. Three different protocols, one cause - 216 frames of floodlit night football
filed as daylight.

**What the staleness check earned.** Five stages re-ran to no effect and one moved a reported
significance result that nothing else would have caught. The check is advisory by design and
reads content from git rather than modification times, which is why it could say *re-run to be
sure* without crying wolf: six candidates, one real. That ratio is what makes it worth running
after any change to the manifest, and it is now the thing to run after one.

## The staleness check could not be cleared by doing what it asked (A34)

`experiments/reproduce_all.py`, `tests/test_reproduce_all.py`

A33 used the staleness check, re-ran the six stages it named, and five of them were **still
listed afterwards**. That was written off in passing as the check being conservative. It is not
conservative, it is broken, and the mechanism is exact:

`stale_inputs` compared an input's last content change against the **output's**. Both come from
git, which only records a change when content changes - so a stage whose rerun produces
identical bytes gets no new timestamp and stays flagged for ever. `h6_zero_shot_gap.csv` last
changed on 2026-09-07 and `h1_h2_baseline_floor.csv` changed today; re-running h6-zero-shot a
hundred times would not have moved either date.

**A warning that cannot be cleared by doing what it asks is one the next person learns to skim
past** - which is the failure this module's own docstring warns about, for modification times,
two paragraphs above the code that repeats it in a different form.

**The fix separates two questions the timestamps had conflated.** "When did this output last
change" is what git answers. "When did this stage last run" is what the check needs, and nothing
recorded it. A run ledger now does: `data/interim/stage_runs.json`, written on success and only
after the outputs are confirmed present, so an exit-0 run that produced nothing cannot clear a
warning. It is machine state rather than a result, which is why it lives under `data/` with the
caches and is gitignored - a fresh clone has no ledger and falls back to the git timestamps,
which is the conservative direction.

After the fix, re-running the five settled all five, and the check now reports nothing stale.

**Two smaller defects fell out of it.**

*The new dependency made a passing test read real global state.* `stale_inputs` consults the
ledger, and `test_a_stage_is_judged_on_its_newest_output_not_its_oldest` uses the stage name
`figures` - so the moment a real `figures` entry existed, that test picked it up, passed its
first assertion for the wrong reason and failed its second. An autouse fixture now points
`RUN_LEDGER` at a temporary file for every test in the module. This is the same mistake made
with `vision/roi`'s store in A16 and fixed the same way; isolating one global and leaving the
next one is how it recurs.

*And the new test read a different module.* `tests/test_reproduce_all.py` loads
`reproduce_all` by path under the bare name; `import experiments.reproduce_all` gives a second
module object that the fixture has not patched. The test passed alone and failed in the suite,
which is the signature of exactly that.

## The review page was running the probe alone, and had been since A14 (A35)

`src/pitch_occupancy/clip_analysis.py`, `src/pitch_occupancy/api/clip_review.py`

Reported from use: the clip review page returned **24 of 24 ACTIVE_PLAY at confidence 1.000**
on the 234-second clip of a floodlit pitch with nobody playing - the same footage
`scripts/run_slot_on_video.py` had been getting right all week. Two paths, one model, opposite
answers.

**Both halves of the cause, and the second is the one that mattered.**

`api/clip_review.py` applied a boundary only when the request named a camera *and* the store
held an outline for it. An uploaded clip names no camera, so `polygon` was `None` and the
branch was skipped - the probe scored the neighbouring pitch, the walkway and the car park as
if they were this pitch. And `analyse_clip` never had the gates at all: it takes a
`classify(frame)` callable and applies majority smoothing, which is all it has done since it
was written. `worker.run_slot` gained the boundary and both gates in A14-A22 and this path
gained none of them, so the page has been showing the probe alone for four amendments.

**The diagnosis was settled by a report from use, not by the code.** The observation that a
*still* from the empty part classifies EMPTY while the video says ACTIVE_PLAY ruled out the
model and the preprocessing at once, and pointed at whatever the two paths do differently.
Sequential reads and timestamp seeks were checked against each other first, because
`analyse_clip` seeks by `CAP_PROP_POS_MSEC` where every experiment here reads sequentially, and
a seek that returns the wrong frame would look exactly like this. It does not:

| t | whole frame | inside the boundary |
|---|---|---|
| 60 s | **ACTIVE_PLAY 0.99** | EMPTY 0.97 |
| 120 s | EMPTY 0.72 | EMPTY 0.92 |
| 200 s | **ACTIVE_PLAY 0.96** | EMPTY 1.00 |

Sequential and seek agree to three decimals at every timestamp. The boundary is the whole
difference - and the 120-second row is why a still can look right while the clip looks wrong:
some frames survive without an outline and most do not.

**What changed.** `analyse_clip` takes `polygon`, `motion_gate` and `person_gate`; passing none
reproduces its old behaviour exactly. The gates could not be applied by wrapping `classify` the
way the boundary was, because `MotionGate` compares a frame to the previous sample and only
that loop sees the samples in order. `clip_review` now derives a boundary from the footage when
the store has none - the same routine `run_slot_on_video.py` uses, and the reason its numbers
were right - and constructs both gates through a `_gates()` factory, so a test about neighbour
smoothing can turn them off rather than watching the person gate empty a synthetic video.

**The same clip, through the page's own code path:**

| | before | after |
|---|---|---|
| ACTIVE_PLAY samples | **24 / 24** | **0 / 24** |
| EMPTY | 0 | 23 |
| MAINTENANCE_NON_SPORTING | 0 | 1 - the sample with one person inside the outline |

**And the page now says which boundary it had.** "No outline", "the stored outline for this
camera" and "one measured from this clip" are three different claims and it made none of them;
a silent fallback to the whole frame is how this went unnoticed. The samples table gained a
*People inside* column and strikes through the probe's verdict where a gate overruled it, kept
separate from the strike-through the neighbour smoothing already used - two mechanisms, and a
reviewer chasing one should not be handed the other.

**What this says about the rest of the system.** The gates were added to `worker.run_slot` and
nothing checked that every path a user can reach classifies the same way. The worker, this
page, and the `/images` and `/roi` walkthrough endpoints each build their own classifier call;
only the worker and now this one have the boundary and the gates. `/images` and `/roi` still
score whole frames, which is correct for what they are - a boundary editor has to show what no
boundary looks like - but nothing states that, and the next report of this kind will come from
there.

- 2026-09-19 | milestone gate check | `python -m experiments.gate_check` | `gate_status.md` | 4 gate(s) met on artefacts, 3 waiting on a person

## The scheduler was running the probe alone too, and the boundary store knew no production camera (A36, WP9-T1)

`src/pitch_occupancy/pipeline.py`, `scheduler.py`, `worker.py`, `vision/roi.py`,
`vision/roi_derive.py`, `vision/rules.py`

A35 found the review page assembling its own inference - probe, no boundary, no gates - and
closed with the observation that the worker, the page, `/images` and `/roi` each built their
own classifier call. Reading the worker's caller for the rebuild (A36) found the same defect
one layer up, and it had been there longer.

**`scheduler.run_due` never passed the gates or the boundary.** It called
`run_slot(slot_id, source, classify, evidence_dir=...)` and nothing else, although `run_slot`
had accepted `polygon_for`, `motion_gate` and `person_gate` since A14-A16 and every number
published for "the system" (A20: false-play 0.0123 at recall 0.9436) was measured with them.
So `worker --source video`, and the live path behind it, ran the bare probe - the arm measured
at 0.6173-0.7684 false-play and a total error of 1.0000 on the 243 held-out empties. The only
production constructions of `MotionGate` and `PersonGate` in `src/` were on the review page,
after A35 put them there.

**And `roi.get` was handed ids the store had never heard of.** `configs/roi.json` holds
`cam` and `cam2`; `configs/roi_derived.json` holds `slot_20260711_1000_camA` and 99 clip ids.
The worker reads a recording's cameras as `file0`/`file1` (`frame_source.discover_slots`) and
the camera config names them `camera_A`/`camera_B` under `venue_01`. None of those is a key.
Had `run_due` passed a `polygon_for`, it would have returned None for every camera anyway.

**What changed.** One seam. `pipeline.assemble(model_key)` returns the classifier, both
gates and the boundary lookup as one object; `run_due`, `worker.main`, `worker._run_live`,
`/clip`, `/images`, `/roi` and `scripts/run_slot_on_video.py` all take theirs from it, and a
test spies on what reaches `run_slot`. `roi.resolve` looks a camera up the way production
names it - `<slot>/<camera>`, `<venue>/<camera>`, the bare id, each through an `_aliases`
block in `roi.json` - and, given a recording key, maps `file0 -> camA` and `file1 -> camB`.

That last mapping was measured rather than assumed, because `discover_slots` is right that
the `(1)` suffix does not name a physical camera across days. Within one recording it does
name the extraction's `camA`/`camB`: a boundary derived from each recording file against the
stored outlines, by IoU -

| recording | file | vs camA | vs camB |
|---|---|---|---|
| slot_20260711_1000 | file0 | **0.85** | 0.54 |
| slot_20260711_1000 | file1 | 0.54 | **0.93** |
| slot_20260712_2030 | file0 | **0.99** | 0.77 |
| slot_20260712_2030 | file1 | 0.77 | **0.98** |

The `_aliases` for `camera_A`/`camera_B` point at the daytime recording's outline of each
physical camera (`db/seed.py` PHYSICAL_CAMERA); `/roi` is where a person confirms or redraws.

**The boundary is now mandatory on the deployed path.** With a pipeline, a camera whose
boundary cannot be resolved contributes no observation: its minute is recorded as an
`UNCERTAIN` sample and, if no camera on the pitch had one, the minute counts as missed, which
the existing capture floor turns into REVIEW. The interactive pages score the whole frame and
say so in the verdict's trace. `worker --derive-roi` measures a missing boundary from the
footage and stores it under the production id (`roi.derive_from_video`, the library home of
what `scripts/derive_roi.py` did - the routine three callers had been reaching through
`sys.path`, sampling the first fifty seconds of an hour-long recording for its median).

Two store defects fixed on the way: `roi.save` merged the derived store into the hand-drawn
one on every write (the first outline drawn in the editor would have copied 99 derived
boundaries into `roi.json`), and `_write` dropped every underscored key but its own comment.

**Reproduced through the seam, on the unseen floodlit clip**, 16 samples at 15 s:

| | empty samples read EMPTY | says PLAY | false-play |
|---|---|---|---|
| deployed path via `pipeline.assemble` | **13 / 13** | 0 | **0.00** |

Of the three samples with one person walking, two read C3 and one read EMPTY - the detector
found the walker on two of three. That miss is recorded, not smoothed: it is the same path
A26 measured, and the same answer. The real-recordings scheduler test still reproduces the
published verdicts. The probe is fitted on 1,578 development frames as before; the 11
`_pending_4d` frames filed under A36 are `source=synthetic` and A13's exclusion keeps them
out of every published probe table.

**What this says.** Every "the system does X" claim in this log between A14 and A35 was true
of `run_slot` and false of the process that calls it. The seam exists so the next such claim
is about one path.

- 2026-09-19 | A36 WP9-T1 the seam | uv run python scripts/run_slot_on_video.py data/raw/venue_unseen_2026-09-15/empty_floodlit_night.mp4 --every 15 --truth-csv configs/unseen_clip_truth.csv | configs/roi.json | run_due had never passed the gates or the boundary; through pipeline.assemble the unseen clip reads 13/13 empty samples EMPTY, 0 PLAY

- 2026-09-19 | WP9-T2 detector audit | uv run python experiments/detector_audit.py --imgsz 1280 --tiles 1 | detector_audit.csv | chosen yolo11n-seg @1280 x1 on false-person and latency only (no hand counts); false-person 0.008; 191 ms/frame; round 17.2 s of 30; machine-dependent

- 2026-09-19 | WP9-T2 detector audit | uv run python experiments/detector_audit.py --keys yolo11n yolo11n-seg yolov8n yolov8n-seg yolov8s --imgsz 1280 --tiles 1 2 | detector_audit.csv | chosen yolov8n @1280 x1 at |d|<=1 0.700 on 100 hand-counted frames; false-person 0.042; 118 ms/frame; round 10.6 s of 30; machine-dependent

## The detector audit picks the detector already in the repo, and tiling trades the count for the ball (A36, WP9-T2)

`experiments/detector_audit.py`, `results/detector_audit.csv`, `results/hand_counts.csv`

Seven detectors, two tilings, against 100 hand-counted frames, 120 recorded venue_01 EMPTY
frames and 147 play frames across ten venues. The selection rule was written into A36 before
any of it ran: among configurations whose 30-camera × 3-frame round fits in 30 s, best rate of
`|detected − truth| ≤ 1` on the hand counts, tie-break on the false-person rate on the empties,
then latency.

| config | \|d\|≤1 | MAE | ≥5 when ≥5 | 0 when 0 | false-person | ball recall | ms |
|---|---|---|---|---|---|---|---|
| **yolov8n ×1** | **0.70** | 1.23 | 0.964 | 0.952 | 0.042 | 0.430 | **118** |
| yolov8s ×1 | 0.66 | 1.29 | 0.964 | 0.952 | 0.075 | 0.410 | 304 |
| yolo11n ×1 | 0.65 | 1.39 | 0.964 | 0.952 | **0.017** | 0.467 | 175 |
| yolov8n-seg ×1 | 0.64 | **1.15** | 0.964 | 0.952 | 0.042 | 0.330 | 163 |
| yolo11n-seg ×1 | 0.62 | 1.40 | 0.964 | 0.952 | **0.008** | 0.355 | 200 |
| yolov8n ×2 | 0.38 | 2.63 | 0.982 | 0.905 | 0.133 | 0.585 | 589 |
| yolo11n ×2 | 0.42 | 2.77 | 0.982 | 0.952 | 0.100 | 0.620 | 684 |

**The winner is the detector that was already here.** `yolov8n` at imgsz 1280 - the setting
A16 measured every published person count at - is best on the registered criterion and also the
fastest. A generation of model development and three times the parameters do not move the
number this rule reads. That is a result about the task, not about the models: counting people
on CCTV at 1280 pixels is not where a COCO detector's capacity goes.

**Tiling buys the ball and loses the count, and it loses more than it buys.** Four overlapping
crops plus the whole frame take ball recall from 0.43 to 0.585 - and at the four venues where
the ball was nearly invisible it roughly triples: `f_outdoor_bldg` 0.17 → 0.58,
`i_outdoor_trees` 0.13 → 0.47, `h_teal_pitch` 0.20 → 0.47, `d_indoor_dome` 0.20 → 0.47. It
also takes `|d| ≤ 1` from 0.70 to 0.38, MAE from 1.23 to 2.63, and the false-person rate on
empty pitches from 0.042 to 0.133, and it costs 589 ms against 118. A person straddling a tile
edge appears as two partial boxes whose overlap is below the merge threshold, so the count
inflates; that is the mechanism, and the same mechanism is why small distant objects are found.
**The rule reads the count and only records the ball (A17), so the trade is refused** - tiles
stay at 1. The ball figures are kept because WP9-T6's `ball_recovery` is the experiment that
decides what, if anything, may use them, and this is the first measurement it has.

**What the registered criterion did not ask, and the table answers anyway.** The rule does not
need an exact count. It needs two thresholds: *is this five or more* and *is this zero*. Every
configuration answers both at **0.964 and 0.952** - the same numbers, 56 frames with five or
more people inside the boundary and 21 with none. So the criterion that separated the models is
the one the method is least sensitive to, and the one it depends on is not discriminating at
all. The choice stands as registered; the observation is recorded here rather than used to
re-choose, because re-ranking on a metric picked after seeing the table is exactly the move
A36's selection rule exists to prevent.

**The counting truth is a first pass by eye and says so.** `results/hand_counts.csv` carries a
`confidence` column and **47 of 100 frames are marked `unsure`** - a far-side player at a clip
venue is a dozen pixels, and "whose feet are inside this hull" is genuinely undecidable on
some of them. Per-venue `|d| ≤ 1` for the winner tracks that directly: venue_01 0.88 and
`h_teal_pitch` 0.86, against `d_indoor_dome` 0.29 and `a_blue_barrier` 0.38. The dome's derived
boundary covers only the near half of its pitch, so most of its players are outside it by
construction and the disagreements there are about the boundary, not the detector. **[H] A
second pass by a person is WP9-T0b and is not done**; every number above is provisional on it.

**A separate finding the audit surfaced, which is about the boundaries and not the detector.**
The hand count recorded people on the pitch as well as people inside the camera's boundary, and
the two differ badly: **18 of 74 play frames have fewer than five people inside their own
camera's boundary** while the pitch plainly holds a match. This is A16's 88-of-278 measured
again on a different sample, and it is why A36 fuses counts at pitch level rather than deciding
per camera. It also says several derived hulls are too tight - `venue_01` camera B repeatedly
shows ten people on the pitch and three inside the outline.

**Three of the 74 play frames hold four or fewer people on the pitch** and are relabelling
candidates under §2.7: `slot_20260711_1000_camA_t000014_m.jpg` (a child and two adults with a
ball, in daylight - one of the six daytime play frames in the whole corpus),
`slot_20260712_2030_camA_t003476.jpg` and `slot_20260712_2030_camA_t001857.jpg`. Listed here;
moved only after the [H] verification pass, because relabelling on a first pass by eye would put
a guess into the ground truth.

- 2026-09-19 | WP9-T2 detector audit | uv run python experiments/detector_audit.py --keys yolo11n yolo11n-seg yolov8n yolov8n-seg yolov8s --imgsz 1280 --tiles 1 2 | detector_audit.csv | yolov8n @1280 x1 chosen by the registered rule at |d|<=1 0.700 on 100 hand-counted frames; false-person 0.042; 118 ms/frame; round 10.6 s of 30; tiling halves the count agreement and raises ball recall 0.43->0.59; machine-dependent

## The detector-first path reproduces the gated probe on the unseen clip, with no probe (A36, WP9-T3)

`src/pitch_occupancy/vision/rules.py`, `pitch_classifier.py`, `slots/fusion.fuse_pitch`

The decision table registered in A36 is now code, and `scripts/run_slot_on_video.py --model
yolov8n` runs it through `worker.run_slot` exactly as the worker would - the same seam, the
same boundary lookup, the same slot aggregation.

On the 234-second floodlit clip with nobody playing, at 15-second spacing:

| path | empty samples read EMPTY | says PLAY | false-play | slot verdict |
|---|---|---|---|---|
| probe + boundary + gates (A26, WP9-T1) | 13 / 13 | 0 | 0.00 | NOTUSED |
| **detector-first (this)** | **13 / 13** | **0** | **0.00** | **NOTUSED** |

Identical, and that is the finding. The second row has **no backbone, no feature cache and no
fitted head** - nothing in it was trained on this project's data, so there is nothing in it
that can have memorised a camera. The first row's EMPTY verdicts come from a probe that
answers EMPTY at an unseen camera 0 times in 243 (A20), rescued by two gates; the second
row's come from counting, and the count is the verdict rather than an overrule on one.

Both paths call the same two of the three walker minutes (9 and 15) and miss the third (12);
the detector finds one person in each of the two it calls and none in the one it misses. That
is the detector's recall on a distant figure at night and it is the same limit in both rows.

The rule file is **unfrozen** (`frozen_at: null`), so this ran with the shipped defaults -
`person_conf` 0.25, `ball_conf` 0.10, no minimum-height filter, no motion thresholds, row 8
off. WP9-T5 fits those on venue_01 camera A and freezes them; this number is what the rule
does before any of its own thresholds have been tuned, which is the honest place to record it.
The burst was one frame per sample here, because the script's own frame source does not
implement `read_burst` - so no motion cue was available and rows 4 and 8 could not have fired
in any case.

- 2026-09-19 | WP9-T3 rule engine on the unseen clip | uv run python scripts/run_slot_on_video.py data/raw/venue_unseen_2026-09-15/empty_floodlit_night.mp4 --every 15 --truth-csv configs/unseen_clip_truth.csv --model yolov8n | configs/rules.json | detector-first reads 13/13 empty samples EMPTY at 0.00 false-play, matching the gated probe with no trained component; rules unfrozen

- 2026-09-19 | WP9-T4 overlay figures | uv run python experiments/make_overlay_figures.py --model yolov8n | figs/overlays/ | 9 redacted overlays from yolov8n; the label agrees with the rule on 7 of them, 1 more differ only between 3 and 4; people drawn 0 and the ball in another

## The detector's explanation is the objects, not a heatmap (A36, WP9-T4)

`src/pitch_occupancy/vision/overlay.py`, `experiments/make_overlay_figures.py`,
`results/figs/overlays/`

The probe's explanation had to be a heatmap: a logistic regression on mean-pooled features has
no objects in it, only positions that push the score (`vision/explain.py`). The detector-first
path has objects, so its explanation is the objects - people in azure, the ball in amber, the
boundary in yellow, and the state, the count and the rule that fired written on the frame. The
two explanations of the same footage are the comparison in visual form: one shows a count a
reader can check against the picture, the other shows a region of the image.

**A detection found outside the boundary is drawn dimmed rather than dropped.** "The detector
found six people and counted three" is the thing this footage most often needs explained -
the hand-count audit found 18 of 74 play frames with fewer than five people inside their own
camera's outline - and an overlay that draws only the counted three cannot explain it.

**Which detections were counted is read back from `vision/counting.py`, not recomputed.** The
first version of the overlay re-derived inside-or-outside from the polygon, and a test caught
it drawing an outside person as a counted one. A second implementation of "is this inside"
drifts from the first, and it drifts *silently*, because a mis-drawn overlay looks exactly like
a correct one. This is the same defect class as the two the seam closed.

**Redaction is two layers and is now shared.** `vision/explain.redact_frame` is the rule
`make_xai_figures.py` has followed since it was written - pixelate each detected person, then
blur the whole frame so a missed detection is still not an identifiable face - lifted out of
that script so A36's figures use one implementation rather than a copy. The overlay is computed
from a verdict taken on the *original* frame and drawn over the redacted copy, which has
identical geometry, and the drawing goes on after the blur so the outline and the text stay
sharp while the frame underneath is destroyed.

Nine figures, three per folder class, spread across venue and lighting. The rule agrees with
the folder label on seven, and one more differs only between `3` and `4`, which A36 accepts.
**The two disagreements are both the boundary, not the detector**: two of venue_01 camera A's
three recorded C3 frames show the person walking on the far half, *outside* that camera's
derived outline, so the rule reads the pitch as empty and says so in the trace. The hand-count
audit recorded the same two frames the same way. The figure shows the outline, the walker
outside it, and the sentence "0 people inside" together, which is what makes it checkable.

- 2026-09-19 | WP9-T4 overlay figures | uv run python experiments/make_overlay_figures.py --per-class 3 --model yolov8n | figs/overlays/ | 9 redacted overlays; label agrees on 7, one more differs only 3<->4; both disagreements are frames where the person is outside the camera's derived boundary
