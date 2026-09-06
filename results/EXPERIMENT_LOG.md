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
