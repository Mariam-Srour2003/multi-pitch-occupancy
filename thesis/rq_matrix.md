# Research-question traceability

Every experiment maps to at least one research question, and every research question names
the evidence that answers it. Anything mapping to no RQ is either a missing RQ or scope to
cut; anything with no evidence is work still to do.

Updated 2026-09-06.

---

## Status at a glance

| RQ | Question | Status |
|---|---|---|
| RQ1 | Frozen backbone + light head at production accuracy on CPU, under night and fog? | **partly answered** |
| RQ2 | Best accuracy / latency / memory trade-off for one Mini-PC? | **answered** (dev hardware) — *but the input-path finding may reverse it* |
| RQ3 | How does leakage-free, multi-venue evaluation change apparent performance? | **answered** |
| RQ4 | Can slot aggregation + booking reconciliation detect record discrepancies? | not started |
| RQ5 | Do purpose-built lightweight architectures beat single-backbone probes? | **baseline established; the fusion answer looks like *no*** |
| RQ6 | Precision / REVIEW-rate trade-off and its operating point? | **blocked by data** |
| RQ7 | Do deep backbones earn their cost over trivial baselines? | **answered** |

---

## RQ1 - production accuracy on CPU under adverse light

| evidence | file | finding |
|---|---|---|
| H3 cross-venue recall | `h3_cross_venue_recall.csv` | DINOv2 0.930, ConvNeXtV2 0.910 play-recall on unseen venues |
| Label efficiency | `label_efficiency.csv` | 10-25 labels already beat the zero-label rule |
| Efficiency | `efficiency_latency.csv` | 20 cameras in 2.5-5.7 s against a 60 s cycle |

**Partly answered, and the limit is the data, not the method.** Active play is detected
reliably across nine venues in both lighting conditions. Whether an *empty* pitch is
recognised at an unseen venue is **unanswerable** - no empty pitch exists outside venue_01
(`preregistration.md`, "not answerable"). Any RQ1 claim in the thesis must be scoped to
active-play detection.

## RQ2 - accuracy / latency / memory trade-off

| evidence | file | finding |
|---|---|---|
| Latency + concurrency | `efficiency_latency.csv` | ConvNeXtV2 151 ms, ViT 303 ms, DINOv2 418 ms (median, 4 threads) |
| 20-camera throughput | same | 2.5 s / 4.5 s / 5.7 s per round; all inside the 60 s cycle |
| Accuracy under honest protocols | `h1_h2_baseline_floor.csv`, `h3_cross_venue_recall.csv` | DINOv2 leads both |

**Answered, and it overturns the pilot's recommendation.** The pilot chose ConvNeXtV2 as
the production lead on latency. Measured properly, **latency is not the binding constraint**
- even the slowest backbone uses under 10% of the sampling cycle for 20 cameras. Once speed
stops discriminating, the choice falls to accuracy under honest evaluation, and that is
DINOv2.

> **Recommendation changed: DINOv2, not ConvNeXtV2.** ConvNeXtV2 remains the fallback if the
> target Mini-PC turns out far slower than the development machine, which WP7-T1 settles.

**H4 was meant to underwrite this and cannot** (2026-09-07, `h4_model_equivalence.csv`). It
predicted ConvNeXtV2 indistinguishable from ViT on macro-F1 *and* >= 2x faster, with the
production choice then resting on latency. Both halves fail to deliver:

- The equivalence is exact (ΔmacroF1 = 0.0000, zero-width interval) and **degenerate**: the
  two models emit *identical* predictions - ACTIVE_PLAY for all 907 frames, EMPTY zero times,
  C1 F1 = 0.000. 0.4975 is the score of a model that never recognises an empty pitch, and both
  sit exactly there. They are equivalent to each other and to a constant predictor.
- The speed clause is **not met under the condition the pre-registration names**: 1.83x on the
  20-camera concurrent median, against the >= 2x required. It reaches 2.01x only on the
  single-frame median, which that document explicitly declined to rely on.

So the DINOv2 recommendation rests on the **cross-venue** evidence, where the three models are
not equivalent at all - and where DINOv2 is the only one that ever predicts EMPTY. See
amendment A9.

> ### ⚠ And that cross-venue evidence is now itself in question (2026-09-08)
>
> `geometry_convention_probe.csv`. Every published cross-venue number comes from caches built
> by handing **raw frames** to the HF processor, which resizes a 1920x1080 frame shortest-edge
> to 256 and centre-crops 224 - keeping roughly the middle *half* of the pitch. Apply
> `preprocess.py`'s letterbox instead and:
>
> | backbone | recall | false-play | balanced |
> |---|---|---|---|
> | **ConvNeXtV2** | **0.9841** | **0.0206** | **+0.9635** |
> | DINOv2 | 0.9595 | 0.2305 | +0.7290 |
> | ViT | 0.9118 | 0.9712 | −0.0594 |
>
> ConvNeXtV2's false-play falls from **0.9918 to 0.0206**, so it leads on *both* axes - and it
> is the fastest of the three. **That would reverse this recommendation back to ConvNeXtV2.**
>
> **Not acted on, deliberately.** The false-play column is 243 frames amounting to three to
> ten distinct scenes, with no CIs, one seed and no paired test. The recommendation stays
> DINOv2 until the input-path question is settled under the full protocol - which is TODO
> WP3-T3 option (b). Recording it here so the next reader of this section knows the ground
> may move.

## RQ3 - what leakage-free evaluation changes

| evidence | file / figure | finding |
|---|---|---|
| Split comparison | `h1_h2_baseline_floor.csv` | macro-F1 roughly **halves**, random -> grouped: ConvNeXtV2 0.988 -> 0.498, ViT 0.994 -> 0.498, DINOv2 0.988 -> 0.579 |
| Split validation | `check_split()` | the honest grouped split is 99% single-class - degenerate on this data |
| Ranking inversion | `figs/ranking_inversion.png` | ViT 1st under the leaky protocol, tied 2nd/3rd under the honest one |

**Answered, and more strongly than "accuracy drops".** The protocol does not merely deflate
scores, it **reverses the decision**: a reader following the pilot's protocol would have
selected ViT, the weakest generaliser of the three. That is the thesis's central
methodological result.

**Restated 2026-09-07 after the estimand fix, and the effect is ~3x larger than first
reported.** The drop was quoted as -0.159 for ConvNeXtV2; on a consistent 2-class metric it is
**-0.490**. The earlier figure came from comparing a random-split macro-F1 that was
intermittently 3-class against a grouped-split one that was 2-class - C3 has support 1 in the
random test set and entered 63% of bootstrap resamples. Leakage does not shave points off this
benchmark, it halves the score. The affected CIs were also ~15x too wide. Grouped-split
numbers are unchanged. See the WP4-T4b entry in `EXPERIMENT_LOG.md`.

Two caveats, both narrower than before. The magnitude still cannot be attributed *purely* to
leakage, because the grouped test set is near-single-class - the direction is sound and the
number is now clean but not attributable. And the ranking inversion is a fact about **rank,
not margin**: ViT's apparent 0.339 lead under the leaky protocol was almost entirely the
single C3 frame it happened to get right; on equal footing that lead is **0.006**.

## RQ4 - reconciliation

No evidence yet. Needs WP6-T4/T5 and, critically, **adjudicated ground truth** - which slots
really were no-shows. Without that, anomaly precision and recall cannot be measured at all
(WP6-T11).

## RQ5 - novel architectures

| evidence | file | finding |
|---|---|---|
| Naive-ensemble baseline (WP5-T9) | `logit_average_baseline.csv` | a parameter-free average takes **74%** of the multi-backbone oracle headroom (0.9524 of 0.9603, best single 0.9297) |
| Same, second axis | same | every ensemble scores **1.0000** false-play - worse than all three backbones alone |

**The baseline is established, and it points at "no" for the fusion half.** The gated-fusion
module (WP5-T2) was the one WP5 module this data can support, and its premise was
complementarity between backbones. Both halves of that premise now have numbers attached:

- The complementarity is **real but small and mostly free**. DINOv2 is strictly beaten on
  three of seven venue folds and the oracle is +0.031 above it, but plain averaging collects
  three quarters of that without a parameter, leaving 0.008 for a learned gate that cannot
  reach the oracle anyway because it has no venue identity.
- ~~Blending is disqualified outright on the second axis~~ — **refuted 2026-09-08.** On raw
  caches every ensemble scored 1.0000 false-play, which looked decisive. On letterboxed
  caches the two-model ensemble scores **0.0288** false-play at **1.0000** recall, the best
  balanced score measured here. Blending was never the problem; blending a model that had
  been shown a central strip of the pitch was.

**WP5-T2's answer is unchanged and its reasoning is now the opposite.** Not "blending is
disqualified so a gate must be a hard router", but **the naive ensemble is already at the
oracle ceiling** - it captures 100% of the headroom on preprocessed caches, up from 74% - so
there is nothing left for a gate to learn. That argument needs no claim about routing or
confounds. Report WP5-T2 as a negative result against the naive ensemble.

Two things that keep it honest. The recall axis is **saturated** (oracle 1.0000), so the
headroom was only +0.0159; "100% of it" is a smaller claim than it sounds. And the ensemble
leads ConvNeXtV2 alone by 0.008 balanced, which is not a difference on three to ten distinct
scenes - under preprocessing the two are indistinguishable, DINOv2 is behind, and ViT is poor.

STAN (WP5-T1) remains gated on real labelled slots: **2 exist, ~30 are needed**, and the 66
clips are 10-14 s highlights with no slot structure. Any STAN result will be reported as
preliminary on synthesised sequences.

## RQ6 - precision / REVIEW-rate trade-off

| evidence | file | finding |
|---|---|---|
| Calibration + risk-coverage | `rq6_calibration.csv`, `rq6_risk_coverage.csv` | apparent "99% precision at 0% review" - an artifact |

**Machinery built and tested; the answer is blocked by the same degenerate split.** With a
test set that is 99% one class, a 99% precision target is met before confidence is
consulted, so the risk-coverage curve has nothing to trade against. Temperature scaling
fitted on the morning recording and applied to night footage made two of three models
worse, and DINOv2's temperature pinned to the grid floor - caught by the boundary warning
rather than reported as a fitted parameter.

`evaluation/calibration.py` is ready. It needs a test set with a real class mix.

## RQ7 - do deep backbones earn their cost?

| evidence | file / figure | finding |
|---|---|---|
| Baseline floor | `h1_h2_baseline_floor.csv` | clock rule 0.4907 vs ConvNeXtV2 0.4975 on the grouped split |
| Baseline floor | same | colour histogram 0.686 vs ConvNeXtV2 0.657 under the random split |
| Cross-venue | `h3_cross_venue_recall.csv` | a lighting-only rule collapses across venues; backbones hold above 0.86 |

**Answered, and the answer is conditional.** *Within* a confounded venue, no - a clock rule
matches ConvNeXtV2 and ViT, and a colour histogram beats them. *Across* venues, emphatically
yes - the trivial baselines collapse while the frozen features transfer.

> **Do not quote 0.219 as the clock rule's cross-venue recall.** It is a lower bound partly
> produced by label error: the `lighting` column for the 396 clip frames is a brightness
> proxy, and `f_outdoor_bldg` is night football filed as `day`. Correcting that one
> twelve-frame fold moves the figure to 0.362. The defensible claim is the *direction* - a
> lighting-only rule collapses across venues while the backbones hold above 0.86 - and the
> exact value waits on the hand relabelling in TODO WP3-T5. See the correction entry in
> `results/EXPERIMENT_LOG.md`.

H2 as pre-registered ("within 2 macro-F1 points of the best probe") is **refuted**: DINOv2
clears the floor by 8.9 points. The narrower version stands and is the more useful finding:
two of three backbones fail to beat a rule that never looks at the image.

---

## Experiments not yet mapped

**One.** The zero-shot **prompt search** (`prompt_search.csv`, `prompt_search_best.json`, 375
rows) has no task in any plan document and no RQ. It belongs to **H6** and RQ1 - the
no-label cold-start cost - and is mapped there rather than left as an orphan artefact. It is
now a stage in the reproduction pipeline (`prompt-search`) and is disclosed as an undeclared
search family in `preregistration.md` A6.

## Coverage gaps to close, in cost order

1. **RQ5 (fusion half)** - baseline done and it points at a negative result. Cheapest
   remaining WP5 work: the lighting-only gate ablation, then WP5-T2 built to be reported
   either way. Needs no new data.
2. **RQ6** - code complete; blocked on a non-degenerate test set, i.e. empty pitches at
   more than one venue. No further engineering will unblock it.
3. **RQ4** - reconciliation. Needs a booking export and adjudicated slots (human).
4. **RQ5 (STAN half)** - needs ~30 real labelled slots, which the current footage cannot
   supply.
5. **H4 and H6** - both pre-registered, neither reported. H4 is a pairwise McNemar on the
   grouped split; H6 is the zero-shot-vs-trained comparison the prompt search's numbers
   already support. Both are hours of work on cached features (`preregistration.md` A5).
