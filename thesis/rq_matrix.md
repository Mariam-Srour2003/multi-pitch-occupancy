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
| RQ2 | Best accuracy / latency / memory trade-off for one Mini-PC? | **answered** (dev hardware) |
| RQ3 | How does leakage-free, multi-venue evaluation change apparent performance? | **answered** |
| RQ4 | Can slot aggregation + booking reconciliation detect record discrepancies? | not started |
| RQ5 | Do purpose-built lightweight architectures beat single-backbone probes? | not started |
| RQ6 | Precision / REVIEW-rate trade-off and its operating point? | not started |
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

## RQ3 - what leakage-free evaluation changes

| evidence | file / figure | finding |
|---|---|---|
| Split comparison | `h1_h2_baseline_floor.csv` | macro-F1 falls for every model, random -> grouped |
| Split validation | `check_split()` | the honest grouped split is 99% single-class - degenerate on this data |
| Ranking inversion | `figs/ranking_inversion.png` | ViT 1st under the leaky protocol, 3rd under the honest one |

**Answered, and more strongly than "accuracy drops".** The protocol does not merely deflate
scores, it **reverses the decision**: a reader following the pilot's protocol would have
selected ViT, the weakest generaliser of the three. That is the thesis's central
methodological result.

Honest caveat: the magnitude of the drop cannot be attributed purely to leakage, because
the grouped test set is near-single-class. Direction is sound; the number is not clean.

## RQ4 - reconciliation

No evidence yet. Needs WP6-T4/T5 and, critically, **adjudicated ground truth** - which slots
really were no-shows. Without that, anomaly precision and recall cannot be measured at all
(WP6-T11).

## RQ5 - novel architectures

No evidence yet. STAN (WP5-T1) is gated on real labelled slots: **2 exist, ~30 are needed**,
and the 66 clips are 10-14 s highlights with no slot structure. Any STAN result will be
reported as preliminary on synthesised sequences.

## RQ6 - precision / REVIEW-rate trade-off

No evidence yet. Needs calibration (WP4-T5) then the risk-coverage sweep (WP4-T9). Both are
cheap on cached features - this is the nearest unstarted RQ.

## RQ7 - do deep backbones earn their cost?

| evidence | file / figure | finding |
|---|---|---|
| Baseline floor | `h1_h2_baseline_floor.csv` | clock rule 0.4907 vs ConvNeXtV2 0.4975 on the grouped split |
| Baseline floor | same | colour histogram 0.686 vs ConvNeXtV2 0.657 under the random split |
| Cross-venue | `h3_cross_venue_recall.csv` | clock rule collapses to 0.219; backbones hold above 0.86 |

**Answered, and the answer is conditional.** *Within* a confounded venue, no - a clock rule
matches ConvNeXtV2 and ViT, and a colour histogram beats them. *Across* venues, emphatically
yes - the trivial baselines collapse while the frozen features transfer.

H2 as pre-registered ("within 2 macro-F1 points of the best probe") is **refuted**: DINOv2
clears the floor by 8.9 points. The narrower version stands and is the more useful finding:
two of three backbones fail to beat a rule that never looks at the image.

---

## Experiments not yet mapped

None. Every experiment run so far answers RQ1, RQ2, RQ3 or RQ7.

## Coverage gaps to close, in cost order

1. **RQ6** - calibration + risk-coverage. Cheap, cached features, no new data.
2. **RQ4** - reconciliation. Needs a booking export and adjudicated slots (human).
3. **RQ5** - STAN. Needs ~30 real labelled slots, which the current footage cannot supply.
