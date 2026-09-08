# Chapter 2 — Related work (skeleton, WP1-T1)

**This is scaffolding, not a draft.** Each strand states *what it has to establish* and *why
this project needs it*, so the literature search has a target rather than a topic. Every
citation is marked `[CITE]`.

> **No references are invented here, and none should be added without being read.** A
> fabricated or half-remembered citation is the one error in a thesis that cannot be defended
> at all, and it is the specific failure an LLM-assisted draft is most likely to introduce.
> Every `[CITE]` is a placeholder for a paper you have opened. The project's own claims ledger
> exists because unchecked claims drift; the same discipline applies to the ones with authors
> attached.

Nine strands, not seven. Two were added because the work produced results that need them:
**§2.8 prior art**, because the gap statement is exposed until someone has looked, and **§2.9
selective prediction**, because the REVIEW band turned out to be a literature-backed design
rather than an engineering convenience.

---

## §2.1 Occupancy and activity recognition from fixed cameras

*What it must establish:* that classifying scene state from a fixed viewpoint is a solved
problem class, and what accuracy the field considers ordinary — so this project's numbers can
be placed rather than admired.

- Fixed-camera occupancy detection in indoor and outdoor settings `[CITE]`
- Sports-field and court activity recognition specifically `[CITE]`
- The standard failure modes: viewpoint change, illumination, occlusion `[CITE]`

*Why this project needs it:* the contribution is not the classifier. Establishing that the
classification step is routine is what licenses spending the thesis on the evaluation and the
reconciliation instead.

## §2.2 Frozen features and linear probes

*What it must establish:* that a frozen backbone with a small trained head is a legitimate
method rather than a shortcut, and where it wins and loses against fine-tuning.

- Linear probing as a standard evaluation of representation quality `[CITE]`
- Self-supervised backbones as general feature extractors `[CITE]`
- When fine-tuning is worth its cost, and when it is not `[CITE]`

*Why this project needs it:* CPU-only is a design constraint, and a frozen backbone plus a
head that retrains in seconds is what makes the ablation programme affordable at all. The
concession — that fine-tuning would probably score higher — has to be made with a citation
behind it rather than as an apology.

## §2.3 Dataset leakage and evaluation protocol

*What it must establish:* that same-scene or near-duplicate leakage is a known and recurring
problem, with prior cases where a benchmark was found to be measuring memorisation.

- Grouped and subject-wise splitting `[CITE]`
- Near-duplicate leakage in image benchmarks, and its measured effect `[CITE]`
- Cases where a published result was revised after a leakage audit `[CITE]`

**This is the strand the thesis's central contribution sits in**, so it needs the most care.
The project's own numbers to place against it: macro-F1 roughly halves from a random to a
grouped split; **100% of leaky-split errors had a near-duplicate on the training side against
0% of honest-split errors**; and a zero-shot control shows about a third of the fall is test-set
composition rather than leakage. If the literature already reports effects of this size, say
so — the contribution is then the *decomposition*, not the discovery.

## §2.4 Trivial baselines and benchmark validity

*What it must establish:* that checking a benchmark against a baseline which ignores the input
is established practice, not an eccentricity of this project.

- Majority-class and shortcut baselines in vision benchmarks `[CITE]`
- Shortcut learning and spurious correlation `[CITE]`
- Cases where a benchmark was retired after a trivial baseline matched it `[CITE]`

*Why this project needs it:* a rule reading only the clock comes within 0.0068 of ConvNeXtV2
on the honest split, and on the cross-venue protocol a constant predictor scores a perfect
macro-F1. Those are the project's most quotable findings and they need a tradition to belong
to, or they read as a curiosity about one dataset.

## §2.5 Edge and CPU-constrained inference

*What it must establish:* what is normal for CPU-only vision deployment, and which
optimisation routes exist that this project did not take.

- Efficient architectures and CPU inference `[CITE]`
- Quantisation, distillation, runtime optimisation (OpenVINO, ONNX) `[CITE]`
- Sampling and duty-cycling rather than continuous processing `[CITE]`

*Why this project needs it:* the deployment argument is that sampling one frame per camera per
minute cuts bandwidth ~99% and fits 20 cameras in 2.5–5.7 s of a 60 s cycle. The strand should
make clear that the *sampling design*, not the model, is what makes that possible — and should
name the optimisations (quantisation, OpenVINO) that remain unexplored, since WP4-T11 leaves
them open.

## §2.6 Calibration and uncertainty for deployed classifiers

*What it must establish:* that a confidence score is an operational quantity when it routes
work to a person, and how calibration is normally assessed.

- Modern neural networks are miscalibrated `[CITE]`
- Temperature scaling and post-hoc calibration `[CITE]`
- Expected calibration error, its bin dependence and its critics `[CITE]`

*Why this project needs it:* RQ6 asks what a REVIEW rate buys. The project's finding is that it
cannot be answered here — 890 of 907 calibrated confidences are identical and 905 of 907 land
in a single reliability bin — so the strand has to establish what a well-posed version of the
question looks like, in order for "this test set cannot pose it" to be a result.

## §2.7 Facility management, audit and record reconciliation

*What it must establish:* that cross-checking sensor observation against an administrative
record is a recognised problem, in this domain or an adjacent one.

- Audit and anomaly detection against administrative records `[CITE]`
- Occupancy sensing in facilities management `[CITE]`
- Human-in-the-loop audit systems, and the cost of a false accusation `[CITE]`

*Why this project needs it:* reconciliation is the operational contribution. The break-even
analysis puts a number on it — a flag must be right ~95% of the time before the feature pays —
and that number is only interpretable against what the literature considers an acceptable
false-alarm rate for an audit tool.

## §2.8 Prior art — who already does this ★

*What it must establish:* whether a commercial or academic system already does facility
occupancy audit, and what it does not do.

- Commercial pitch/court occupancy and utilisation products `[CITE]`
- Academic systems for sports-facility monitoring `[CITE]`
- Booking-system integrations that claim utilisation analytics `[CITE]`

**The gap statement is exposed until someone has actually looked.** If a product already does
this, the gap narrows to the parts it does not — CPU-only on existing CCTV, leakage-free
evaluation, and reconciliation against the booking record with a human in the loop. That is
still a gap, and stating it deliberately is much better than being told at the defence.

*Search terms to start from:* pitch utilisation analytics, court occupancy monitoring, sports
facility CCTV analytics, booking no-show detection.

## §2.9 Selective prediction and learning to defer ★

*What it must establish:* that abstaining and handing a case to a human is a studied design,
with its own metrics.

- Selective prediction, the risk–coverage trade-off `[CITE]`
- Learning to defer / rejection learning `[CITE]`
- Human-in-the-loop decision systems and their evaluation `[CITE]`

*Why this project needs it:* the REVIEW band **is** selective prediction with a human
fallback. Citing this literature upgrades it from an engineering convenience to a principled
choice, and it supplies the vocabulary for the risk–coverage work — including the reason the
curve here has to be drawn as a band rather than a line.

---

## What this chapter has to do for the thesis

Three jobs, and the strands map onto them:

| job | strands |
|---|---|
| establish that the classification step is routine, so the contribution lies elsewhere | §2.1, §2.2, §2.5 |
| establish that the evaluation contribution belongs to a tradition | §2.3, §2.4, §2.6, §2.9 |
| establish that the gap is real and correctly bounded | §2.7, §2.8 |

## How to work through it

1. Take one strand at a time and find 3–5 papers you have read.
2. Replace each `[CITE]` with the real reference and one sentence on what it establishes.
3. Where the literature already reports what this project found, **say so and narrow the
   claim** — that is a stronger chapter than one that implies novelty by omission.
4. §2.8 first if time is short. It is the only strand that can change what the thesis claims.
