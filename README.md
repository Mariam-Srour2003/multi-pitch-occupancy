# Multi-Pitch Occupancy & Booking Verification

Low-bandwidth, CPU-only computer vision that audits how a network of synthetic 5-a-side football
pitches is actually used — and reconciles what it sees against what the facility's booking records
claim.

Master's thesis project.

---

## The idea

Rather than continuously decoding 20–30 video streams, the system samples **one frame per camera
per minute**, classifies it as *empty*, *active play*, or *maintenance / non-sporting*, and
aggregates a rental slot's worth of predictions into a verdict — `USED`, `NOTUSED`, or `REVIEW` —
bound to three evidence images for one-click human audit.

That cuts network load by roughly 99% (from ~60–90 Mbps to under 1 Mbps) and runs entirely on a
single Intel Mini-PC with **no discrete GPU**. CPU-only is a design constraint of the thesis, not a
limitation to be worked around.

The operational payoff is reconciliation: the booking system holds what *should* have happened and
what staff *recorded*; the cameras observe what *actually* happened. Cross-checking the three
surfaces no-shows, unbooked usage, and data-entry errors — with a human confirming every anomaly.

---

## Status

**The system is built and the evaluation is the finding.** Sampling, classification, slot
aggregation, reconciliation, the API and the dashboard all run; every experiment regenerates from
the manifest and the cached features through one reproduction pipeline. All six pre-registered
hypotheses have been reported.

**What is not done is the data.** Every EMPTY frame in the corpus comes from one venue, two
cameras, two days — so the questions that matter operationally (can it recognise an empty pitch
somewhere new? what does a REVIEW threshold cost?) are blocked on footage, not on method. The
honest headline of this project is a set of results about *how to evaluate this problem*, several
of which are negative and most of which were found by checking whether a guard actually guarded.

<!-- status:start -->
**61 source modules · 43 experiment scripts · 65 test files
· 49 committed result files.** Counts come from git, so this line cannot drift from
the repository; the assessment above it is written by hand. What each module and experiment
does is in [docs/CODEBASE.md](docs/CODEBASE.md); what each run found is in
[results/EXPERIMENT_LOG.md](results/EXPERIMENT_LOG.md).
<!-- status:end -->

| Branch | Contains |
|---|---|
| `main` | The whole project — plan, implementation, experiments and results |
| `pilot/model-selection` | The first-step bake-off that chose the models — code, results, and a full run guide in its own README |

To read or re-run the pilot:

```bash
git checkout pilot/model-selection
```

---

## What has been found

The contribution is the evaluation. Each of these was measured, and each is written up with its
own reasoning in [results/EXPERIMENT_LOG.md](results/EXPERIMENT_LOG.md).

**On evaluating this problem**

- **Same-scene evaluation roughly halves when made honest** — macro-F1 falls from ~0.99 to ~0.50
  moving from a random split to one that holds whole slots out. It also *reverses the ranking*: a
  reader following the pilot's protocol would have shipped the weakest generaliser.
- **But only about two thirds of that fall is leakage.** A model that never trains cannot leak, so
  OpenCLIP scored zero-shot on the identical test sets measures what changing the test set does on
  its own — it drops 0.183. The leakage-attributable part is 0.33–0.40, not 0.52–0.58.
- **On the cross-venue protocol a constant predictor wins.** `majority` reads no pixels, always
  answers "playing", and scores macro-F1 **1.000** — ahead of every backbone — because every
  held-out venue is 100% active play. There is no ranking there to reverse.
- **Cross-venue recall is free unless you measure false play too.** Held-out folds contain no empty
  pitch, so a model that always says "playing" scores perfectly; the control shows one backbone
  calling 99.2% of held-out empty frames a match while a rule that never looks at the image calls
  2.1%.
- **Frame counts overstate the evidence.** 907 frames are 95 distinct scenes, and 243 held-out
  empty frames are three to ten. Most frame-level significance in this project does not survive
  being recounted by scene, and that is reported rather than caveated.

**On the models**

- **DINOv2** is the production recommendation, on cross-venue evidence rather than latency —
  latency is not the binding constraint, with 10–24× headroom in the 60 s cycle. The pick was
  challenged on an input-path finding and the challenge was tested and did not survive.
- **The prompt matters ~9× more than the model.** Zero-shot with a pre-declared prompt beats every
  trained probe, but across the whole prompt space macro-F1 spans 0.021–0.747 where the three
  backbones span 0.082. The cost of a no-label deployment is a distribution, not a penalty.
- **Selective prediction cannot be drawn as a curve here.** One model's calibrated confidences are
  890/907 identical, so its risk–coverage "curve" is a band; the two models with the best-looking
  curves are the two that never predict EMPTY.
  ([figure](results/figs/risk_coverage_band.png))

**On the code that produced the numbers**

Several results moved because a guard existed and did not operate: a final-test-set lock that
failed open from any other directory, a seeded split that returned a different row *order* every
process (so every bootstrap interval was a different draw), a false-play control scored against
training data, a figure with its ranks hardcoded, and a site export that had silently stopped
covering the thesis. Each is fixed, tested, and written up — including one finding of my own that
a later measurement retracted.

---

## Documents

| File | What it is |
|---|---|
| **[TODO.md](TODO.md)** | **The working checklist.** Every task, WP0→WP8, with milestone gates. Start here. |
| **[results/EXPERIMENT_LOG.md](results/EXPERIMENT_LOG.md)** | **What every run found**, in order, with the reasoning and the retractions. |
| [results/gate_status.md](results/gate_status.md) | **Milestone status**, checked against the artefacts rather than ticked by hand |
| [thesis/claims.md](thesis/claims.md) | **The claims ledger** — every quantitative claim mapped to the artefact that produced it, re-derived on every run |
| [thesis/defence_redteam.md](thesis/defence_redteam.md) | The ten hardest questions and the evidence for each answer |
| [thesis/ch2_related_work.md](thesis/ch2_related_work.md) | Chapter 2 scaffolding — nine strands, what each must establish, and no invented citations |
| [thesis/alternatives.md](thesis/alternatives.md) | Why computer vision rather than a motion sensor, a turnstile or an app check-in |
| [thesis/data_requests.md](thesis/data_requests.md) | What to ask the facility for, in priority order |
| [docs/backup.md](docs/backup.md) | The backup policy for the one irreplaceable thing |
| [docs/runbook.md](docs/runbook.md) | What happens when a camera dies, and which rows of that are enforced |
| [thesis/preregistration.md](thesis/preregistration.md) | The six hypotheses, their decision rules, and every amendment — including the two decision rules that were disowned |
| [thesis/rq_matrix.md](thesis/rq_matrix.md) | Each research question, the evidence for it, and how far it is answered |
| [thesis/mvt.md](thesis/mvt.md) | **The minimum viable thesis** — the four load-bearing items, and what the floor deliberately does *not* require. Read before any scope decision. |
| [docs/PM_REVIEW_2026-09-08.md](docs/PM_REVIEW_2026-09-08.md) | **Latest review.** What day two overturned — including two of day one's conclusions — and the six open decisions |
| [docs/PM_REVIEW_2026-09-07.md](docs/PM_REVIEW_2026-09-07.md) | The first review, kept as written |
| [SUPER_PLAN.md](SUPER_PLAN.md) | Reference plan — research questions, architecture, hard-won gotchas |
| [PLAN.md](PLAN.md) | The original engineering plan from the pilot phase, kept for history |
| `Thesis_Plan_Multi-Pitch_Occupancy (4).docx` | Full thesis plan — abstract, related work, methodology, contributions |
| `Thesis_Schedule_and_Effort_Plan.docx` | Companion schedule: work packages, effort, dependencies, gates |

---

## What the pilot established

Seven models were benchmarked on 1,296 labelled frames from one venue. Four were carried forward:

| Model | Role the pilot assigned | Accuracy | ms/frame |
|---|---|---|---|
| ConvNeXtV2-Tiny + head | production lead | 98.1% | 102 |
| ViT-Base + head | accuracy reference | 99.2% | 203 |
| DINOv2-Base + head | robustness reference | 98.5% | 254 |
| OpenCLIP B/32 | zero-shot cold-start baseline | 75.8% | 247 |

Each is a **frozen** backbone with a small logistic-regression head that retrains in seconds on
cached features. No fine-tuning, no GPU.

**These are not thesis results.** They come from a same-scene random split on a single venue with
two starved classes — a feasibility signal only. The main line of work replaces them with
grouped and leave-one-venue-out splits, confidence intervals, and significance testing. Details in
the pilot branch README.

> **The pilot's production pick has since been overturned.** Measured properly, latency is not
> the binding constraint — even the slowest backbone uses under 10% of the 60 s sampling cycle
> for 20 cameras — so the choice falls to accuracy under honest evaluation, and that is
> **DINOv2**, with ConvNeXtV2 the fallback if the target Mini-PC proves far slower (WP7-T1).
> The "Role" column above is what the pilot concluded, kept for the record; `thesis/rq_matrix.md`
> (RQ2, RQ7) holds the current answer.
>
> **That answer was challenged and the challenge was tested** (2026-09-08). ConvNeXtV2's
> headline weakness — calling 99.2% of held-out empty pitches a match, against DINOv2's 30.9%
> — looked like an artefact of the **input path** rather than the model: every published
> number comes from caches built by handing *raw* frames to the HF processor, which keeps
> roughly the middle half of a 16:9 pitch, and letterboxing instead moves ConvNeXtV2 to 2.1%
> false-play at 98.4% recall. Put through the full protocol, that does not hold up. The whole
> column is one measurement in one direction; swap the two cameras it is built from and
> ConvNeXtV2 has no defect to repair (raw false-play 2.8%), while DINOv2 reverses from perfect
> to 97.6% wrong. The recommendation stays **DINOv2**. What the question actually needs is
> empty-pitch footage from a second venue. See `results/input_path_protocol.csv` and
> `docs/PM_REVIEW_2026-09-08.md`.

---

## Beyond the benchmark

Three purpose-built, edge-constrained modules are proposed and ablated against strong baselines,
each keeping the backbones frozen and adding only small trainable parts:

- **STAN** — a slot-temporal aggregation network that replaces hand-tuned ratio thresholds with a
  learned, calibrated decision layer over the per-minute sequence.
- **Gated multi-backbone fusion** — a tiny gate conditioned on cheap image statistics picks or
  blends backbones per condition, with a conditional-compute variant that only invokes the heavier
  backbone when the cheap one is uncertain.
- **Context-aware multimodal head** — visual features fused with temporal and booking priors, with
  booking-flag dropout so the audit can never simply trust the record it is meant to check.

---

## Ethics

The system classifies **scene state, not identities**. No face recognition, no re-identification.
Evidence images follow a defined retention policy and exist for audit only. Because reconciliation
can implicate individual staff, a human stays in the loop for every anomaly — the system produces
decision support and **never takes an automated financial action**. Faces are blurred in every
published figure.

---

## Where the code is

**`main` carries the whole project.** Clone it and everything is there — no branch to check
out first.

It was built on a chain of stacked feature branches, and they are all still there, because
that history is worth reading: each branch is one step, with its own diff and a commit
message explaining what it found. **[docs/CODEBASE.md](docs/CODEBASE.md)** lists them in
order.

**[docs/CODEBASE.md](docs/CODEBASE.md)** is the reference: what each branch contains, what
every module does, and how to run all of it.

## Getting started

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). CPU only — no GPU is used anywhere,
and `pyproject.toml` pins torch to the CPU wheel index so a CUDA build can never slip into the
lockfile and quietly invalidate the reported latencies.

```bash
uv sync                 # create the venv and install from uv.lock
uv run pitch info       # show resolved config and check the expected paths exist
uv run pytest           # run the test suite
```

Serve the API and dashboard:

```bash
uv run pitch serve --reload
# or directly:
uv run uvicorn pitch_occupancy.api.app:app --reload
```

`uv.lock` and `.python-version` are committed deliberately: they are what lets another machine —
or an examiner — reproduce the numbers.

## Layout

```
src/pitch_occupancy/     the library: importable, testable, one code path
├── config.py            paths and settings (env-driven, PITCH_* / .env)
├── data/                manifest, splits, taxonomy, feature cache
├── vision/              preprocessing, ROI, frozen backbones, heads
├── slots/               two-camera fusion, aggregation, STAN, reconciliation
├── evaluation/          metrics, statistics, calibration, latency
├── db/                  SQLite schema and access
├── api/                 FastAPI app — the only part uvicorn serves
├── worker.py            the sampling scheduler
└── cli.py               `pitch` command
experiments/             one script per thesis experiment, results to results/
configs/                 camera ROIs, slot schedule
tests/
```

The pipeline is a scheduled batch process, not a web app: `worker.py` samples and classifies
independently, and `api/` only reads what it wrote. Restarting the dashboard never interrupts
sampling.

Footage is not distributed with this repository — it shows identifiable people at a client
facility. The dataset is referenced by manifest rather than stored in git.
