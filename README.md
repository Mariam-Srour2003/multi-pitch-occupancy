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

**Planning complete. Implementation starting.**

The model-selection pilot is finished — it chose four models to build on and produced an end-to-end
prototype. That work lives on its own branch and is not carried into `main`; the real system is
built from scratch against the plan, with leakage-free evaluation from the start.

| Branch | Contains |
|---|---|
| `main` | The thesis plan and the real implementation as it is built |
| `pilot/model-selection` | The first-step bake-off that chose the models — code, results, and a full run guide in its own README |

To read or re-run the pilot:

```bash
git checkout pilot/model-selection
```

---

## Planning documents

| File | What it is |
|---|---|
| **[TODO.md](TODO.md)** | **The working checklist.** Every task, WP0→WP8, with milestone gates. Start here. |
| [SUPER_PLAN.md](SUPER_PLAN.md) | Reference plan — research questions, architecture, hard-won gotchas |
| [PLAN.md](PLAN.md) | The original engineering plan from the pilot phase, kept for history |
| `Thesis_Plan_Multi-Pitch_Occupancy (4).docx` | Full thesis plan — abstract, related work, methodology, contributions |
| `Thesis_Schedule_and_Effort_Plan.docx` | Companion schedule: work packages, effort, dependencies, gates |

---

## What the pilot established

Seven models were benchmarked on 1,296 labelled frames from one venue. Four were carried forward:

| Model | Role | Accuracy | ms/frame |
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
