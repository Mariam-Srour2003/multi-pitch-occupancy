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

**What is not done is the data, and the shape of the gap is now precise.** Across the 1,692
recorded frames, daylight is 98% empty pitches and night is 99% football — so
*"night means play, day means not-play"* is correct on **99.1%** of the corpus, and on 98.8%
within the one venue that has both classes. Every daylight frame comes from that venue. Nothing
measured on this data can separate *recognises an empty pitch* from *recognises the time of day*,
and the two cells that would break the tie hold **9 frames** (empty at night) and **6** (play in
daylight).

**The sharpest result is what that does to a benchmark.** A rule that reads the clock and never
looks at the image beats all three frozen backbones on the project's own cross-venue protocol —
perfect play-recall, and a false-play rate fifteen times lower than the best of them. On a
234-second clip of a floodlit pitch with nobody playing, the same rule is wrong on **every one of
16 minutes** — false-play **1.00** — while the deployed system's false-play is **0.00** and the
probe alone sits at 0.38. A 1,692-frame benchmark ranked them one way; one clip containing the
missing cell reversed it completely.

So the honest headline is a set of results about *how to evaluate this problem*, several of them
negative, most found by checking whether a guard actually guarded — and the most useful single
sentence in the thesis is that no protocol compensates for a cell the data never fills.

<!-- status:start -->
**83 source modules · 75 experiment scripts · 84 test files
· 93 committed result files.** Counts come from git, so this line cannot drift from
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
  latency is not the binding constraint, with 14–31× headroom in the 60 s cycle. The pick was
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
| [thesis/presentation/defence_slides.html](thesis/presentation/defence_slides.html) | **The oral defence deck** — 19 slides, self-contained, opens in a browser. Speaker notes carry what is said; the slides carry only the point |
| [thesis/defence_deck.md](thesis/defence_deck.md) | The slide *plan* the deck was built from — the argument, the framing decision, and what not to say |
| [thesis/ch2_related_work.md](thesis/ch2_related_work.md) | Chapter 2 scaffolding — nine strands, what each must establish, and no invented citations |
| [thesis/alternatives.md](thesis/alternatives.md) | Why computer vision rather than a motion sensor, a turnstile or an app check-in |
| [thesis/data_requests.md](thesis/data_requests.md) | What to ask the facility for, in priority order |
| [docs/backup.md](docs/backup.md) | The backup policy for the one irreplaceable thing |
| [docs/runbook.md](docs/runbook.md) | What happens when a camera dies, and which rows of that are enforced |
| [thesis/preregistration.md](thesis/preregistration.md) | The six hypotheses, their decision rules, and every amendment — including the two decision rules that were disowned |
| [thesis/rq_matrix.md](thesis/rq_matrix.md) | Each research question, the evidence for it, and how far it is answered |
| [thesis/mvt.md](thesis/mvt.md) | **The minimum viable thesis** — the four load-bearing items, and what the floor deliberately does *not* require. Read before any scope decision. |
| [docs/PM_REVIEW_2026-09-11.md](docs/PM_REVIEW_2026-09-11.md) | **Latest review.** Where "existence is not health" cost three artefacts, two results that were artefacts of their axis, a headline that did not survive a second seed, and the six decisions waiting on a person |
| [docs/PM_REVIEW_2026-09-08.md](docs/PM_REVIEW_2026-09-08.md) | What day two overturned — including two of day one's conclusions |
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
decision support and **never takes an automated financial action**. Faces are pixelated in
every figure the experiments publish, by a redaction step the scripts cannot skip and a test
enforces — five venue-audit sheets committed in September predate it, and `thesis/ethics.md`
names them rather than implying otherwise.

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

### Pulling the 2026-09-21 changes

The detector-first rebuild (A36/A40, WP9–WP10) is **merged into `main`**. If you have a clone
from before 2026-09-21, three things changed underneath you and one of them needs a command.

**There is nothing to train, and that is the point.** The deployed path counts people with an
off-the-shelf detector and decides with a written rule, so there is no fitting step and no
checkpoint to restore. The DINOv2 comparator has no saved model either — its logistic head is
refitted from the cached features on every load, so "retraining" it means rebuilding the
cache, nothing more.

```bash
git pull origin main
uv sync                                              # ultralytics; the lockfile is committed
uv run pitch fetch-weights                           # ~80 MB of YOLO weights, the one download
uv run python scripts/collapse_label_folders.py      # DRY RUN - see the next section
uv run pitch info                                    # every path should read [ok]
uv run pytest -m "not slow"
```

#### If you already have a `data/` directory, migrate it

The labelling folders collapsed from four to three: `3_people_not_playing` and `4_maintenance`
are now one folder, `3_maintenance_non_sporting`. A clone's `data/` is gitignored, so **git
will not do this for you** and nothing will fail loudly — the manifest will simply report
"unexpected class folder" and skip those frames.

```bash
uv run python scripts/collapse_label_folders.py          # reports what it would move
uv run python scripts/collapse_label_folders.py --apply  # moves frames AND every sidecar
uv run pitch manifest                                    # rebuild manifest.csv
uv run python scripts/assign_scene_ids.py                # rebuild the scene sidecar
uv run pitch cache dinov2                                # ~10 min; only for the probe arm
```

The collapse script also rewrites `labels.csv`, `scene_ids.csv`, `results/hand_counts.csv` and
the `.npz` feature caches, which key their rows by filename and would otherwise drop 175
frames silently. It is idempotent — running it twice is safe and the second run reports
nothing to do. Nothing under `results/` that records a *measurement* is touched; those say
what was true when they were written.

Fresh clone with no `data/` yet? Skip the collapse; restore `data/` from the backup
(`docs/backup.md`) and it already has the three folders.

Three things a clone does not carry, in the order they bite:

| what | why it is missing | how to get it |
|---|---|---|
| **detector weights** (`*.pt`, ~80 MB) | gitignored; third-party binaries | `uv run pitch fetch-weights` |
| **`data/`** (~4.2 GB) | gitignored in full — footage of identifiable people | restore from the backup (`docs/backup.md`) |
| **`data/cache/*.npz`** | regenerable, so not worth storing | `uv run pitch cache dinov2` (~10 min) |

The feature cache is needed **only for the DINOv2 comparator**, not for the detector-first
path. To see the rebuild work, skip it.

What a clone *does* carry, and should: `configs/rules.json` (the decision rule's numbers and
switches), `configs/roi.json` and `roi_derived.json` (the pitch boundaries),
`configs/davinci_venues.csv` (which clip came from which venue, with the evidence) and
`results/hand_counts.csv` (a person's count of 100 frames). A boundary and a rule are part of
what a deployment *is*, and a truth file a clone loses is a model selection nobody can
re-derive.

Run it on a clip — this needs the weights and nothing else:

```bash
uv run python scripts/run_slot_on_video.py <clip.mp4> --every 15 --model yolov8n
```

Reproduce the rebuild's tables — these need `data/` as well:

```bash
uv run python experiments/reproduce_all.py --check     # what is stale; runs nothing
uv run python experiments/rule_frame_eval.py           # the rule against the probe
uv run python experiments/make_overlay_figures.py --model yolov8n
```

`configs/rules.json` ships with `frozen_at: null`, so the rule **refuses to be deployed**
until WP9-T5 fits its thresholds on venue_01 camera A and freezes them. Experiments run
against it regardless — measuring an unfrozen rule is how it gets frozen — and
`settings.default_model_key` is still `dinov2` until then, so `--model yolov8n` is how you
ask for the detector path today.

Serve the API and dashboard:

```bash
uv run pitch serve --reload
# or directly:
uv run uvicorn pitch_occupancy.api.app:app --reload
```

Run the pipeline over the recorded slots — sample, classify, fuse, aggregate, reconcile, and
write the verdicts to the database:

```bash
uv run python -m pitch_occupancy.worker --source video --dry-run   # list, load no model
uv run python -m pitch_occupancy.worker --source video             # verdicts to the database
uv run python -m pitch_occupancy.worker --source video --evidence-dir data/evidence
```

Evidence images are off by default because they are frames of identifiable people. With
`--evidence-dir` the winning camera's frame is written for every observed minute and all but
the three that justify the verdict are deleted once the slot is complete.

**Against real cameras** — copy `configs/cameras.example.json` to `configs/cameras.json`
(gitignored: an RTSP URL usually carries the camera's credentials) and:

```bash
uv run python -m pitch_occupancy.worker --source live --dry-run    # what it would open
uv run python -m pitch_occupancy.worker --source live              # asks before connecting
```

This follows `configs/slots_schedule.json` in real time, taking one frame per camera per
minute. It has **never been run against a camera** — the path is tested against an injected
capture opener, which is the right way to test it and is not the same thing. `docs/runbook.md`
lists what happens when one dies mid-slot and which rows of that are enforced.

### Reviewing a single clip

`/clip` takes a video, samples a frame at a fixed interval, classifies each one and reports
when the state changed. There is no schedule, booking or slot behind it — it is for a file
somebody has in their hand: a disputed hour exported after the fact, or a camera being
checked before it goes into the schedule. **The upload is deleted as soon as it has been
read**, and nothing is written to the database.

**Watch it work.** The same page's "Watch it work" button streams the analysis step by step —
one newline-delimited JSON record per frame, flushed before the next is computed, so the
browser renders step *k* while the backbone is still on *k+1*. Each explained step shows the
frame beside its evidence map.

That map is **not a saliency heuristic**. The probe is a logistic regression over mean-pooled
frozen features, so a class score is exactly the mean of per-position contributions plus a
constant — `class_evidence_map` returns those summands, and the map therefore *sums to the
score*. The page prints `score_from_map`, `score_direct` and the error between them (about
10⁻¹⁵) so the claim is checkable rather than asserted. It also reports how much positive
evidence lands on detected people against the area they occupy: far above 1 means the model
is reading *players*, near 1 means it is reading the scene around them.

The slow motion is a pause the page adds between steps and the button drops it — the backbone
runs at the same speed either way. The control that changes the actual work is *explain in
detail*, since an explained frame costs about twice a bare prediction.

A lone sample that disagrees with both its neighbours is usually a misread frame, so a
median filter (`slots.stan.majority_smooth`, the same one the STAN baseline uses) corrects
it. **The correction is always shown, never applied silently** — the timeline stripes it, the
table strikes the raw prediction through, and a toggle switches back to raw. That matters
because the same filter erases real brief events: on the ten-minute clip this was built
against it correctly absorbed one flicker *and* deleted a genuine 15-second maintenance
event that the labels confirm was real. The samples alone cannot tell those apart; a person
looking at the footage can.

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
