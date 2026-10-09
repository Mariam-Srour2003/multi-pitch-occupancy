# Pitch Occupancy

Low-bandwidth, CPU-only computer vision that checks how a network of 5-a-side football pitches
is actually used, and reconciles what it sees against what the facility's booking records claim.

## The idea

Rather than continuously decoding 20–30 video streams, the system samples **one frame per camera
per minute**, decides whether the pitch is *empty*, in *active play*, or in use for something else
(*maintenance / non-sporting*), and aggregates a rental slot's worth of minutes into a verdict —
`USED`, `NOTUSED`, or `REVIEW` — bound to three evidence images for one-click human audit.

That cuts network load by roughly 99% (from ~60–90 Mbps to under 1 Mbps) and runs on a single
Intel Mini-PC with **no discrete GPU**.

The operational payoff is reconciliation: the booking system holds what *should* have happened
and what staff *recorded*; the cameras observe what *actually* happened. Cross-checking the three
surfaces no-shows, unbooked usage and data-entry errors — with a person confirming every anomaly.
The system never bills or acts on money by itself (`slots/authority.py`).

## How a minute is decided

Two classifiers sit behind one seam (`pipeline.py`):

- **Detector-first (`yolov8n`)** — counts people (by their feet) and the ball inside the pitch
  boundary over a short burst of frames, then applies a written decision table
  (`vision/rules.py`, numbers in `configs/rules.json`): nobody → empty; 1–4 people → not playing;
  5 or more people (with the ball and movement the rule file requires) → playing.
- **DINOv2 probe** — frozen DINOv2 features pooled inside the boundary with a logistic head,
  then a motion gate and a person gate that may only weaken its verdict. This is still
  `settings.default_model_key`.

**Every pitch has two cameras**, each seeing about half. A pitch verdict adds the people both
cameras count, takes the ball from either, and applies the table once (`slots/fusion.py`).

## Getting started

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). CPU only — `pyproject.toml` pins
torch to the CPU wheel index.

```bash
uv sync                      # create the venv and install from uv.lock
uv run pitch fetch-weights   # ~80 MB of YOLO weights
uv run pitch info            # show resolved config and check the expected paths exist
uv run pytest -m "not slow"  # run the fast tests
```

What a clone does not carry:

| what | why | how to get it |
|---|---|---|
| detector weights (`*.pt`) | third-party binaries, gitignored | `uv run pitch fetch-weights` |
| `data/` | footage of identifiable people, gitignored | restore from the backup (`docs/backup.md`) |
| `data/cache/*.npz` | regenerable | `uv run pitch cache dinov2` (~10 min; DINOv2 probe only) |

## Running it

Serve the API, dashboard and review pages:

```bash
uv run pitch serve --reload
# or directly:
uv run python -m uvicorn pitch_occupancy.api.app:app --reload
```

| page | what it does |
|---|---|
| `/` (also `/client`) | operator dashboard: pitches hour by hour, slot verdicts, evidence, overrides |
| `/clip` | one video in, a timeline of what the pitch was doing; "Watch it work" streams each step |
| `/clip/pair` | **both cameras of one pitch**: two videos, people added, ball from either, one verdict |
| `/images` | one still or a batch, each explained, least confident first |
| `/roi` | draw the pitch boundary for a camera, with a before/after preview |
| `/docs` | the API |

Uploads to the review pages are deleted as soon as they have been read; nothing is written to
the database.

Run the pipeline over the recorded slots — sample, classify, fuse, aggregate, reconcile, and
write the verdicts to the database:

```bash
uv run python -m pitch_occupancy.worker --source video --dry-run   # list, load no model
uv run python -m pitch_occupancy.worker --source video             # verdicts to the database
uv run python -m pitch_occupancy.worker --source video --evidence-dir data/evidence
```

Evidence images are off by default because they are frames of identifiable people. With
`--evidence-dir` the winning camera's frame is written for every observed minute and all but the
three that justify the verdict are deleted once the slot is complete.

**Against real cameras** — copy `configs/cameras.example.json` to `configs/cameras.json`
(gitignored: an RTSP URL usually carries credentials) and:

```bash
uv run python -m pitch_occupancy.worker --source live --dry-run    # what it would open
uv run python -m pitch_occupancy.worker --source live
```

This follows `configs/slots_schedule.json` in real time. It has not yet been run against a live
camera; `docs/runbook.md` lists what happens when one dies mid-slot.

Run one clip through the slot path exactly as the worker would:

```bash
uv run python scripts/run_slot_on_video.py <clip.mp4> --every 15 --model yolov8n
```

`configs/rules.json` ships with `frozen_at: null`, so the worker refuses to deploy the
detector-first rule until its thresholds are fitted and frozen; the review pages and scripts run
against it regardless.

## Layout

```
src/pitch_occupancy/
├── config.py            paths and settings (env-driven, PITCH_* / .env)
├── pipeline.py          the one seam every surface classifies through
├── worker.py            samples and classifies slots
├── scheduler.py         follows the slot schedule in real time
├── clip_analysis.py     one video -> timeline
├── pair_analysis.py     two cameras of one pitch -> one timeline
├── bookings.py          read-only booking records
├── retention.py         deletes sampled frames after 7 days, evidence after 365
├── data/                manifest, splits, taxonomy, feature cache
├── vision/              detector, counting, rules, ROI, backbones, probe, explanations
├── slots/               two-camera fusion, aggregation, reconciliation, evidence
├── db/                  SQLite schema and access
├── api/                 FastAPI app and pages — the only part uvicorn serves
└── cli.py               `pitch` command
configs/                 rules, pitch boundaries, cameras, slot schedule, splits
scripts/                 one-off tools: ingest footage, derive boundaries, run a clip
docs/                    runbook, database, data layout, backup
tests/
```

The pipeline is a scheduled batch process, not a web app: `worker.py` samples and classifies
independently, and `api/` only reads what it wrote. Restarting the dashboard never interrupts
sampling.

Footage is not distributed with this repository — it shows identifiable people at a client
facility. The dataset is referenced by manifest rather than stored in git.

The thesis that produced this system — experiments, results, write-up and defence material — is
preserved on the `thesis-archive` branch.
