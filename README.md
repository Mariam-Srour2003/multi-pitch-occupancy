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

**Sign-in.** The server listens on this machine only (`127.0.0.1`) by default. To open it to
the facility's network, set a password first — every page and endpoint then asks for it
(`/health` stays open for monitoring):

```bash
# in .env (gitignored) or the service environment
PITCH_ACCESS_PASSWORD=choose-a-long-password
PITCH_ACCESS_USER=manager          # optional, default "admin"
uv run pitch serve --host 0.0.0.0
```

`pitch serve` refuses a network address without a password unless you add `--no-password`.
The browser sends the password with each request, encoded rather than encrypted, so outside a
trusted local network put the service behind HTTPS or a VPN.

Every page shares one navigation bar: **Home · Cameras · Records · More**. The first three
are the simple pages for everyday use; everything else is under **More**.

| page | what it does |
|---|---|
| `/` | **Home**: every camera as a card with how many people it saw at the last check; click one to watch it live at `/live/<place>/<camera>` (picture, number of people, re-checked every 30 s) |
| `/cameras/add` | **Cameras**: add a camera with just a name, IP and password (username, port, stream and a full link under "More options"); it is tested on saving. Test one or all, remove |
| `/records` | **Records**, the database: for each camera and each 1.5 hours, were there people and the most at once; a CSV download; and the **On/Off** switch for automatic checking (`monitor.py`: every camera every 1-90 minutes, set under "Settings"; stored in the gitignored `configs/monitor.json`) |
| `/more` | the pages below, one line each |
| `/client` (also `/dashboard`) | operator dashboard: a "Getting started" checklist (cameras → boundaries → schedule → detector → bookings → sign-in → recording → alerts), pitches hour by hour, slot verdicts, evidence, overrides |
| `/testing` | the testing hub: try the system on your own footage and photos; running and queued tests with a Cancel button |
| `/testing/day` | **booking-day test**: upload a day's video (or pick a recording already on this computer, from `data/recordings/` or `PITCH_RECORDINGS_DIR`), say when it starts (e.g. 08:00) and how long a slot is (e.g. 90 min); get every slot's minutes playing / empty / maintenance and USED / NOT USED / REVIEW, adjustable slot rules (re-judge a saved run without re-running it), CSV download, print / save as PDF, and an optional comparison with booking records |
| `/clip` | one video in, a timeline of what the pitch was doing; "Watch it work" streams each step |
| `/clip/pair` | both cameras of one pitch: two videos, people added, ball from either, one verdict |
| `/images` | one still or a batch, each explained, least confident first |
| `/reports` | from what the live system recorded: hours played and utilisation per pitch per day, used / not used / review, no-shows and unbooked play, a busy-hours heatmap (weekday × hour), and any recorded day cut into your own slot length; CSV for each |
| `/cameras` | add and test camera connections ("Test all cameras", each camera's last recorded picture), check a pitch live ("what is happening now?"), plan the recorded day (start time + slot length → live schedule), and the connection guide (also `docs/cameras.md`) |
| `/bookings` | connect booking records (optional, read-only: CSV, SQLite or another SQL database), check their quality, and compare a day with the camera verdicts |
| `/roi` | draw the pitch boundary for a camera, with a before/after preview |
| `/alerts` | what gets sent and when (no-show or unbooked play, a silent camera, recording stopped, an optional daily summary), a "Send a test alert" button, and the alert log |
| `/docs` | the API |

Uploads to the testing pages are deleted as soon as they have been read; nothing is written to
the database. A booking-day test keeps only its results, under `data/test_runs/`. Uploads are
limited to 16 GB per video (`PITCH_TESTING_MAX_UPLOAD_MB`); an hour of recording is about
630 MB, so for a whole day it is quicker to copy the files into the recordings folder and pick
them from the page.

**How a booking-day slot is judged** (`timetable.py`): each slot's main activity is whatever took
most of the time seen. Mostly playing is **USED**; mostly empty or maintenance is **NOT USED**. A
slot is **REVIEW** when less than half of it was seen, or when playing was between 40% and 60% of
the time seen. Compared with bookings, a REVIEW slot is never counted against anyone.

**Bookings** are optional. To connect them, copy `configs/bookings_source.example.json` to
`configs/bookings_source.json` (gitignored — it may hold a database password) and pick a type.
Give the system a database login that can only read the bookings table.

**Cameras** live in `configs/cameras.json` (gitignored), edited from `/cameras`. After saving
a schedule there, start or restart the live worker; it reads the schedule at start-up.

**Alerts** go by webhook (Slack, Microsoft Teams, Discord, or WhatsApp / Telegram / SMS through
an automation service) or email. Copy `configs/alerts.example.json` to `configs/alerts.json`
(gitignored — it holds webhook URLs and SMTP passwords), enable a channel, restart the server,
and press "Send a test alert" on `/alerts`. The server checks every few minutes and does not
repeat the same alert within the cooldown.

**Backups**: `uv run pitch backup` copies the database safely while the worker runs, checks the
copy and keeps the newest 14 (`data/backups/`). Config files with passwords are included only
with `--with-secrets`. Scheduling it daily and restoring are in `docs/backup.md`.

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
uv run python -m pitch_occupancy.worker --source live --model yolov8n --dry-run  # what it would open
uv run python -m pitch_occupancy.worker --source live --model yolov8n
```

This follows `configs/slots_schedule.json` in real time. It has not yet been run against a live
camera; `docs/runbook.md` lists what happens when one dies mid-slot.

Run one clip through the slot path exactly as the worker would:

```bash
uv run python scripts/run_slot_on_video.py <clip.mp4> --every 15 --model yolov8n
```

`configs/rules.json` ships with `frozen_at: null`: its motion thresholds are not fitted yet,
and the worker's log line labels the detector-first rule "UNFROZEN - not for deployment" until
they are. It still runs; the label is a reminder, not a lock.

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
