# Failure-mode runbook

What happens when things break, what the system does about it, and what a person has to do.

**The single rule everything below follows: the system never fabricates a verdict.** When it
cannot see enough to decide, the answer is `REVIEW` — not a guess with the confidence of an
observation. A missing minute is recorded as missing, never filled in.

Each row says whether the behaviour is *enforced in code* or *not yet implemented*, because a
runbook that does not distinguish those is a wish list.

---

## The degraded-mode ladder

| # | what breaks | what the system does | status |
|---|---|---|---|
| 1 | one camera drops a frame | records the minute as missed for that camera; the other camera still contributes | **enforced** |
| 2 | one camera dies mid-slot | the slot's `cameras_seen` falls to 1; `concerns()` flags *"half the pitch is unseen"* | **enforced** |
| 3 | **both cameras die mid-slot** | capture rate falls below 0.5 → **verdict is `REVIEW`**, with the reason naming how many minutes of how many were captured | **enforced** (`Thresholds.review_below_capture`) |
| 4 | no frames at all | `REVIEW`, *"no samples captured for this slot"* | **enforced** |
| 5 | frames arrive but are unreadable — fog, glare, a dirty lens | `SlotConditions.concerns()` reports low contrast and low confidence to the operator | **advisory only** |
| 6 | confidence collapses facility-wide | nothing automatic — `review_below_confidence` is **0.0 and therefore inert** | **not implemented, deliberately** |
| 7 | disk fills | `run_slot` checks free space **once, before the first write**, and disables evidence images for that slot rather than filling the disk mid-run — the verdict is still produced. A disk it cannot measure is written to anyway | **enforced** (`retention.has_room`, 500 MB floor) |
| 8 | the booking export is stale or absent | a day outside the export's span reconciles to **NEEDS_REVIEW (info)**, never to an anomaly — `bookings.covers` answers whether the export reaches the date, and the check runs before anything that can return SERIOUS. An empty export covers nothing rather than everything | **enforced** (`reconcile(..., records_cover_this_day=)`) |
| 9 | **a camera has no pitch boundary** | every minute of that camera is `UNCERTAIN` — the detector never runs and no verdict is formed for it. The other camera still decides the pitch, at **half the confidence**; if neither has one, the minutes count as not captured and row 3 makes the slot `REVIEW`. The run names the cameras (`SlotRun.uncertain_cameras`) and the startup line says so before a stream is opened | **enforced** (`pipeline.require_boundary`) |
| 10 | the detector's weights are missing or will not load | the same: `UNCERTAIN`, never EMPTY. "Not checked" and "checked, found nobody" are different values throughout (`detect` returns `None` against `[]`), so a broken install cannot report every pitch empty | **enforced** (`vision/detector.py`) |
| 11 | **two pitches have slots at the same time**, or a slot starts while another is running | every due slot runs in its own thread (`scheduler.run_concurrently`); the loop keeps ticking and starts each slot instance once, skipping any whose verdict is already in the database after a restart. Until 2026-10-10 slots ran one after another and every overlapping slot was silently lost | **enforced** (`tests/test_live_concurrency.py`) |
| 12 | **the worker starts late**, in the middle of a slot | the slot is still recorded from its *scheduled* start: minutes already past are missing, never read now and labelled as then, and the slot ends at its scheduled end. A mostly-missed slot falls under row 3's floor and is `REVIEW`. Until 2026-10-10 it recorded a full slot length from the moment it started, overrunning into the next booking | **enforced** (`RTSPSource.wait_for`, `live_sources`) |
| 13 | **the worker process dies** (crash, power cut, closed window) | nothing records, and nothing in the database says so. The worker's heartbeat, `data/worker_status.json`, stops being refreshed; the *Recording status* card on `/cameras` turns to "Not running" once it is 3 minutes old. Keeping the worker running is the service manager's job — `docs/service.md` | **enforced** (detection, `status.py`); restart depends on the service set-up |

**Fixing row 9.** Draw the outline at `/roi`, or let the worker measure one:
`uv run python -m pitch_occupancy.worker --source live --derive-roi` writes a boundary derived
from the footage to `configs/roi_derived.json` under the production camera id, and prints what
it kept. A derived outline is a fallback and weaker than a drawn one — confirm it at `/roi`,
where drawing over it is how you correct it.

## Why rows 3 and 6 are treated differently

They look like the same kind of guard and are not, and the difference is worth being able to
explain.

**Capture rate is not a model quantity.** It is how much of the hour was observed. *"We saw
eleven minutes of this hour"* is not a statement about a pitch whatever the classifier says
about those eleven minutes, so a floor can be set from first principles: below half the slot,
the system stops asserting. **0.5 is a judgement, and is recorded as one** — it is the weakest
defensible reading rather than a value calibrated against real degraded slots, of which there
are none in the corpus.

**A confidence threshold is a model hyper-parameter**, and picking one by hand is exactly the
mistake `aggregate.py`'s own header warns about. It needs calibrated confidences first, which
need a test set with a real class mix. It sits at 0.0 with a note saying it is inert —
because three claims in this project turned out to describe guards that were not guarding, and
a fourth silently-zero threshold would be the same defect.

**Protection against a false accusation does not rest on either.** A human confirms every
anomaly and the
system takes no automated financial action (`slots/authority.py`); these are
defence-in-depth, not the thing standing between the audit and a false accusation.

## Camera offline — the worked case

The most likely outage, and the one with a test behind it.

```
minute 0-11   camera A returns frames; pitch is empty
minute 12-59  camera A returns None (RTSP timeout, retried twice, gave up)
```

- 12 samples recorded, 48 minutes counted as **missed**. No frame is invented for the gap.
- `capture_rate` = 0.20.
- Ratios over the surviving fragment would read *empty 100%* → `NOTUSED`, which is a confident
  claim about an hour from a fifth of it — and the pitch may well have filled up at 12:15.
- The gate fires instead: **`REVIEW`, "only 12 of 60 minutes captured (20%); too little of the
  slot was observed to decide"**.

Tested in `tests/test_worker.py`, including the case that would have made it vacuous: the
worker has to pass the slot's real length through, or the ratios are taken over whatever
arrived and the check never fires.

## What an operator does

| symptom on the dashboard | first check | then |
|---|---|---|
| no new verdicts at all | the *Recording status* card on `/cameras` (or `data/worker_status.json`): is the worker running, when was it last seen? | if "Not running", start it again (`docs/service.md`) and read the worker's log for why it stopped. Slots missed while it was down are not recoverable |
| a slot in `REVIEW` citing captured minutes | is the camera up? `pitch info` shows resolved paths | if the outage is over, the slot cannot be recovered — the footage was not taken. Adjudicate from the booking record. |
| many slots in `REVIEW` on one field | that field's camera or its network leg | a single field failing is a mount, a cable or a lens |
| many slots in `REVIEW` across all fields | the network, the disk, or the host | restart the worker; the API only reads what the worker wrote, so the dashboard stays up either way |
| a verdict that looks wrong | open the three evidence frames | override it — the override is recorded and feeds the retraining loop |

**Restarting the dashboard never interrupts sampling.** The worker and the API are separate
processes sharing a database; `api/` only reads what `worker.py` wrote. That is a design
property worth knowing during an incident, not a coincidence.

## What is not handled, and what it would take

Stated so the gaps are visible rather than discovered in production.

- **Disk full.** The retention worker exists (`pitch retention`, dry run by default) and
  bounds growth: sampled frames are purged after 7 days, evidence kept 365. Row 7 stops a
  slot writing evidence onto a full disk, but nothing runs retention on a schedule; that
  belongs with the service units that run the worker.
- **Facility-wide confidence collapse.** Row 6. Needs calibrated confidences, which need data.
- **A stale booking export.** Reconciliation compares against whatever the export last said. No
  freshness check, and no alert if the export stops arriving.
- **Camera moved or re-aimed.** Comparing a camera's background against an earlier recording
  detects this after the fact (that is how the `(1).mp4` suffix was found swapping between
  venue_01's two days), but nothing watches for it live.
- **None of this has run against a real outage.** The camera-offline case is tested against a
  stub. A shadow-mode run alongside the facility's own records is where these rows stop being
  predictions.

## The heartbeat and stopping the worker

The live worker writes `data/worker_status.json` when it starts, on every loop tick (once a
minute), after every camera read of every running slot, when a slot finishes and on error. It
holds what is recording now (minutes done of expected, and per camera the last picture, pictures
taken and minutes missed), the next five slots, the last ten finished slots and the last ten
errors — camera ids only, never URLs. On a clean exit it writes `"state": "stopped"`; a
crash leaves an old `updated_at`, which is how a dead worker is told from a quiet one.

**Ctrl+C cancels the slots that are running**: no verdict is written for a slot cut short,
because a part of a slot is not the slot. Slots already finished are in the database. Restarting
during a slot records the rest of it (row 12), usually as `REVIEW`. With `--iterations N` the
loop stops after N ticks and *lets running slots finish* before exiting.

## Before the live run

1. Verify the restore procedure works from the off-site backup (`docs/backup.md`).
2. Re-measure latency on the target hardware — **on an idle machine**; a busy CPU makes the
   number meaningless.
3. Confirm the disk has room for 7 days of raw frames plus 365 days of evidence.
4. Agree with the facility who adjudicates a `REVIEW` and how fast.
5. Copy `configs/cameras.example.json` to `configs/cameras.json` and fill in one RTSP URL per
   camera. The real file is gitignored: an RTSP URL usually carries the camera's credentials.
   Every camera named by a slot in `configs/slots_schedule.json` needs a URL — the run
   refuses to start otherwise, because half a pitch reported as the whole one is a wrong
   verdict rather than a missing one.
6. `uv run python -m pitch_occupancy.worker --source live --dry-run` first. It lists what it
   would open and connects to nothing. The run without `--dry-run` prints each camera as
   `scheme://host` — credentials withheld — and waits for you to type `yes`.
