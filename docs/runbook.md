# Failure-mode runbook (WP7-T5)

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
mistake `aggregate.py`'s own header warns about. It is blocked on RQ6's calibration, which is
blocked on a test set with a real class mix. It sits at 0.0 with a note saying it is inert —
because three claims in this project turned out to describe guards that were not guarding, and
a fourth silently-zero threshold would be the same defect.

**The ethics commitment does not rest on either.** A human confirms every anomaly and the
system takes no automated financial action (`slots/authority.py`); these are
defence-in-depth, not the thing standing between the audit and a false accusation.

## Camera offline — the worked case

The one WP7-T5 names, and the one with a test behind it.

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
| a slot in `REVIEW` citing captured minutes | is the camera up? `pitch info` shows resolved paths | if the outage is over, the slot cannot be recovered — the footage was not taken. Adjudicate from the booking record. |
| many slots in `REVIEW` on one field | that field's camera or its network leg | a single field failing is a mount, a cable or a lens |
| many slots in `REVIEW` across all fields | the network, the disk, or the host | restart the worker; the API only reads what the worker wrote, so the dashboard stays up either way |
| a verdict that looks wrong | open the three evidence frames | override it — the override is recorded and feeds the retraining loop (WP6-T7) |

**Restarting the dashboard never interrupts sampling.** The worker and the API are separate
processes sharing a database; `api/` only reads what `worker.py` wrote. That is a design
property worth knowing during an incident, not a coincidence.

## What is not handled, and what it would take

Stated so the gaps are visible rather than discovered in production.

- **Disk full.** The retention worker exists (`pitch retention`, dry run by default) and
  bounds growth: sampled frames are purged after 7 days, evidence kept 365, both read from
  `thesis/ethics.md`. What is still missing is a *reaction* — nothing checks free space before
  writing evidence, and nothing runs retention on a schedule. Both belong with WP7-T2's
  service units.
- **Facility-wide confidence collapse.** Row 6. Needs RQ6's calibration, which needs data.
- **A stale booking export.** Reconciliation compares against whatever the export last said. No
  freshness check, and no alert if the export stops arriving.
- **Camera moved or re-aimed.** The camera-fingerprint audit can detect this after the fact
  (it found the `(1).mp4` suffix swapping between venue_01's two days) but nothing watches for
  it live.
- **None of this has run against a real outage.** The camera-offline case is tested against a
  stub. WP7-T3's shadow-mode run is where these rows stop being predictions.

## Before the live run

1. Verify the restore procedure works from the off-site backup (`docs/backup.md`).
2. Run `uv run python -m experiments.reproduce_all --check` on the target machine.
3. Re-measure latency on the target hardware (WP7-T1) — **on an idle machine**; the
   reproduction pipeline holds that stage back from `--force` for this reason.
4. Confirm the disk has room for 7 days of raw frames plus 365 days of evidence.
5. Agree with the facility who adjudicates a `REVIEW` and how fast.
