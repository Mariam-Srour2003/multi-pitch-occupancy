# Data request (WP2-T2)

**One conversation, one list, in priority order.** Everything here is cheap for the facility
to supply and expensive for this project to do without. The ordering is by what each item
unblocks, not by how much of it there is — the first two items are small and each unblocks
more than everything below them combined.

Send this with `results/coverage.md` attached, which is generated and shows the gaps as a
table.

---

## What the corpus contains today

1,692 labelled frames, 10 venues. The shape of the problem is in two rows:

| | EMPTY | ACTIVE_PLAY | MAINTENANCE / non-sporting |
|---|---|---|---|
| venue_01 | 494 | 796 | **6** |
| the nine clip venues, combined | **0** | 396 | **0** |

Every empty pitch in the corpus is one venue, two cameras, two days. Every other venue is
active play only. That single fact is what most of this request is about.

---

## 1 · Empty pitches at any venue other than venue_01 ★

**The highest-value frames in this request, and the cheapest to produce.** An empty pitch
needs no scheduling and no event — just a camera pointed at one for a few minutes, at two or
three sites, in daylight and under floodlights.

*What it unblocks, and it is four separate things:*

- **RQ1's central claim.** Whether an *empty* pitch is recognised at an unseen venue is
  currently unanswerable, so every accuracy figure in the thesis has to be scoped to
  active-play detection.
- **RQ6, the operating point.** Calibration and the REVIEW-rate curve are computed on a test
  set that is 99% one class; the risk–coverage curve cannot be identified on it.
- **The two confidence thresholds** in the production path, which cannot be set without it.
- **The input-path decision (WP3-T3).** The evidence for it reverses when the only two
  cameras that see an empty pitch are swapped, and no test on this data can separate the two
  explanations.

*Ask:* 20–30 minutes of an unoccupied pitch, at ≥ 2 venues, day **and** night.

## 2 · Complete slots with a known real verdict ★

**Frames are not the bottleneck — slots are.** The slot-level decision layer is the
system's operational contribution, and its real test set is currently **n = 2**. No
statistical claim survives n=2, and it is the first thing an examiner will notice.

Slot labels are nearly free: one label per hour of footage, taken from the facility's own
booking sheet — *was this slot actually used?*

*Ask:* **30–40 complete slots**, continuous, with the facility's own record of whether each
was used. Spread across fields and times of day if possible; consecutive slots from one week
are fine and better than none.

## 3 · The booking export, in the same conversation ★

Reconciliation is the operational contribution and cannot be evaluated without the record it
reconciles against. Asking for this now rather than in three months costs the same
conversation.

*Ask:* a read-only export covering whatever period the footage covers — field, date, start
time, booked yes/no, and the staff-recorded outcome if one exists. CSV is fine. No names or
customer details are needed; see §7.

★ **And one extra column, for a reason found on 2026-09-11: whether the slot was blocked for
maintenance, separately from its booking status.** The reconciliation matrix has a SERIOUS
anomaly for a pitch that was closed for maintenance and sold anyway — and as the importer is
written it can never fire, because `booked` and `maintenance_window` are both derived from a
single `status` column and that column cannot hold two values at once. One field cannot say
"blocked" and "confirmed" simultaneously, so the conflict is inexpressible and every instance
of it in this repository is a hand-written fixture. A separate `blocked`/`closed` flag makes
the anomaly reachable from real data; without it, that row of the matrix is a design that
cannot be evaluated. It costs the facility one more column of an export they already run.

## 4 · Maintenance and non-sporting activity

Currently **6 frames** in the whole corpus, all from one moment at one camera — so the third
class has no evaluable support and every macro-average in the thesis is effectively
two-class.

*Ask:* brooming, line-painting, mowing, seeding, net repair, or any groundstaff activity on
the pitch. Day **and** night. Also idle-people slots — walk-throughs, photo sessions,
someone crossing the pitch — which is the harder and more useful half of this class, because
the system must not call a person crossing a pitch a match in progress.

## 5 · More pitches at the same facility

Two cameras at one venue is what makes the current cross-venue evaluation one-directional.

*Ask:* ≥ 2 further pitches at the existing facility, any activity.

## 6 · Two external venues, both lighting regimes

*Ask:* ≥ 2 venues beyond the current set, ≥ 2 slots each, one daylight and one floodlit.
Rain or fog days if they happen to occur — not worth scheduling for, worth keeping if they
arrive.

## 7 · What is *not* being asked for

Stated explicitly because an over-broad data request is harder to approve than a narrow one,
and because the answer to "why do you need this?" should be short.

- **No customer names, contact details or payment information.** The booking export needs
  field, date, time and a used/not-used flag; nothing that identifies a person who booked.
- **No staff identifiers for analysis.** Discrepancies are reported per field, never per
  person — that is enforced in code and tested (`slots/authority.py`).
- **No continuous recording.** One frame per camera per minute is the system's own sampling
  rate and is sufficient.
- **No footage retained beyond the agreed period.** See `thesis/ethics.md`.

---

## If only one thing can be supplied

**Item 1** — empty pitches somewhere other than venue_01. It is 20 minutes of a camera
pointing at nothing, and it is the difference between a thesis that can make a claim about
recognising an empty pitch and one that cannot.

**If two: items 1 and 2.** Together they turn the two weakest parts of the evaluation — the
class that cannot be tested, and the slot-level test set of two — into ordinary ones.
