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

**And as of 2026-09-17 there is a sharper way to say it.** The lighting labels for the clip
venues were wrong - a brightness threshold filed 216 frames of floodlit night football as
daylight - and correcting them exposes a confound that the venue table hides:

| | EMPTY | ACTIVE_PLAY | MAINTENANCE |
|---|---|---|---|
| **daylight** | **485** | 6 | 6 |
| **night / floodlit** | **9** | **1186** | 0 |

**"Night means play, day means not-play" is correct on 99.1% of the recorded corpus**, and on
98.8% within venue_01 alone. So nothing measured on this data can separate *recognises an empty
pitch* from *recognises the time of day* — and that is not a labelling mistake, it is how
five-a-side pitches are used. People play in the evening.

What that costs, concretely: a rule that reads only the clock and never looks at the image
**beats all three frozen backbones** on this project's cross-venue protocol, on recall and on
false-play at once. On a 234-second clip of a floodlit pitch with nobody on it, the same rule is
wrong on **every one of 16 minutes**. The corpus cannot tell those two facts apart. Two cells
would:

| the two frames that matter most | currently |
|---|---|
| **an empty pitch at night** | 9 frames, all venue_01 |
| **active play in daylight** | 6 frames, all venue_01 |

Item 1 below is both of those cells. Everything else in this list is worth less than either of
them, and the first of the two is worth more than the second.

---

## 1 · Empty pitches at another venue — and, above all, at night ★

> **Sharper as of 2026-09-17, and the reason changed.** This was originally about what the
> model cannot *learn*. It is now also about what nothing can be *checked* against. Six
> separate measurements this week ended at the same wall:
>
> | what was being measured | why it stalled |
> |---|---|
> | the false-play collapse when clip venues enter training | no empty pitch to measure it on but venue_01's |
> | H3's cross-venue recall | its folds are 100% play, so false-play is unmeasurable |
> | the motion gate's threshold | not calibrated across venues, and nothing to calibrate it on |
> | ROI pooling, train against serve | the two tests disagree and only venue_01 can arbitrate |
> | scene deduplication | its cost shows on venue_01 and its benefit only on borrowed footage |
> | **the person count** | **100% recall at nine venues; false-play checkable at one** |
>
> The last is the sharpest. A pretrained detector counting people inside the boundary misses
> **no frame of real play at any of nine venues**, which makes it the strongest component in
> the system - and its error rate can be measured at exactly one site, because that is the
> only site with an empty pitch on record.
>
> Twenty minutes of an unoccupied pitch at two other venues would settle all six.

**The highest-value frames in this request, and the cheapest to produce.** An empty pitch
needs no scheduling and no event — just a camera pointed at one for a few minutes, at two or
three sites, in daylight and under floodlights.

> **Ask for the floodlit half first.** After the 2026-09-17 correction, *empty at night* is the
> single cell that breaks the confound: it is the combination the clock rule must get wrong and
> a working model must get right. Twenty minutes of an unused pitch after dark is worth more
> than an hour of one in the afternoon, because the afternoon frames are the 485 the corpus
> already has.
>
> **And if any of it can come from `clipvenue_b_floodlit_track` or `clipvenue_c_teal_boards`,
> ask for that.** Those two are the locked final test set: 114 frames, every one ACTIVE_PLAY,
> every one night. As it stands a model that answers ACTIVE_PLAY unconditionally scores 1.000
> on it, so opening it buys a confirmation rather than a test (A27). Empty frames from *those
> two venues specifically* would turn it into a final test set that can refute something —
> footage from anywhere else is development data and cannot join a locked set afterwards.

**The other half of the cell, and it is not free like the first.** *Active play in daylight* -
the corpus has six frames - needs a booked game in the afternoon, which is a scheduling request
rather than a camera pointed at nothing. It is worth asking for because it closes the confound
from the other side: with empty-at-night alone a model could still be reading "is it dark", and
daylight play is what rules that out. But if only one of the two can be had, take the empty
night pitch: it is free, and it is the combination the trivial baseline must get wrong.

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
