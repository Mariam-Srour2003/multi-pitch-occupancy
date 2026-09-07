# Labelling protocol (WP1-T2)

**Status: draft for supervisor approval — the M1 gate artefact.**
Written 2026-09-07 by consolidating rules that already existed but were scattered across
`TODO.md`, `SUPER_PLAN.md`, `PLAN.md`, `src/pitch_occupancy/data/taxonomy.py` and
`thesis/ethics.md`. Nothing here is invented; where a rule was never decided it is marked
**OPEN** rather than filled in.

Why it matters more than its length suggests: **all 1,692 labels in this dataset already rest
on these rules**, and until now they existed only in one person's head. `thesis/mvt.md`
identifies this as the single genuine hole in the minimum viable thesis. A single-annotator
dataset *with* a written protocol is defensible; without one, every number downstream rests on
an undocumented definition — which is the first thing an examiner will probe about a
one-person labelling effort.

> Not to be confused with **[`protocol.md`](protocol.md)**, which is the *measurement* protocol
> — what is done to a frame's pixels. This document is about what a frame *means*.

---

## 1. What is being labelled

**Scene state, never identity.** No face recognition, no re-identification, no tracking a
person between frames or cameras (`ethics.md`). The question a label answers is only: *what
kind of activity is happening on this pitch right now?*

Frames are labelled into **four folders** and reported as **three classes**. The finer
distinction costs nothing at labelling time and keeps a 4-class ablation available
(`taxonomy.py`).

| Folder on disk | Reporting class | Slot meaning |
|---|---|---|
| `1_empty` | **C1 EMPTY** | NOTUSED |
| `2_playing` | **C2 ACTIVE_PLAY** | USED |
| `3_people_not_playing` | **C3 MAINTENANCE_NON_SPORTING** | NOTUSED / REVIEW |
| `4_maintenance` | **C3 MAINTENANCE_NON_SPORTING** | NOTUSED / REVIEW |

**The folder is authoritative**, not `labels.csv` — the CSV supplies provenance only
(`manifest.py`). A frame's class is where it sits on disk.

---

## 2. Frame-level rules

Apply in order. The first rule that matches decides.

### 2.1 The region of interest

**People outside the pitch do not count.** Spectators behind the fence, passers-by, staff on
an adjacent pitch, and traffic in the background are all irrelevant to whether *this* pitch is
in use.

> **Honest caveat on enforcement.** ROI polygons have **not** been drawn (WP3-T1 is open, and
> `preprocess.py` deliberately omits `roi` from its defaults because `roi_mask` with no polygon
> returns the frame untouched). So this rule is currently applied by the annotator's judgement
> and is **not** enforced in code. Every label in the dataset already depends on a boundary
> that only the annotator can see. Drawing the polygons would turn this from a convention into
> a constraint, and is the single change that would most improve label reliability.

### 2.2 ACTIVE_PLAY (`2_playing`)

Ball-and-athletic activity on the pitch, in any of these forms — all are usage:

- a match, at any number of players;
- warm-up, drills, shooting practice;
- **academy or youth training** — counts as ACTIVE_PLAY, not as a separate class;
- **goalkeeper-only, while a real match is running.** This is the case two-camera fusion
  exists for: one camera can see a nearly-empty half while play is genuinely happening on the
  other. Label the *frame* by what the frame shows only if the match is not visible in it —
  see 2.6 — and never let a goalkeeper-only half make a live slot read as empty.

### 2.3 EMPTY (`1_empty`)

No people within the ROI, and no activity. An unattended ball, cones left out, or open goals
do **not** make a pitch occupied.

### 2.4 People present but not playing (`3_people_not_playing`)

Humans inside the ROI without athletic activity:

- walk-throughs, someone crossing the pitch;
- a **coach carrying gear**;
- photo sessions, events, a birthday party, standing around talking;
- anyone inside the ROI whose activity is not sport.

### 2.5 Maintenance (`4_maintenance`)

Groundskeeping. The discriminating cue is equipment or workwear, not posture:

- **hi-vis clothing plus a tool** — broom, rake, line-marker, seed spreader;
- a mower or other machinery on the surface;
- line-painting, brushing, seeding, net or goal repair.

**`hi-vis + tool = C3`** is the operative test. Hi-vis alone (a referee's bib, a bright kit) is
not maintenance; a tool alone with no person is not either.

### 2.6 Ties and uncertainty — the rule the annotator follows

Written explicitly so that Cohen's κ (WP2-T5) measures genuine disagreement rather than
missing instructions.

1. **Activity beats emptiness.** If any athletic activity is visible in the ROI, the frame is
   ACTIVE_PLAY, however few people are involved.
2. **Sport beats non-sport.** A maintenance worker at the edge while a match is on is
   ACTIVE_PLAY. This matches the fusion module's priority order
   (`playing > maintenance > people > empty`), so labels and code agree.
3. **Non-sport beats emptiness.** Any person in the ROI rules out EMPTY.
4. **When genuinely undecidable, do not guess — set it aside.** Put the frame in a
   `_ambiguous/` holding folder with a one-line reason instead of forcing a class. An honest
   count of ambiguous frames is a *result*; a forced label is silent noise in every number
   downstream.
5. **Never label from context you cannot see in the frame.** Not the filename, not the
   timestamp, not the neighbouring frame. The models see one frame; so does the annotator.
   This one is load-bearing: the day/night confound and the `lighting` mislabelling both came
   from metadata standing in for pixels, and labelling from a clock would bake that confound
   into the ground truth itself.

---

## 3. Slot-level rules

**OPEN — and the highest-priority gap in this document.**

STAN (WP5-T1) is trained on *slot* labels, so ambiguity here propagates straight into the
thesis's headline novelty. Only **2 real labelled slots** exist, which is why these rules have
never been exercised — but they must be written *before* slot labelling starts, not after.

The frame→slot aggregation baseline is in `slots/aggregate.py` and is explicitly a set of
hyper-parameters, not constants:

```
ACTIVE_PLAY ratio >= 0.35                          -> USED
ACTIVE_PLAY ratio <  0.10 and EMPTY ratio >= 0.75  -> NOTUSED
otherwise                                          -> REVIEW
```

A *human* slot label is the ground truth those thresholds are tuned against, so it must be
decided independently of them. The cases that need a written answer:

| Case | Question | Proposed rule |
|---|---|---|
| **Late arrival** | Players arrive 20 min into a 60 min slot | **USED.** The slot was used; partial use is use. Any duration rule invites arguing about the cutoff. |
| **Early finish / abandoned** | Play for 20 min, then empty for 40 | **USED**, with the transition recorded. `slots/evidence.py` already selects transition-aware evidence, so the shape is visible to a reviewer. |
| **Back-to-back overlap** | Players of two consecutive slots overlap the boundary | Assign each frame to the slot **whose scheduled window contains its timestamp**, with no grace period. A fixed arbitrary rule beats a judgement call, because a judgement call cannot be reproduced. |
| **Non-sporting booked use** | Booked slot used for a birthday party | **USED for billing, C3 for vision.** These genuinely disagree, and the disagreement is the reconciliation layer's job to surface — not something to hide by picking one. |
| **No-show** | Booked, nobody came | **NOTUSED.** |
| **Unbooked usage** | Not booked, played anyway | **USED.** The booking status must never influence the vision label — that independence is the whole audit. |
| **Camera failure mid-slot** | One or both cameras drop out | **REVIEW**, never a verdict. Degraded input must not produce an accusation (WP7-T5). |

**Requires a decision from the client, not from us:** whether a slot used for a non-sporting
booked purpose counts as USED for their billing. Everything else above can be fixed by fiat as
long as it is written down and applied consistently.

---

## 4. Provenance: how a label was produced

Recorded per frame in the manifest's `labeled_by` column, because the two are not equally
strong evidence:

| value | meaning | count |
|---|---|---|
| `human` | inspected and labelled individually | 1,058 |
| `bulk` | filed as a batch — the 396 clip frames, verified with YOLO plus visual review of outliers, not one by one | 634 |

**`bulk` is not a synonym for `human`**, and no table should present them as one. The 396 clip
frames are still awaiting an individual spot-check (WP2-T3).

### Known label errors, uncorrected

Two frames in `2_playing/` show an empty pitch:
`slot_20260711_1000_camA_t000021_m.jpg` and `..._t000027_m.jpg`. Both are human-labelled. They
are 2 of only 6 daytime ACTIVE_PLAY frames at `venue_01`, so **correcting them makes the
day/night confound more absolute, not less**. Deferred because moving them invalidates every
feature cache (WP2-T4); do it between search runs, then re-run `reproduce_all.py`.

The `lighting` column for the 396 clip frames is a **brightness proxy, not a clock**, and is
wrong for at least `b_floodlit_track`, `f_outdoor_bldg` and `c_teal_boards` — night football
filed as `day`. Hand relabelling is ~1 hour (WP3-T5) and it moves a reported figure, so it is
not cosmetic.

---

## 5. Agreement measurement (WP2-T5)

- **10% sample**, labelled blind by a second annotator, κ computed by `tools/kappa.py`.
- **Target κ ≥ 0.85.** Below that, the disagreements are read, this document is amended, and
  the sample is relabelled — the protocol is the thing that gets fixed, not the annotator.
- **Disagreements are logged, not silently resolved.** Which classes get confused tells you
  where the definitions are weak; C1-vs-C3 is the expected boundary.
- **Human ceiling (WP2-T9):** the second annotator also labels the held-out test set, so
  human-vs-consensus accuracy can be reported as the ceiling. This is what makes a model
  accuracy interpretable — if humans reach 97% on the same frames, further accuracy chasing
  is wasted effort, and *that* is a defensible finding.

---

## 6. Open items blocking approval

| | Item | Who decides |
|---|---|---|
| 1 | Slot-level rules in §3 — confirm or amend before any slot labelling | Supervisor + client |
| 2 | Non-sporting booked use: USED or NOTUSED for billing | Client |
| 3 | ROI polygons drawn, so §2.1 becomes a constraint rather than a convention | Us (WP3-T1) |
| 4 | Second annotator identified for the κ sample and the human ceiling | Supervisor |
| 5 | C3 taxonomy decision (WP2-T11) — with 6 frames and 0 maintenance, the recommendation is Option C: report 2-class with C3 as a flagged exception, and state the scope reduction plainly | Supervisor |

Once §3 is confirmed and this document is approved, the M1 gate's labelling half is closed.
