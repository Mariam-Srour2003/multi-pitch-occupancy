# The minimum viable thesis

**Written 2026-09-07**, at the request in `TODO.md` §"Capacity check". Its purpose is stated
there: *"Knowing your floor makes every later scope decision calm instead of panicked."*

This document is the floor, not the plan. `TODO.md` remains the plan of record. Nothing here is
an instruction to stop early — it is the answer to *"if everything else fell through, is what I
already have a thesis?"*

**The answer is yes.** That is worth knowing before the next scope decision, not after it.

---

## The floor: four things, all already in hand

A complete, defensible master's thesis exists if these four hold, and as of today all four do.

### 1. A leakage-free evaluation of frozen-backbone occupancy classification, with honest statistics

Four models, grouped and leave-one-venue-out splits, bootstrap CIs, McNemar with an exact
regime, Holm–Bonferroni, effect sizes beside every p-value, and effective sample size beside
every frame-level number.

*Status: done and defensible.* `h1_h2_baseline_floor.csv`, `h3_cross_venue_recall.csv`,
`h3_sensitivity_merged_venues.csv`, `evaluation/stats.py` with 25 tests.

### 2. The methodological result: honest evaluation *reverses the decision*

The headline is not "accuracy drops under a harder split" — every reader expects that. It is
that a reader following the pilot's protocol would have **selected the weakest generaliser**:
ViT ranks 1st under the leaky split and 3rd under the honest one, and the production
recommendation moves from ConvNeXtV2 to DINOv2 once latency stops discriminating.

Paired with the trivial-baseline floor — a clock rule within 0.007 macro-F1 of ConvNeXtV2 using
no pixels, a colour histogram statistically indistinguishable from DINOv2 on the leaky split's
62 distinct scenes — this is a contribution about *measurement*, and measurement contributions
survive the data limits below intact.

*Status: done.* `figs/ranking_inversion.png`, `rq_matrix.md` RQ3 and RQ7.

### 3. The data limits, reported as findings rather than discovered at the defence

That EMPTY exists at one venue, that C3 has six frames and maintenance zero, that 98.5% of
frames have a near-duplicate, that the honest grouped split is 99% single-class, that a global
blur filter flags exactly one venue, that `lighting` is a brightness proxy for the clip venues —
each of these was measured, written down, and used to scope a claim.

**This is the part most theses get wrong, and doing it well is itself the answer to the hardest
examiner question.** `preregistration.md` §"Not answerable" plus its amendment log is the
artefact.

*Status: done.*

### 4. A working end-to-end system on recorded slots

Sampling worker, two-camera fusion, slot aggregation to USED/NOTUSED/REVIEW with evidence,
reconciliation against booking records with anomalies per field and never per person, SQLite
persistence, and an operator surface. Validated end-to-end on the two real slots that exist.

*Status: built; validated at n = 2 and reported as such.*

---

## What the floor deliberately does **not** require

Each of these is desirable, in the plan, and **not** load-bearing. If any fails, the thesis
still stands.

| Not required | Why the thesis survives without it |
|---|---|
| STAN as a headline result | Its real test set is n = 2. `preregistration.md` §4 already commits to reporting it as preliminary on synthesised sequences. WP5-T8's hard gate says the same. |
| ≥30 real labelled slots | Only STAN and RQ4's precision/recall need them. Both are already scoped as preliminary/open. |
| A non-degenerate test set for RQ6 | The calibration and risk–coverage machinery is built and tested. That it *cannot be answered on this data* is itself a reportable finding about the dataset, and no engineering unblocks it. |
| Live deployment on the Mini-PC (WP7) | Reported as a deployment plan with dev-hardware latency and a stated hardware caveat. M6 is the most droppable gate in the plan. |
| A released dataset | `ethics.md` already removed it: operator-supplied footage of identifiable people is not releasable. Features-only, or nothing. |
| The context head and distillation (5.C, 5.D) | Explicitly stretch tier. |

---

## The one thing the floor *does* require that is not yet done

**A labelling protocol (WP1-T2).**

Everything above rests on 1,692 labels. The document defining what a label *means* — people
outside the ROI don't count, goalkeeper-only during live play is ACTIVE_PLAY, coach carrying
gear is C3 — has not been written, and it is the M1 gate artefact that was due in week 4.

**Drafted 2026-09-07 → [`labelling_protocol.md`](labelling_protocol.md).** It was the only
genuine hole in the floor, it was a writing task rather than a compute task, and it was cheap.
Until it existed, every number above rested on a definition that lived only in one person's
head — exactly the question a sharp examiner asks about a single-annotator dataset.

**The hole is not fully closed, and the remainder is not ours.** The document needs supervisor
sign-off, and two of its open items need the client (the slot-level rules, and whether a booked
slot used for a non-sporting purpose counts as USED). One item *is* ours and would materially
improve label reliability: no ROI polygons have been drawn (WP3-T1), so "people outside the
pitch don't count" is currently the annotator's judgement rather than a constraint in code.

Its companion, WP2-T5's second-annotator κ, is *not* in the floor: it would be better to have,
but a single-annotator dataset with a written protocol is defensible, while a
multi-annotator one without a protocol is not.

---

## How to use this document

When a scope decision arrives — a module that will not converge, footage that will not come, a
week lost — the question is not *"can I still do everything?"* It is:

> **Does this threaten items 1–4, or the protocol?**

If no, the answer is to note it, scope the claim, and move on. That covers almost every decision
this project will face between now and submission.

If yes, escalate to the supervisor the same week, because that is the only category of problem
that can actually cost the thesis.
