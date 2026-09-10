# Defence deck (WP8-T4)

**A slide plan, not slides.** Each entry is one slide: what is on it, what you say, and — where
it matters — what *not* to say. Figures are named by file so the deck is regenerated rather
than redrawn, and every number cites a claim id from `thesis/claims.md`, so a result that
changes breaks this document instead of quietly outdating it.

**Twenty minutes of talk, ten of demo, thirty of questions** is the shape this is built for.
Cut from §4 first if you are short; never cut §5.

> **The framing decision, and make it consciously.** There is a version of this defence that
> apologises: *we tried a gated fusion head and it did not work, we tried STAN and the test set
> was too small*. There is another that is true and much stronger: **this is a thesis about how
> to evaluate this problem, and the evaluation kept turning out to be measuring something other
> than what it claimed.** Every negative result below is evidence for that thesis rather than a
> gap in it. Decide now which one you are giving, because the two need different opening
> sentences and you cannot switch halfway.

---

## §1 · The problem (3 slides, 3 min)

### 1. Title

Multi-pitch occupancy and booking verification from existing cameras. Your name, supervisor,
date. Nothing else.

### 2. What the facility actually needs

One line of the problem: a facility sells hours on pitches, and nobody knows which sold hours
were used. Cameras are already installed for highlights.

*Say:* the output is not a classification, it is a **billing conversation** — "this booking
looks unused, here are three frames from it".

*Do not say* "we detect football". The taxonomy is EMPTY / ACTIVE_PLAY / MAINTENANCE, and
maintenance is the class that stops a mower being billed as a match.

### 3. Why cameras and not a €20 motion sensor

**Concede first.** A PIR is more robust in fog, darkness and against a dirty lens. Then: a
sensor answers *was something moving*; the audit needs *was this booking used, and here is the
picture*, and it has to tell a five-a-side match from a groundsman.

→ `thesis/alternatives.md`. This is the question most likely to open, per WP8-T6.

---

## §2 · The pilot, and the two bugs that set the direction (3 slides, 4 min)

This section exists because it is honest and because it is *interesting*. It also
pre-empts "how do we know your harness is right".

### 4. The spec's chosen model lost

The proposal named ConvNeXtV2. Under leakage-free evaluation **DINOv2 leads cross-venue play
recall at 0.9297** (`h3-cross-venue-dinov2`) and ConvNeXtV2 does not.

*Say:* the decision was made on a measurement, and the measurement was made after the proposal.
That is the process working.

### 5. The pooler-output bug, and the lesson

An early harness read `pooler_output` — a randomly-initialised projection for these
checkpoints — instead of pooling the hidden states. Every early number was noise dressed as a
result.

*Say the lesson, not just the bug:* **a wrong result that looks plausible is invisible.** That
observation is the reason for everything in §3; it is where the claims ledger and the
break-the-guard practice come from.

### 6. Leakage: the number that reframed the project

Frames are sampled every 15 s from fixed cameras, so **98.5% have a direct near-duplicate**. A
random split puts 37.1% of near-duplicate pairs across the train/test boundary; a grouped split
puts 0.8%.

Cost of being honest: **ConvNeXtV2 loses 0.4904 macro-F1** (`h1-drop-convnextv2`) — the score
roughly halves. On the leaky split **every one of the 12 errors** was a frame whose
near-duplicate was in training (`leaky-errors-had-a-near-duplicate`).

Figure: `figs/leakage_decomposition`.

---

## §3 · The contribution: the evaluation kept lying (5 slides, 6 min)

**The heart of the defence.** Four independent cases, each found by checking rather than by
theorising. Deliver them as a sequence — the pattern is the point.

### 7. A rule that never looks at the image

The clock rule — majority class per lighting condition, no pixels — scores **0.4907** on the
honest grouped split, within **0.0068** of ConvNeXtV2 (`floor-clock-rule-grouped`).

*Say:* this is why four trivial baselines run in every protocol. The rule does collapse across
venues, to **0.219** (`h3-cross-venue-clock-rule`) — which is the strongest single piece of
evidence that the backbones learn something transferable.

### 8. A protocol a constant predictor wins

Under leave-one-venue-out, **a constant predictor scores macro-F1 1.000**
(`constant-predictor-wins-cross-venue`), ahead of every backbone, because every held-out venue
is 100% ACTIVE_PLAY.

*Say:* stronger than "the protocol changes the ranking" — on that protocol there is no ranking
to change. Figure: `figs/ranking_inversion`.

### 9. **The false-play rate measures something else** *(the strongest slide)*

The rate has been quoted throughout as how well a model recognises an empty pitch. Ask what the
models answer *instead* of ACTIVE_PLAY on the 243 held-out empty frames:

> **DINOv2 answers ACTIVE_PLAY 75 times and MAINTENANCE 168 times. It is correct zero times**
> (`false-play-is-not-accuracy`). No model exceeds 0.165.

The protocol trains on camera A and tests on camera B: it is a **camera-transfer test**, not a
specificity test. The published numbers stand; the inference drawn from them does not.

*Say:* we found this by adding a column to check whether a rate of 0.000 was real. It was not —
that model answers MAINTENANCE on all 243.

### 10. What it costs to onboard a camera

The same finding, turned into the deployment answer. A probe trained on one camera scores
**0.36–0.51** macro-F1 on the other camera watching the same pitch. **One labelled frame takes
every backbone to ~0.98** (`onboarding-one-frame`).

And the control: **from five frames, training on those alone matches those plus 775 from the
source camera** (`onboarding-source-stops-helping`).

*Say the weaker claim, because it is the true one:* what buys the accuracy is having *any*
labels from the new camera, not a large corpus elsewhere. Figure: `figs/onboarding_cost`.

### 11. Guards that were not guarding

Six examples, one line each: a test-set lock resolved against the working directory; a
processor-geometry fingerprint that never changed; a confidence threshold at 0.0 that can never
fire; a gate criterion that grepped for a word instead of running the verifier; a live camera
class whose `n_minutes` raised, so it could never run; evidence selection that recorded a path
of `None` for every frame it chose.

*Say:* each new guard is now verified by **deliberately breaking the thing it guards**. That
practice is a contribution of this thesis and it is what found half of §3.

---

## §4 · The novel modules, reported as they came out (3 slides, 4 min)

*Cut this section first if short. Do not cut §5.*

### 12. Gated multi-backbone fusion — negative

The specified architecture, built. **Routing is worth −0.0238 cross-venue recall** against the
same head with the gate off, on one informative fold of seven, p = 1.000.

*Say two things.* The ablation is a three-rung ladder (uniform → constant → mlp) because an
on/off switch credited *routing* with what is a learned constant — the gate puts ~0.70 of its
weight on DINOv2 in every fold and moves it by 0.086. And the implementation **can** route: on
a fixture where the useful backbone flips, the lower rungs score 0.671 and routing scores
1.000. The null is about the data.

### 13. STAN — preliminary, and the benchmark is saturated in three draws of five

A 1,651-parameter temporal convolution, against four tuned baselines including an HMM. It
scores **1.0000** on 200 composed slots, beating the HMM by 0.105 at p < 0.0001.

*Say the reading, not the number:* that is an exhausted test set, not a win. The composed label
is a deterministic function of five templates, so a model that reads contiguity recovers the
generating process. **The real test set is two slots.** The ≥30-slot rule is enforced in code —
`assert_preliminary` raises, and the M4 gate criterion checks the caveat is in the results.

*If asked whether that 1.0000 is stable:* it is not, and the answer is measured. Re-composing
the slots from four further draws gives 1.0000 three times, then 0.9100 and 0.8000 — mean
0.9420, sd 0.0884. **The ordering is what survived**: STAN is first in five draws of five and
no baseline matches it in any. Volunteer this before it is asked; it is the same check that
retracted the augmentation headline, and running it on your own headline is the point.

### 14. The decision layer, which is not a classifier

Fusion of two cameras → slot aggregation → reconciliation against bookings → advisory.
`slots/authority.py` makes "the system must never bill" structural: `Advisory` is the only
output, `requires_human_confirmation` is a property rather than a parameter, and there is no
code path that acts.

*Say:* the ethics commitment is enforced, not stated.

---

## §5 · Demo (10 min) — **never cut this**

### 15. Demo slide

Live, from a terminal beside the deck.

```
uv run pitch serve                     # then open http://127.0.0.1:8000/client
```

**The run order, and stop after each:**

1. **Meters and anomalies** — the manager's daily view. One unbooked-usage anomaly, one
   consistent slot.
2. **Open a slot** → the evidence inspector. **Three photos**, the reason the model gave, and
   the per-minute table. *Say: this is what a billing conversation is made of.*
3. **Override it** with an operator name. Show the audit fields — the model's original verdict
   is retained, never replaced.
4. `uv run pitch retraining` — the override's evidence frames staged for re-labelling.
   *Say: staged **unfiled** — a slot verdict is not a frame label.*
5. `uv run pitch bookings` — the booking export shape being asked of the client.

**If the demo fails**, say so and move on; do not debug in front of the panel. Fallback:
`results/project_site.html` opens offline and contains every figure and table.

> **Rehearse the demo twice on the machine you will present from**, with the database seeded
> (`uv run pitch seed`). The most common defence failure is a demo that worked yesterday.

---

## §6 · Limits, and what would change them (3 slides, 3 min)

### 16. What this cannot claim

From `thesis/threats_to_validity.md`, the three that an examiner will find anyway — so name
them first:

- **EMPTY exists at exactly one venue.** Cross-venue three-class evaluation is impossible, and
  no frame-disjoint train-synthetic/test-real split exists — the two recorded slots hold 1,296
  of 1,692 frames including every EMPTY and every MAINTENANCE frame.
- **The effective sample is scenes, not frames.** 243 held-out empty frames are **three
  distinct scenes**; the corpus is ~150. Six comparisons significant to p = 8e-53 did not
  survive being recounted that way.
- **Nothing has run on the target hardware.** Every latency number is from a laptop.

### 17. Tests that could not have been significant

Several comparisons run over seven venue folds, where the smallest attainable two-sided p is
**0.0156** — and 1.000 when only one fold differs. `sign_flip_test` reports its own resolution
floor beside every p-value.

*Say:* "not significant" and "the design could not have produced significance" are different
statements, and we report which one applies.

### 18. What would actually change this

Ordered by value, not effort:

1. **≥30 real labelled slots** — one conversation with the facility. Lifts STAN out of
   preliminary and gives reconciliation a ground truth.
2. **Empty-pitch footage at a second venue** — fixes the confound, the external-validity limit
   and the C3 scope reduction at once.
3. **Five labelled frames per camera** — measured, cheap, and now the onboarding recipe.

**Close on this:** the limitations are mostly a data-access problem with a known and
inexpensive solution, not a methodological one. That is a much better last sentence than an
apology.

---

## §7 · Reserve slides (not presented; have them ready)

Keep these after the final slide and jump to them on the matching question. The full answers
are in `thesis/defence_redteam.md`.

| Question | Slide |
|---|---|
| "Would a colour histogram have done this?" | 0.9616 on the leaky split (`floor-histogram-random`); indistinguishable from ConvNeXtV2 on the 62 distinct scenes |
| "Is your novelty just temporal smoothing?" | The four tuned baselines, including the HMM — and STAN's margin over it, with the saturation caveat |
| "How many comparisons before p < 0.05?" | Holm within declared families; the pre-registration and its amendments A1–A12 |
| "Is auditing staff by camera ethical?" | Per-field reporting only; `entered_by` carried and never aggregated; `authority.py` |
| "Can anyone reproduce this?" | `reproduce_all.py` with per-stage dependencies; every claim in the ledger re-derived from its artefact on every run |
| "What is the human ceiling?" | **No inter-annotator figure exists.** Say so plainly — it is a real gap (WP2-T5) |
| "Why is your XAI trustworthy?" | The decomposition is exact, not Grad-CAM: the head is linear over a mean pool, and the run aborts if the map fails to reconstruct the score |

---

## Rehearsal notes

- **Time §3 with a clock.** It is the section you will want to over-run and the one worth
  protecting.
- **Practise saying "we do not know" three times.** The human ceiling, the target hardware, and
  whether any of this transfers to a second venue. An examiner trusts a candidate who has
  bounded their ignorance more than one who has not noticed it.
- **The strongest sentence you have** is a version of: *"we measured that, and here is how large
  it is."* Most of §3 and §6 is that sentence. Where it is not available — §6's three limits —
  the second-strongest is *"that is a real limitation and here is exactly what it costs."*
- Rewrite every argument here in your own words before the mock exam (WP8-T7). The numbers are
  checked by the ledger; the phrasing is not, and phrasing you did not choose collapses on the
  first follow-up.
