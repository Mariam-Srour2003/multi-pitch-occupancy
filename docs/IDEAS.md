# Ideas backlog

Things worth doing that are **not** in the current plan, with enough reasoning attached
that a later reader can judge them without rediscovering the argument. Nothing here is
committed to; the plan of record is `TODO.md`.

Each entry says what it would take and what would make it *not* worth doing, because an
idea without a stopping condition tends to get built anyway.

---

## 1. Weather as an explanation, not just a condition

**The idea.** Detect rain (and later fog, wind, snow) from the frame, and attach it to the
verdict. So instead of `NOTUSED`, the system reports *"NOTUSED - heavy rain observed
throughout"*, or *"USED - play continued through rain"*.

**Why it is more than a nicety.** The current verdict answers *what happened*. It cannot
answer *why*, and for the audit the why decides whether an anomaly is real:

| booking | staff record | vision | weather | what it actually is |
|---|---|---|---|---|
| booked | used | NOTUSED | clear | possible no-show or over-recording - **flag it** |
| booked | used | NOTUSED | heavy rain | almost certainly a weather cancellation - **do not flag** |
| booked | not used | USED | heavy rain | played through rain, check-in missed - flag, but sympathetically |

Right now rows 1 and 2 are indistinguishable, so the reconciliation layer would raise the
same `NO_SHOW_OR_OVERRECORDED` warning for both. **Weather is the single cheapest thing
that would improve anomaly precision** (RQ4), because it removes a whole class of false
accusations - and false accusations are the failure mode the ethics position cares most
about.

**A second use: abandoned matches.** A slot that reads PLAY for twenty minutes and then
EMPTY for forty is a different event from one that was never used. Pair that transition
with rain and the system can say *"play started, then stopped - rain began at minute 22"*,
which is exactly the evidence a disputed booking needs. STAN (WP5-T1) would learn this
shape from the sequence anyway; weather makes it explainable rather than merely detected.

**How it could be detected, cheapest first.**

1. **Zero-shot VLM prompt.** OpenCLIP is already in the stack. Prompt *"a football pitch in
   heavy rain"* vs *"a football pitch in dry weather"*. Needs no labels, and it is the
   fastest way to find out whether the signal is there at all. Try this before building
   anything.
2. **Reuse the quality metrics.** WP3-T5 already plans blur (Laplacian variance) and
   contrast measures for camera health. Rain on the lens looks exactly like a camera-health
   problem - lowered contrast, localised blur blobs - so the same numbers may separate wet
   from dry with no new machinery. **Note the ambiguity this creates**: "the lens is dirty"
   and "it is raining" would produce a similar signature, and confusing them means either
   dispatching a cleaner into a downpour or ignoring a genuinely dirty lens.
3. **Temporal signature.** Rain streaks are fast, oriented, high-frequency, and *move*
   between consecutive frames; wet turf changes specular reflection more slowly. Frame
   differencing over a few consecutive frames would separate them, at the cost of sampling
   more than one frame per minute.
4. **A learned head**, only if the above are insufficient - the same frozen-backbone +
   linear-probe recipe on a `weather` label.

**The trap, stated up front.** Rain will correlate with empty pitches in any real dataset,
because people stop playing when it rains. That is *the same confound the project already
found the hard way* with day-vs-night: a model could learn "wet-looking frame -> EMPTY" and
score well without ever detecting a person. Any weather work must therefore be evaluated
with weather and occupancy **crossed** - rain-with-play and rain-without-play both present -
or it will manufacture a second confound on top of the first.

**What it needs before it can start.** Footage in the rain, with both outcomes. The current
1,692 frames contain **no weather axis at all** - `lighting` is day/night only, and nothing
in the manifest records precipitation. This is a genuine "later", not a "soon".

**Stop condition.** If the zero-shot probe cannot separate wet from dry on whatever rain
footage eventually arrives, stop - the downstream value does not justify a labelled weather
dataset on top of the occupancy one.

---

## 2. Synthetic rain as training augmentation — BUILT (untested)

**The idea.** WP3-T6 already lists synthetic fog. Add rain: oriented motion-blurred streaks
at varying density and angle, plus a slight contrast reduction and a wet-turf specular
lift.

**Why it is the right shape of fix.** The input ablation established that *removing*
information has a floor - grayscale plus crop fell below the untouched baseline, and (once the
false-play control was added on 2026-09-11) that only grayscale is above it: the crop's larger
apparent gain was 0.998 recall at **empty accuracy 0.000**. Augmentation
is the complement: it makes the model invariant to a nuisance factor **without discarding
anything at inference time**, so it cannot cross that floor. Where grayscale threw colour
away permanently, colour jitter and synthetic rain leave the pixels intact and teach the
model not to lean on them.

**Built** in `vision/augment.py`: `AugmentConfig` (brightness, contrast, saturation, hue,
gamma, noise, fog, rain, horizontal flip, each gated at probability `p`), `synthetic_fog`,
`synthetic_rain`, and five presets - `none`, `colour`, `weather`, `light`, `full`. Covered
by 16 property tests. Two design decisions are pinned by tests rather than left to comment:

- **No rotations, warps or perspective changes.** The cameras are bolted to a post. A
  rotated pitch is not a harder example, it is an impossible one.
- **Horizontal flip is the one exception, deliberately.** It produces a mirror the camera
  never sees, which is why it helps: it cannot change whether people are playing, but it
  breaks memorisation of *this* pitch's layout - the exact failure the cross-venue
  evaluation exists to catch.

**Not yet measured, and the reason matters.** Everything downstream trains a linear probe on
**cached** embeddings, one vector per frame. Augmentation happens before the backbone, so
every augmented view needs its own forward pass - the cache stops being a cache. At the
measured extraction rate that is roughly 8 minutes per model per epoch-equivalent of views,
against seconds for a probe fit on cached features. So this is not a switch to flip inside
the existing experiments; it needs its own extraction budget, which is why the module ships
tested but unclaimed.

**Cheap to validate.** Train on clean frames with and without synthetic rain, evaluate on
whatever real rain footage exists. If synthetic rain does not improve real-rain recall, it
is decoration. Note the ordering trap: **there is no real rain footage** (see idea 1), so
today synthetic rain can only be validated against synthetic rain, which proves nothing
about weather and everything about the generator. The honest claim available now is the
narrower one - whether `colour` jitter improves *cross-venue* recall, which needs no
weather at all and is directly motivated by the ablation.

---

## 3. Every verdict should carry the conditions it was formed under — BUILT

**The idea.** Generalise the weather idea: attach an observed-conditions record to each
slot evaluation - mean contrast, lighting, capture rate, person counts, weather once it
exists - and surface it in the evidence inspector.

**Built** in `slots/conditions.py`, produced by `run_slot` and covered by tests. A slot now
carries capture rate, confidence distribution, camera disagreement rate, cameras seen,
lighting and contrast, plus `concerns()` which turns them into plain-language reasons
ordered by how much each undermines the verdict. `weather` is reserved and unset, so idea 1
has somewhere to land.

**Why.** `REVIEW` is currently a verdict with a sentence attached. An operator opening a
REVIEW slot has to work out from three images why the system hesitated. If the record said
*"mean confidence 0.55, contrast 30% below this camera's baseline, 8 of 60 minutes
missing"*, the REVIEW becomes actionable in seconds rather than a puzzle. Most of these
numbers are already computed and then thrown away.

---

## 4. Evidence should show the moment a slot changed, not fixed thirds — BUILT

**The idea.** Evidence selection currently takes the most confident frame from each third.
Where a slot contains a **state transition** - play starting late, stopping early,
maintenance arriving - the most informative frame is the one either side of the change.

**Built** in `slots/evidence.py` as `find_transitions` and
`select_evidence_around_transitions`, wired into `run_slot`. A change must hold for five
minutes on both sides to count, which is what separates a match ending from a player
walking through frame; thirds still apply when nothing changed. An abandoned match now
reports "active play to empty at minute 22" and brackets its evidence around that minute.

**Why.** For a disputed slot, "here is minute 20 and here is minute 22" settles the
question far better than three frames that all show the same thing. Thirds remain the right
default for a uniform slot; the transition rule would apply only when a change is detected.

---

## 5. Sweep the preprocessing space rather than hand-picking variants

**The idea.** The ablation tested five hand-chosen variants and found a non-monotone
surface: crop50 best, grayscale second, both together worse than neither. That surface was
sampled at five points. A small sweep - crop fraction 0.4/0.6/0.8 x colour on/off - would
locate the optimum rather than guess it.

**Cost.** One embedding pass per cell (~8 min each on this hardware). Worth it only once
real ROI polygons exist, since a hand-drawn polygon should beat any centre crop and would
change the surface anyway.

---

## 6. Ideas already captured elsewhere

Not repeated here, to keep one list per idea:

- DINOv3, quantisation/OpenVINO, active learning, self-supervised domain adaptation,
  NL explanations for REVIEW, camera-health prediction, test-time augmentation,
  multi-tenant productisation - see the extensions backlog in `TODO.md`.
- ROI polygons (WP3-T1), quality filter and camera health (WP3-T5), train-time augmentation
  (WP3-T6) - these are *plan of record*, not ideas; they live in `TODO.md`.
