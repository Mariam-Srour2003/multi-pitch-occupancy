# Ideas backlog

Things worth doing that are **not** in the plan of record (`TODO.md`). Each live idea carries
a **stop condition** - what would make it not worth doing - because an idea without one tends
to get built anyway.

The numbering is stable: `conditions.py`, `augment.py`, `augmentation_transfer.py` and
`TODO.md` all cite these as `IDEAS #n`. Entries marked BUILT are kept as pointers, short,
because the code and its tests are the record now.

---

## 1. Weather as an explanation, not just a condition

Detect rain from the frame and attach it to the verdict - *"NOTUSED - heavy rain observed
throughout"* rather than a bare `NOTUSED`.

**Why.** `booked / used / NOTUSED` in clear weather is a possible no-show; the same row in
heavy rain is almost certainly a cancellation. Today they are indistinguishable and both
raise `NO_SHOW_OR_OVERRECORDED`. This is the cheapest available improvement to anomaly
precision (RQ4), and a false accusation is the failure mode the ethics position cares most
about.

**The trap.** Rain correlates with empty pitches, because people stop playing when it rains -
the day-versus-night confound arriving a second time. Any weather work must be evaluated with
weather and occupancy **crossed** (rain-with-play and rain-without-play both present) or it
manufactures a second shortcut on top of the first.

**Blocked on data**, genuinely later rather than soon: the corpus has no weather axis at all,
`lighting` is day/night only. First probe when footage exists is a zero-shot OpenCLIP prompt,
wet versus dry - no labels, and it answers whether the signal is there before anything is
built.

**Stop condition.** If zero-shot cannot separate wet from dry, stop. The downstream value
does not justify a labelled weather dataset stacked on the occupancy one.

---

## 2. Synthetic rain as training augmentation — BUILT, unmeasured

`vision/augment.py`: `AugmentConfig`, `synthetic_fog`, `synthetic_rain`, five presets, 16
property tests. No rotations or warps - the cameras are bolted to a post, so a rotated pitch
is not a harder example but an impossible one. Horizontal flip is the deliberate exception:
it cannot change whether people are playing, but it breaks memorisation of *this* pitch's
layout.

**Why it is unmeasured.** Augmentation runs before the backbone, so every view needs its own
forward pass and the feature cache stops being a cache - roughly 8 minutes per model per
epoch-equivalent, against seconds for a probe fit on cached features. It is not a switch to
flip inside the existing experiments; it needs its own extraction budget.

**Stop condition.** There is no real rain footage (#1), so synthetic rain can today only be
validated against synthetic rain - which proves something about the generator and nothing
about weather. The honest question available now is whether `colour` jitter improves
cross-venue recall. If it does not, the module stays shipped and unclaimed.

---

## 3. Verdicts carry the conditions they were formed under — BUILT

`slots/conditions.py`, produced by `run_slot`: capture rate, confidence distribution, camera
disagreement, cameras seen, lighting, contrast, plus `concerns()`, which orders them by how
much each undermines the verdict. `weather` is reserved and unset, so #1 has somewhere to
land.

It turns `REVIEW` from a puzzle into something actionable - *"mean confidence 0.55, contrast
30% below this camera's baseline, 8 of 60 minutes missing"* - out of numbers that were
already computed and then thrown away.

---

## 4. Evidence brackets the moment a slot changed — BUILT

`slots/evidence.py`: `find_transitions` and `select_evidence_around_transitions`, wired into
`run_slot`. A change must hold for five minutes on both sides, which separates a match ending
from a player walking through frame; fixed thirds still apply when nothing changed. For a
disputed slot, "here is minute 20 and here is minute 22" settles the question better than
three frames that all show the same thing.

---

## 5. Sweep the preprocessing space rather than hand-picking variants

The input ablation found a non-monotone surface - crop50 best, grayscale second, both
together worse than neither - sampled at five hand-chosen points. A small sweep, crop
fraction 0.4/0.6/0.8 x colour on/off, would locate the optimum rather than guess it.

**Stop condition.** One embedding pass per cell, ~8 minutes each on this hardware, and a
hand-drawn ROI polygon should beat any centre crop and would change the surface anyway. Not
worth running until real polygons exist.

---

## 6. Captured elsewhere

- DINOv3, quantisation/OpenVINO, active learning, self-supervised domain adaptation, NL
  explanations for REVIEW, camera-health prediction, test-time augmentation, multi-tenant
  productisation - the extensions backlog in `TODO.md`.
- ROI polygons (WP3-T1), quality filter and camera health (WP3-T5), train-time augmentation
  (WP3-T6) - plan of record, not ideas.
