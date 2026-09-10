# Measurement protocol

What is applied to a frame, in what order, and by whom. Written because the answer turned
out not to be the one the code appeared to give.

---

## The pipeline, as intended

```
raw frame  →  preprocess.py  →  HF processor  →  backbone  →  mean pool  →  probe
              (searched)         (normalise)      (frozen)
```

`preprocess.py` is described throughout this project as **the single preprocessing path**:
one place where geometry and photometry are decided, with ten switches that the
preprocessing search explores. The Hugging Face processor was understood to do only what a
model needs to interpret its input — rescale to `[0, 1]` and normalise with that model's
training statistics.

> **Correction, 2026-09-08 — for the main feature caches it is not a path at all.**
> `build_cache` opens **raw frames** and hands them straight to the HF processor, whose
> resize is the only thing making them model-sized. Its `preproc` argument is a dict of
> *labels for the fingerprint*; it has never driven a transform, and `preprocess` is never
> imported there. So the letterbox of WP3-T2, ROI masking, CLAHE and the rest of the ten
> switches **do not touch any cached feature the headline experiments read**. The
> preprocessing search and the input ablation are the exception — those scripts call
> `preprocess()` themselves, which is why they keep their own cache family.
>
> The diagram above is therefore the *intended* pipeline and the one the search operates in,
> not the one behind the cross-venue numbers. This document should not be read as saying the
> two agree.

> **Decision, 2026-09-08 — the two paths stay separate, and the write-up says so.**
> `results/input_path_protocol.csv` tested whether adopting `preprocess.py` in `build_cache`
> (TODO WP3-T3 option (b)) is carried by the evidence. It is not: the finding that motivated
> it reverses when the two cameras the false-play control is built from are swapped, the
> cross-venue recall axis cannot reach significance with seven venue folds, and no frame-level
> comparison survives being recounted by distinct scene. So **option (c) stands** — the
> default cache path keeps the processor's own geometry on raw frames, and every claim about
> a searched preprocessing switch is scoped to the search and ablation cache family.
>
> This is a decision under uncertainty, not a finding that preprocessing does not matter. The
> mechanism is real — a 1920×1080 frame resized shortest-edge to 256 and centre-cropped to
> 224 keeps roughly the middle half of the pitch. What is missing is footage that could
> measure its cost: **every EMPTY frame in the corpus is venue_01**, so the input path and
> the camera pair cannot be separated. Revisit when a second venue's empty footage arrives.

## The pipeline, as measured (WP3-T3, 2026-09-07)

The photometric half is correct. Each model is normalised with its own statistics, because
each model's own processor is called:

| model | rescale | mean | std | correct |
|---|---|---|---|---|
| ConvNeXtV2 | 1/255 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | yes — ImageNet |
| ViT | 1/255 | (0.5, 0.5, 0.5) | (0.5, 0.5, 0.5) | yes — its own convention |
| DINOv2 | 1/255 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | yes — ImageNet |

**The geometric half is not.** The processor does not only normalise. Given a frame that
`preprocess.py` has already resized to 224×224 with letterbox padding, it resizes and crops
it again:

| model | what the processor does to a 224×224 frame | frame kept | cosine vs. no-geometry |
|---|---|---|---|
| **ConvNeXtV2** | `crop_pct=0.875` → resize to **256**, centre-crop 224 | **76.6%** | **0.886** |
| ViT | resize to exactly 224×224 | 100% | 1.000 |
| **DINOv2** | resize shortest edge **256**, centre-crop 224 | **76.6%** | **0.961** |

Two of the three upscale by 1.143 and then cut a 16-pixel border from every side —
**23.4% of the frame area, discarded after preprocessing has run**. The two affected models
are the default model (DINOv2) and the one the 740-minute preprocessing search ran on
(ConvNeXtV2). ViT is the only backbone that sees what preprocessing produced.

### Why it was not noticed

ConvNeXtV2 is the instructive case. Its configuration reads:

```python
{'size': {'shortest_edge': 224}, 'crop_pct': 0.875, ...}
```

`size` says 224, the input is 224, so it reads as a no-op — and the first pass of this audit
recorded it as one. `crop_pct` is what makes it resize to `224 / 0.875 = 256` first and then
crop back. There is no `do_center_crop` flag to notice. Reading `size` alone gives the wrong
answer, and `size` is the field one naturally reads.

DINOv2 declares it openly (`do_center_crop`, `crop_size`), which is why it was found first.

## What this affects

1. **The letterboxing of WP3-T2 is partly undone.** Aspect-preserving resize with grey
   padding was added, tested, and then had its padding cropped off again for two of three
   models — along with a slice of real image.
2. **The preprocessing search optimised a transform that was partly overwritten.** Every
   geometric switch — `centre_crop`, `top_crop`, `letterbox`, `undistort` — was applied and
   then re-cropped. The switches are not meaningless, but they do not mean exactly what they
   say.
3. **It supplies a mechanism for a result already recorded as a puzzle.** The search found
   `centre_crop=0.5` at −0.029 alone and **−0.200** combined with `sharpen`, logged as
   non-additivity. With a hidden crop already discarding 23.4%, an explicit half-crop is a
   crop applied to a crop, and the model is left looking at a small central patch of pitch.
   The input ablation's `crop50` gain (+0.038 on DINOv2) is the same compound.
4. **It does not invalidate the cross-venue findings.** Every model was measured under the
   same pipeline, so comparisons between them are sound, and the confound results rest on
   labels rather than on pixels.

## The decision, and why it is not yet made

`embed_batch(..., processor_geometry=False)` disables the processor's resize and crop while
keeping rescale and normalisation, so the model sees exactly what `preprocess.py` produced.
That is the coherent choice for a project where preprocessing is a *searched variable*: a
search over transforms that are then partly overwritten is optimising the wrong function.

It is **not** the default yet, for two reasons.

- The 256-then-crop transform is the canonical inference recipe for both models — how they
  were evaluated when published. Departing from it is a choice that should be **measured**,
  not applied quietly as a bug fix.
- Flipping it invalidates every cached feature and every result built on one. The flag is
  part of the cache fingerprint, so the two conventions can never silently mix.

> **Correction, 2026-09-08.** That last sentence was not true when it was written. The flag
> was **not** in the fingerprint, and `build_cache` had no parameter for it at all — so the
> alternative convention could not be cached, and had it been, both conventions would have
> collided on one cache key *and* one filename. `embed_batch` accepted the flag; nothing
> above it did.
>
> Now: `build_cache(..., processor_geometry=False)` threads it through, the flag is in the
> fingerprint, and the non-default convention gets its own file (`<backbone>_nogeom.npz`) so
> the two can exist side by side — which is what "compare them" requires. Six tests pin it.
> The existing caches were verified to be `processor_geometry=True` builds bit-for-bit
> before their stored fingerprint was migrated to the new formula, so no published number
> moves.

**What settles it:** one cross-venue run per model under each convention. If disabling the
processor geometry improves transfer, the preprocessing path becomes genuinely single and
the search results should be regenerated under it. If it does not, the finding stands as a
documented property of the pipeline and the switches keep their asterisk.

---

## WP2-T11: the starved third class stays, and not for the reason it was created

**Decision taken 2026-09-10, on measurement rather than on the plan.** Pending supervisor
confirmation at the next check-in, which is the only part of this item a person still owns.

C3 (`3_people_not_playing` + `4_maintenance` → MAINTENANCE_NON_SPORTING) holds **6 frames**,
all from one moment on one camera. The fallback tree written in week 3 offered three ways out
and the risk register recommended **Option C** — report the benchmark as 2-class and state the
scope reduction. `experiments/empty_recognition.py` measured what that would cost, on the 243
held-out empty frames of the camera-transfer control:

| configuration | DINOv2 false-play rate | absorbed by C3 |
|---|---|---|
| 3-class, balanced | **0.309** | 0.691 |
| 3-class, unbalanced | 0.881 | 0.119 |
| **2-class** (Option C) | **1.000** | — |

**Option C is refuted for the production model.** Dropping C3 does not leave the false-play
axis unchanged; it takes DINOv2 from calling 31% of unfamiliar empty pitches *play* to calling
**all of them** play. The six frames are absorbing 69% of the frames the model cannot place.

**Why that matters operationally rather than statistically.** `empty_accuracy` is 0.000 in
every configuration — the model never correctly identifies an empty pitch on this transfer
set, and no arrangement of classes changes that. What changes is **where the errors land**.
C3 maps to NOTUSED/REVIEW and ACTIVE_PLAY maps to USED, so a misplaced empty frame is either a
slot sent to a human, or a slot billed as used. The starved class converts 69% of would-be
billing errors into correct-or-reviewable verdicts. It earns its place as a *none of the
above* sink, which is not the job it was defined to do.

**So the taxonomy is unchanged and the claim is what shrinks.** Three classes stay. C3 is
**not a claimable class**: 6 frames from one moment cannot support a per-class metric, no
leakage-free split puts it on both sides (WP3-T7), and `taxonomy.py` already forbids reporting
a 4-class macro-F1 improvement while its support is near zero. Every table that reports macro
over three classes must name C3's support beside it, and no sentence anywhere may describe the
system as detecting maintenance. It does not. It has a sink, and the sink is load-bearing.

**What would change this.** Real maintenance footage (WP2-T8, `data_requests.md`) turns C3
into a class that can be claimed and makes the question a genuine three-class one. Until then
the honest statement is the one above, and it is stronger than the scope reduction it
replaces: the class is kept for a measured operational reason rather than dropped for a
presentational one.

**What was rejected, and why.** *Option A* (compositing hi-vis workers onto empty pitches) was
already ruled out on 2026-09-09: no configuration of C3 recovers empty-pitch accuracy, and what
does is one labelled frame of the target camera, so synthesising maintenance crops answers a
question the data says is not binding. *Option B* (an open-vocabulary detector as a rule-based
C3 branch) remains available and is now better motivated than before — the sink works, and a
detector would make it deliberate rather than incidental — but it is a new module in write-up
week, and nothing currently rests on it.
