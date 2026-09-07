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

**What settles it:** one cross-venue run per model under each convention. If disabling the
processor geometry improves transfer, the preprocessing path becomes genuinely single and
the search results should be regenerated under it. If it does not, the finding stands as a
documented property of the pipeline and the switches keep their asterisk.
