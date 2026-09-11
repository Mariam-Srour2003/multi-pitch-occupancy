# Prompt: "Draw what was trained and what was frozen"

Paste everything below the line into Claude. Every number in it was read out of this
repository, not remembered — the sources are named so the diagram can be checked against them.

---

## Task

Draw me a single diagram of the machine-learning pipeline in my MSc thesis project (pitch
occupancy detection from CCTV frames of football pitches). The diagram's **one job** is to
make this unmistakable at a glance:

> **The three vision backbones were never trained on my data. They are frozen, off-the-shelf
> pretrained feature extractors. What was trained is a small classifier head sitting on top of
> their output — plus two small downstream models further along the pipeline.**

My supervisor is going to ask "so did you actually train anything?" and I need a picture that
answers that in five seconds, and then rewards a closer look with the reasons.

Use a **frozen/trained visual split that is impossible to misread** — e.g. a cool desaturated
palette with a padlock motif for frozen blocks, a warm saturated palette with a spark motif for
trained blocks, and a legend that states the rule once. Do not rely on colour alone; put a
literal "FROZEN — 0 parameters updated" / "TRAINED — N parameters fitted" label inside every
block.

## The pipeline, stage by stage

Lay it out left to right in five stages. Show that the boundary between stage 2 and stage 3 is
**the only place where my labels enter the picture** — this is the single most important line
in the diagram. Consider drawing it as a literal vertical seam labelled *"labels enter here,
and nowhere to the left of this line."*

### Stage 1 — Input (not trained, not learned)

- 1,692 labelled frames from 10 venues. 1,578 are *development* rows across 8 venues; the
  remaining rows at 2 venues are **locked** and deliberately never evaluated until development
  is finished.
- Class counts across all rows: `C1_EMPTY` 494, `C2_ACTIVE_PLAY` 1,192,
  `C3_MAINTENANCE_NON_SPORTING` **6**. Show that "6" prominently — it drives design decisions
  further down the diagram.
- Deterministic preprocessing: aspect-preserving resize and letterbox to 224×224. No learning.

### Stage 2 — Three frozen backbones, run once each (THE KEY STAGE)

Draw these as three parallel lanes. Each lane is one padlocked block. Inside each, show:

| | ConvNeXtV2 | ViT | DINOv2 |
|---|---|---|---|
| checkpoint | `facebook/convnextv2-tiny-22k-224` | `google/vit-base-patch16-224` | `facebook/dinov2-base` |
| architecture | convnet, stages `[3,3,9,3]`, widths `[96,192,384,768]` | transformer, 12 layers, hidden 768, patch 16 | transformer, 12 layers, hidden 768 |
| **parameters, ALL FROZEN** | **27,866,496** | **86,389,248** | **86,580,480** |
| how it was pretrained | ImageNet-22k, masked-autoencoder + supervised | supervised ImageNet-21k → 1k | self-supervised, LVD-142M, **no labels at all** |
| its role in the thesis | production lead — smallest and fastest | accuracy reference | robustness reference |
| pooling | `last_hidden_state.mean(dim=(2,3))` global average pool | `last_hidden_state.mean(dim=1)` over patch tokens | `last_hidden_state.mean(dim=1)` over patch tokens |
| output | one 768-d vector per frame | one 768-d vector per frame | one 768-d vector per frame |

Together that is **200,836,224 frozen parameters**, and **zero** of them received a gradient at
any point in this project. Put that total somewhere large.

Three details inside this stage that the diagram should carry as small annotations, because
each one is a real finding:

1. **The pooling is explicit and deliberate.** Hugging Face exposes a `pooler_output` on these
   models; on a plain ViT that pooling head is **randomly initialised and never trained** —
   loading the checkpoint literally reports `pooler.dense.weight | MISSING`. Reading it cost an
   earlier pilot a false 38% score and hours spent debugging a "model problem" that was not
   one. Every extractor here therefore mean-pools explicitly, and every cached feature file is
   stamped `pooling="mean"` so a stale cache is rejected loudly. Draw this as a red-flagged
   "trap avoided" note on the ViT lane, or as a crossed-out `pooler_output` box.
2. **Two of the three silently re-crop the frame.** ConvNeXtV2 (`crop_pct 0.875`: resize to 256
   → centre-crop 224) and DINOv2 (resize shortest edge 256 → centre-crop 224) each discard
   **23.4% of the frame area** — exactly where the letterbox padding lives. ViT resizes to
   exactly 224×224, a true no-op. Measured cosine similarity between embeddings with and
   without that crop: **0.886 for ConvNeXtV2, 0.961 for DINOv2, exact for ViT**. Show the crop
   as a visible bite taken out of the frame on two of the three lanes.
3. **Each backbone is run exactly once over all 1,692 frames**, into a cached feature matrix of
   shape `(1692, 768)`. Draw the cache as an explicit artefact between stage 2 and stage 3.

### Stage 3 — The trained classifier head (small, and the point)

One block, fed by the cache. This is what "training" means in this project:

- `StandardScaler` → `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)`
- **768 features × 3 classes + 3 biases = 2,307 trained parameters.**
- Refitted from scratch for every backbone × split protocol × seed. Takes under a second.
- `class_weight="balanced"` exists because C3 has 6 frames in the entire dataset and an
  unweighted fit simply never predicts it.

Make the size contrast between stage 2 and stage 3 **visually literal** — 200.8M frozen versus
2,307 trained is a ratio of about **87,000 : 1**. If the blocks are drawn to scale the trained
box is a speck. Do that deliberately, and annotate it: *"this speck is the only thing that ever
saw a label."*

### Stage 4 — Two more genuinely trained models, further downstream

These matter because they are the answer to "so you trained nothing?" — draw them, smaller, as
a second row or a branch:

**(a) Gated fusion head** (research question 5 — can a router pick the best backbone per frame?)

- Input: 3 cheap per-frame statistics — brightness, contrast, edge density.
- Gate: tiny MLP, one hidden layer of width 8 → softmax weights over the K backbones.
- Body: `concat_k(w_k · standardised features_k)` → one shared linear head → 3 classes.
- **≈ 7,000 trained parameters.** Full-batch Adam on CPU, 300 epochs, lr 0.01, weight decay
  1e-3, seed 42 — reproducible to the bit.
- Ablation ladder with everything else held identical: `uniform` (weights fixed at 1/K) →
  `constant` (K learned weights shared by all frames) → `mlp` (weights learned per frame).
  `constant − uniform` is the value of learned mixing; `mlp − constant` is the value of
  *routing*, which is the only thing a gate is for.
- **Result to show honestly:** it puts ≈ 0.70 of its weight on DINOv2 in every fold and moves
  that weight by only 0.086 between frames. It is a learned constant wearing a router's
  costume. Against `uniform` alone it would have read as "the gate works."

**(b) STAN — spatio-temporal aggregation network** (the thesis's headline novelty)

- Input: a slot's ordered sequence of per-minute class probabilities.
- Two dilated 1-D convolutions, dilations 1 and 2, 16 channels, kernel 5 → receptive field
  ≈ 13 minutes. Mean **and** max pooling concatenated → 3-way slot status.
- **≈ 3,000 trained parameters**, against a planned ceiling of 100k. Variable-length slots are
  padded and masked, never truncated.
- Trained on 200 composed slots built from 1,578 frames.
- **Draw its guard rail.** Only **2** real labelled slots exist, and the project sets a hard
  gate of 30 before any headline claim. The code enforces this itself rather than trusting a
  note in a plan. Show it as a barrier: "PRELIMINARY — gated at 30 real slots, have 2".
- Its baselines are trained too, and that is the point: tuned thresholds, median smoothing, a
  3-state HMM with counted transitions and Viterbi decoding, and a logistic regression on four
  summary numbers. Beating unlearned constants would prove nothing.

### Stage 5 — Controls that are trained on *nothing* (the comparison floor)

A separate, visually distinct column. These exist so that "the deep model works" is a claim
that can fail:

- **Majority class** — predicts the most common training label. The true zero point.
- **Clock rule** — reads only the lighting field, no pixels at all. It learns "night means a
  match, day means an empty pitch" and scored **98.4%**, matching the trained backbones. Give
  this its own callout; it is the most uncomfortable number in the project.
- **Mean intensity** and **colour histogram** probes — one number, and a coarse RGB histogram.
  Neither can represent "are there players on the pitch."
- **OpenCLIP zero-shot** (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`) — **never fitted at all**.
  Its `fit()` is a deliberate no-op, so the same per-frame label is scored under every split
  protocol. That makes it a true control: any movement in its score is caused purely by which
  frames are in the test set. Draw it as a padlocked block with a struck-through `fit()`.

## The reasoning the diagram must carry

Do not just label blocks frozen and trained — show **why**, and **what each choice bought**.
Put these as annotated callouts attached to the relevant blocks, as cause → effect pairs.

**Why the backbones are FROZEN — four reasons, each with its consequence:**

1. **Sample size.** 1,578 development frames across 8 venues cannot support fine-tuning 86M
   parameters. *Effect:* a fine-tuned model would memorise the venues, and the finding would be
   about my 8 sites rather than about the models.
2. **It makes the comparison fair.** Freezing holds the representation fixed, so when DINOv2
   beats ConvNeXtV2 the only thing that differs **is the representation**. *Effect:* fine-tune
   all three and I am comparing three optimisation runs — three learning rates, three
   schedules, three random seeds — not three representations. The research question would
   quietly change underneath me.
3. **Cost and reproducibility.** Embed once, reuse forever: embedding 1,692 frames takes
   minutes, fitting the head takes under a second. *Effect:* this is the only reason 4 split
   protocols × 5 seeds × 8 predictors, plus every ablation and label-efficiency curve, was
   runnable at all. Freezing is what bought the experimental breadth.
4. **It makes the headline finding legible.** *Effect:* a leaked split plus a fine-tuned 86M
   network gives 0.99 and no explanation. A leaked split plus a 2,307-weight linear probe gives
   0.99 **and a mechanism you can point at** — the probe is too small to do anything clever, so
   near-duplicate frames are the only available explanation, and that was then measured
   directly.

**Why the heads ARE trained:** they are the only part small enough to fit honestly on this much
data, and they are the part that is actually task-specific. A 768→3 logistic regression cannot
memorise 1,578 frames; an 86M transformer can.

**What freezing does NOT excuse** — draw this as a caveat box, because it is the obvious
counter-question. The backbones' pretraining sets (ImageNet-21k/22k, LVD-142M) contain sports
imagery I did not choose and cannot audit. Freezing removes *my* training leakage; it does not
remove *their* pretraining exposure. This is a stated limitation, not a solved problem.

## The anticipated objection — give it its own panel

Features are extracted **once over every frame, before any split exists**. Somebody will say:
*"then the model has seen your test set."* The diagram should pre-empt this:

- The backbone is frozen and **never sees a label or a split**. Embedding a test frame is
  therefore just preprocessing — mathematically identical to embedding it after the split.
- The real leakage is elsewhere, and it is in the **split**, not the model: frames sampled
  seconds apart are near-identical, so a random shuffle puts the same moment on both sides.
- Evidence, measured directly: **100%** of the errors on the leaky split had a near-duplicate
  frame on the training side; on the honest grouped split, **0%** did.

Draw this as two miniature split diagrams side by side — random shuffle with duplicate frames
straddling the train/test line, versus grouped-by-scene with a clean cut.

## The results panel — get the metric right

If you include the headline numbers, they are **macro-F1**, not accuracy, and that distinction
is the whole point. Seed 42, three probes on frozen features:

| backbone | random split (leaky) | grouped-by-slot (honest) |
|---|---|---|
| ConvNeXtV2 | 0.9879 | 0.4975 |
| ViT | 0.9940 | 0.4975 |
| DINOv2 | 0.9879 | 0.5794 |

Four things to annotate, because each is load-bearing:

- **Accuracy barely moves** — it stays around 0.99 on both. Only macro-F1 collapses. Label the
  axis explicitly so nobody reads this as an accuracy drop.
- **0.4975 is the single-class floor.** The honest test set is **99.01% one class**, so
  predicting that class for every frame yields accuracy 0.9901 and macro-F1 0.4975 — *exactly*
  what ConvNeXtV2 and ViT score. So the honest statement is **"they collapse to the majority
  class"**, which is stronger than "the score halves", and it explains why two models tie at
  joint-last. DINOv2's 0.5794 is the only score above the floor.
- **The control subtraction:** an untrained model drops 0.183 across the same change in
  test-set composition, so the leakage-attributable part is **0.332–0.395**, not the full drop.
- **The ranking reversal:** ViT ranks first under the leaky protocol and joint-last under the
  honest one.

## Style and format

- One page, landscape, self-contained — readable printed in greyscale and on a projector.
- Left-to-right flow with the five stages clearly delineated and the "labels enter here" seam
  drawn as a real line.
- Blocks drawn roughly to parameter scale wherever it does not destroy legibility; where it
  would, break the scale explicitly and say so rather than silently fudging it.
- A legend defining frozen vs trained, stated once, using shape and texture as well as colour.
- Every quantitative claim on the diagram should be one a reader could go and verify.
- Prefer clean vector-style output (SVG, or an HTML artifact) over a raster image.
- Do not invent numbers. If something is needed that is not specified above, leave a clearly
  marked placeholder rather than filling it in plausibly.
