# Speaker script — 25 minutes

**This script follows the site.** Start the server, open <http://127.0.0.1:8000>, and move
through the eight tabs in order, scrolling each one.

```bash
uv run pitch serve
```

Headings below are `Tab.Block` and repeat each block's own heading, so you always know where
on the page you are. The site carries **no** delivery notes — this file is the only place
the talk is written down.

**Read only the plain paragraphs aloud.** Blockquoted boxes are prepared answers for
pushback. Passages marked **⟨cut if short⟩** can go if you are running over — they are
depth, not structure.

Things still yours to fill in are marked **[LIKE THIS]**.

| Tab | Blocks | Starts | Runs |
|---|---|---|---|
| 1. Summary | 4 | 0:00 | 2:10 |
| 2. Plan | 2 | 2:10 | 0:50 |
| 3. General Introduction | 5 | 3:00 | 2:35 |
| 4. Ch.1 State of the Art | 4 | 5:35 | 2:25 |
| 5. Ch.2 Methods & Model | 8 | 8:00 | 4:30 |
| 6. Ch.3 YOLOv8 | 6 | 12:30 | 4:25 |
| 7. Ch.4 Applications | 7 | 16:55 | 5:40 |
| 8. Conclusion | 5 | 22:35 | 2:05 |

**Measured, not guessed.** At 140 words a minute with the **⟨cut if short⟩**
passages dropped, the spoken text is **24 minutes 40**; reading every word including them is
**26**. Tabs 5, 6 and 7 are together three fifths of the talk, which is right &mdash; they
are the three chapters with results in them. Time yourself once and adjust.

---

# Tab 1 — Summary · 0:00

### 1.0 · Opening

Good morning, distinguished examiners, professors and colleagues.

My name is Mariam Srour, and I am a Master's student in **[PROGRAMME]** at
**[INSTITUTION]**. My supervisor is **[NAME]**.

The title of my work is *Multi-Pitch Occupancy and Booking Verification from Existing
Cameras*.

### 1.1 · "The abstract"

In one paragraph: a facility rents five-a-side pitches by the hour and cannot verify which
slots were actually used. This work samples **one frame per camera per minute** from the
CCTV that is already installed, classifies each frame as empty, active play or maintenance,
aggregates an hour into a verdict, and reconciles that verdict against the booking record —
on a single mini-PC with **no GPU**, and with a person confirming every flag.

### 1.2 · "The study in numbers"

The scale, once, so nobody has to guess it. Sixteen hundred and ninety-two labelled frames,
three classes, seven venue folds for cross-venue testing, and no GPU anywhere in the design.

### 1.3 · "Three findings"

And three findings, which I will build up to over the next twenty minutes.

**First, the evaluation was measuring the wrong thing.** Lighting and occupancy are
confounded in this data: a rule that reads only the clock is right on 99.1% of the corpus,
and beats all three deep backbones across venues they have never seen.

**Second, one short clip reversed a benchmark of sixteen hundred frames** — because it
contained the one case the benchmark was missing.

**Third, a result we published did not survive being re-run.** An augmentation headline of
0.855 turned out to be the maximum of five random draws; four of the five were worse than
using no augmentation at all. We retracted it and restated it.

### 1.4 · "What it contributes"

So the contribution is on three levels: an **evaluation method**, a **deployment recipe** of
five labelled frames per new camera, and a **decision layer** that can only ever advise —
enforced by the type system rather than by a policy document.

---

# Tab 2 — Plan · 2:10

### 2.1 · "The eight parts"

The report is in eight parts: this summary, the plan, a general introduction, four chapters,
and a conclusion.

**Chapter 1** is the state of the art — playgrounds and deep learning. **Chapter 2** is the
methods and our model. **Chapter 3** is YOLOv8, the detector. **Chapter 4** is the
applications and the data augmentation. And the conclusion sets out what remains.

### 2.2 · "What each chapter answers"

Put as four questions: Chapter 1 asks *what is already done, and what is missing?* Chapter 2
asks *which model, and why that one?* Chapter 3 asks *how do we count people and a ball on a
pitch?* And Chapter 4 asks *does it work in practice, and what did we do about the data?*

---

# Tab 3 — General Introduction · 3:00

### 3.1 · "The problem in general"

The setting is a facility renting twenty or thirty synthetic five-a-side pitches in hourly
slots, all day, most of them after dark.

The record of which slots were used is written **by hand**, by staff, and nothing checks it.
And nothing can: one manager physically cannot watch twenty pitches at once, so they
spot-check — and a spot check is a sample, not an audit.

Three errors therefore go unseen: **no-shows**, where a slot is paid for and never played;
**unbooked use**, where a pitch is played on and never paid for; and plain **data-entry
error**, where the record and the reality simply drift apart.

### 3.2 · "Authors who have worked on it"

The literature this work sits in has nine strands. Fixed-camera occupancy and activity
recognition. Frozen features and linear probes. Dataset leakage and evaluation protocol.
Trivial baselines and benchmark validity. CPU-constrained inference. Object detection for
people and small objects. Data augmentation under domain shift. Prior art in facility and
booking verification. And selective prediction — the literature behind our REVIEW band.

> **Say this plainly, and do not skip it:**
>
> "The references on this page are shown as placeholders on purpose. My rule in this project
> has been that no citation is written down until I have opened and read the paper, because
> a half-remembered reference is the one error in a thesis that cannot be defended. The nine
> strands state what each must establish; completing them is the next step in the write-up."

**⟨cut if short⟩** The strand that carries the most weight is the third one — dataset
leakage — because that is where my central result sits. If the literature already reports
leakage effects of the size I measure, then my contribution is the *decomposition*, not the
discovery, and I would want to say so.

### 3.3 · "The problem to be resolved"

Stated precisely: given only the cameras a facility already owns and a computer with **no
GPU**, decide for each booked hour whether the pitch was used — accurately enough to raise a
billing conversation, and with evidence a customer can be shown.

Four constraints sharpen that. It must recognise an **empty** pitch, not just a busy one,
because the money rests entirely on that class. It must work at a venue it has **never
seen**. It must run twenty to thirty cameras inside a **sixty-second** cycle. And it must
never act on its own.

### 3.4 · "How we resolve it"

Five steps, and I will spend the rest of the talk on them.

**Sample sparsely** — one frame per camera per minute instead of decoding video, which is
about ninety-nine per cent less network traffic. **Classify with frozen features** — three
pretrained backbones used as fixed extractors with a small trained head. **Count with a
detector** — YOLOv8 finds people and the ball inside the pitch boundary, and cheap rules can
overrule the classifier. **Aggregate and reconcile** against the booking. And **evaluate
honestly** — grouped splits, trivial baselines, and every safeguard verified by breaking it.

### 3.5 · "Decomposition of the report"

Which maps onto the four chapters you see here.

---

# Tab 4 — Chapter 1, State of the Art · 5:35

### 4.1 · "Playgrounds — how occupancy is measured today"

Before the deep learning, the domain. There are five ways a facility can already answer
"was this pitch used", and I costed all of them.

A **PIR motion sensor** is about twenty euros and measures motion in a cone. A **turnstile**
is three hundred to two thousand and counts people through a gate. A **clamp meter on the
floodlight circuit** is about sixty euros and tells you the lights are on. An **app
check-in** is free. And **manual logging** costs staff time.

*(Point at the last row.)* This system reuses the CCTV that is already there, and measures
scene state per minute, with evidence.

### 4.2 · "Why the alternatives do not close the case"

None of them closes the case, and it is worth being precise about why.

A sensor answers *did something move*. An audit needs *was this booking used, and here is
the picture*. **None of them can tell a five-a-side match from a groundsman on a mower** —
which is exactly the difference between billing and not billing. Only the camera produces
evidence a customer can be shown. And the camera is already installed; every alternative is
new hardware on every pitch.

Note the fifth row, because it is the sharpest one: manual logging cannot be the answer,
since **it is one of the three records being audited**.

> **Concede this before anyone raises it:** a PIR sensor genuinely is more robust in fog, in
> darkness and against a dirty lens. The camera's advantage is the evidence trail and the
> ability to distinguish activities — not robustness.

### 4.3 · "Deep learning — what the field has established"

On the deep learning side, four things are established.

Scene classification from a fixed camera is **routine** — which is what licenses me to spend
this thesis on the evaluation rather than on the classifier. **Frozen backbones with linear
probes work** from very few labels. **Object detection finds people reliably**, the YOLO
family in particular, though small objects at distance remain hard. And **leakage and
shortcut learning are known failure modes**, with grouped splitting and trivial baselines as
established practice.

### 4.4 · "The gap this work sits in"

And here is the gap.

Almost all reported occupancy accuracy is scored on frames drawn from the **same scenes** as
training — so it measures memory of a place rather than recognition of an activity. Trivial
baselines are rarely reported, so a benchmark that a clock rule could win looks healthy
until somebody runs the clock rule. And the reconciliation problem — three records that
disagree — has the least prior art of all.

---

# Tab 5 — Chapter 2, Methods and our Model · 8:00

### 5.1 · "The families considered"

Four families were considered. A modern **convolutional network**, ConvNeXtV2. A **vision
transformer**, ViT-Base. A **self-supervised transformer**, DINOv2, trained with no labels
at all. And **vision-language models** — CLIP, OpenCLIP, SigLIP — used zero-shot with
written class descriptions.

### 5.2 · "Our model — almost nothing is trained"

And here is the model, which surprises people.

**Two hundred million frozen parameters, and about eleven thousand trained ones.** The three
backbones never saw this data during training; they are fixed feature extractors. The only
thing that learns is a classifier small enough to read on one screen.

### 5.3 · "Why the backbones are frozen"

That was a decision, not a shortcut, and this is the block I most want to land.

Fine-tuning two hundred million parameters on roughly **a hundred and fifty distinct scenes**
does not teach a network football. It teaches it *these pitches*, under *these floodlights*,
from *these camera angles*. Then either the cross-venue number collapses — or, far worse, it
does not collapse and I believe it.

And I could not simply collect my way out, because this is footage of **identifiable
people**: players, staff, sometimes children. Every additional venue is a consent and
data-protection conversation, not a download. The bottleneck is **permission**.

The evidence that the risk is real is on the block: 98.5% of frames have a near-duplicate,
and accuracy **falls** when labels rise from three hundred to six hundred and seventy-one.

So freezing is a **defence against overfitting first**, and an efficiency win second.

### 5.4 · "The two novel modules, reported as they came out"

The work proposed two novel modules, and I report both the way they came out.

The **gated fusion head** is a negative result. Routing between backbones is worth minus
0.024 cross-venue recall against the same head with the gate switched off. It puts about
seventy per cent of its weight on DINOv2 in every single fold — a learned constant in a
router's costume.

**STAN**, a small temporal model, scores a perfect 1.000 — and I am going to argue against
my own number. The composed labels are a deterministic function of five templates, so any
model that reads contiguity recovers the generating process. That is an exhausted test set,
not a win. **The real test set is two slots.**

### 5.5 · "The architecture"

*(Pause — let them read the diagram.)*

The thing to notice is that **every stage can refuse rather than guess**. That is what makes
REVIEW a real outcome instead of a low-confidence "used".

### 5.6 · "What the system stores"

One row per camera per sampled minute is the only thing actually observed; everything above
it is derived and can be recomputed.

Two decisions worth naming. A verdict and its evidence frames are written in **one
transaction**, so no verdict can exist without its evidence. And a dropped minute is stored
**as a gap** rather than filled — because no footage is not evidence that a pitch was
unused.

### 5.7 · "How the model is evaluated"

Splits grouped by venue, leave-one-venue-out over seven folds, four trivial baselines in
every protocol, and hypotheses pre-registered before the runs.

⟨cut if short⟩ And I report the **resolution** of every test: over seven folds the smallest
p-value the design can produce is 0.0156, so "not significant" and "the design could not
have produced significance" are different statements, and I say which one applies.

### 5.8 · "And the evaluation is where the result was"

Which brings me to the central finding of the thesis. **Slow down here.**

In this data, **98% of daylight frames are an empty pitch and 99% of night frames are a
match**. So "night means play" is correct on 99.1% of my corpus without looking at a single
pixel. A constant predictor that always answers "playing" scores a perfect macro-F1 across
venues. And on the 243 held-out empty frames, DINOv2 is correct **zero times** — it answers
"playing" 75 times and "maintenance" 168 times.

That last one is the result I am proudest of finding, and it came from adding one column to
check whether a number meant what it said. A low false-alarm rate had been quoted throughout
as evidence that a model recognises an empty pitch. It is not. The published numbers stand;
the inference drawn from them does not.

This is a statement about the **dataset**, not about the models. Identifying it, rather than
reporting around it, is the contribution.

---

# Tab 6 — Chapter 3, YOLOv8 · 12:30

### 6.1 · "Why a detector at all"

Chapter 3 is the detector, and the first question is why have one at all when there is
already a classifier.

Four reasons. **A classifier cannot be argued with** — "this frame looks like play" is not a
reason, but "six people and a ball inside the boundary" is. **The facility's rule is a
count** — five people make a game, four or fewer do not, and that is a number a manager can
check. **It gates the model** — the boundary, motion, people and ball checks sit in front of
the classifier and can veto it. And **it is explainable**: every verdict carries the boxes it
was made from, drawn on the evidence frame.

### 6.2 · "Why YOLOv8"

And why YOLOv8 specifically.

It is **single-stage and real-time** — one forward pass per frame, which is what a
sixty-second cycle over twenty to thirty cameras can afford on a CPU. It is **pretrained on
COCO**, which already contains *person* and *sports ball*, so the two classes this system
needs come for free, with no training of my own. **Segmentation variants exist**, so masks
can be drawn for the reader without changing how anything is counted. And in this
implementation it is a **registry, not a hard-coded string** — seven candidates are named in
code, so changing detector is a setting rather than an edit.

### 6.3 · "The bake-off — seven candidates, one chosen"

Which is what made this comparison possible. Seven detectors, scored against a hundred
hand-counted frames.

**YOLOv8n was chosen**: the lowest count error at 1.23 people, the best within-one rate at
0.70, and the fastest that fits the cycle at 118 milliseconds a frame.

Two things worth pointing at. The **segmentation variants count no better** — yolov8n-seg
has a marginally lower error but a worse within-one rate, and costs 40% more time for masks
that only the reader sees. And **the larger model does not win**: yolov8s is more than twice
as slow for a slightly worse count.

### 6.4 · "Tiling — measured, and rejected"

Then a negative result I want to report properly.

A far-side player is only a few pixels tall at the detector's input size, so the obvious
idea is **tiling** — run the frame as four overlapping quarters plus the whole, and merge.
It should find the small ones.

And it does: tiling improves ball recall substantially, from 0.43 to 0.59, and the ball is
exactly the small object it should help with. But it costs roughly **five forward passes**.
Every tiled configuration **fails the sixty-second budget** — and the count error gets
*worse*, not better, roughly doubling.

So tiling stays in the registry, switched off, with the measurement beside it. It is a
candidate, not a default.

### 6.5 · "From boxes to a verdict — the rule table"

The boxes then become a verdict through a table that a facility manager can read and argue
with. First matching row wins.

No detector or no boundary gives **uncertain** — a missing detector is not an empty pitch.
Nobody and nothing moving is **empty**. One to four people is **not a game**. More than four
people **with a ball and motion** is active play, and that is the only way in. More than four
people otherwise is a crowd standing on a pitch, which is not a booking being used.

Two details matter. A person is placed by the **foot of their box**, so detection and
segmentation models count identically. And five and four are the **facility's** numbers, not
ones I fitted.

### 6.6 · "What the detector cannot do"

And the honest limitation.

**It often cannot see the ball.** Cross-venue ball recall is 0.40, and ranges from 0.06 to
0.89 depending on the venue. Which makes the ball requirement expensive: turning it on drops
play recall from 0.996 to **0.308** at the worst venues. That is the facility's rule, and I
measured the cost rather than arguing about it.

⟨cut if short⟩ One design note underneath: a failed detection returns `None`, never an empty
list. "Not checked" and "nothing found" are different, and collapsing them is how a broken
install comes to report every pitch empty.

---

# Tab 7 — Chapter 4, Applications and Augmentation · 16:55

### 7.1 · "The application — a booked hour, end to end"

Chapter 4 is what it does in practice.

The booking says pitch three was sold for an hour and staff marked it used. The cameras
sampled sixteen frames, and thirteen show an empty pitch under floodlights. The system
raises **one advisory with three photographs** — never "do not bill", only "look at this".

A manager opens the slot, sees the evidence frames and the per-minute table, and confirms or
overrides. The model's original verdict is **retained, never replaced**.

### 7.2 · "Preprocessing — before the model sees a frame"

Before the model sees anything, a frame goes through preprocessing: letterbox, undistort,
crop, standardise, CLAHE, gamma and saturation, denoise, sharpen — and blur.

Two to point at. **CLAHE** is the switch the proposal predicted would fix floodlight glare;
measured honestly, it **costs** 0.27 macro-F1. And **blur is a control, not a candidate** —
at high strength it removes the people, and seeing what that does to the score tells you
what the score is actually reading.

The experiments and the live system share **one** preprocessing module. The pilot had two,
they drifted, and the numbers stopped meaning the same thing.

### 7.3 · "Data augmentation — the presets"

Four augmentation presets: colour, light, weather, and everything at once.

And two geometric decisions worth defending. **No rotations, warps or perspective changes** —
the cameras are bolted to a post and see one view forever, so a rotated pitch is not a harder
example, it is an impossible one. **Horizontal flip is the one exception**, deliberately,
because a mirror cannot change whether people are playing but does break memorisation of
*this* pitch's layout.

### 7.4 · "The augmentation experiment — and its retraction"

Now the experiment I most want to tell you about.

A model trained on camera A scores 0.441 on camera B. Can augmentation close that gap for
free, with no labels from B? The first answer was **yes** — one draw reached 0.855, with
empty-pitch recall going from zero to 0.687. It went into the write-up as a finding, with
one note attached: *only one draw has been taken.*

*(Pause.)*

That note turned out to be the finding. I ran it four more times — same frames, same preset,
same probe seed, **only the random draw changed**. 0.414, 0.351, 0.350, 0.348.

The published number was the **maximum of five**. The standard deviation is 0.221 on a
metric bounded between zero and one. And four of the five draws are **worse than using no
augmentation at all**.

The transferable lesson is this: the row in the results file is unchanged and still
reproduces exactly. A test pinning the published value would have passed forever.
**Reproducibility checked the wrong thing** — only re-drawing the randomness caught it.
Every headline now runs five draws, and I ran it on my own strongest result before anybody
asked.

### 7.5 · "Generating data with AI, from a real starting point"

The dataset is missing one case entirely — an empty pitch at night — so I generated some,
and I want to be careful about how I describe that.

I did **not** generate a dataset from text prompts. **Every generated item starts from a real
anchor**, an actual frame of an actual pitch, and the generation changes one thing about it.
For images I used **Gemini** and **ChatGPT**. For video I took still frames and **animated
them**, adding motion so that there is movement for the motion check to read, using two AI
video tools: **[VIDEO TOOL 1]** and **[VIDEO TOOL 2]**.

It worked: thirty-one generated empty frames took cross-venue false alarms from **0.768 to
0.024**, and play-recall went *up*, which is the sign it added signal rather than noise.

And the guards are on the block. They are **excluded from every corpus count**, because
counting them would make a gap look filled that is not. ⟨cut if short⟩ One line I did not
cross: I did not generate rain. There is no wet footage here, so synthetic rain could only
be validated against synthetic rain — which tests the generator, not the weather.

### 7.6 · "Two configuration searches"

Two configuration searches, with opposite costs. Prompts are cheap, so that one is
exhaustive — 375 sets. Preprocessing re-embeds the whole dataset per candidate, so that one
is greedy — 88 runs.

The prompt search gave me a number I find genuinely uncomfortable. Worst wording to best
spans **0.726** macro-F1. Choosing between the three backbones — the decision the whole of
Chapter 2 is about — moves it by **0.082**. Wording matters about **nine times more** than
architecture.

And underneath, the reason both searches are weaker than they look: **one frame** moving in
the smallest venue fold shifts the headline by 0.0119, and the confidence band is 0.092
wide. Almost every gap the search ranked on is smaller than its own error bar.

### 7.7 · "What the applications achieved"

So what actually worked. Rules in front of the model take false alarms from 0.617 to 0.012.
Generated empty frames take them from 0.768 to 0.024. And one labelled frame of a new camera
takes it from 0.441 to 0.990.

**Not one of those is a bigger model.** And the control that reframes the last one: from five
frames, training on those five *alone* matches training on those five plus 775 from the old
camera. What buys the accuracy is having **any** labels from the new camera.

---

# Tab 8 — Conclusion · 22:35

### 8.1 · "What was achieved"

In conclusion. A complete system — sampling, classification, detection, aggregation,
reconciliation, an API and a dashboard, all on a CPU. A reproducible programme, where every
result regenerates through one pipeline and every claim is re-derived from its artefact. A
deployment recipe of five labelled frames per camera. And an ethical boundary that is
enforced rather than promised.

### 8.2 · "What was found"

The findings indicate that the system works end to end, and that **the evaluation is the
real result**: lighting and occupancy are the same variable in this data, one short clip
reversed a benchmark of sixteen hundred frames, a low false-alarm rate is not accuracy, and
a reproducible number is not a reproducible result.

### 8.3 · "What this study cannot claim"

Four limits, named first because you will find them anyway.

Empty pitches exist at **one venue only**. The 243 held-out empty frames are **three distinct
scenes**, and six comparisons stopped being significant when I recounted them that way.
**Nothing has run on the target hardware**. And there is **no human ceiling** — no
inter-annotator figure exists, which is a real gap and I would rather say so than have it
found.

*(Point at the diagram.)* One absent cell propagates into four separate research questions.
They are blocked on data, not on work.

### 8.4 · "Perspectives"

Which is why the perspectives are mostly a recording trip rather than an engineering plan.

Twenty to thirty minutes of **empty-pitch footage at a second venue** resolves the confound,
the external-validity limit and the scope reduction at once. **Thirty real labelled slots**
lifts the temporal model out of preliminary. An **inter-annotator study** would give the
human ceiling this work lacks. **Deployment on the target mini-PC** would make the production
claim testable. And **a ball detector that can actually see the ball** would remove the most
expensive rule in the system.

### 8.5 · The closing line

*(Pause.)*

**No protocol compensates for a case the data never contains.**

Thank you for your time and attention. I welcome any questions, comments, suggestions or
feedback you may have.

---

## Questions to expect

> **Would a colour histogram have done this?** 0.9616 on the leaky split — indistinguishable
> from ConvNeXtV2 once you count 62 scenes rather than 394 frames.
>
> **Why not fine-tune?** Block 5.3 is the answer: 150 distinct scenes, sensitive data that
> cannot be grown, and accuracy that already falls as labels rise.
>
> **Why YOLOv8 and not YOLO11?** Block 6.3 — YOLO11n was tested and lost on both count error
> and latency. The registry makes switching a setting, so the choice is revisitable.
>
> **How many comparisons before p < 0.05?** Holm within declared families; the undeclared
> search families are disclosed as numbered amendments.
>
> **Is auditing staff by camera ethical?** Per-field reporting only, never per person, and
> the decision layer can structurally only advise.
>
> **What is the human ceiling?** No inter-annotator figure exists. Say so plainly — it is a
> real gap.
>
> **Where are your references?** Block 3.2 — say the honest version: none are written down
> until the paper has been read, and completing them is the next step.

## Delivery notes

- **Practise saying "we do not know" three times**: the human ceiling, the target hardware,
  and whether any of this transfers to a second venue.
- **Time Tabs 5 and 7 with a clock.** Together they are two fifths of the talk.
- **Slow down on 5.8, 7.4 and 8.5.** Everything else can move.
- **Running over?** Drop every **⟨cut if short⟩** passage first — about 90 seconds — then
  5.5, 5.6 and 3.5, which are diagrams and a recap that work without commentary.
- **Your strongest sentence** is a version of *"we measured that, and here is how large it
  is."* Where that is not available — the four limits — the second strongest is *"that is a
  real limitation, and here is exactly what it costs."*
- Practise with a strong, clear voice and a confident tone. Use simple language for the
  complex ideas, and end on the statistic rather than an apology.

## Before the day

- [ ] Fill in **[PROGRAMME]**, **[INSTITUTION]** and **[NAME]** (block 1.0).
- [ ] Fill in **[VIDEO TOOL 1]** and **[VIDEO TOOL 2]** (block 7.5, and the same placeholder
      on the site).
- [ ] **Fill the nine `[CITE]` strands** in block 3.2 with papers you have read. This is the
      largest remaining gap, and the one an examiner is most likely to open with.
- [ ] Re-read [`thesis/claims.md`](../claims.md) — every number above comes from it, and it
      is regenerated from the result artefacts. If a result moved, this script is stale.
- [ ] Rewrite at least Tabs 1 and 8 in your own words.
- [ ] Run `uv run pitch serve` on the machine you will present from, and check Chapters 2, 3
      and 4 render their diagrams, tables and figures.
