# Speaker script — 25 minutes

**This script follows the site**, not the PowerPoint. Start the server, open
<http://127.0.0.1:8000>, and page with `→`. Slide numbers are cumulative across all eleven
tabs, and each heading repeats the slide's own title so you always know where you are.

```bash
uv run pitch serve
```

Press **`N`** to show the notes on screen if you lose your place.

**Read only the plain paragraphs aloud.** Blockquoted boxes are prepared answers for
pushback, not part of the talk. Passages marked **⟨cut if short⟩** can go if you are running
over — they are depth, not structure.

Read it aloud three times before the day, and rewrite anything that does not sound like you.
Phrasing you did not choose collapses on the first follow-up question.

Things still yours to fill in are marked **[LIKE THIS]**.

| Tab | Slides | Starts | Runs |
|---|---|---|---|
| Start | 1–3 | 0:00 | 1:05 |
| Purpose | 4–6 | 1:05 | 1:50 |
| How it works | 7–12 | 2:55 | 2:55 |
| The problem | 13–21 | 5:50 | 4:20 |
| Data | 22–28 | 10:10 | 3:45 |
| Models | 29–31 | 13:55 | 2:00 |
| Rules | 32–34 | 15:55 | 2:20 |
| Searches | 35–38 | 18:15 | 1:35 |
| Augmentation | 39–43 | 19:50 | 2:00 |
| Solution | 44–47 | 21:50 | 2:20 |
| Summary | 48–51 | 24:10 | 2:00 |

**Measured, not guessed.** Read straight through at 140 words a minute with the
**⟨cut if short⟩** passages dropped, this is **26 minutes**; at 150 it is **24½**. Reading
every word including the optional passages adds 90 seconds. Time yourself once and adjust —
if you run slower than 140, drop slides 17, 26 and 43 as well: they are pictures that work
without commentary.

---

# Tab 1 — Start · 0:00

### Slide 1 · "Who actually used the pitch?"

Good morning, distinguished examiners, professors and colleagues.

My name is Mariam Srour, and I am a Master's student in **[PROGRAMME]** at
**[INSTITUTION]**.

The title of my research is *Multi-Pitch Occupancy and Booking Verification from Existing
Cameras*. I will use the shorter version on the screen: **who actually used the pitch?**

### Slide 2 · "Where we are going"

I will take you through the purpose of the research; how it works, with a short scenario;
the problem it tackles; the data and the models; the experiments; and a summary.

Two things I will spend real time on: the configuration searches, and the augmentation
experiment.

### Slide 3 · "A system that checks every booked minute, and never gets bored"

The scale, once. Sixteen hundred labelled frames, three classes, **one frame per camera per
minute**, and no GPU anywhere in the design.

That sampling rate is what everything rests on: roughly a **99% cut in network traffic**
against decoding twenty or thirty live streams, which is what lets the whole facility run on
one small computer.

---

# Tab 2 — Purpose · 1:05

### Slide 4 · "Find out which booked hours were really used"

The purpose of this study is to find out **which booked pitch hours were actually used**,
using the cameras a facility already owns.

Three constraints shape everything: no new hardware, no GPU, and a person decides every
case.

### Slide 5 · "What the owner gets"

Let me say plainly why anyone would want this.

A facility rents twenty or thirty pitches by the hour. Staff write down which slots were
used, nobody verifies it — and **nobody can**. One manager cannot watch twenty pitches at
once. So they spot-check, and they trust the sheet.

This gives the owner three things.

**Time.** It checks every booked minute on every pitch, and shows the manager only the
bookings where the record and the camera disagree.

**Consistency.** It applies the same rule at eight in the morning and eleven at night. It
does not get tired, and it does not skip the far pitch because it is raining. A person
samples; this does not.

**Evidence.** Every flag arrives with three photographs. The conversation with a customer is
about a picture on a screen, not somebody's memory of Tuesday.

That is why I call it **more trustworthy than a manual check** — not cleverer than a person,
but consistent, complete, and able to show you what it decided on.

> **If pushed on "more trustworthy than a person" — do not defend the strong version:**
>
> "To be precise: it is more **consistent** and more **auditable** than a manual check, and
> it covers every minute rather than a sample. I am not claiming it is more accurate than a
> human looking at the same photograph. I have no inter-annotator figure, and that is a real
> gap in this work. Accuracy stays a person's job — which is exactly why the system only ever
> advises."

### Slide 6 · "One rule carried everywhere"

One design rule ran through all of it: **the vision path must never see the booking record as
an input.** A model that has read the booking flag cannot give evidence independent of the
record it audits. That is the difference between an audit and a rubber stamp.

---

# Tab 3 — How it works · 2:55

### Slide 7 · "Five stages, one frame a minute"

Five stages.

**Sample** — one frame per camera per minute. A still image, not a stream.

**Classify** — empty, active play, or maintenance. Maintenance exists so a mower is not
billed as a match.

**Check** — cheap rules on the same frame: is anyone inside the pitch boundary, did anything
move, are there enough people, is there a ball. These rules can **overrule** the neural
network. Remember that — it matters more than the choice of network.

**Aggregate** — about sixty predictions become one verdict for the hour: used, not used, or
review.

**Compare** — that verdict is checked against the booking, and disagreements are flagged
with their evidence.

### Slide 8 · "Tuesday, 20:00 — Pitch 3"

Concretely. The booking says pitch three was sold for an hour and staff marked it used. The
cameras sampled sixteen frames, and **thirteen show an empty pitch under floodlights**.

The system does not cancel the charge. It raises one advisory with three photographs, and a
manager decides.

In the code that is structural, not a promise: the only thing the decision layer can return
is an *Advisory*, and there is no code path that acts.

### Slide 9 · "Where a verdict can be refused"

*(Pause — let them read the diagram.)*

Every stage can **refuse** rather than guess. That is what makes "review" a real outcome
instead of a low-confidence "used", and why a camera that drops out cannot silently become an
empty pitch.

### Slide 10 · "What gets stored"

One row per camera per minute is the only thing observed; everything above it is derived.

A verdict and its evidence are written in **one transaction**, so no verdict can exist
without its evidence. And a dropped minute is stored **as a gap** rather than filled —
because no footage is not evidence that a pitch was unused.

### Slide 11 · "How the data were collected"

Fixed CCTV at nine venues, sampled every fifteen seconds, hand-labelled against a protocol
written before the runs.

The line at the bottom is what matters: because the cameras are fixed and the interval is
short, **98.5% of frames have a near-duplicate**. That single property invalidated my own
pilot.

### Slide 12 · "How the data were analysed"

Frozen backbones with a small trained head, under splits grouped by venue.

Two choices here are the method contribution. **Four trivial baselines run in every
protocol** — a clock rule, a colour histogram, a constant predictor, a random one. If a
trivial baseline wins, that is information about the protocol, not a curiosity to leave out.

And I report the **resolution** of every test. Over seven folds the smallest possible p-value
is 0.0156 — so "not significant" and "the design could not have produced significance" are
different statements, and I say which applies.

---

# Tab 4 — The problem · 5:50

### Slide 13 · "A facility sells hours. Nobody checks which ones were used."

The problem is that a facility sells hours with no verified record of which were used. Three
errors follow: no-shows, unbooked usage, and data-entry mistakes.

Let me concede the obvious objection first. A twenty-euro motion sensor is more robust in
fog, darkness and against a dirty lens. But a sensor answers *did something move*. An audit
needs *was this booking used, and here is the picture* — and it has to tell a five-a-side
match from a groundsman on a mower.

### Slide 14 · "What is known — and what is missing"

Much is known about frozen backbones and about occupancy from CCTV. The gap is that almost
all of it is evaluated on frames from the **same scenes** as training — so the score measures
memory of a place, not recognition of an activity.

That matters because the money rests on one class. Getting "playing" right is easy and worth
nothing. If the system cannot recognise an **empty** pitch, it cannot flag a single unused
booking.

### Slide 15 · "Honest splitting halves the score"

Under a random split, **37% of near-duplicate pairs** land on opposite sides of the
train/test line. Grouped by venue and slot, **below 1%**.

The price of that honesty is severe — ConvNeXtV2's score roughly halves.

⟨cut if short⟩ But only about **two thirds** of that fall is leakage. A model that never
trains cannot leak, so scoring a zero-shot model on the same test sets isolates the rest:
0.183 is simply the test set getting harder. Reporting the bigger number would have made my
story stronger, and it would have been wrong.

### Slide 16 · "In this data, the time of day is the answer"

**Slow down. This is the most important slide in the talk.**

Across my data, **98% of daylight frames are an empty pitch, and 99% of night frames are a
match**. So "night means play, day means not-play" is correct on **99.1%** of my corpus —
without looking at a single pixel.

Which means nothing measured on this data can separate a model that recognises an empty pitch
from one that recognises the time of day.

The two cells that would break the tie hold **nine frames** and **six frames**.

That is a statement about the **dataset**, not the models. And identifying it, rather than
reporting around it, is the contribution of this thesis.

### Slide 17 · "The same model, judged two ways"

Identical model, identical data. Two ways of drawing the train/test line — and the answer
changes.

### Slide 18 · "A rule that never looks at the image wins"

Here is what that does to a benchmark. A rule reading only the clock **beats all three
backbones** across unseen venues — on finding the match and on false alarms, at the same
time.

Expect: *so are your models worthless?* No. They are indistinguishable from a light meter
**on this dataset**.

⟨cut if short⟩ And I should volunteer one thing: an earlier version of this slide said the
opposite. That number came from a label error — a brightness threshold had filed 216 frames
of floodlit night football as daylight. Corrected, the rule wins.

### Slide 19 · "Then one clip reversed the ranking"

So I went and got the missing case: four minutes of a floodlit pitch at night with nobody on
it, scored minute by minute.

The clock rule calls **all sixteen minutes** active play — wrong on every one. The model
alone sits at 0.38. The full deployed system is at **zero**, and gets all thirteen empty
minutes right.

The benchmark had sixteen hundred frames; the clip had a few hundred. **The clip won, because
it contained the case the benchmark was missing.** More data of the same kind would not have
found this.

### Slide 20 · "What a low false-play rate is actually measuring"

This is the strongest result in the work, and it came from adding one column.

The false-alarm rate had been quoted as how well a model recognises an empty pitch. So I
asked what the models answer **instead**. On 243 held-out empty frames, DINOv2 answers
"playing" 75 times and "maintenance" 168 times. It is correct **zero** times.

The protocol trains on camera A and tests on camera B — a **camera-transfer test**, not a
specificity test. The published numbers stand; the inference drawn from them does not.

### Slide 21 · "Two more things the protocol was hiding"

One more, because the pattern is the point. A **constant predictor** that always answers
"playing" scores a perfect macro-F1 cross-venue, ahead of every backbone, because every
held-out venue is 100% active play. There is no ranking to change.

Both were found the same way: by checking whether a number meant what it said.

---

# Tab 5 — Data · 10:10

### Slide 22 · "What we have"

Sixteen hundred and ninety-two labelled frames. Three classes. Every empty pitch in the
corpus comes from **one venue**, and maintenance has too few frames to evaluate at all.

### Slide 23 · "Why this data is hard to get"

The obvious question: why not collect more?

Because this is **footage of identifiable people** — players, staff, sometimes children. That
is sensitive personal data. Every additional venue is a consent and data-protection
conversation with a facility, not a dataset download. The bottleneck is **permission**, not
storage and not labelling effort.

And the task needs **variety** — many venues, lightings, angles, weathers — not volume. Ten
thousand more frames of the same pitch on the same evening add nothing, because they are
near-duplicates of what I already have.

### Slide 24 · "So we made some, starting from something real"

So I generated some data, and I want to be careful how I describe it.

I did **not** generate a dataset from text prompts. **Every generated item starts from a real
anchor** — a real frame of a real pitch — and the generation changes one thing about it.

For images I used **Gemini** and **ChatGPT**, to produce the scene the corpus does not
contain: an empty pitch under floodlights at night.

For video I took still frames and **animated them** — adding motion to a still image, so
there is movement for the motion check to read. That used two AI video tools:
**[VIDEO TOOL 1]** and **[VIDEO TOOL 2]**.

The result: **31 generated empty frames** took cross-venue false alarms from **0.768 to
0.024** — and play-recall went *up*, which is the sign it added signal rather than noise.

### Slide 25 · "And we are honest about what generated data is"

Generated data is a legitimate tool and an easy way to fool yourself, so the guards are on
the slide.

They are **excluded from every corpus count**; counting them would make a gap look filled
that is not. They are a training aid, not evidence the system works on real footage of that
case. And I checked the effect against 26 real frames, so it is not an artefact of the
generator.

One line I did not cross: **I did not generate rain.** There is no wet footage in this
dataset, so synthetic rain could only be validated against synthetic rain — and that tests
the generator, not the weather.

### Slide 26 · "The confound, as a picture"

Read straight off the manifest, so it cannot drift from the corpus. The two nearly-empty
cells are the whole problem.

### Slide 27 · "One missing cell, four unanswerable questions"

And this is why that gap is not a footnote in a limitations chapter. One absent cell
propagates into **four separate research questions**. They are blocked on data, not on work.

### Slide 28 · "Small data has a specific danger"

The risk with a dataset this size is **not** that the numbers come out low. It is that they
come out **high for the wrong reason** — the model memorises these pitches, and that looks
like success. That is overfitting, and the opposite of what we need, which is
**generalisation** to a venue it has never seen.

The evidence this is real and not theoretical: label efficiency here is **not monotone**.
Every backbone peaks at one to three hundred labels and gets **worse** at 671, because the
extra labels are near-duplicates that add redundancy rather than information.

That is the whole reason the evaluation is built the way it is.

---

# Tab 6 — Models · 13:55

### Slide 29 · "We did not train the models"

Which brings me to a question I expect: why did you not fine-tune anything?

**Two hundred million frozen parameters, and about eleven thousand trained ones.** The three
backbones never saw this data during training. The only thing that learns is a classifier
small enough to read on one screen.

### Slide 30 · "Why not — and it is the data, not the compute"

That was a decision, not a shortcut.

Fine-tuning two hundred million parameters on roughly **150 distinct scenes** does not teach
it football. It teaches it *these pitches*, under *these floodlights*, from *these angles*.
Then either the cross-venue number collapses — or, far worse, it does not collapse and I
believe it.

And I could not collect my way out, for the reason I gave: the data is sensitive and the
bottleneck is permission.

So **freezing is a defence against overfitting first**, and an efficiency win second. Embed
each frame once and reuse it forever — which is how seventy-five experiments fit into one
thesis — but that is the second reason, not the first.

### Slide 31 · "Two novel modules, reported as they came out"

The thesis proposed two novel modules, and I report both as they came out.

The **gated fusion head** is a negative result: routing is worth minus 0.024 cross-venue
recall against the gate switched off. It puts about 70% of its weight on DINOv2 in every
fold — a learned constant in a router's costume. It *can* route, on a fixture where the
useful backbone flips. So the null is about the data, not the code.

**STAN** scores a perfect 1.000 — and I will argue against my own number. The composed labels
are a deterministic function of five templates, so any model reading contiguity recovers the
generating process. That is an exhausted test set, not a win. The real test set is **two
slots**.

⟨cut if short⟩ One more thing: the proposal named ConvNeXtV2. Measured without leakage,
DINOv2 leads. The decision was made on a measurement taken after the proposal — that is the
process working.

---

# Tab 7 — Rules · 15:55

### Slide 32 · "A table, not a black box"

The deployed decision layer is deliberately a table a facility manager can read and argue
with. First matching row wins.

No detector or no boundary gives **uncertain** — a missing detector is not an empty pitch.
Nobody and nothing moving is **empty**. One to four people is **not a game**. More than four
people **with a ball and motion** is active play — the only way in. More than four people
otherwise is a crowd standing on a pitch, which is not a booking being used.

Two of those numbers are the **facility's rule**, not something I fitted: five people make a
game, four or fewer do not.

The last two rows changed late, and they invert what I pre-registered. Play now has to be
**shown**, not assumed.

### Slide 33 · "Every switch, and what it costs"

Each switch exists because its cost is a decision somebody should take deliberately.

The **ball requirement** is the clearest case. Cross-venue ball recall is 0.40, and between
0.06 and 0.89 by venue. Turning it on drops play recall from 0.996 to 0.308 at the worst
venues. That is expensive, it is the facility's rule, and whoever switches it off should see
exactly what they are buying.

And a rule about rules: **a cue that cannot be measured is reported, never assumed.** A
requirement that silently never fires is the shape of every safeguard this project found not
working.

### Slide 34 · "What happens to a frame before the model sees it"

Before any of that: letterbox, undistort, crop, standardise, CLAHE, gamma and saturation,
denoise and sharpen, and blur.

Two to point at. **CLAHE** is the switch the proposal predicted would fix floodlight glare;
measured honestly, it **costs** 0.27 macro-F1. And **blur** is a *control*, not a candidate —
at high strength it removes the people, and seeing what that does to the score tells you what
the score is reading.

The experiments and the live system share **one** preprocessing module. The pilot had two,
they drifted, and the numbers stopped meaning the same thing.

---

# Tab 8 — Searches · 18:15

### Slide 35 · "We tuned hard. Twice."

Two searches with opposite costs. Prompts are cheap, so that one is **exhaustive** — 375
sets. Preprocessing re-embeds the whole dataset per candidate, so that one is **greedy** — 88
runs.

Both guarded the same way: every cross-venue fold is 100% active play, so recall alone can be
bought by saying "playing" more often. Everything is ranked on recall minus false alarms.

### Slide 36 · "The winner sits inside the noise"

The preprocessing search found a clear winner with a perfect score. Then I measured the
resolution of the search itself.

### Slide 37 · "The search is finer than the data"

**One frame** moving in the smallest venue fold shifts the headline by 0.0119. The confidence
band at the median setting is 0.092 wide.

Almost every difference the search ranked on is **smaller than its own error bar**. The
ranking is real arithmetic on unreal precision.

### Slide 38 · "Wording moved the score nine times more than the model did"

The prompt search gave me the number I find genuinely uncomfortable.

Worst wording to best is **0.726** macro-F1. Choosing between the three backbones — the
decision the entire model chapter is about — moves it by **0.082**. Wording matters about
**nine times more** than architecture.

Two guards, and I state both first. Only 22.9% of prompt sets beat the trained probe and the
median loses — so "zero-shot works" is only true if you already know the wording, which needs
labels. And the winner's lead of 0.447 is measured on the folds it was selected from.

---

# Tab 9 — Augmentation · 19:50

### Slide 39 · "A good question, cheaply asked"

Now augmentation — the experiment I most want to tell you about.

A model trained on camera A scores 0.441 on camera B, watching the same pitch from a
different angle. One labelled frame of B takes it to 0.99. **Can augmentation close that gap
for free**, with no labels from B at all?

### Slide 40 · "It worked. We wrote it up."

The first answer was yes. One draw of the brightness-and-gamma preset took empty-pitch recall
on the unseen camera from zero to **0.687**. Macro-F1 0.855.

It went into the write-up as a finding, with one note attached: *only one draw has been
taken.*

*(Pause.)*

That note turned out to be the finding.

### Slide 41 · "Then we ran it four more times"

Same frames, same preset, same probe seed, same test set. **Only the random draw changed.**

0.414. 0.351. 0.350. 0.348.

The published 0.855 was the **maximum of five**. The standard deviation is 0.221 on a metric
bounded between zero and one. And four of the five draws are **worse than using no
augmentation at all**.

⟨cut if short⟩ One detail explains the shape: three draws score exactly 0.3479, which is the
score of answering "active play" to every test frame. It does not degrade gracefully — it is
either a working classifier or the trivial one.

### Slide 42 · "The number reproduced perfectly. The result did not."

So the headline was retracted, and I want to be precise about what was retracted.

The row in the results file is **unchanged and still reproduces exactly**. What was withdrawn
is the claim about the *method*.

The transferable lesson: a unit test pinning the published value would have passed forever,
because the artefact was reproducible. **Reproducibility checked the wrong thing.** Only
re-drawing the randomness caught it.

That is now standing practice — and I ran it on my own strongest result before anybody asked.

### Slide 43 · "Why the pictures are the argument"

Augmentation code **fails silently**. A preset that does nothing, a crop that removes the
goalmouth — all of them pass a shape and type check. The only reliable check is a person
looking.

⟨cut if short⟩ On the first run these sheets caught rain streaks written in absolute pixels:
at low resolution they were white poles a tenth of the frame wide. No assertion could have
found that.

---

# Tab 10 — Solution · 21:50

### Slide 44 · "The fix was never a bigger model."

So what did I do about all this? Three things — and **not one of them is a bigger neural
network.**

### Slide 45 · "Three fixes, all cheap"

**Cheap rules in front of the model** — false alarms on empty pitches fall from 0.617 to
0.012.

**Generate the missing case** — thirty-one synthetic empty frames take cross-venue false
alarms from 0.768 to 0.024, while play-recall goes up.

**Label a handful of frames per camera** — one labelled frame takes every backbone to about
0.99. And the control that reframes it: from five frames, training on those five **alone**
matches training on those five plus 775 from the old camera.

So I state the weaker claim, because it is the true one: what buys the accuracy is **any**
labels from the new camera, not a large corpus from an old one. That is the onboarding recipe
I would hand to the facility.

### Slide 46 · "Every guard is verified by breaking it"

One slide on method, because this is the part that transfers beyond football pitches.

**Six safeguards in this project were doing nothing at all** — a lock resolved against the
wrong directory, a threshold set to zero that can never fire, a criterion that searched for a
word instead of running the verifier.

Every one passed whatever test existed. So each new safeguard is now verified by
**deliberately breaking the thing it guards**. That practice found half the results in this
talk.

The root of it: **a wrong result that looks plausible is invisible.**

### Slide 47 · "What it will not do"

And one slide on what the system deliberately will not do, because auditing how pitches are
used is close to auditing the people who work there.

It **never bills and never acts** — structurally, not as a policy sentence. Anomalies are
reported per **field**, never per person. Faces are redacted.

And **review never becomes an anomaly**. If the model is unsure, that uncertainty belongs to
the model. Turning it into a flag against a member of staff would launder the system's own
weakness into someone else's record.

---

# Tab 11 — Summary · 24:10

### Slide 48 · "What this study cannot claim"

Three limits, named first because you will find them anyway.

Empty pitches exist at **one venue only**. The 243 held-out empty frames are **three distinct
scenes** — six comparisons stopped being significant when I recounted them that way. And
**nothing has run on the target hardware**.

There is also no human ceiling — no inter-annotator figure exists. That is a real gap, and I
would rather say so than have it found.

### Slide 49 · "What would actually change this"

What would fix them, ordered by value rather than effort.

Twenty to thirty minutes of **empty-pitch footage at a second venue** — that half-hour
resolves the confound, the external-validity limit and the scope reduction at once. About
**thirty real labelled slots**. And **five labelled frames per camera**, already measured.

The limitations of this thesis are mostly a **data-access problem with a known and
inexpensive solution** — not a methodological one.

### Slide 50 · "In summary"

This study **investigated** whether cameras a facility already owns can verify which booked
hours were actually used — on ordinary hardware, with a human deciding every case.

The findings **indicate** that the system works end to end, and that the evaluation is the
real result. Lighting and occupancy are confounded so tightly that a rule reading only the
clock outperforms three modern backbones — and one short clip reverses that ranking
completely.

These findings **contribute** a set of evaluation practices: trivial baselines in every
protocol, grouped splits, re-drawing randomness rather than re-running it, and every safeguard
verified by breaking what it guards. And for the facility, a concrete recipe for onboarding a
new camera.

*(Pause.)*

**No protocol compensates for a case the data never contains.**

### Slide 51 · "Questions, comments and suggestions welcome."

Thank you for your time and attention. I welcome any questions, comments, suggestions or
feedback you may have.

---

## Delivery notes

- **Practise saying "we do not know" three times**: the human ceiling, the target hardware,
  and whether any of this transfers to a second venue. An examiner trusts a candidate who has
  bounded their ignorance more than one who has not noticed it.
- **Time Tab 4 with a clock.** It is the section you will want to over-run and the one worth
  protecting.
- **Your strongest sentence** is a version of *"we measured that, and here is how large it
  is."* Where that is not available — the three limits — the second strongest is *"that is a
  real limitation, and here is exactly what it costs."*
- **Slow down on slides 16, 19 and 50.** Everything else can move.
- **Running over?** Drop every **⟨cut if short⟩** passage first — about 90 seconds — then
  slides 17, 26 and 43, which are pictures that work without commentary.
- The four questions on the closing slide carry their own short answers. The fifth — the
  human ceiling — has no number, and the right answer is to say so plainly.

## Before the day

- [ ] Fill in **[PROGRAMME]**, **[INSTITUTION]**, and the supervisor's name (slide 1).
- [ ] Fill in **[VIDEO TOOL 1]** and **[VIDEO TOOL 2]** (slide 24).
- [ ] Re-read [`thesis/claims.md`](../claims.md) — every number above comes from it, and it is
      regenerated from the result artefacts. If a result moved, this script is stale.
- [ ] Rewrite at least Tab 2 and Tab 11 in your own words.
- [ ] Run `uv run pitch serve` on the machine you will present from, and check the Data and
      Augmentation tabs render their figures.
