# Speaker script — 25 minutes

**This script follows the site.** Start the server, open <http://127.0.0.1:8000>, and move
through the nine tabs in order, scrolling each one.

```bash
uv run pitch serve
```

Headings below are `Tab.Block` and repeat the block's own heading, so you always know where
on the page you are. The site itself carries **no** delivery notes — this file is the only
place the talk is written down.

**Read only the plain paragraphs aloud.** Blockquoted boxes are prepared answers for
pushback. Passages marked **⟨cut if short⟩** can go if you are running over — they are
depth, not structure.

Things still yours to fill in are marked **[LIKE THIS]**.

| Tab | Blocks | Starts | Runs |
|---|---|---|---|
| 1. Introduce | 2 | 0:00 | 0:30 |
| 2. Topic | 3 | 0:30 | 1:15 |
| 3. Roadmap | 2 | 1:45 | 0:40 |
| 4. Purpose | 3 | 2:25 | 2:05 |
| 5. How it works | 11 | 4:30 | 7:00 |
| 6. The problem | 9 | 11:30 | 4:30 |
| 7. The solution | 6 | 16:00 | 4:50 |
| 8. Summary | 4 | 20:50 | 2:00 |
| 9. Thank you | 2 | 22:50 | 0:45 |

**Measured, not guessed.** Read straight through at 140 words a minute with the
**⟨cut if short⟩** passages dropped, the spoken text is about **23 minutes**;
every word including them is about **25**. The remaining two minutes are the pauses, the
scrolling and the breath between tabs &mdash; which is what makes it a 25-minute talk rather
than a 23-minute one. Time yourself once and adjust.

---

# Tab 1 — Introduce · 0:00

### 1.0 · The hero

Good morning, distinguished examiners, professors and colleagues.

My name is Mariam Srour, and I am a Master's student in **[PROGRAMME]** at
**[INSTITUTION]**.

### 1.1 · "Who and where"

My supervisor is **[NAME]**, and this work was carried out over **[PERIOD]**.

### 1.2 · "The work, in one sentence"

In one sentence: this is a system that checks, from cameras a facility already owns, which
booked pitch hours were actually used — on one small computer, with a person deciding every
case.

---

# Tab 2 — Topic · 0:30

### 2.0 · The hero

The title of my research is *Multi-Pitch Occupancy and Booking Verification from Existing
Cameras*.

For the rest of this talk I will use the shorter version: **who actually used the pitch?**

### 2.1 · "What that means"

Three things to fix in your mind before we go further.

**The question** is a business question: a facility sells pitch hours, and which of them
were actually played?

**The constraint** is that nothing new may be installed. Existing cameras, one small
computer, no GPU — and no footage leaves the site.

**The output** is not a classification. It is an advisory with three evidence photographs.

### 2.2 · "The scale of it"

The scale, once, so nobody has to guess it. Sixteen hundred and ninety-two labelled frames,
three classes, seven venue folds for cross-venue testing, and no GPU anywhere in the design.

### 2.3 · "One design rule, carried everywhere"

And one design rule that runs through all of it: **the vision path must never see the
booking record as an input.**

A model that has read the booking flag cannot give evidence independent of the record it is
auditing. That is the difference between an audit and a rubber stamp.

---

# Tab 3 — Roadmap · 1:45

### 3.1 · The five stops

In this presentation I will briefly take you through the **purpose** of the research; **how
it works**, including a short hypothetical scenario; **what problem** it tackles; **how it
provides a solution**; and I will conclude with a **summary**.

### 3.2 · "Two things I will spend real time on"

Two things I want to give proper time, because they are where the method contribution of
this thesis actually sits.

The **configuration searches** — both found a winner, and neither winner meant what it
looked like.

And the **augmentation experiment** — a result we published and then withdrew, and the
check that caught it.

---

# Tab 4 — Purpose · 2:25

### 4.0 · The hero

The purpose of this study is to find out **which booked hours were really used**, using the
cameras a facility already owns.

### 4.1 · "The study focuses on"

The study focuses on a network of synthetic five-a-side football pitches. The population is
every booked hour on every pitch, sampled once a minute. The context is existing CCTV, a
CPU-only mini-PC, and the facility's own booking records. And the variables are three scene
classes — empty, active play, maintenance — set against what the booking claims.

### 4.2 · "What the owner gets"

Let me say plainly why anyone would want this.

A facility rents twenty or thirty pitches by the hour. Staff write down which slots were
used, nobody verifies it — and **nobody can**. One manager cannot watch twenty pitches at
once. So they spot-check, and they trust the sheet.

This gives the owner three things. **Time** — it checks every booked minute and shows the
manager only where the record and the camera disagree. **Consistency** — the same rule at
eight in the morning and eleven at night, never tired, never skipping the far pitch because
it is raining. And **evidence** — every flag arrives with three photographs, so the
conversation with a customer is about a picture, not somebody's memory of Tuesday.

That is why I call it **more trustworthy than a manual check**: not cleverer than a person,
but consistent, complete, and able to show you what it decided on.

> **If pushed on "more trustworthy than a person" — do not defend the strong version:**
>
> "To be precise: it is more **consistent** and more **auditable** than a manual check, and
> it covers every minute rather than a sample. I am not claiming it is more accurate than a
> human looking at the same photograph. I have no inter-annotator figure, and that is a real
> gap in this work. Accuracy stays a person's job — which is exactly why the system only
> ever advises."

### 4.3 · "And three things it refuses to do"

And three things it deliberately will not do, because auditing how pitches are used is close
to auditing the people who work there.

It **never bills and never acts** — *Advisory* is the only output the decision layer can
produce. Anomalies are reported **per field, never per person**. And **review never becomes
an anomaly**: if the model is unsure, that uncertainty belongs to the model, not to a member
of staff.

---

# Tab 5 — How it works · 4:30

### 5.0 · The hero

To achieve the purpose of the study, I built a five-stage system and then evaluated it. It
samples one frame per camera per minute — a still image, not a video stream. That is what
makes the whole facility fit on one small computer.

### 5.1 · "The design"

**Sample** — one frame per camera per minute. About **99% less** network traffic than
decoding twenty or thirty live streams.

**Classify** — empty, active play, or maintenance. Maintenance exists so a mower is not
billed as a match.

**Check** — pitch boundary, motion, people, ball. These rules can **overrule** the neural
network. Remember that; it matters more than the choice of network.

**Aggregate** — about sixty predictions become one verdict for the booked hour.

**Compare** — that verdict against the booking record, with disagreements flagged.

### 5.2 · "A hypothetical scenario — Tuesday, 20:00, Pitch 3"

Concretely. The booking says pitch three was sold for an hour and staff marked it used. The
cameras sampled sixteen frames, and **thirteen show an empty pitch under floodlights**.

The system does not cancel the charge. It raises one advisory with three photographs, and a
manager decides.

### 5.3 · "Where a verdict can be refused"

*(Pause — let them read the diagram.)*

Every stage can **refuse** rather than guess. That is what makes "review" a real outcome
instead of a low-confidence "used", and why a camera that drops out cannot silently become
an empty pitch.

### 5.4 · "What gets stored"

One row per camera per sampled minute is the only thing observed; everything above it is
derived.

A verdict and its evidence are written in **one transaction**, so no verdict can exist
without its evidence. And a dropped minute is stored **as a gap** rather than filled —
because no footage is not evidence that a pitch was unused.

### 5.5 · "The data collected"

The data are frames from fixed CCTV at nine venues, sampled every fifteen seconds and
hand-labelled against a protocol written before the runs.

The number to notice is the last one. Because the cameras are fixed and the interval is
short, **98.5% of frames have a near-duplicate**. That single property invalidated my own
pilot, and it is why the evaluation is built the way it is.

⟨cut if short⟩ And notice the split underneath: empty pitches are daytime, active play is
night. I will come back to what that does.

### 5.6 · "Why more data was not simply collected"

The obvious question is: why not just collect more?

Because this is **footage of identifiable people** — players, staff, sometimes children.
That is sensitive personal data. Every additional venue is a consent and data-protection
conversation with a facility, not a dataset download. The bottleneck is **permission**, not
storage and not labelling effort.

And the task needs **variety** — many venues, lightings, angles, weathers — not volume. Ten
thousand more frames of the same pitch on the same evening add nothing, because they are
near-duplicates of what I already have.

### 5.7 · "So some data was generated, from a real starting point"

So I generated some, and I want to be careful about how I describe it.

I did **not** generate a dataset from text prompts. **Every generated item starts from a
real anchor** — an actual frame of an actual pitch — and the generation changes one thing
about it.

For images I used **Gemini** and **ChatGPT**, to produce the scene the corpus does not
contain: an empty pitch under floodlights at night. For video I took still frames and
**animated them**, adding motion to a still image so there is movement for the motion check
to read. That used two AI video tools: **[VIDEO TOOL 1]** and **[VIDEO TOOL 2]**.

The result: **31 generated empty frames** took cross-venue false alarms from **0.768 to
0.024** — and play-recall went *up*, which is the sign it added signal rather than noise.

And the guards are on the slide. They are **excluded from every corpus count**; counting
them would make a gap look filled that is not. ⟨cut if short⟩ One line I did not cross: I
did not generate rain. There is no wet footage here, so synthetic rain could only be
validated against synthetic rain — which tests the generator, not the weather.

### 5.8 · "The models — almost nothing here is trained"

Which brings me to a question I expect: why did you not fine-tune anything?

**Two hundred million frozen parameters, and about eleven thousand trained ones.** The three
backbones never saw this data during training.

That was a decision, not a shortcut. Fine-tuning two hundred million parameters on roughly
**150 distinct scenes** does not teach it football — it teaches it *these pitches*, under
*these floodlights*, from *these angles*. Then either the cross-venue number collapses, or,
far worse, it does not collapse and I believe it.

The evidence that the risk is real: accuracy here **falls** when labels rise from three
hundred to 671, because the extra labels are near-duplicates.

So **freezing is a defence against overfitting first**, and an efficiency win second.

### 5.9 · "The rule table — first matching row wins"

The deployed decision layer is deliberately a table a facility manager can read and argue
with.

No detector or no boundary gives **uncertain** — a missing detector is not an empty pitch.
Nobody and nothing moving is **empty**. One to four people is **not a game**. More than four
people **with a ball and motion** is active play — the only way in. More than four people
otherwise is a crowd standing on a pitch, which is not a booking being used.

Two of those numbers are the **facility's rule**, not something I fitted: five people make a
game, four or fewer do not.

⟨cut if short⟩ The last two rows changed late and invert what I pre-registered — play now
has to be **shown**, not assumed. That is stricter, and it costs recall exactly where the
detector cannot see the ball. Which I measured rather than argued about: 0.996 down to
0.308 at the worst venues.

### 5.10 · "Preprocessing — what happens before the model sees a frame"

Before any of that, a frame goes through preprocessing: letterbox, undistort, crop,
standardise, CLAHE, gamma and saturation, denoise, sharpen — and blur.

Two to point at. **CLAHE** is the switch the proposal predicted would fix floodlight glare;
measured honestly, it **costs** 0.27 macro-F1. And **blur** is a *control*, not a candidate
— at high strength it removes the people, and seeing what that does to the score tells you
what the score is actually reading.

The experiments and the live system share **one** preprocessing module. The pilot had two,
they drifted, and the numbers stopped meaning the same thing.

### 5.11 · "How the data were analysed"

The collected data were analysed with frozen backbones and a small trained head, under
splits grouped by venue, with leave-one-venue-out across seven folds.

Two choices here are the method contribution. **Four trivial baselines run in every
protocol** — a clock rule, a colour histogram, a constant predictor, a random one. If a
trivial baseline wins, that is information about the protocol, not a curiosity to leave out.

And I report the **resolution** of every test. Over seven folds the smallest possible
p-value is 0.0156 — so "not significant" and "the design could not have produced
significance" are different statements, and I say which applies.

---

# Tab 6 — The problem · 11:30

### 6.0 · The hero

This research addresses the problem that a facility sells hours and has no verified record
of which ones were used. Staff write it down, the records are unverified, and three kinds of
error follow.

### 6.1 · "The problem, in three failure modes"

**No-shows** — paid for, never played. **Unbooked use** — played, never paid for. And plain
**data-entry error**, where the record and the reality drift apart.

Let me concede the obvious objection first. A twenty-euro motion sensor is more robust in
fog, in darkness, and against a dirty lens. But a sensor answers *did something move*. An
audit needs *was this booking used, and here is the picture* — and it has to tell a
five-a-side match from a groundsman on a mower.

### 6.2 · "What is already known — and the gap"

Although a great deal is known about frozen backbones and about occupancy from CCTV, there
remains a gap: almost all of it is evaluated on frames drawn from the **same scenes** as
training.

So the reported score measures memory of a **place**, not recognition of an **activity** —
and nobody runs a trivial rule beside it to check.

### 6.3 · "This gap matters because the money rests on one class"

And this gap is important because the facility's money rests entirely on one class. Getting
"playing" right is easy and worth nothing. If the system cannot reliably recognise an
**empty** pitch, it cannot flag a single unused booking.

### 6.4 · "First consequence — honest splitting halves the score"

Here is the first consequence, measured. Under a random split, **37% of near-duplicate
pairs** land on opposite sides of the train/test line. Grouped by venue and slot, **below
1%**.

The price of that honesty is severe — ConvNeXtV2's score roughly halves.

⟨cut if short⟩ But only about two thirds of that fall is leakage. A model that never trains
cannot leak, so scoring a zero-shot model on the same test sets isolates the rest at 0.183.
Reporting the bigger number would have made my story stronger, and it would have been wrong.

### 6.5 · "The real gap — in this data, the time of day is the answer"

**Slow down. This is the most important block in the talk.**

Across my data, **98% of daylight frames are an empty pitch, and 99% of night frames are a
match**. So "night means play, day means not-play" is correct on **99.1%** of my corpus —
without looking at a single pixel.

Which means nothing measured on this data can separate a model that recognises an empty
pitch from one that recognises the time of day.

The two cells that would break the tie hold **nine frames** and **six frames**.

That is a statement about the **dataset**, not about the models. And identifying it, rather
than reporting around it, is the contribution of this thesis.

### 6.6 · "The same model, judged two ways"

Identical model, identical data. Two ways of drawing the train/test line — and the answer
changes.

### 6.7 · "What that does to a benchmark"

And here is what the confound does to a benchmark. A rule reading only the clock **beats all
three backbones** across unseen venues — on finding the match and on false alarms at the
same time.

Expect the question: *so are your models worthless?* No. They are indistinguishable from a
light meter **on this dataset**.

Then the pair of numbers underneath. I went and got the missing case — four minutes of a
floodlit pitch at night with nobody on it. The clock rule is wrong on **all sixteen
minutes**. The full deployed system is at **zero**.

The benchmark had sixteen hundred frames; the clip had a few hundred. **The clip won,
because it contained the case the benchmark was missing.**

⟨cut if short⟩ One honesty note: an earlier version of this said the opposite. That came
from a label error — a brightness threshold had filed 216 frames of floodlit night football
as daylight. Corrected, the rule wins.

### 6.8 · "And two more things the protocol was hiding"

Two more, and the pattern is the point — each was found by adding a column to check whether
a number meant what it said.

A **constant predictor** that always answers "playing" scores a perfect macro-F1
cross-venue, ahead of every backbone, because every held-out venue is 100% active play.
There is no ranking to change.

And a low false-alarm rate is **not** accuracy. On 243 held-out empty frames, DINOv2 answers
"playing" 75 times and "maintenance" 168 times. It is correct **zero** times. The published
numbers stand; the inference drawn from them does not.

### 6.9 · "Which is why one missing cell blocks four questions"

And this is why that gap is not a footnote in a limitations chapter. One absent cell
propagates into **four separate research questions**. They are blocked on data, not on work.

---

# Tab 7 — The solution · 16:00

### 7.0 · The hero

So what did I do about it? Three interventions — and I want to say before I show them that
**not one of them is a bigger or better neural network.**

### 7.1 · "Three fixes, all cheap"

**Cheap rules in front of the model** — false alarms on empty pitches fall from 0.617 to
0.012.

**Generate the missing case** — thirty-one synthetic empty frames take cross-venue false
alarms from 0.768 to 0.024, while play-recall goes up.

**Label a handful of frames per camera** — one labelled frame takes every backbone to about
0.99. And the control that reframes it: from five frames, training on those five **alone**
matches training on those five plus 775 from the old camera.

So I state the weaker claim, because it is the true one: what buys the accuracy is **any**
labels from the new camera, not a large corpus from an old one. That is the onboarding
recipe I would hand to the facility.

### 7.2 · "What was investigated — two configuration searches"

Specifically, the investigation had two configuration searches with opposite costs. Prompts
are cheap, so that one is exhaustive — 375 sets. Preprocessing re-embeds the whole dataset
per candidate, so that one is greedy — 88 runs.

And the prompt search gave me the number I find genuinely uncomfortable. Worst wording to
best is **0.726** macro-F1. Choosing between the three backbones — the decision the entire
model chapter is about — moves it by **0.082**. Wording matters about **nine times more**
than architecture.

Two guards, and I state both before anyone else does. Only 22.9% of prompt sets beat the
trained probe and the median loses — so "zero-shot works" is only true if you already know
the wording, which needs labels. And the winner's lead of 0.447 is measured on the very
folds it was selected from.

### 7.3 · "Because both searches are finer than the data"

**One frame** moving in the smallest venue fold shifts the headline by 0.0119. The
confidence band at the median setting is 0.092 wide.

Almost every difference the search ranked on is **smaller than its own error bar**. The
ranking is real arithmetic on unreal precision.

### 7.4 · "The augmentation experiment — and its retraction"

Now the experiment I most want to tell you about.

A model trained on camera A scores 0.441 on camera B. Can augmentation close that gap for
free, with no labels from B? The first answer was yes — one draw reached **0.855**, with
empty-pitch recall going from zero to 0.687. It went into the write-up as a finding, with
one note attached: *only one draw has been taken.*

*(Pause.)*

That note turned out to be the finding. I ran it four more times — same frames, same preset,
same probe seed, **only the random draw changed**. 0.414, 0.351, 0.350, 0.348.

The published number was the **maximum of five**. The standard deviation is 0.221 on a
metric bounded between zero and one. And four of the five draws are **worse than using no
augmentation at all**.

The transferable lesson is on the block: the row is unchanged and still reproduces exactly.
A test pinning the published value would have passed forever. **Reproducibility checked the
wrong thing** — only re-drawing the randomness caught it. Every headline now runs five
draws, and I ran it on my own strongest result before anybody asked.

### 7.5 · "And the practice that found half of this"

Which generalises into the methodological half of the contribution.

**Six safeguards in this project were doing nothing at all** — a lock resolved against the
wrong directory, a threshold set to zero that can never fire, a criterion that searched for
a word instead of running the verifier.

Every one passed whatever test existed. So each new safeguard is now verified by
**deliberately breaking the thing it guards**. That practice found half the results in this
talk, and the root of it is one observation: **a wrong result that looks plausible is
invisible.**

### 7.6 · "What the findings contribute"

So the findings contribute on two levels. **To practice** — a five-frame onboarding recipe
for every new camera, and a rule table a facility manager can read and argue with. **To the
field** — an evaluation method: trivial baselines in every protocol, grouped splits,
re-drawing randomness rather than re-running it, and every guard verified by breaking it.

---

# Tab 8 — Summary · 20:50

### 8.1 · Investigated, found, contributed, implies

In summary, this study **investigated** whether cameras a facility already owns can verify
which booked hours were actually used — on a CPU, with a human deciding every case.

The findings **indicate** that the system works end to end, and that the evaluation is the
real result: on this data a clock beats a neural network, because lighting and occupancy are
confounded so tightly that no protocol here separates them.

These findings **contribute** an evaluation method — trivial baselines everywhere, grouped
splits, re-drawing randomness, and every guard verified by breaking it. And they **imply**,
for the facility, a five-frame onboarding recipe; and for the field, that a benchmark can
rank a light meter first and still look perfectly healthy.

### 8.2 · "What this study cannot claim"

Four limits, named first because you will find them anyway.

Empty pitches exist at **one venue only**. The 243 held-out empty frames are **three
distinct scenes** — six comparisons stopped being significant when I recounted them that
way. **Nothing has run on the target hardware**. And there is **no human ceiling** — no
inter-annotator figure exists, which is a real gap and I would rather say so than have it
found.

### 8.3 · "What would actually change it"

And here is what would fix them, ordered by value rather than effort.

Twenty to thirty minutes of **empty-pitch footage at a second venue** — that half-hour
resolves the confound, the external-validity limit and the scope reduction at once. About
**thirty real labelled slots**, which is one conversation with the facility. And **five
labelled frames per camera**, which is already measured.

The limitations of this thesis are mostly a **data-access problem with a known and
inexpensive solution** — not a methodological one.

### 8.4 · The closing line

*(Pause.)*

**No protocol compensates for a case the data never contains.**

---

# Tab 9 — Thank you · 22:50

### 9.0 · The hero

Thank you for your time and attention. I welcome any questions, comments, suggestions or
feedback you may have.

### 9.1 · "Questions I expect, and the short answers"

*(Do not read these out. They are on screen so you can point at one while you answer.)*

> **Would a colour histogram have done this?** 0.9616 on the leaky split — indistinguishable
> from ConvNeXtV2 once you count 62 scenes rather than 394 frames.
>
> **Is the novelty just temporal smoothing?** Tested against four tuned baselines including
> an HMM, with the saturation caveat enforced in code.
>
> **How many comparisons before p < 0.05?** Holm within declared families; the undeclared
> ones disclosed as numbered amendments.
>
> **Is auditing staff by camera ethical?** Per-field reporting only, and the decision layer
> can structurally only advise.
>
> **Can anyone reproduce this?** One pipeline re-derives every claim from its artefact on
> every run.
>
> **What is the human ceiling?** No inter-annotator figure exists. Say so plainly — it is a
> real gap.

### 9.2 · "Closing"

*(Only if the room is quiet and nobody has moved yet.)*

I will leave you with the one sentence this whole thesis reduces to: **no protocol
compensates for a case the data never contains.**

Thank you.

---

## Delivery notes

- **Practise saying "we do not know" three times**: the human ceiling, the target hardware,
  and whether any of this transfers to a second venue. An examiner trusts a candidate who
  has bounded their ignorance more than one who has not noticed it.
- **Time Tab 5 and Tab 6 with a clock.** Together they are half the talk, and Tab 6 is the
  one you will want to over-run.
- **Your strongest sentence** is a version of *"we measured that, and here is how large it
  is."* Where that is not available — the four limits — the second strongest is *"that is a
  real limitation, and here is exactly what it costs."*
- **Slow down on 6.5, 6.7 and 8.4.** Everything else can move.
- **Running over?** Drop every **⟨cut if short⟩** passage first — about 90 seconds — then
  5.3, 6.6 and 6.9, which are diagrams that work without commentary.
- Practise with a strong, clear voice and a confident tone. Use simple language for the
  complex ideas, and end on the statistic rather than an apology.

## Before the day

- [ ] Fill in **[PROGRAMME]**, **[INSTITUTION]**, **[NAME]** and **[PERIOD]** (Tab 1).
- [ ] Fill in **[VIDEO TOOL 1]** and **[VIDEO TOOL 2]** (block 5.7, and the same placeholder
      on the site).
- [ ] Re-read [`thesis/claims.md`](../claims.md) — every number above comes from it, and it
      is regenerated from the result artefacts. If a result moved, this script is stale.
- [ ] Rewrite at least Tab 4 and Tab 8 in your own words.
- [ ] Run `uv run pitch serve` on the machine you will present from, and check Tab 5 and
      Tab 6 render their diagrams and figures.
