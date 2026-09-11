# First supervision meeting — what to say

**Purpose of this file.** Not a script to read out. It is here so that you know what is in
your own repository well enough to *answer questions about it*, which is the only thing that
matters in the room. Read section 1 and 2 until you could say them with the file closed.

Every number below was verified against the repository on 2026-09-11. Numbers not in here
should not be quoted from memory — several earlier headlines were revised, and one was
retracted entirely.

---

## 1. What the project is — say this in your own words, in about 30 seconds

> A facility runs a network of 5-a-side football pitches. They have cameras, and they have a
> booking system. The booking system records what *should* have happened and what staff say
> *did* happen. The cameras can show what *actually* happened.
>
> I am building a system that samples **one frame per camera per minute**, classifies each
> frame as *empty*, *active play*, or *maintenance*, and turns an hour of those frames into
> one verdict for the rental slot: **USED, NOT USED, or REVIEW**. Then it compares that
> verdict against the booking record and flags the disagreements — no-shows, unbooked usage,
> data-entry errors.
>
> The constraint that shapes everything is that it must run on **one small PC with no
> graphics card**. That is why it samples one frame a minute instead of decoding video
> continuously — it cuts the network load by about 99% — and why every model is a **frozen**
> backbone with a small trainable head on top, rather than a fine-tuned network.

If they ask "why is a human still involved?" — because the system produces **decision
support, never an automated action**. A person confirms every anomaly. That is deliberate:
the tool can implicate a member of staff, so it must never bill or accuse on its own.

---

## 2. What actually exists

Say this plainly; it is all verifiable in the repo.

| | |
|---|---|
| Dataset | **1,692 labelled frames** across **10 venues** |
| Code | **64 source modules, 43 experiment scripts** (git-derived, in the README) |
| Tests | **1,084 tests** |
| Results | **50 committed result files**, each produced by a named script |
| Hypotheses | **all six pre-registered hypotheses reported** — including the two whose decision rules you disowned and said why |
| Reproducibility | one command regenerates every table and figure (`reproduce_all.py`) |

**What the pipeline does end to end:** samples frames → preprocesses them → runs a frozen
backbone to get a feature vector → a small logistic-regression head classifies the frame →
two cameras watching the same pitch are fused → a slot's worth of predictions is aggregated
into USED / NOT USED / REVIEW → three evidence images are attached so a human can check it →
the verdict is reconciled against the booking record.

**Two honest qualifiers, and say them yourself rather than being asked:**

- The reconciliation layer is built and tested but has only ever run on **two real slots**,
  because that is all the footage contains.
- The system has been validated on recorded footage, not on a live deployment yet.

---

## 3. The data situation — say this early, do not let it be discovered

This is the single most important thing for your supervisor to understand, and owning it
early is what makes the rest of the meeting go well.

> The footage I have is **1,692 frames from 10 venues, but the classes are not spread across
> them.** Empty pitches exist at essentially **one venue only**. Of the three classes,
> maintenance has **6 frames** in the entire dataset.
>
> That means several of my research questions are **not answerable with this data**, and I
> wrote that down in advance rather than discovering it at the end.

The sharpest way to show why it matters — this one sentence usually lands:

> On the cross-venue test, **a model that always answers "playing" scores a perfect 1.000**,
> because every held-out venue contains only active play. So a high score there proves
> nothing on its own, and every result I report has a second column to catch exactly that.

**What would fix it:** empty-pitch footage from two or three other venues. It is the cheapest
footage a facility owns — a pitch is empty most of the day — and it has almost no privacy
concern because nobody is in the frame. **Ask your supervisor to help you get it.** It
unblocks more of the project than anything else you could do.

---

## 4. The results you can actually defend

Lead with this one. It is the strongest thing in the project and it is stable.

### The main finding: how you evaluate changes the answer, not just the score

> The standard way to split this data — shuffle the frames randomly — is wrong, because
> frames sampled seconds apart are nearly identical, so the same moment ends up in both
> training and test. The model is being tested on copies of what it memorised.
>
> When I split honestly instead, by scene and by venue, **the score roughly halves**:
> ConvNeXtV2 goes from 0.988 to 0.498, ViT from 0.994 to 0.498, DINOv2 from 0.988 to 0.579.
>
> And it does not only lower the numbers — **it reverses the decision**. ViT ranks first under
> the sloppy protocol and joint-last under the honest one. Anyone following the original
> protocol would have chosen the model that generalises worst.

Three pieces of evidence behind it, if they push:

- **100% of the errors on the leaky split had a near-duplicate frame on the training side.
  On the honest split, 0% did.** That is the mechanism, measured directly.
- I ran a **control** for the possibility that the drop is just the test set changing
  composition: an untrained model drops 0.183 on the same change. So the part attributable to
  leakage is **0.332–0.395**, not the full drop. *(This is the kind of detail that shows you
  are not overselling — mention it.)*
- The honest split is **99% one class**, which I report as a limitation rather than hiding.

### The second finding: do the deep models earn their cost?

> I compared the deep backbones against deliberately trivial baselines — a rule that only
> looks at the clock, and a 16-bin colour histogram.
>
> **Within one venue, no.** A clock rule that never looks at the image comes within 0.007 of
> ConvNeXtV2. **Across venues, emphatically yes** — the trivial baselines collapse while the
> frozen features transfer.

That conditional answer is more interesting than a yes or a no, and it is honest.

### A third result worth having ready: the prompt matters more than the model

> Using a vision-language model with **no training labels at all**, the wording of the text
> prompt moves the score about **nine times more than the choice of backbone does** —
> macro-F1 spans 0.021 to 0.747 across the prompt space, where the three backbones span 0.082.
>
> So "what does it cost to deploy at a new site with no labels?" has no single answer. It is a
> distribution, and which prompt you happen to write matters more than which model you buy.

### What you are recommending, and why

> **DINOv2** for production. The pilot had chosen ConvNeXtV2 on speed, but measured properly,
> **speed is not the binding constraint** — even the slowest model uses under 10% of the
> one-minute sampling cycle for 20 cameras. Once speed stops discriminating, the choice falls
> to accuracy under honest evaluation, and that is DINOv2.

---

## 5. The thing that will impress them most

Most students present only what worked. Say this, and say it as **rigour, not apology**:

> I built the project so that mistakes surface, and it has been working. Three examples:
>
> 1. **A headline result of mine turned out to be the best of five random seeds.** I had
>    reported an augmentation setting at 0.855. Run with four other seeds it gives 0.348,
>    0.350, 0.351, 0.414 — and four of the five are *worse than using no augmentation at all*.
>    I retracted it everywhere it appeared.
>
> 2. **One of my own statistical tests was measuring the wrong thing.** I was reporting a
>    macro-F1 difference next to a p-value computed on accuracy. Fixed — and the correction
>    made my main result about three times *larger*, not smaller.
>
> 3. **I found what looked like a major result and then killed it myself.** One model appeared
>    to have a catastrophic flaw that a preprocessing change fixed. Before reporting it I ran
>    it through the full protocol: it was one measurement in one direction. Swap the two
>    cameras and it reverses. So it is not in my results.

Then the sentence that ties it together:

> None of these were found by a test failing. They were found by asking whether a number could
> be reproduced. Most of those checks are now tests, so they cannot come back.

**Framing matters.** This is "my method catches errors", never "I made a lot of mistakes".

---

## 6. What you need from them — ask for all of it

Bring a recommendation for each. Supervisors approve proposals much more easily than they
invent answers.

1. **Sign off M1** — the protocol, taxonomy and evaluation design. This is the gate that was
   due in week 4. The labelling protocol is drafted and waiting at
   `thesis/labelling_protocol.md`. *This is the main transaction of the meeting.*
2. **The ethics question.** The facility gave me the footage and holds the consent for
   collecting it. Does the programme need its own separate sign-off for my *research use* of
   it? It is likely a short form, and it is much cheaper to ask now than at submission.
3. **May pixels ever be published?** Images of real players are in my repository history.
   Whether any frame can appear in the thesis decides whether I can show a worked
   reconciliation example, and whether a dataset can be listed as a contribution.
4. **Help me ask the facility for empty-pitch footage** at two or three other venues — see
   section 3. Biggest single unblock available.
5. **Tell them about the C3 decision.** The maintenance class has 6 frames, so I cannot make
   any claim about it. I decided to report the benchmark as 2-class with maintenance handled
   as a flagged exception, and to state the scope reduction openly. **This went against the
   plan's standing recommendation, so it needs saying out loud rather than being found.**
6. **Agree the framing:** does the thesis lead with results or with method? Your own evidence
   says method. Get them to agree, because it shapes the whole write-up.

---

## 7. Questions you will be asked — and the honest answers

**"Why computer vision? Why not a €20 motion sensor?"**
The most dangerous question, and it is fair. Short answer: a motion sensor tells you something
moved, not *what* — it cannot tell a match from a groundskeeper from a fox, and it cannot
produce evidence images for a disputed booking.

**You have a full written answer to this already: `thesis/alternatives.md`.** It compares
sensors, turnstiles, floodlight power draw and app check-ins on cost, accuracy, failure modes
and retrofit difficulty, and it concedes the €20 point before answering it. Read it before the
meeting — it is a page, and it is the single best-prepared answer you own.

**"How many real slots have you tested the slot logic on?"**
Two. Say the number. It is why the slot-level module is reported as preliminary.

**"Is your accuracy good?"**
Refuse the framing politely: on this test set a constant predictor scores 1.000, so a high
number means nothing by itself. That is why every result carries a second column.

**"How do you know your splits do not leak?"**
Measured, not assumed. Frames are grouped by scene and by venue, near-duplicates are detected
by perceptual hash, and the audit showed 100% of leaky-split errors had a near-duplicate in
training against 0% on the honest split.

**"What is your contribution?"**
The method: a leakage-free, pre-registered evaluation of frozen-backbone occupancy
classification on CPU, plus the reconciliation idea. Say it that way round.

**If you are asked something you do not know:** say *"I would have to check — it is in the
experiment log and I will confirm after the meeting."* That is a completely normal and
respectable answer. Guessing is the only thing that will damage you.

> **You have already written the long version of this section.**
> `thesis/defence_redteam.md` holds **twelve** hard questions with the evidence for each
> answer. This list is the meeting-sized subset. If you only have time to read one thing
> before you walk in, read that file.

---

## 8. Before the meeting

- [ ] Read sections 1, 2 and 3 until you can say them with the file closed.
- [ ] Open `thesis/labelling_protocol.md` and skim it — it is what you are asking to be signed
      off, so you should know what is in it.
- [ ] Fix the stale first line of `docs/backup.md`: it still says the backup has not been made,
      and it was completed on 2026-09-09. Do not let a stale document be the one they open.
- [ ] Decide your one sentence for "what is the contribution". Practise it aloud.

**If the meeting is only 15 minutes:** sections 1, 3 and 6. The project, the data limit, and
the decisions. Everything else can go in writing.
