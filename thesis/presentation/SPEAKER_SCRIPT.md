# Speaker script — 25 minutes

**Read this out loud three times before the defence.** The wording is a starting point, not a
cage: rewrite any sentence that does not sound like you, because phrasing you did not choose
collapses on the first follow-up question.

Timings are cumulative. `[SLIDE]` markers match the tabs on the site (`uv run pitch serve`)
and the slides in `Thesis_Defence_25min.pptx`.

Things that are still yours to fill in are marked **[LIKE THIS]**.

---

## 0:00 — Start

> **[SLIDE: Start — title]**

Good morning, distinguished examiners, professors and colleagues.

My name is Mariam Srour, and I am a Master's student in **[PROGRAMME]** at
**[INSTITUTION]**.

The title of my research is *Multi-Pitch Occupancy and Booking Verification from Existing
Cameras*. Through the talk I will use the shorter version: **who actually used the pitch?**

> **[SLIDE: Roadmap]**

In this presentation I will briefly take you through the purpose of the research; how the
research works, with a short scenario; what problem it tackles; the data and the models;
the experiments; and I will conclude with a summary.

Two things I will spend real time on, because they are where the method contribution sits:
the two configuration searches, and the augmentation experiment.

---

## 1:30 — Purpose, and who it helps

> **[SLIDE: Purpose — the statement]**

The purpose of this study is to find out **which booked pitch hours were actually used**,
using the cameras a facility already owns.

Three constraints shape everything: no new hardware, no GPU, and a person decides every
case.

> **[SLIDE: Purpose — what the owner gets]**

Let me say plainly why anyone would want this, because it is easy to lose in the
methodology.

A facility rents out twenty or thirty pitches by the hour. Staff write down which slots were
used. Nobody verifies that. And **nobody can**: one manager physically cannot watch twenty
pitches at once. So they spot-check, they trust the sheet, and the sheet is unverified.

This system gives the owner three things.

**First, time.** It checks every booked minute on every pitch, and shows the manager only
the bookings where the record and the camera disagree. The manager's attention goes where it
is worth something instead of on a walk-round.

**Second, consistency.** It applies the same rule at eight in the morning and at eleven at
night. It does not get tired, it does not rush the last pitch, and it does not skip the one
at the far end because it is raining. A person samples; this does not.

**Third, evidence.** Every flag arrives with three photographs. So the conversation with a
customer is about a picture on a screen, not about somebody's memory of Tuesday.

That is also why I would say it is **more trustworthy than a manual check** — not because it
is cleverer than a person, but because it is consistent, it covers everything, and it shows
you what it decided on. A person can be argued with; a photograph is harder to argue with.

> *If an examiner pushes on "more trustworthy than a person" — and they might — do not
> defend the strong version. Say this:* "To be precise: it is more **consistent** and more
> **auditable** than a manual check, and it covers every minute rather than a sample. I am
> not claiming it is more accurate than a human looking at the same photograph. I have no
> inter-annotator figure, and that is a real gap in this work. Accuracy stays a person's
> job — which is exactly why the system only ever advises."

> **[SLIDE: Purpose — one rule carried everywhere]**

One design rule carried through the whole system: **the vision path must never see the
booking record as an input.** A model that has read the booking flag cannot give you
evidence that is independent of the record it is auditing. That is the difference between an
audit and a rubber stamp.

---

## 4:30 — How it works

> **[SLIDE: How it works — five stages]**

To achieve this, I built a five-stage system and then evaluated it.

**Stage one, sample.** One frame per camera per minute. Not a video stream — a single still
image. That is roughly a **99% cut in network traffic**, and it is what lets the whole
facility run on one small computer with no graphics card.

**Stage two, classify.** Each frame is labelled empty, active play, or maintenance.
Maintenance is there so that a mower is not billed as a football match.

**Stage three, check.** Cheap rules look at the same frame: is anyone inside the pitch
boundary, did anything move, are there enough people, is there a ball. These rules can
**overrule** the neural network. I will come back to that, because it turns out to matter
more than the choice of network.

**Stage four, aggregate.** About sixty predictions become one verdict for the booked hour:
used, not used, or review.

**Stage five, compare.** That verdict is checked against the booking record, and
disagreements are flagged with their evidence.

> **[SLIDE: How it works — pipeline diagram]**

The thing to notice in this diagram is that every stage can **refuse** rather than guess.
That is what makes "review" a real outcome instead of a low-confidence "used" — and it is
why a camera that drops out cannot silently become an empty pitch.

> **[SLIDE: How it works — the scenario]**

Let me make it concrete.

It is Tuesday evening, eight o'clock, pitch three. The booking system says the pitch was
sold for an hour, and a member of staff marked it used. The cameras sampled sixteen frames
across that hour, and thirteen of them show an empty pitch under floodlights.

The system does **not** cancel the charge. It raises one advisory with three photographs,
and a manager decides.

In the code that is structural rather than a promise: the only thing the decision layer can
return is an *Advisory*, human confirmation is a property rather than a setting, and there
is no code path that acts on a verdict.

> **[SLIDE: How it works — what gets stored, then the two method slides]**

On method: the data are frames from fixed CCTV cameras at nine venues, sampled every fifteen
seconds and hand-labelled against a protocol written before the runs.

They are analysed with frozen vision backbones and a small trained head, under splits
grouped by venue, with four trivial baselines running alongside in every protocol, and every
comparison reported with the smallest p-value the design could have produced.

That last part matters, and I will show you why.

---

## 8:00 — The problem, and the gap

> **[SLIDE: The problem]**

This research addresses the problem that a facility sells hours and has no verified record
of which ones were used. Three kinds of error follow: no-shows, unbooked usage, and plain
data-entry mistakes.

Let me concede the obvious objection first. A twenty-euro motion sensor is more robust than
a camera in fog, in darkness, and against a dirty lens. But a sensor answers *did something
move*. An audit needs *was this booking used, and here is the picture* — and it has to tell
a five-a-side match from a groundsman on a mower.

> **[SLIDE: The gap — what is known]**

A great deal is already known about frozen vision backbones and about occupancy from CCTV.
The gap is that almost all of it is evaluated on frames drawn from the **same scenes** as
training. So the reported score measures memory of a place rather than recognition of an
activity.

That matters here because the facility's money rests entirely on one class. Getting
"playing" right is easy and worth nothing. If the system cannot reliably recognise an
**empty** pitch, it cannot flag a single unused booking.

> **[SLIDE: Honest splitting halves the score]**

Here is the first consequence, measured. Under a random split, 37% of near-duplicate frame
pairs land on opposite sides of the train/test line. Group the split by venue and slot, and
that drops below 1%.

The price of that honesty is severe — ConvNeXtV2's score roughly halves.

But I checked my own result, and only about **two thirds** of that fall is leakage. A model
that never trains cannot leak, so scoring a zero-shot model on the same test sets isolates
the rest: 0.183 of it is simply the test set getting harder. Reporting the bigger number
would have made my story stronger, and it would have been wrong.

> **[SLIDE: The confound — slow down here]**

And now the real gap, which is the most important slide in this talk.

Across my recorded data, **98% of daylight frames are an empty pitch, and 99% of night
frames are a match**. Which means the sentence "night means play, day means not-play" is
correct on **99.1%** of my corpus — without looking at a single pixel.

The consequence is that nothing measured on this data can separate a model that recognises
an empty pitch from a model that recognises the time of day. The two cells that would break
the tie hold **nine frames** and **six frames**.

I want to be precise about what kind of claim that is. It is a statement about the
**dataset**, not about the models. And identifying it, rather than reporting around it, is
the contribution of this thesis.

> **[SLIDE: The clock rule beats the models]**

Here is what that does to a benchmark. A rule that reads only the clock and never looks at
the image beats all three modern backbones across unseen venues — on finding the match, and
on false alarms, at the same time.

Expect the question: *so are your models worthless?* No. They are indistinguishable from a
light meter **on this dataset**.

I should volunteer one thing here. An earlier version of this slide said the opposite. That
number came from a label error — a brightness threshold had filed 216 frames of floodlit
night football as daylight. Corrected, the rule wins and the finding reversed.

> **[SLIDE: The clip that reversed the ranking]**

So I went and got the missing case: about four minutes of a floodlit pitch at night with
nobody on it, scored minute by minute.

The clock rule calls all sixteen minutes active play — wrong on every single one. The neural
model on its own sits at 0.38. The full deployed system is at **zero**, and gets all thirteen
empty minutes right.

The point: the benchmark had 1,692 frames and the clip had a few hundred. **The clip won,
because it contained the case the benchmark was missing.** More data of the same kind would
not have found this. I think that is the most useful sentence in the thesis.

> **[SLIDE: Two more things the protocol was hiding]**

Two more, quickly, and the pattern is the point — each was found by checking rather than by
theorising.

A **constant predictor** that always answers "playing" scores a perfect macro-F1 on the
cross-venue protocol, ahead of every backbone, because every held-out venue is 100% active
play. That is stronger than saying the protocol changes the ranking — there is no ranking.

And a low false-alarm rate is **not** accuracy. On 243 held-out empty frames, DINOv2 answers
"playing" 75 times and "maintenance" 168 times. It is correct **zero** times. The published
numbers stand; the inference drawn from them does not.

---

## 13:00 — The data: why it is small, and what we made

> **[SLIDE: Data — what we have]**

Now the data, because everything I have just shown you is a consequence of it.

1,692 labelled frames. Three classes. Every empty pitch in the corpus comes from **one
venue**, and maintenance has too few frames to be a class you can evaluate at all.

> **[SLIDE: Data — why this data is hard to get]**

The obvious question is: why not just collect more?

Because this is **footage of identifiable people** on a pitch — players, staff, and
sometimes children. That is sensitive personal data. Every additional venue is a consent and
data-protection conversation with a facility, not a dataset download. The bottleneck is
**permission**, not storage and not labelling effort.

And note what the task needs. It needs **variety** — many venues, many lighting conditions,
many camera angles, many weathers. It does not need volume of the same scene. Ten thousand
more frames of the same pitch on the same evening would add nothing, because they are
near-duplicates of what I already have.

> **[SLIDE: Data — so we made some]**

So I generated some data, and I want to be careful about how I describe that.

I did not generate a dataset from text prompts. **Every generated item starts from a real
anchor** — a real frame of a real pitch — and the generation changes one thing about it.

For images, I used **Gemini** and **ChatGPT**'s image models, to produce the scene the
corpus does not contain: an empty pitch under floodlights at night.

For video, I took still frames and animated them — adding motion to a still image, so that
there is movement for the motion check to read. That used two AI video tools:
**[VIDEO TOOL 1]** and **[VIDEO TOOL 2]**.

The result: **31 generated empty frames** took cross-venue false alarms from **0.768 down to
0.024** — and play-recall went *up*, not down, which is the sign that it added signal rather
than noise.

> **[SLIDE: Data — and we are honest about what generated data is]**

Generated data is a legitimate tool and a very easy way to fool yourself, so the guards are
on the slide.

They are **excluded from every corpus count** in this thesis. Counting them would make a gap
look filled that is not. They are a training aid, not evidence that the system works on real
footage of that case. And I checked the effect against 26 real frames, so it is not an
artefact of the generator.

One line I did not cross: **I did not generate rain.** There is no wet footage in this
dataset at all, so synthetic rain could only ever be validated against synthetic rain — and
that tests the generator, not the weather.

> **[SLIDE: Data — the confound as a picture, then what it blocks]**

This diagram is read straight off the manifest, so it cannot drift from the corpus. The two
nearly-empty cells are the whole problem — and the next diagram shows that one missing cell
propagating into **four separate research questions** that no amount of further engineering
can reach.

> **[SLIDE: Data — small data has a specific danger]**

And here is the danger with a dataset this size, stated precisely.

The risk is **not** that the numbers come out low. It is that they come out **high for the
wrong reason** — the model memorises these specific pitches, and that looks like success.
That is overfitting, and the opposite of what we need, which is **generalisation** to a venue
the system has never seen.

The evidence that this is real and not theoretical is in the middle of the slide: label
efficiency here is **not monotone**. Every backbone peaks at one to three hundred labels and
then gets **worse** at 671, because the extra labels are near-duplicates that add redundancy
rather than information.

That is the whole reason the evaluation is built the way it is.

---

## 17:00 — Models, and why we trained nothing

> **[SLIDE: Models — we did not train the models]**

Which brings me to the models, and to a question I expect: why did you not fine-tune
anything?

The honest answer is on the slide. **Two hundred million frozen parameters, and about eleven
thousand trained ones.** The three backbones — DINOv2, ConvNeXtV2 and ViT — never saw this
data during training. They are fixed feature extractors. The only thing that learns is a
classifier small enough to read on one screen.

> **[SLIDE: Models — why not]**

And that was a decision, not a shortcut.

Fine-tuning a two-hundred-million-parameter network on roughly **150 distinct scenes** does
not teach it football. It teaches it *these pitches*, under *these floodlights*, from *these
camera angles*. Then either the cross-venue number collapses — or, far worse, it does not
collapse and I believe it.

I could not collect my way out of that, for the reason I gave: the data is sensitive and the
bottleneck is permission.

So **freezing the backbones is a defence against overfitting first**, and an efficiency win
second. The efficiency is real and very welcome — embed each frame once, reuse it forever,
which is how seventy-five experiments fit into one thesis — but it is the second reason, not
the first.

> **[SLIDE: Models — two novel modules]**

The thesis proposed two novel modules, and I report both the way they came out.

The **gated fusion head** is a negative result. Routing between backbones is worth minus
0.024 cross-venue recall against the same head with the gate switched off. The gate puts
about 70% of its weight on DINOv2 in every single fold — it is a learned constant wearing a
router's costume. It *can* route: on a test fixture where the useful backbone flips, it
scores 1.000 against 0.671. So the null result is about the data, not the code.

**STAN**, the temporal model, scores a perfect 1.000 — and I am going to argue against my own
number. The composed labels are a deterministic function of five templates, so any model that
reads contiguity recovers the generating process. That is an exhausted test set, not a win.
The real test set is **two slots**, and the thirty-slot threshold raises an exception in code
rather than sitting in a footnote.

One more thing worth saying: the proposal named ConvNeXtV2. Measured without leakage, DINOv2
leads. The decision was made on a measurement taken after the proposal — that is the process
working.

---

## 19:30 — Rules and preprocessing

> **[SLIDE: Rules — the decision table]**

The deployed decision layer is deliberately a table a facility manager can read and argue
with. First matching row wins.

1. **Detector unavailable** → uncertain. A missing detector is not an empty pitch.
2. **No pitch boundary** → uncertain. Mandatory on the deployed path.
3. **Nobody, and nothing moving** → **empty**.
4. **Nobody, but something moved** → uncertain.
5. **One to four people** → not a game. Too few, ball or no ball.
6. **More than four people, and a ball, and motion** → **active play**. The only way in.
7. **More than four people, otherwise** → not a game. A crowd standing on a pitch.

Two numbers in that table are the **facility's rule**, not something I fitted: five people
make a game, four or fewer do not. The rest are fitted on one camera and frozen along with
the commit that froze them.

Rows six and seven changed late, and they invert what I pre-registered. Originally play was
the default above the head count. The facility's rule is the opposite — a game has a ball in
it and people moving, and a crowd standing on a pitch is not a booking being used. So play
now has to be **shown**, not assumed.

> **[SLIDE: Rules — every switch, and what it costs]**

Each switch exists because its cost is a decision somebody should be able to take
deliberately.

The **ball requirement** is the clearest case. Cross-venue ball recall is 0.40, and between
0.06 and 0.89 depending on the venue. Turning it on drops play recall from 0.996 to 0.308 at
the worst venues. That is expensive, it is the facility's rule, and whoever switches it off
should be able to see exactly what they are buying.

And there is a rule about rules at the bottom: **a cue that cannot be measured is reported,
never assumed.** If motion cannot be computed, the clause is skipped and the trace says so —
because a requirement that silently never fires is the shape of every safeguard this project
found not working.

> **[SLIDE: Preprocessing]**

Before any of that, a frame goes through preprocessing, and this is the list:

- **Letterbox** to 224×224 — pad rather than squash.
- **Undistort** — take the barrel out of a wide CCTV lens.
- **Crop** — centre or top, to drop sky and car park.
- **Per-image standardise** — normalise each frame to its own statistics.
- **CLAHE** — local contrast, for floodlight glare.
- **Gamma and saturation** — brightness curve, and colour down to grayscale.
- **Denoise and sharpen**.
- **Blur** — in there as a *control*, not a candidate.

Two to point at. **CLAHE** is the switch the proposal predicted would fix floodlight glare;
measured honestly, it **costs** 0.27 macro-F1. And **blur** is a control because at high
strength it removes the people — and seeing what that does to the score tells you what the
score is actually reading.

The experiments and the live system share **one** preprocessing module. That is not
tidiness: the pilot had two, they drifted, and the numbers stopped meaning the same thing.

---

## 21:00 — The searches, and augmentation

> **[SLIDE: The searches]**

Two configuration searches, with opposite costs. Prompts are cheap, so that search is
exhaustive — 375 sets. Preprocessing is expensive, because every candidate re-embeds the
whole dataset, so that one is greedy — 88 runs.

Both are guarded the same way: every cross-venue fold is 100% active play, so recall alone
can be bought by saying "playing" more often. Everything is ranked on recall minus false
alarms instead.

> **[SLIDE: Search 1, then the resolution floor]**

The preprocessing search found a clear winner with a perfect score. Then I measured the
resolution of the search itself.

**One frame** moving in the smallest venue fold shifts the headline by 0.0119. The confidence
band at the median setting is 0.092 wide. Almost every difference the search ranked on is
**smaller than its own error bar**.

So the ranking is real arithmetic on unreal precision. The search is not wrong — it is
finer-grained than the evidence underneath it, and saying so is more useful than announcing a
winner.

> **[SLIDE: Search 2 — prompts]**

The prompt search gave me the number I find genuinely uncomfortable.

The span from the worst wording to the best is **0.726** macro-F1. Choosing between the three
trained backbones — the decision the entire model chapter is about — moves the score by
**0.082**. How you write the sentence matters about **nine times more** than which model you
pick.

Two guards, and I state both before anyone else does. Only 22.9% of prompt sets beat the
trained probe and the median one loses — so "zero-shot works" is only true if you already
know which wording to use, and knowing that requires labels. And the winner's lead of 0.447 is
measured on the very folds it was selected from. That is disclosed in the pre-registration as
an undeclared search family.

> **[SLIDE: Augmentation — the question]**

Now augmentation, which is the experiment I most want to tell you about.

The question is the deployment question. A model trained on camera A scores 0.441 on camera
B, which watches the same pitch from a different angle. One labelled frame of camera B takes
it to 0.99. **Can augmentation close that gap for free** — by varying brightness, gamma and
sensor noise during training, with no labels from B at all?

> **[SLIDE: Augmentation — it worked]**

The first answer was yes. One draw of the brightness-and-gamma preset took empty-pitch recall
on the unseen camera from zero to **0.687**, with no labels from it. Macro-F1 0.855.

It went into the write-up as a finding, with one note attached: *only one draw has been
taken.*

That note turned out to be the finding.

> **[SLIDE: Augmentation — five draws]**

I ran the identical experiment four more times. Same frames, same preset, same probe seed,
same test set. **Only the random draw changed.**

0.414. 0.351. 0.350. 0.348.

The published 0.855 was the **maximum of five**. The standard deviation is 0.221 on a metric
that only runs from zero to one. And four of the five draws are **worse than using no
augmentation at all**.

One detail explains the shape: three draws score exactly 0.3479, which is the score of
answering "active play" to every single test frame. What moves between draws is whether the
fitted boundary reaches camera B's empty pitch at all. It does not degrade gracefully — it is
either a working classifier or the trivial one.

> **[SLIDE: Augmentation — the lesson]**

So the headline was retracted, and I want to be precise about what was retracted.

The row in the results file is **unchanged and still reproduces exactly**. What was withdrawn
is the claim about the *method*. It is now restated as a claim about that one draw, and only
ever quoted beside the spread.

And the transferable lesson is this: a unit test that pinned the published value would have
passed forever, because the artefact was reproducible. **Reproducibility checked the wrong
thing.** Only re-drawing the randomness caught it.

That is now standing practice, and I ran it on my own strongest result — the temporal model —
and volunteered the spread before anybody asked for it.

---

## 23:00 — The solution, and the limits

> **[SLIDE: Solution — the fix was never a bigger model]**

So what did I do about all of this? Three things — and none of them is a bigger neural
network.

**One: put cheap rules in front of the model.** False alarms on empty pitches fall from 0.617
to 0.012.

**Two: generate the missing case.** Thirty-one synthetic empty-pitch frames take cross-venue
false alarms from 0.768 to 0.024, while play-recall goes up.

**Three: label a handful of frames per camera.** One labelled frame of a new camera takes
every backbone to about 0.99. And the control that reframes it: from five frames, training on
those five **alone** matches training on those five plus 775 frames from the old camera. The
old data stops contributing.

So I state the weaker claim, because it is the true one: what buys the accuracy is having
**any** labels from the new camera — not a large corpus from an old one. That is the
onboarding recipe I would hand to the facility.

> **[SLIDE: Solution — every guard is verified by breaking it]**

One slide on method, because I think this is the part that transfers beyond football pitches.

**Six safeguards in this project were doing nothing at all.** A test-set lock resolved against
the wrong directory. A confidence threshold set to zero, which can never fire. A criterion
that searched for a word instead of running the verifier. Evidence selection that recorded a
file path of *None* for every frame it chose.

Every one of them passed whatever test existed. So each new safeguard is now verified by
**deliberately breaking the thing it guards** and confirming it fires. That practice found
half the results in this talk.

The root of it is one observation from early on: **a wrong result that looks plausible is
invisible.**

> **[SLIDE: Solution — what it will not do]**

And one slide on what the system deliberately will not do, because auditing how a facility's
pitches are used is close to auditing the people who work there.

It never bills and never acts. Anomalies are reported per **field**, never per person.
And review never becomes an anomaly — if the model is unsure, that uncertainty belongs to the
model. Turning it into a flag against a member of staff would be laundering the system's own
weakness into someone else's record.

> **[SLIDE: Limits]**

Three limits, named first, because you will find them anyway.

Empty pitches exist at **one venue only**, so a three-class cross-venue evaluation cannot be
run on this corpus. The 243 held-out empty frames are **three distinct scenes** — six
comparisons stopped being significant when I recounted them that way. And **nothing has run
on the target hardware**; every speed number is from a laptop.

There is also no human ceiling. No inter-annotator figure exists. That is a real gap, and I
would rather say so than have it found.

> **[SLIDE: What would change this]**

And here is what would fix them, ordered by value rather than effort.

Twenty to thirty minutes of **empty-pitch footage at a second venue** — that one half-hour
resolves the confound, the external-validity limit and the scope reduction at once. About
**thirty real labelled slots**, which is one conversation with the facility. And **five
labelled frames per camera**, which is already measured.

The limitations of this thesis are mostly a **data-access problem with a known and
inexpensive solution** — not a methodological one.

---

## 24:30 — Summary and close

> **[SLIDE: Summary]**

In summary.

This study **investigated** whether the cameras a facility already owns can verify which
booked hours were actually used — on ordinary hardware, with a human deciding every case.

The findings **indicate** that the system works end to end, and that the evaluation is the
real result. Lighting and occupancy are confounded so tightly in this data that a rule
reading only the clock outperforms three modern backbones — and one short clip containing the
missing case reverses that ranking completely.

These findings **contribute** a set of evaluation practices that apply well beyond this
problem: run trivial baselines in every protocol, group your splits, re-draw randomness
rather than re-running it, and verify every safeguard by breaking the thing it guards. And
for the facility, they give a concrete and inexpensive recipe for onboarding a new camera.

*(pause)*

**No protocol compensates for a case the data never contains.**

> **[SLIDE: Thank you]**

Thank you for your time and attention. I welcome any questions, comments, suggestions or
feedback you may have.

---

## Delivery notes

- **Practise saying "we do not know" three times**: the human ceiling, the target hardware,
  and whether any of this transfers to a second venue. An examiner trusts a candidate who has
  bounded their ignorance more than one who has not noticed it.
- **Time the problem section with a clock.** It is the one you will want to over-run and the
  one worth protecting.
- **Your strongest sentence** is a version of *"we measured that, and here is how large it
  is."* Where that is not available — the three limits — the second strongest is *"that is a
  real limitation, and here is exactly what it costs."*
- Speak slowly on the confound slide and on the closing line. Everything else can move.

## Before the day

- [ ] Fill in **[PROGRAMME]**, **[INSTITUTION]**, and the supervisor's name.
- [ ] Fill in **[VIDEO TOOL 1]** and **[VIDEO TOOL 2]**.
- [ ] Re-read [`thesis/claims.md`](../claims.md) — every number above comes from it, and it
      is regenerated from the result artefacts. If a result moved, this script is stale.
- [ ] Rewrite at least the purpose and summary sections in your own words.
