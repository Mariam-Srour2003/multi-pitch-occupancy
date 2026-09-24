# Speaker script — 15 minutes

**This script follows the site.** Start the server, open <http://127.0.0.1:8000>, and move
through the eight tabs in order.

```bash
uv run pitch serve
```

Headings below are `Tab.Block` and repeat each block's own heading, so you always know where
on the page you are. The site carries **no** delivery notes — this file is the only place
the talk is written down.

**Read only the plain paragraphs aloud.** Blockquoted boxes are prepared answers for
questions. They are not part of the talk.

Things still yours to fill in are marked **[LIKE THIS]**.

| Tab | Starts | Runs |
|---|---|---|
| 1. Summary | 0:00 | 3:00 |
| 2. Plan | 3:00 | 0:30 |
| 3. Introduction | 3:30 | 1:00 |
| 4. Ch.1 State of the Art | 4:30 | 1:30 |
| 5. Ch.2 Methods & Model | 6:00 | 3:00 |
| 6. Ch.3 YOLOv8 | 9:00 | 2:30 |
| 7. Ch.4 Applications | 11:30 | 2:00 |
| 8. Conclusion | 13:30 | 1:00 |

**About the timing.** The spoken text is **1,948 words** — counted, not estimated. That is
**13:55** at 140 words a minute and **14:59** at 130, so it fits 15 minutes only if you do
not stop to add explanations. When a slide has a picture on it, let the picture do the work
and keep to the words here. Time yourself once. If you run long, the two cuts that cost
least are 4.2 and 7.3.

---

# Tab 1 — Summary · 0:00

### 1.0 · Opening

Good morning. My name is Mariam Srour. I am a Master's student in **[PROGRAMME]** at
**[INSTITUTION]**. My supervisor is **[SUPERVISOR NAME]**.

My work is called *Playground Activity Detection using Deep Learning*.

### 1.1 · "The abstract"

Let me start with the problem.

Think of a man who owns seven football playgrounds. He wants to know which pitches are
being used and which are empty. He cannot sit and watch seven cameras all day. And he is
not alone. Lebanon has around one thousand two hundred football playgrounds with the same
problem.

Some of them sell their hours online. But a booking only says the hour was **sold**. It
does not say anyone came.

So this work uses deep learning to answer the question by itself. The cameras are already
on the wall. We take **one picture per camera per minute**. We ask a model what it sees:
an empty pitch, people playing, or maintenance. We turn sixty of those answers into one
answer for the hour, and compare it with the booking sheet.

The owner stops watching cameras. He only sees the hours where the sheet and the camera
**disagree**. A person always checks before anything happens.

And it all runs on a normal **CPU**. No GPU, no cloud bill. That was a choice, not a
limit. It keeps the price near the price of the cameras he already owns.

> **If asked where 1,200 comes from:** give the source, or say plainly that it is an
> order-of-magnitude figure you have not been able to verify. Do not invent a precise
> number in the room.

### 1.2 · "The study in numbers"

The size of the study, so nobody has to guess. One thousand six hundred and ninety-two
labelled pictures. Three classes. Seven venues for testing. And zero GPUs.

### 1.3 · "Three findings"

Now the three things I found. The third one is about my own work.

**First.** In this footage, people play in the evening and the pitch is empty in the day.
So *dark means busy, daylight means empty* is right ninety-nine per cent of the time. I
wrote that rule as a test. It only reads the clock. It never looks at the picture. It
matched all three deep models. At venues they had never seen, it beat them. So a good
score on this data does not prove the model can see a pitch.

**Second.** I filmed four minutes of a floodlit pitch at night with nobody on it. That is
the one situation my dataset never recorded. I took sixteen moments from it. The clock
rule said "a match is happening" sixteen times out of sixteen, and was wrong every time.
The full system raised no false alarm at all. Four minutes of video turned over a
benchmark of one thousand six hundred pictures.

**Third.** I published an image-augmentation result of zero point eight five five. Then I
ran it four more times and changed nothing but the random number. The other four came out
between zero point three five and zero point four one. Using no augmentation at all gives
zero point four four. So my number was simply the best of five. I withdrew it the same
day.

---

# Tab 2 — Plan · 3:00

### 2.1 · "The eight parts"

Eight parts. A summary, this plan, an introduction, four chapters, and a conclusion.

Chapter one is what already exists. Chapter two is the model that reads the picture.
Chapter three is the detector that counts people. Chapter four is what happens in real
conditions, and what I did about the data.

---

# Tab 3 — Introduction · 3:30

### 3.1 · "The problem in general"

Six questions, and today nobody can answer them without a person watching.

Was this hour booked? The system shows the booking, but not what actually happened on the
pitch. Did they actually come? A customer may book the pitch and never show up. Did they
actually play? People may be on the pitch without actually playing a game. Is the pitch
empty right now? The owner cannot know unless someone checks the pitch or the camera. Did
someone play without booking? Then the owner loses money on an hour the facility gave away. And
did the staff record it correctly? Manual records can be entered incorrectly, or
forgotten.

All six have one honest source: the cameras the facility already owns.

---

# Tab 4 — Ch.1 State of the Art · 4:30

### 4.1 · "Playgrounds — how occupancy is measured today"

This table is how facilities measure occupancy today, and how each one breaks.

A motion sensor costs about twenty euros, and rain sets it off. So does wind in the
netting, or a member of staff walking past. A door counter is expensive, and several
pitches share one entrance. Floodlight power tells you the lights are on, not that anyone
is playing. An app check-in measures who tapped the button — but the no-show is exactly
the person who does not tap. And manual logging is the sheet we are trying to audit.

My system reuses cameras that already exist. It needs a view of the pitch, and it can get
the class wrong. Those limits are on the table too.

### 4.2 · "Deep learning — what the field must establish"

On the research side I read nine areas. Three matter most here. Leakage, because
near-duplicate pictures make a benchmark measure memory instead of skill. Trivial
baselines, because testing a benchmark with a rule that ignores the picture is established
practice. And frozen features, because using a pretrained model without training it is a
recognised method, not a shortcut.

---

# Tab 5 — Ch.2 Methods & Model · 6:00

### 5.1 · "The families considered"

I looked at four families. A modern convolutional network, ConvNeXtV2. A vision
transformer, ViT. A self-supervised transformer, DINOv2. And vision-language models like
CLIP, used with written descriptions and no training.

### 5.2 · "Why the backbones are frozen"

This is the most important choice in the chapter, so let me be clear about it.

I do not train the big model. I freeze it. Here is why.

My data has about one hundred and fifty genuinely different scenes. If I update two
hundred million parameters on that, the model does not learn football. It learns *these
pitches*, under *these floodlights*.

And I cannot simply collect more. This is video of real people, so every new venue is a
permission conversation, not a download.

My own numbers agree. Ninety-eight per cent of my pictures have a near-twin in the set.
And when I raised the labels from three hundred to six hundred and seventy-one, accuracy
went **down**.

So freezing is protection against overfitting first, and speed second.

### 5.3 · "DINOv2 — what actually runs"

This diagram is the whole model.

DINOv2 was trained on one hundred and forty-two million images with no labels. It has no
idea what an empty pitch is. It has no classifier at all. What it produces is a
description of a picture.

The picture is cut into small squares. Twelve transformer layers let every square see
every other square. Out comes one list of seven hundred and sixty-eight numbers.

Then, and only then, a decision. Those numbers go into a small logistic regression with
three outputs. That regression is the only thing I train. Everything in the dashed box is
downloaded and never touched.

Look at the ratio. Eighty-six million frozen parameters. Two thousand three hundred
trained ones. Thirty-seven thousand frozen for every one that moves.

### 5.4 · "The full model comparison"

This table compares them. Every column says what the number is, and whether higher or
lower is better.

Look at the first column. On a shuffled split everything scores about ninety-nine per
cent, including the clock rule. That column tells you nothing.

Now the columns that matter. DINOv2 leads on the grouped split, at zero point five eight.
It finds ninety-three per cent of play at a venue it has never seen.

But look at false-play — how often a model calls an **empty** pitch a match. ConvNeXtV2 is
at zero point nine nine. It says "playing" to almost everything. Its high recall was
bought, not earned. ViT is at zero point eight four. DINOv2 is at zero point three one.

The last column, balanced, is recall minus false-play. DINOv2 is the only backbone with a
healthy number there. That is why it is the one I deploy — and it costs two hundred and
nineteen milliseconds a frame, which on a one-minute cycle does not matter.

> **If asked whether the lead is significant:** say no. The intervals overlap and a paired
> test comes back inconclusive. Something has to be deployed and this is the best
> estimate, but it is not a measured gap.

---

# Tab 6 — Ch.3 YOLOv8 · 9:00

### 6.1 · "The architecture — one forward pass"

The classifier says what the scene looks like. The detector says how many people are
standing on it.

This is YOLOv8, in one pass, with COCO weights. Nothing here is trained by me. I ask it
for two classes only: person, and sports ball. It ends at boxes. It is never asked whether
a match is happening. That is decided afterwards, by counting — which means I can show a
manager the objects the verdict was made from.

### 6.2 · "The three cues, and what each is allowed to decide"

Three cues.

People, counted by the **feet**, not the middle of the box. Someone at the touchline has
their body over the pitch and their feet outside it, and the feet are the truth. At my
main venue an empty pitch has zero people in eighty-nine per cent of pictures. A match has
a median of six.

Motion — did anything change since the last picture. It says something moved. It never
says who.

And the ball. It must appear in at least two of three quick frames, because a false ball
appears once — a bright shoe stud, a bin lid, a painted line. And it must move more than
its own width. A ball lying on the grass while three people work around it is
furniture, not a game.

### 6.3 · "One direction only"

This is the rule I am most confident about.

The cues can turn a "playing" verdict into **empty**. They can never turn empty into
playing. Finding nobody is strong evidence against a match. Finding somebody is not
evidence for one.

### 6.4 · "From boxes to a verdict — the rule table"

And here is the whole decision, in seven lines.

If the detector is missing, the answer is uncertain — not empty. If there is no pitch
boundary, uncertain. Nobody and nothing moving: empty. Four people or fewer: not a game.
There is one way, and only one way, to reach "active play": five or more people, a ball in
play, and motion.

Five and four are the facility's numbers, not numbers I fitted. Five make a game. Four do
not.

---

# Tab 7 — Ch.4 Applications · 11:30

### 7.1 · "Preprocessing — six switches, before and after"

Preprocessing code fails quietly. A switch that does nothing, or a crop that removes the
goal, still passes every shape check. The only real check is a person looking. So here are
six switches, before and after, on the same frame.

Each one says how much of the picture it actually moved. Cutting the sky moves twenty-one
points out of two hundred and fifty-five and throws away a third of the frame. And the
blur at the bottom has destroyed the people — which is the point of showing it.

### 7.2 · "The augmentation argument in full"

Now the argument of this chapter, and it is one sentence.

Preprocessing **removes** information, forever. Augmentation **varies** it and keeps every
pixel at the end.

Look at the left side. Grayscale helps a little. But grayscale plus a fifty per cent crop
falls **below** the untouched frame. That is the floor. Once you have thrown information
away, no amount of tuning brings it back.

Now the right side. Augmentation makes several views of the same frame, and the model
trains on all of them. At the end it still sees the original. Nothing was discarded, so
there is no floor to cross.

### 7.3 · "The preset sheets"

And these are the sheets I actually look at. Every preset, three draws each, on a night
match and an empty day pitch. Then one effect at a time.

This is the check that caught a real bug. My rain was written in fixed pixels instead of a
share of the frame, so it drew white poles across the picture. No shape test would have
found that. A person looking at the sheet found it at once.

---

# Tab 8 — Conclusion · 13:30

### 8.1 · Closing

Let me finish with what I claim and what I do not.

I claim a working system that runs on a CPU, reuses cameras a facility already owns, and
gives a manager a short list of hours to check instead of seven screens to watch.

I do not claim my numbers prove the model sees a pitch. A rule that only reads the clock
matched my deep models on my own benchmark. That is a fact about my data, and I put it on
the first page of this talk rather than in a footnote.

What would settle it is empty-pitch footage from more than one venue. That is the next
piece of work, and it is a permission problem more than a research one.

Thank you. I am happy to take your questions.

### 8.2 · "Questions"

> Take questions here. When they are finished, do not go back to a slide — run the system
> live on a frame instead.
