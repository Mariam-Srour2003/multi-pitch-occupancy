# Why computer vision, and not something cheaper (WP1-T6)

*Draft for Chapter 1. One page in the thesis; the reasoning is here in full so the page can be
cut from it rather than composed under pressure.*

**"Why do you need computer vision at all?" is the most dangerous question in the defence**,
because the honest first answer — *a motion sensor is €20 and this is not* — is correct as far
as it goes. The argument below concedes that and then says what the €20 buys and what it does
not.

---

## What the system is actually for

Not "is anyone on the pitch". That framing is what makes the cheap alternatives look
sufficient, and it is the wrong problem.

The deliverable is a **reconciliation**: the booking system holds what *should* have happened,
staff record what they *say* happened, and the cameras observe what *did*. The output is a
per-slot verdict — `USED` / `NOTUSED` / `REVIEW` — **bound to three evidence images a human
can look at**, so that a disputed slot is settled by someone looking at a picture rather than
by trusting one of the two records that disagree.

Every alternative below can answer *was there movement*. The question is which can produce
something a person can adjudicate, and which fail in ways that matter for an audit.

## The alternatives

| | hardware cost per pitch | what it measures | how it fails | retrofit |
|---|---|---|---|---|
| **PIR / motion sensor** | ~€20 | motion in a cone | rain, wind-blown netting, foxes, staff crossing; no evidence trail | easy, but needs power and a mount per pitch |
| **Door counter / turnstile** | €300–2,000 | people through a gate | multi-pitch venues share an entrance; nobody counts *out*; a group of ten entering says nothing about which pitch | hard: civil works |
| **Floodlight power draw** | ~€60 (clamp meter) | lights on | daylight slots invisible; lights left on; lights on for an adjacent pitch | easy where a per-pitch circuit exists — it usually does not |
| **App check-in** | ~€0 marginal | that someone tapped a button | measures compliance with the app, not use of the pitch; the no-show is precisely the case where nobody taps | easy, and it is the alternative most likely to be proposed |
| **Manual logging** | staff time | what staff wrote down | **it is one of the three records being audited**; using it as ground truth assumes what the project is testing | none needed |
| **This system** | reuses existing CCTV | scene state per minute, with evidence | needs a camera view; classification error; see the whole evaluation | **the strongest single argument — see below** |

## The four arguments that survive scrutiny

**1 · The cameras are already there, and the alternatives are not.** This is the decisive
practical point. The facility has CCTV for security; the marginal hardware cost of this system
is **zero**, against €20 × *n* pitches plus mounting, power and maintenance for sensors that
do nothing else. A €20 sensor is only cheap before someone has to install, power and maintain
one per pitch across a network of venues — and then keep them working.

**2 · A motion sensor cannot produce evidence, and an audit needs it.** The system's output
implicates a booking record and sometimes a member of staff. A binary "motion at 10:07" cannot
be adjudicated: a person disputing it has nothing to look at. Three evidence frames can be,
and the design requires a human to confirm every anomaly (`slots/authority.py`). **The
evidence trail is not a feature added to the classifier — it is the reason a classifier is
used instead of a sensor.**

**3 · Only vision distinguishes the states that matter.** The taxonomy is not
occupied/unoccupied. It is *empty* / *active play* / *maintenance or non-sporting*, and the
third is where the money is: a pitch with two groundstaff on it is occupied, has motion, and
is **not a used booking**. A PIR fires identically for a five-a-side match and a mower. That
distinction is the operational contribution, and no cheap sensor can make it.

**4 · Bandwidth, which is what makes it deployable at all.** Continuous decoding of 20–30
streams is 60–90 Mbps and a GPU. Sampling one frame per camera per minute is under 1 Mbps on a
CPU-only mini-PC — roughly a **99% reduction**, and the reason this runs on hardware a
facility already owns rather than a server it would have to buy. The engineering contribution
is as much the sampling design as the classifier.

## What the alternatives are genuinely better at

Conceding this is what makes the rest credible.

- **A PIR is more robust.** It works in fog, in darkness, with a dirty lens, and at a camera
  angle that shows nothing useful. This system's own quality filter exists because frames go
  bad; a sensor does not have that failure mode.
- **A turnstile counts people; this does not.** The system classifies scene state, never
  identities and never headcount, deliberately — see `thesis/ethics.md`.
- **App check-in is instant and free.** Where compliance is high it is a better answer, and
  the honest position is that this system is for facilities where it is not.
- **A power clamp is nearly unfalsifiable.** If a venue has per-pitch floodlight circuits and
  only ever books at night, it is a better instrument than any of this.

## The one-sentence answer

> A motion sensor answers *was something moving*. The audit needs *was this booking used, and
> here is the picture* — and it has to tell a match from a mower. The cameras are already
> installed, so the marginal hardware cost is zero and the marginal bandwidth is under 1 Mbps.

## Where this is weakest

Recorded here rather than discovered at the examination.

- **The comparison is argued, not measured.** No PIR was deployed alongside the cameras, so
  the failure rates in the table are from the literature and from reasoning, not from this
  facility. A one-week side-by-side would settle it and has not been done.
- **"The cameras are already there" is a property of this client**, not of the problem. At a
  venue without CCTV the cost argument reverses, and the honest scope is *facilities with
  existing camera infrastructure*.
- **The evidence-trail argument assumes the evidence is usable.** It rests on the frames
  showing enough to adjudicate — which the quality filter exists to check and which the
  cross-venue evaluation shows is not uniform across sites.
