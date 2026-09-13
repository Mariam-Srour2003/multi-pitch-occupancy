# Synthetic frame generation — protocol and prompt pack (A13)

**Status:** approved to proceed. Ethics cleared with the supervisor and the facility operator
(`thesis/ethics.md`, amendment 2026-09-13). Pre-registration amended (`preregistration.md`,
A13). Training side only, enforced by `splits.check_split`.

---

## 1 · The gap this addresses

From `results/coverage.md` and the frame counts on disk:

| class directory | frames | note |
|---|---|---|
| `1_empty` | 494 | all venue_01, 98% daylight |
| `2_playing` | 1192 | |
| `3_people_not_playing` | **6** | one moment, one camera, daylight |
| `4_maintenance` | **0** | *the four-class taxonomy has an empty class* |

`MAINTENANCE × night` has no frames at all. The four-class scheme currently cannot be trained
and cannot be reported; the three-class scheme rests on six frames.

---

## 2 · The one design decision that makes this defensible

> **Condition every generation on an EMPTY frame from `1_empty`, and ask only for the
> activity to be added.**

Three things follow from it, and they are the reasons to do it this way rather than any other:

- **Little that is identifiable is uploaded — but not *nothing*, and that was wrong when
  first written here.** `EMPTY` means the **pitch** is empty, not the **frame**. Inspection of
  `slot_20260711_1000_camA_t000029.jpg` and `slot_20260712_2030_camA_t000074.jpg` found
  people sitting in the dugout in both, including what appear to be children. Every frame
  must therefore be redacted by hand before it leaves the project — see §3a.
- **The background is genuinely yours.** Camera height, lens distortion, pitch markings,
  floodlight positions, barrier colour, grass wear — all real, all venue_01, all unchanged.
  Only the person and the equipment are generated.
- **The failure mode is bounded.** If a real-vs-generated probe later separates the sets, you
  know it is separating *people*, not scenes, which is a far easier finding to interpret and
  to report honestly.

Use the **edit / inpaint** mode, not plain text-to-image. Mask the area of pitch where the
activity should appear and leave the rest of the frame untouched. Text-to-image redraws the
whole scene and you lose the only thing that made this worth doing.

---

## 3a · Redact before uploading — the detector will not do it for you

**`vision.explain.redact_people` returns zero detections on frames that visibly contain
people.** Measured on the two frames above: 0 boxes at the default `confidence=0.25`, 0 at
0.15, and at 0.10 it finds **one** of the four people visible in the daylight frame and
nothing more down to 0.02. Its own docstring warns that "zero detections does not mean zero
people"; this is that warning coming true on the first real use.

So automatic redaction is **not** an acceptable gate for an upload, which is irreversible.
Redact by **generous hand-drawn regions** covering whole dugout and bench strips, not by
per-person boxes and not by detector output, then **look at the result** before sending it.

Also redact anything that **geolocates the venue**. The night frame carries an illuminated
hotel sign in the skyline; the facility's consent covers research use of the footage, not
publishing where the facility is.

## 3 · Frames to upload

Keep this list updated as you go. It is the audit trail the ethics amendment promises.

| # | source frame | lighting | used for | uploaded on |
|---|---|---|---|---|
| 1 | | day | maintenance | |
| 2 | | day | people-not-playing | |
| 3 | | night | maintenance | |
| 4 | | night | people-not-playing | |

Pick day frames from `1_empty` freely — there are 485. **Night is the constraint: only 9
empty night frames exist**, so choose from those deliberately and record which.

Vary the source frame across generations. Twenty images from one background is one image
twenty times over as far as a frozen backbone is concerned.

---

## 4 · Prompt pack

Preamble to paste **once** at the start of the ChatGPT conversation:

> I am going to give you frames from a fixed CCTV camera overlooking a football pitch, for a
> master's thesis on pitch-occupancy classification. For each one I want you to edit the
> image so that specific activity appears on the pitch. Critical constraints, which apply to
> every edit: do not move, rotate, zoom or re-frame the camera. Do not change the time of
> day, the weather, the lighting, the pitch markings, the surrounding buildings, fences or
> vegetation. Keep the existing image quality exactly as it is — this is a low-bitrate
> security camera, so the added content must match that softness, noise and compression, not
> look sharper than the scene around it. People should be the correct size for their distance
> from the camera and should be standing on the ground plane with their feet on the surface,
> not floating. Edit only the region I mask; leave every other pixel unchanged.

### 4a · `4_maintenance` — hi-vis and tools (the textbook case)

Generate **8-10 per prompt**, varying position on the pitch and distance from the camera.

| # | Prompt |
|---|---|
| M1 | Add one groundskeeper in dark work clothing and a hi-vis vest, walking slowly across the pitch pushing a wide **broom**. Show faint drag marks on the grass behind them. |
| M2 | Add one worker operating a **ride-on mower** on the pitch, with a visible stripe of freshly cut grass behind it running to the edge of frame. |
| M3 | Add one worker pushing a **white line-marking machine** along a pitch line, with a short section of brighter, freshly painted line behind them. |
| M4 | Add two groundstaff carrying a **folded goal net** between them across the pitch, and a small pile of equipment on the grass near the touchline. |
| M5 | Add one worker kneeling at the edge of the penalty area doing **turf repair**, with a bag of seed and a hand tool on the grass beside them. |
| M6 | Add a **small utility vehicle or tractor** parked on the pitch with a trailer, and one person standing beside it. |
| M7 | Add two workers **setting out cones and a portable goal** near one end of the pitch, with equipment stacked on the grass. |
| M8 | Add one worker dragging a **hose or sprinkler line** across the pitch, with a visibly wet patch of grass behind them. |
| M9 | Add one person using a **leaf blower** along the touchline, with debris visible in the air. |
| M10 | Add two workers on a **ladder at the goal**, one holding it steady, repairing the net or the frame. |

### 4b · `4_maintenance` — **work clothes, no hi-vis** (the real-world case)

The operative test in `labelling_protocol.md` §2.5 is **"hi-vis + tool = C3"**. On this
facility that test is too narrow: staff frequently work in ordinary clothes. A model trained
only on 4a learns *hi-vis vest*, not *work*, and will miss every real maintenance frame where
nobody wore one. **These prompts matter more than 4a.**

| # | Prompt |
|---|---|
| W1 | Add one man in a **plain t-shirt and jeans**, no hi-vis, pushing a broom across the pitch. |
| W2 | Add one person in **ordinary dark casual clothes** carrying a bucket and a hand tool, walking towards the goal. |
| W3 | Add two people in **tracksuits** dragging a heavy roller or drag-mat across the surface. |
| W4 | Add one person in a **hoodie** crouched at the edge of the pitch working on the artificial turf seam with a hand tool. |
| W5 | Add one person in **normal clothes** carrying a stack of cones and a bag of equipment across the pitch. |
| W6 | Add two people in **casual clothes** lifting and repositioning a portable goal. |
| W7 | Add one person in ordinary clothes **inspecting the pitch surface**, walking slowly and looking down, holding a clipboard or phone. |
| W8 | Add one person in plain clothes **brushing sand or rubber crumb** into the artificial turf with a stiff broom. |
| W9 | Add one person in a **plain shirt sweeping the touchline area** with a household-style broom, a wheelbarrow parked nearby. |
| W10 | Add two people **coiling a hose** at the side of the pitch, wearing ordinary summer clothes. |

### 4c · `3_people_not_playing` — people, no ball

This is the class that decides whether the system calls a person crossing a pitch a match in
progress. It matters more than the count suggests.

| # | Prompt |
|---|---|
| P1 | Add one person in ordinary outdoor clothes **walking diagonally across** the pitch carrying a bag, clearly not playing. |
| P2 | Add three or four people **standing in a loose group** near the centre circle, talking, no ball anywhere in the frame. |
| P3 | Add two people **sitting on the grass** near the touchline, and one person standing looking at a phone. |
| P4 | Add one person walking a **dog** across the pitch. |
| P5 | Add a small group being shown around - one person **gesturing towards the goal** while three others stand and watch. A site visit, not a game. |
| P6 | Add two people taking **photographs** on the pitch, one posing near the goal and one holding a camera. |
| P7 | Add one person **jogging alone** around the perimeter of the pitch, no ball, no other players. |
| P8 | Add a single person **standing motionless** near the penalty spot, looking away from the camera. |
| P9 | Add two people **walking side by side across the pitch talking**, carrying nothing. |
| P10 | Add one person **sitting alone on the grass** near the centre of the pitch looking at a phone. |
| P11 | Add a group of five or six people **standing in a circle talking**, clearly a meeting rather than a warm-up, no ball. |
| P12 | Add one adult and two children **walking across the pitch holding hands**, not playing. |

### 4d · Small group with a ball - **not a game** (contested label, see §4e)

Two to four people knocking a ball about is the hardest boundary in the whole taxonomy and
the corpus almost certainly has none of it. Generate these regardless of how the label
question settles - the frames are needed either way.

| # | Prompt |
|---|---|
| B1 | Add **two people** casually passing a football back and forth near the centre of the pitch, standing still, relaxed posture, not running. |
| B2 | Add **three people** standing in a loose triangle knocking a ball between them, in ordinary clothes rather than kit. |
| B3 | Add **one person alone** taking shots at an empty goal, no goalkeeper, no other players. |
| B4 | Add **two children** kicking a ball to each other near one corner of the pitch while an adult stands watching. |
| B5 | Add **four people** in ordinary clothes with one ball, two standing and talking, two lightly passing - clearly not a match. |
| B6 | Add **one person** juggling a football on their foot in the middle of the pitch, nobody else present. |
| B7 | Add **two people** sitting on the grass with a football resting beside them, not playing with it. |
| B8 | Add **three people** walking onto the pitch, one carrying a ball under their arm, none of them playing yet. |

### 4e · The label conflict these expose - **decide before labelling them**

**`labelling_protocol.md` says the opposite of what was described on 2026-09-13, twice.**

| | The protocol as written | As described |
|---|---|---|
| small group with a ball | §2.2 ACTIVE_PLAY "**at any number of players**"; §2.6 rule 1 "however few people are involved" | 2-4 people with a ball is **not** a game, so **not playing** |
| maintenance clothing | §2.5 "**hi-vis + tool = C3**" is "the operative test" | staff often work in ordinary clothes; the work is what counts |

These are not small. **1,192 frames are already labelled C2 under the current rule**, so
adopting a player-count threshold is a relabelling job and a pre-registration amendment, not
an edit to one sentence. And a count threshold needs a defensible number - why 4 and not 5 -
or it becomes the kind of after-the-fact cutoff this project's whole method exists to avoid.

**Until a supervisor decision:** generate 4d, save it to `data/processed/_pending_4d/`, and
label nothing. An honest holding folder is what §2.6 rule 4 already prescribes for exactly
this situation. The hi-vis question is the easier of the two and can be fixed by widening
§2.5 to "**work activity, with or without workwear**", with the 4b frames as its support.

### 4f · Night — the empty cell

Run the same prompts against a **night** source frame, adding to each:

> This is a floodlit night scene. Keep the existing floodlight positions and the pools of
> light and shadow exactly as they are. People must be lit only by those floodlights, cast
> shadows in directions consistent with them, and be darker and less distinct than they would
> be in daylight. Do not brighten the scene.

Priority order if time is short: **M1, M2, M3 at night** — that fills
`MAINTENANCE × night`, the cell the coverage report names as a claim the dataset cannot
support.

---

## 5 · Bringing them back in

1. Save to `data/processed/4_maintenance/` and `data/processed/3_people_not_playing/`.
2. **Name them so they can never be mistaken for real frames:**
   `syn_<sourceframe>_<promptid>_<nn>.jpg` — e.g. `syn_slot_20260711_1000_camA_t000029_M1_03.jpg`.
3. Manifest rows must carry `source=synthetic`, and **`venue` copied from the source frame**
   — never a new venue value. A13 condition 2.
4. Re-run `uv run pitch coverage` and confirm the generated frames appear as their own row.
5. Run `check_split` on every split you intend to use. If any reports generated frames on the
   test side, the split is unreportable — fix the split, do not fix the message.

---

## 6 · The check that decides whether any of this is usable

Pre-declared in A13, before the frames exist:

> Train a probe on frozen features to classify **real vs generated**. If it reaches
> macro-F1 > 0.90, the generated frames are a distinguishable distribution rather than an
> augmentation of this one, and the augmentation is withdrawn.

Run it first, before any classifier is retrained. If it fails, you have lost an afternoon and
gained a genuine methodological finding worth a paragraph in the thesis — which is a better
outcome than a silent contamination.

---

## 7 · What this does not fix

Stated here because the temptation to believe otherwise is the whole risk:

- **Empty pitches at a second venue** — test-side need, must be real. RQ1 stays blocked.
- **Complete slots with a real verdict** — still n=2. The operational contribution stays
  untested.
- **The two production confidence thresholds** — still unsettable.

Items 1 and 2 of `thesis/data_requests.md` are unchanged in priority by anything in this
document. This makes the third and fourth classes *trainable*. Only real footage makes them
*evaluable*.
