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

- **Nothing identifiable is uploaded.** Empty-pitch frames contain no people, so the
  narrowest possible reading of the facility's consent is satisfied without relying on the
  approval at all.
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

### 4a · `4_maintenance` — the empty class, highest priority

Generate **8–10 per prompt**, varying position on the pitch and distance from the camera.

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

### 4b · `3_people_not_playing` — the harder half

This is the class that decides whether the system calls a person crossing a pitch a match in
progress. It matters more than the count suggests.

| # | Prompt |
|---|---|
| P1 | Add one person in ordinary outdoor clothes **walking diagonally across** the pitch carrying a bag, clearly not playing. |
| P2 | Add three or four people **standing in a loose group** near the centre circle, talking, no ball anywhere in the frame. |
| P3 | Add two people **sitting on the grass** near the touchline, and one person standing looking at a phone. |
| P4 | Add one person walking a **dog** across the pitch. |
| P5 | Add a small group being shown around — one person **gesturing towards the goal** while three others stand and watch. A site visit, not a game. |
| P6 | Add two people taking **photographs** on the pitch, one posing near the goal and one holding a camera. |
| P7 | Add one person **jogging alone** around the perimeter of the pitch, no ball, no other players. |
| P8 | Add a single person **standing motionless** near the penalty spot, looking away from the camera. |

### 4c · Night — the empty cell

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
