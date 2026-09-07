# THESIS MASTER TODO — Multi-Pitch Occupancy & Booking Verification

> **What this file is.** The single checkable work list for the whole thesis. It merges
> `SUPER_PLAN.md`, `Thesis_Plan_Multi-Pitch_Occupancy (4).docx`, and
> `Thesis_Schedule_and_Effort_Plan.docx`, plus a set of additions that close gaps found while
> reviewing all three. Tick boxes as you go. Detail/rationale for the original tasks lives in
> `SUPER_PLAN.md`; this file is what you *work from*.
>
> **Legend:** `★` = new, not in any of the three source documents. `[H]` = only a human can do it
> (you, supervisor, or client). `[B]` = blocking — something else waits on it.

---

## 0. Do this week (critical path — longest lead times first)

These five come before everything else because they either protect work already done or start an
external clock you don't control.

- [x] **0.1 ★ Ethics — settled.** The facility operator supplied the footage and holds the approval
      for its collection; the data is never published or shared. Recorded in `thesis/ethics.md`,
      along with the consequences: **no dataset release** (so it comes out of the contributions
      list, or becomes features-only), faces blurred in figures, retention enforced in code.
      Two items remain open in that file: **(a)** one question to the supervisor confirming the
      programme doesn't need its own sign-off for *research use* of operator-supplied footage;
      **(b)** the WP1-T4 decision on per-staff reporting — recommendation is to report anomalies
      **per field only**, which removes the workplace-monitoring exposure at no scientific cost.
- [ ] **0.2 ★ [B] Back up `data/`.** Structure work is **finished**, so this is unblocked — do it
      now. 4.2 GB. External drive **plus** a cloud folder (two places, not one), then open a file
      from each copy to confirm the restore actually works.
- [x] **0.3 Footage request — closed; working with what we have.** The 66 clips landed: ~9 venues,
      day and night, all active play. Decision taken not to request more. The consequences are
      pinned down in `thesis/preregistration.md` §"Not answerable" — cross-venue *three-class*
      evaluation is impossible (no empty pitch exists outside venue_01), C3 stays unclaimable, and
      STAN can only be reported as preliminary on synthesised sequences. Scope reduced deliberately
      and documented, rather than discovered at the defence.
- [x] **0.4 ★ Final test set locked.** 66 clips grouped into 9 venues
      (`configs/clip_venues.csv`); `clipvenue_b_floodlit_track` + `clipvenue_c_teal_boards`
      (19 clips, 29%) held out in `results/splits/FINAL_TESTSET_venues.csv`, chosen by venue
      identity with a fixed seed **before any model was fitted**. Not evaluated until the end.
  - [x] **[H] Venue grouping confirmed** (2026-09-06, visual audit; sheets in
        `results/figs/venue_check/`, findings in `EXPERIMENT_LOG.md`). `cg` is one facility
        (adjacent pitches 4+5 — the pitch-4 camera sees pitch 5's sign in frame), raised to
        `high`; `ch` plausibly one venue, raised to `medium`; `cg` and `ch` confirmed
        *different* facilities, and no two of the nine groups are secretly one.
  - [x] **Sensitivity check closed the residual doubt.**
        `experiments/h3_sensitivity_merged_venues.py` reruns H3 with `cg`+`ch` merged into a
        single fold — the pessimistic assumption. **H3 survives**: DINOv2 0.967
        [0.932, 0.993], ConvNeXtV2 0.927 [0.843, 0.989], clock rule still collapses at
        0.243. The grouping can no longer change the H3 conclusion, so the worst-fold
        caveat is safe to quote. Main H3 stays the primary number; this is the robustness
        check (the higher means are partly fold arithmetic — see the log).
  - [x] Lock enforced in code (WP0-T4). Every split strategy drops locked-venue rows; reaching
        them requires `final_test_rows(..., i_have_finished_all_development=True)`, which is
        deliberately awkward and greppable — one call site, at the end.
- [x] **0.5 ★ Analysis plan pre-registered.** `thesis/preregistration.md` — six hypotheses with
      primary metric, test, correction and decision rule; standing rules (macro-F1 leads, CIs on
      everything, Holm correction, effect sizes, report either direction); an explicit
      **"not answerable with this data"** section; and the locked test set with its amendment
      disclosed. Committed and dated.

---

## 1. Research questions & traceability

The RQs exist in the thesis-plan docx but were dropped from `SUPER_PLAN.md`, and no experiment is
tagged with the question it answers. Fix that first — it is what turns a build log into a thesis.

- [x] **1.1 RQ1-RQ5 copied into `SUPER_PLAN.md` §0.5**, with RQ6 and RQ7 added.
- [x] **1.2 ★ RQ traceability matrix** → `thesis/rq_matrix.md`. RQ1, RQ2, RQ3 and RQ7 now have
      evidence; RQ4, RQ5 and RQ6 are open. No experiment is unmapped. **It already earned its
      keep: RQ2's answer overturns the pilot's production choice** — latency is not the binding
      constraint, so the pick falls to accuracy under honest evaluation, which is DINOv2.
- [x] **1.3 ★ RQ6 added.** The current five RQs never ask the *operational* question a
      facility manager actually cares about: *"How much human review must be budgeted to reach a
      given verdict reliability?"* Phrase as: **RQ6 — What is the achievable trade-off between
      automated-verdict precision and REVIEW rate, and where is the operating point that maximises
      recovered revenue per hour of human review?** This is answered by WP4-T9 and WP6-T10 below and
      is one of the more publishable angles in the whole project.
- [x] **1.4 ★ RQ7 added — and answered.** Conditionally: *within* a confounded venue the deep
      backbones do **not** earn their cost (a clock rule matches two of three; a colour histogram
      beats them). *Across* venues they emphatically do. Exactly the question a sharp examiner
      asks first, now with a two-sided answer.

---

## WP0 — Repo & rigour hardening
*Weeks 1–2 · ~40 h · feeds M1*

### 0.A Version control & safety
- [x] **WP0-T1 Git init.** Repo on GitHub (`multi-pitch-occupancy`, private) with the pilot
      preserved on its own branch. `.gitignore` anchors `/data/` to the root - unanchored it also
      swallowed the `src/pitch_occupancy/data/` package.
- [ ] **WP0-T8 ★ Backup policy documented.** Formalise 0.2 into `docs/backup.md`: what is backed up,
      where, how often, and the last verified restore date. Re-verify monthly.
- [x] **WP0-T9 ★ Environment pinning.** `uv.lock` + `.python-version` committed. torch pinned to
      the **CPU wheel index** on every platform, so a CUDA build cannot silently invalidate a
      reported latency. scikit-learn and scipy pinned to 1.8.0 / 1.16.3: Windows Smart App Control
      blocks the newer builds' native extensions outright on this machine.

### 0.B Data plumbing
- [x] **WP0-T2 Dataset manifest.** `pitch_occupancy/data/manifest.py`, `uv run pitch manifest`.
      **1,692 rows over 9 venues**, handling both slot recordings and clip frames. The class folder
      is authoritative; `labels.csv` supplies provenance only, so the 238 bulk-filed frames are
      marked `labeled_by=bulk` rather than counted as hand-labelled.
  - [x] ★ `slot_id`, `quality` and `split_role` columns added. `slot_id` deliberately omits the
        camera: both cameras of a pitch see the same scene at the same moment.
- [x] **WP0-T3 Taxonomy layer.** `pitch_occupancy/data/taxonomy.py` - 4 folders in, 3 reporting
      classes out, with a test that fails if the two-into-C3 collapse is ever made one-to-one and
      the 4-class ablation quietly disappears.
- [x] **WP0-T4 Split module.** `pitch_occupancy/data/splits.py`: `grouped_split`,
      `leave_one_group_out`, `temporal_split`, and a deliberately leaky `random_split` kept only
      as H1's control arm. Materialised to `results/splits/`, referenced by name, never
      re-randomised. 16 tests.
  - [x] ★ `temporal_split` added — drift was listed as a risk but never measured.
  - [x] ★ Final-venue lock enforced in code, not discipline (see 0.4).
  - [x] ★ `check_split()` reports what would make results misleading: group overlap, duplicate
        frames, a class present in train but absent from test, and a near-single-class test set.
        **On the current data it fires immediately:** the honest grouped split yields a test set
        that is 99% ACTIVE_PLAY with C3 absent, while the leaky random split looks clean. That
        contrast is H1's evidence, and it is now produced by the tooling rather than asserted.
- [x] **WP0-T5 Feature cache.** `data/feature_cache.py` + `vision/backbones.py`,
      `uv run pitch cache`. Pooling is explicit and **stamped**; `load_cache` refuses anything not
      mean-pooled, so the pilot's `pooler_output` bug cannot recur silently. Preprocessing
      fingerprint refuses cross-preprocessing comparison; uncached frames are dropped and reported,
      never zero-filled.

### 0.C Measurement & bookkeeping
- [x] **WP0-T6 Stats utilities.** `evaluation/stats.py` + `evaluation/metrics.py`. Bootstrap CIs,
      metric-level bootstrap for macro-F1, paired bootstrap for slot level, McNemar with an exact
      binomial regime for small discordant counts, Cohen's g, Holm-Bonferroni. Metrics name
      zero-support classes as **absent** rather than folding an undefined F1 into the headline.
  - [x] ★ Holm-Bonferroni implemented and applied in the H1/H2 experiment.
  - [x] ★ Effect size beside every p-value. Already earning its keep: H2's clock-rule gap is
        significant at p_holm 1.2e-07 and negligible in size (0.007 macro-F1).
- [x] **WP0-T10 ★ Latency-measurement harness — done and run.** `evaluation/latency.py`.
      Median/p95, warm-up discarded, threads recorded. **All three backbones fit the 60 s
      cycle with 10-24x headroom**, so latency is *not* the binding constraint and the model
      choice falls to accuracy. Concurrency proved *faster* than naive 20x extrapolation, the
      opposite of the expectation the harness was built to test.
- [ ] **WP0-T10b Repeat on the target Mini-PC** (WP7-T1). Nothing above settles deployment.
- [ ] ~~WP0-T10 original~~ `tools/bench_latency.py`: discard warm-up runs,
      N≥50 reps, report **median and p95** (not mean), declare thread count, pin CPU affinity,
      measure with nothing else running. The pilot's ms/frame numbers were probably measured
      casually — the whole "20–30 cameras on one Mini-PC" claim rests on them.
  - [ ] ★ **Concurrency test, not multiplication.** Measure 20 cameras *actually running
        concurrently*, not `20 × single-frame latency`. Memory-bandwidth contention on a Mini-PC
        makes those two numbers different, and the honest one is the one you must report.
- [x] **WP0-T7 Experiment log.** `results/EXPERIMENT_LOG.md`, pilot backfilled, appended
      automatically by every experiment script.
- [ ] **WP0-T11 ★ Reproduction script.** `make reproduce` (or `tools/reproduce_all.py`) that
      regenerates **every table and figure** in the thesis from the manifest + cached features.
      *Accept:* runs clean from scratch. Doubles as insurance against the laptop-sleep risk, since
      every stage is resumable and cached.

---

## WP1 — Literature, protocol & clearances
*Weeks 1–4 · ~70 h · M1 gate*

### 1.A Clearances (do first — they block WP2)
- [ ] **WP1-T3 ★ [H][B] Ethics/DPIA outcome documented** → `thesis/ethics.md`: legal basis for
      recording, retention periods, signage/notification of players and staff, whether a DPIA is
      required and (if so) its completion, works-council/HR position on the staff-audit feature.
- [ ] **WP1-T4 ★ [H] Staff-audit framing decision.** Decide *with the client* whether per-staff
      discrepancy reporting is in scope or whether reconciliation reports only per-field anomalies.
      This is an ethics question and a client-relationship question before it is a technical one.
      Whatever you decide, the thesis needs a paragraph defending it.
- [ ] **WP1-T5 ★ [H] Data-release decision.** The plan lists "a documented dataset" as a
      contribution — but footage of identifiable people at a client site is very likely **not
      releasable**. Decide now between: (a) release **cached embeddings + labels + manifest, no
      pixels** (GDPR-friendly, still fully reproducible for anyone rerunning your heads — a
      genuinely good answer), (b) release a small blurred/ROI-cropped subset, (c) no release, and
      remove "dataset" from the contributions list. Do not claim a contribution you cannot deliver.

### 1.B Literature
- [ ] **WP1-T1 Related-work skeleton** → `thesis/ch2_related_work.md`, the 7 strands from the thesis
      doc §2.1–2.7, 3–5 bullet claims + candidate citations each, placeholders marked `[CITE]`.
  - [ ] ★ **Strand 8 — prior art / competitors.** Search for existing commercial and academic
        sports-facility occupancy systems. The §2.7 "research gap" claim is exposed until you have
        actually looked. If a product already does this, your gap statement must narrow to the parts
        it does not do (CPU-only, leakage-free evaluation, reconciliation) — which is fine, but you
        must say it deliberately rather than be told it at the defence.
  - [ ] ★ **Strand 9 — selective prediction / learning-to-defer.** Your REVIEW band *is*
        selective prediction with a human fallback. There is a literature for it, and citing it
        upgrades REVIEW from an engineering hack to a principled design choice (supports RQ6).
- [ ] **WP1-T6 ★ Alternative-solutions analysis** → one page in Chapter 1. Why computer vision
      rather than: PIR/motion sensors, door counters/turnstiles, floodlight power draw, app
      check-ins, or manual logging? Compare on cost, accuracy, failure modes, retrofit difficulty.
      **"Why do you need CV at all?" is the single most dangerous question you can be asked**, and it
      is trivially cheap to answer in advance.

### 1.C Protocol (M1 artefact)
- [ ] **WP1-T2 Protocol document** → `thesis/protocol.md`: the 3-class taxonomy with labelling rules
      (people outside ROI don't count; goalkeeper-only during live match = ACTIVE_PLAY; academy =
      ACTIVE_PLAY; coach carrying gear = C3; hi-vis + tool = C3), split definitions, metrics, seeds.
  - [ ] ★ Add **slot-level labelling rules**, not just frame-level: what makes a real slot USED when
        players arrive 20 min late, when two slots run back-to-back and players overlap the
        boundary, when a slot is booked but used for a birthday party. STAN is trained on slot
        labels, so ambiguity here propagates straight into your headline novelty.
  - [ ] ★ Add the **decision rule for ties/uncertainty** the annotator follows, so κ is measuring
        genuine disagreement rather than missing instructions.
- [ ] **M1 GATE [H]** — protocol, taxonomy, evaluation design approved by supervisor. *By week 4.*

---

## WP2 — Data collection & labelling
*Weeks 3–9 · ~150 h · M2 gate · **the critical path — protect this package***

### 2.A Know what you're missing
- [x] **WP2-T1 Coverage tracker.** `uv run pitch coverage` -> `results/coverage.md`. Absent cells
      render as `-` rather than `0`, and a **concentration table** reports each class's largest
      venue and lighting share - the check that predicts a degenerate split before one is built.
      It currently reads EMPTY 100% venue_01 / 98% day.

### 2.B The collection request
- [ ] **WP2-T2 [H] Collection request doc** → `thesis/data_requests.md`. Ask for:
  - [ ] Maintenance windows (brooming, line-painting, mowing, seeding), day **and** night — currently
        **0 frames**, and this is the class that makes macro-F1 meaningful.
  - [ ] Idle-people slots (walk-throughs, events, photo sessions) — currently **6 frames**.
  - [ ] ≥ 2 more pitches at the same facility.
  - [ ] ≥ 2 external venues, 2 slots per lighting regime.
  - [ ] Rain/fog days when they occur.
  - [ ] ★ **≥ 30–40 distinct complete slots with a known real verdict.** This is the single most
        important line in the whole request. **Frames are not your bottleneck — slots are.** STAN
        (your headline novelty, WP5-T1) is a *slot* classifier, so its real-world test set is
        currently **n = 2**. No statistical test survives n=2, and a committee will say so in the
        first five minutes. Slot labels are also nearly free to produce: one label per hour of
        footage, from the facility's own booking sheet.
  - [ ] ★ Ask for the **booking export at the same time** (WP6-T4) rather than in a second request
        three months later — same person, same conversation, half the waiting.
- [ ] **WP2-T8 ★ [H] Slot-verdict ground truth.** For every recorded slot, capture the true
      USED/NOTUSED/REVIEW verdict from facility records or a human skim → `data/slot_labels.csv`.
      *Accept:* ≥ 30 real slots labelled before WP5-T1 reports headline numbers.

### 2.C Ingest & quality
- [x] **WP2-T3 Ingest loop — clips batch done.** `uv run pitch extract-clips` → 396 frames from
      66 clips (6 per clip, middle 80%), venue assigned from `configs/clip_venues.csv`, lighting
      **measured from pixels** because clip filenames carry no timestamp. Metadata lands in a
      sidecar (`data/interim/clip_frames.csv`) that the manifest joins on. Manifest now holds
      **1,692 frames over 9 venues**; ACTIVE_PLAY's confound warning has cleared. Filed as
      `labeled_by=bulk` — verified per frame with YOLO plus visual review of the outliers, not
      hand-labelled one by one.
  - [ ] **[H] Spot-check the 396 clip frames** before they back a headline number. They are in
        `data/processed/2_playing/` as `clip_*.jpg`.
- [ ] **WP2-T3b Ingest loop for future batches.** Same path for any further footage.
- [ ] **WP2-T4 De-duplication.** Perceptual-hash near-duplicate pass; drop near-identical
      consecutive frames within a class. *Accept:* duplicate rate reported per batch.
- [ ] **WP2-T5 Double-labelling & κ.** 10% sample → second annotator, blind → `tools/kappa.py`
      computes Cohen's κ, logs disagreements → resolve, amend `protocol.md`. *Accept:* κ ≥ 0.85.
  - [ ] **WP2-T9 ★ Human ceiling on the test set.** While the second annotator is labelling, have
        them label the **held-out test set** too. Report human-vs-consensus accuracy as the ceiling.
        This is what makes "98.1%" interpretable: if humans hit 97% on the same frames, your model is
        at ceiling and further accuracy chasing is wasted effort — and *that* is a defensible finding.
        Costs almost nothing on top of the κ work you are already doing.
- [ ] **WP2-T6 Screenshot rescue (optional).** Crop/inpaint the red burned-in labels from the 14
      `data/ss data/` screenshots → held-out qualitative set. *Accept:* no legible label text remains.
- [ ] **WP2-T10 ★ Camera fingerprinting.** Gotcha §2.7 says the `(1).mp4` ↔ camera-side mapping is
      inconsistent across days and is currently keyed by hand. That does not survive 5 venues.
      Auto-identify a camera view by background histogram / keypoint match against a reference frame.
      *Accept:* all existing footage auto-assigned correctly.

### 2.D Contingency for the C3 class (the #1 scientific risk)
- [ ] **WP2-T11 ★ C3 fallback decision tree.** Decide by **week 6** — do not drift past it. If C3
      is still under ~100 real frames:
  - [ ] **Option A — copy-paste augmentation.** Composite hi-vis worker / mower crops onto real
        empty-pitch backgrounds. A recognised, citable augmentation technique (the "cut, paste and
        learn" line of work). Must be reported honestly as synthetic, and evaluated separately on
        whatever real C3 frames exist.
  - [ ] **Option B — detector-based C3.** Use open-vocabulary detection (YOLO-World: "person in
        high-visibility vest", "lawnmower") as a rule-based C3 branch, evaluated as its own module
        rather than as a class of the main classifier.
  - [ ] **Option C — reframe the taxonomy.** Report the main benchmark as 2-class (EMPTY /
        ACTIVE_PLAY) with C3 handled as a flagged exception, and state the scope reduction plainly.
  - [ ] *Whichever you pick:* write the justification into `protocol.md` and tell your supervisor at
        the next check-in. A documented, defended scope reduction costs you nothing; an undocumented
        starved class costs you the macro-F1 claim.
- [ ] **WP2-T7 M2 gate check.** Every class ≥ target in ≥ 2 lighting regimes and ≥ 2 venues; grouped
      splits regenerate cleanly. *Accept:* M2 checklist in `EXPERIMENT_LOG.md`.
- [ ] **M2 GATE [H]** — dataset balanced across classes/conditions/venues, grouped-split-ready.
      *By week 9.*

---

## WP3 — Preprocessing pipeline & ablations
*Weeks 6–10 · ~80 h*

> Build every stage as an independent switchable step in `engine/preprocess.py`, single entry point
> `preprocess(frame_bgr, camera_tag, *, roi=True, letterbox=True, clahe='auto', ...)`, used by
> **both** `benchmark.py` and the live pipeline. One code path — this is gotcha §2.1's lesson.

- [ ] **WP3-T1 [H] ROI polygons.** Run `tools/draw_roi.py` for every camera view (current 4 + each
      new venue camera). *Accept:* a polygon per camera tag in `config/cameras.json`; masked previews
      visually correct.
- [x] **WP3-T2 Letterbox resize.** Aspect-preserving with grey padding, tested against the
      squashing it replaces.
- [ ] ~~WP3-T2 original~~ Aspect-preserving, replaces naive thumbnail. *Accept:* unit test —
      224×224, no distortion, grey padding.
- [ ] **WP3-T3 Photometric normalisation audit.** Verify each model's HF processor stats are applied;
      document per model in `protocol.md`. *Accept:* table in protocol.md.
- [x] **WP3-T4 Low-light / fog branch.** CLAHE on the LAB lightness channel, `on|off|auto`
      gated on RMS contrast.
- [ ] ~~WP3-T4 original~~ RMS contrast on ROI; below threshold → CLAHE/gamma variant;
      expose `clahe on|off|auto`. *Accept:* toggleable; before/after visuals saved.
- [x] **WP3-T5 Quality filter & camera health.** Over/under-exposure, Laplacian-variance blur,
      lens-dirt proxy (persistent contrast drop vs the camera's own 7-day baseline) → `quality=bad`
      in manifest; emit `camera_health.csv`. *Accept:* known-bad frames flagged.
  - [x] `vision/quality.py` + `experiments/camera_health.py` → `camera_health.csv`,
        `frame_quality.csv`. Thresholds are per (physical camera × lighting), never global.
  - [x] **The finding:** a global blur cutoff flags 169 frames and **100% are `venue_01`** —
        it measures which camera took the frame, not whether the frame is usable. Same
        confound as day-vs-night, third appearance.
  - [x] *Accept* met by construction, not by real positives: **this dataset has no known-bad
        frames**, so the filter is proved on deliberately degraded frames in
        `tests/test_quality.py` and the absence is recorded rather than tuned away.
  - [ ] ★ **Relabel the 396 clip-venue frames' `lighting` by hand** (~1 hour). It is currently
        a brightness proxy, and it is wrong for at least `b_floodlit_track`, `f_outdoor_bldg`
        and `c_teal_boards` — all night football labelled `day`. The clock rule reads this
        column and nothing else, so H3's 0.219 is partly label error; correcting one 12-frame
        fold alone moves it to 0.362. See the correction entry in `results/EXPERIMENT_LOG.md`.
- [ ] **WP3-T6 Train-time augmentation.** Photometric jitter, synthetic fog (gaussian haze),
      night-gamma, horizontal flip. **No rotations/warps** — cameras are fixed. *Accept:* flag in
      benchmark; visual grid saved.
  - [x] Module built: `vision/augment.py` — 9 effects, 5 presets, 18 property tests.
  - [x] Visual grid saved: `experiments/augmentation_grid.py` → `results/figs/augmentation_grid.jpg`.
        It earned its keep immediately: rain streak geometry was in absolute pixels, so it
        rendered as white poles at 320×180 and would have been invisible hairlines at 1080p
        after the resize to 224. Now every dimension is a fraction of frame height, pinned by
        a test. **No shape/dtype check could have caught that** — only looking at it.
  - [ ] Flag in benchmark — *deliberately not done yet.* Augmenting means a backbone forward
        pass per view, so the feature cache every experiment relies on stops applying
        (~8 min/model/epoch-equivalent vs seconds for a probe fit). This needs its own
        extraction budget, not a switch. See `docs/IDEAS.md` #2.
- [ ] **WP3-T7 Class balancing.** `class_weight='balanced'` + optional weighted sampling, default ON
      for 3-class runs. *Accept:* C3 recall improves on validation vs unweighted.
- [ ] **WP3-T8 Preprocessing ablation (E-PRE).** Best model + grouped split; toggle
      {ROI, letterbox-vs-thumbnail, CLAHE, augmentation, balancing} one at a time; deltas with CIs
      → `results/ablation_preprocessing.csv`. *Accept:* table + one-paragraph finding per switch.
- [ ] **WP3-T9 ★ ROI ablation gets its own figure.** Of all preprocessing steps, ROI masking is the
      one with a *visual* story (adjacent pitches firing false ACTIVE_PLAY). Pair the number with
      side-by-side example frames — it will be one of the most quoted figures in your defence.

---

## WP4 — Leakage-free benchmark & statistics
*Weeks 9–13 · ~110 h · M3 gate*

### 4.A The headline experiments
- [ ] **WP4-T1 Same-scene vs grouped vs leave-one-venue-out.** 4 models × 3 split strategies ×
      3 classes. *Accept:* `results/benchmark_v2.csv`; the random-vs-grouped delta is the thesis's
      first key figure. **(RQ3)**
  - [ ] ★ Add the **temporal split** as a 4th strategy (from WP0-T4) — measures drift, costs one
        extra run on cached features. **(RQ3)**
  - [ ] ★ Report **mean ± std over ≥5 grouped splits**, not one split. A single split's number is a
        sample of size one, and reviewers treat it as such.
- [x] **WP4-T10 ★ Trivial-baseline floor — done, and it fired.** `experiments/h1_h2_baseline_floor.py`.
      Under the random split a **16-bin colour histogram beat ConvNeXtV2 on macro-F1** (0.686 vs
      0.657); under the grouped split the **clock rule came within 0.007** of it using no pixels at
      all. The comparison was indeed measuring the wrong thing — established in week 1, not at the
      defence. **(RQ7)**
  - [x] ★ The complement: `experiments/h3_cross_venue_recall.py` shows the clock rule collapsing to
        0.219 play-recall on unseen venues while ConvNeXtV2 and DINOv2 hold at 0.910 / 0.930. The
        evaluation was uninformative; the models were not the problem. **(RQ3, RQ7)**
- [ ] **WP4-T2 Label-efficiency curves.** Training sizes {10, 25, 50, 100, 300, 1000, all} × 5 seeds
      × 4 models; zero-shot OpenCLIP as the 0-label horizontal line. *Accept:*
      `results/label_efficiency.csv` + `results/figs/label_curve.png`. **(RQ1, RQ2)**
- [ ] **WP4-T3 Cross-venue few-shot adaptation.** Leave-one-venue-out, then add {0, 25, 100, 300}
      target-venue frames. *Accept:* adaptation curve per venue — answers "what does onboarding a new
      client site cost?" **(RQ3)**
- [ ] **WP4-T9 ★ Risk–coverage (REVIEW-rate) curve.** Sweep the confidence threshold and plot
      automated-verdict error rate against the fraction of slots sent to REVIEW. Report the operating
      point where automated verdicts reach ≥99% precision, and the REVIEW rate it costs. **This is
      the number a facility manager actually buys**, it directly answers RQ6, and it is the natural
      home for the selective-prediction citations from WP1-T1. Nearly free once calibration is done.

### 4.B Making the numbers defensible
- [ ] **WP4-T4 Statistical testing.** Bootstrap CIs + McNemar (frame level) + paired bootstrap (slot
      level) for every headline pair. *Accept:* every claim carries CI + p-value columns.
  - [ ] ★ Apply the Holm–Bonferroni correction from WP0-T6 and say so in the caption.
  - [ ] ★ Report effect sizes next to p-values.
- [x] **WP4-T5 Calibration — built, answer blocked.** `evaluation/calibration.py` (ECE,
      reliability bins, temperature scaling, risk-coverage), 16 tests. The run produces
      "99% precision at 0% review", which is an artifact of the 99% single-class test set.
      A boundary warning caught DINOv2's temperature pinning at the grid floor. **No further
      engineering unblocks this** — it needs a test set with a real class mix.
- [ ] ~~WP4-T5 original~~ Reliability diagrams + ECE per model; temperature scaling fitted
      on validation (within training venues only); effect on REVIEW-band volume. *Accept:*
      `results/calibration.csv` + figures; calibrated heads saved with `temperature` in the pkl.
- [ ] **WP4-T6 Error taxonomy & explainability.** Categorise every grouped-split misclassification by
      (condition, confusion pair, camera) → `results/error_taxonomy.csv`. Attention rollout
      (ViT/DINOv2) + Grad-CAM (ConvNeXt) on 20 errors and 20 correct frames → `results/figs/xai/`.
      Confirm reliance on pitch, not background. *Accept:* figures + a finding sentence per class.
- [ ] **WP4-T11 ★ Efficiency table with honest methodology.** Using WP0-T10: median and p95 ms/frame,
      peak RAM, concurrent-20-camera throughput, PyTorch vs OpenVINO, on both the dev laptop and the
      target Mini-PC. **(RQ2)**
- [ ] **WP4-T12 ★ Measure actual power draw.** A €15 plug meter on the Mini-PC gives W under load →
      kWh/year → €/year and gCO₂/year. Trivial effort, and it substantiates the "low-bandwidth,
      low-power edge" framing with a real measurement instead of a datasheet quote.
- [ ] **WP4-T7 DINOv3** (if licence access granted) — add as a 5th row to every WP4 table, same
      protocol, or log explicitly as blocked. Requires [H] HuggingFace login + Meta licence accept.
- [ ] **WP4-T8 M3 gate check.** Benchmark v2 complete, stats attached, figures exported.
- [ ] **M3 GATE [H]** — four-model leakage-free benchmark with CIs and significance tests. *Week 13.*

---

## WP5 — Novel architectures
*Weeks 12–18 · ~150 h · M4 gate · **this is where the thesis earns its originality***

> Rules for every module: frozen backbones, only small trainable parts, ablated against a *strong*
> baseline with significance tests, latency and memory reported next to accuracy. A rigorous negative
> result is a publishable result — but only if the baseline was strong enough to make losing to it
> interesting.

### 5.A STAN — Slot-Temporal Aggregation Network *(Core tier — do this one first)*
- [ ] **WP5-T1 STAN implementation.** `engine/stan.py`: input = ordered per-minute fused class
      probabilities (optionally + embeddings); model = 1-D TCN or 2-layer Transformer encoder
      (< 100k params); output = 3-way slot status + calibrated confidence. **(RQ5)**
- [ ] **WP5-T6 ★ Feed both camera halves separately instead of pre-fusing them.** Current design
      fuses camera A/B with a max-activity rule *before* aggregation, which throws away information:
      **disagreement between the two halves is itself a signal** (occlusion, dirty lens, play
      confined to one half). Let STAN consume both sequences and learn the fusion. This gives you a
      genuinely stronger architecture *and* a clean extra ablation (fuse-then-aggregate vs
      aggregate-jointly) — one of the cheapest real novelty gains available in this plan.
- [ ] **WP5-T7 ★ Strengthen the STAN baselines.** Beating hand-set thresholds proves very little —
      of course a learned model beats an unlearned rule. Also compare against: **(a)** thresholds
      *tuned* on the same training slots, **(b)** majority-vote / median smoothing over a sliding
      window, **(c)** an HMM over the per-minute states, **(d)** logistic regression on summary
      statistics of the sequence (ratios, longest run, first/last active minute). **If STAN beats a
      tuned HMM, that is a real result.** If it only beats fixed thresholds, an examiner will
      discount it — and this is the most likely single point of attack on your novelty claim.
- [ ] **WP5-T8 ★ Slot dataset with an honest synthetic/real boundary.**
      `tools/make_slot_dataset.py` composes realistic slot sequences from manifest frames (templates:
      full match, no-show, late start, maintenance window, intermittent). **Train on synthetic; the
      test set must be real slots only.**
  - [ ] ★ **Hard gate:** do not report STAN as a headline result until ≥ 30 real labelled slots exist
        (WP2-T8). Below that, label it explicitly as a preliminary/pilot result in the thesis. Write
        this rule down now, while it is still easy to be honest about.
  - [ ] ★ Report a **synthetic-vs-real generalisation gap** measurement (train synthetic → test
        synthetic vs train synthetic → test real). That gap is itself an interesting, honest finding
        about training sequence models on composed data.
- [ ] *Accept:* STAN vs best baseline on slot macro-F1, paired bootstrap, p reported in either
      direction; REVIEW rate ≤ baseline; latency negligible.

### 5.B Gated multi-backbone fusion *(Target tier)*
- [ ] **WP5-T2 Fusion head.** `engine/fusion_head.py`: features from {ConvNeXtV2, DINOv2} (option
      +ViT); gate = tiny MLP on cheap image statistics (contrast, brightness, edge density) → fusion
      weights → shared linear head. **(RQ5)**
- [ ] **WP5-T2b Conditional-compute variant.** Run ConvNeXt first; invoke DINOv2 only when confidence
      < τ. Report accuracy **and** average ms/frame vs always-both. *Accept:* ablation table with CIs,
      p-values, latency; adopt/reject decision logged.
- [ ] **WP5-T9 ★ Sanity baseline for fusion: plain logit averaging.** If a naive ensemble of the two
      backbones matches the learned gate, the gate is not the contribution — the ensemble is. Better
      to discover that yourself and report it than to have it asked.

### 5.C Context-aware multimodal head *(Stretch tier)*
- [ ] **WP5-T3 Context head.** `engine/context_head.py`: visual embedding + context vector
      (hour-of-day sin/cos, day-of-week, booked flag, previous-slot state, optional weather) → small
      MLP. **(RQ5)**
  - [ ] Booking-flag dropout (p=0.5) during training.
  - [ ] Adversarial ablation: train with the flag, evaluate without it.
  - [ ] Always keep the vision-only path in production — the audit signal must stay independent of
        the record it audits. *(This rigour note from the thesis doc §4.4 is one of the sharpest
        points in the whole plan; make sure it survives into the write-up.)*
  - [ ] *Accept:* ablation vision-only vs +context vs +context-minus-flag; leakage check documented.

### 5.D Distillation & quantisation *(Stretch tier)*
- [ ] **WP5-T4 Distillation + quantisation.** ViT-head soft labels distilled into the ConvNeXt head
      (KL on cached logits — cheap); OpenVINO export FP32/INT8 of the ConvNeXtV2 backbone; measure
      accuracy delta + latency on the actual target CPU. *Accept:* `results/efficiency.csv`
      (PyTorch vs OV-FP32 vs OV-INT8: ms, RAM, accuracy). **(RQ2)**
- [ ] **WP5-T5 M4 gate check** — ≥ STAN + one fusion module fully ablated with significance.
- [ ] **M4 GATE [H]** *By week 18.*

---

## WP6 — System integration, reconciliation, dashboard
*Weeks 14–19 · ~120 h · M5 gate*

> ⚠️ **Scope warning.** This is the largest package that contributes least to thesis *novelty*
> marks. If the schedule slips, compress **here**, not in WP5. Decide the minimum viable version of
> each item up front.

### 6.A Runtime
- [ ] **WP6-T1 Simulator API.** `engine/simulator_api.py` (FastAPI):
      `GET /api/v1/cameras/{camera_id}/snapshot?mode=simulation`, Bearer token from config.
      *Accept:* curl returns JPEG + JSON metadata.
- [ ] **WP6-T2 Scheduler service.** `main.py`: reads `config/slots_schedule.json`; during active
      slots pulls 1 frame/min per camera (VIDEO_SIM | API_SIM | RTSP_LIVE), classifies, fuses, writes
      DB; at slot end runs the aggregator (threshold or STAN per config) + evidence selection.
      *Accept:* runs continuously; DB fills; verdicts correct on recorded slots.
- [ ] **WP6-T3 RTSP source.** `frame_source.RTSPSource` — snapshot per minute, retry + timeout.
      *Accept:* tested against a reachable RTSP or a local ffmpeg loop of the mp4s.
- [ ] **WP6-T8 Retention worker.** Purge raw frames > 7 days, keep evidence 365 days; disk usage
      bounded and logged. *Accept:* dry-run prints the correct purge set.
  - [ ] ★ Make retention periods match whatever `thesis/ethics.md` (WP1-T3) actually committed to.

### 6.B Reconciliation — the operational contribution
- [ ] **WP6-T4 [H] Booking schema + connector.** `engine/bookings.py`: normalised
      `bookings(field_id, date, start, end, customer_ref, status, entered_by, source)`; CSV importer
      first, SQL/REST stubs behind one interface. **READ-ONLY — never write to client systems.**
      *Accept:* sample CSV imports; unit tests.
- [x] **WP6-T5 Reconciliation.** `slots/reconcile.py`, 16 tests. REVIEW never becomes an
      anomaly; low-confidence slots are downgraded before any rule runs; anomalies are **per
      field, never per person** (`entered_by` never reaches the output — asserted by test),
      which settles WP1-T4 in code. Validated end-to-end on both real slots.
- [ ] ~~WP6-T5 original~~ `engine/reconcile.py`: join slot_evaluations × bookings on
      (field, date, slot) → typed anomalies: `NO_SHOW_OR_OVERRECORDED`, `PLAYED_NOT_RECORDED`,
      `UNBOOKED_USAGE`, `BLOCKED_SLOT_SOLD`; REVIEW routes to an inspector, never a hard anomaly.
      Low-confidence / low-contrast slots → REVIEW, so the audit never over-accuses on poor footage.
      *Accept:* synthetic fixtures produce exactly the expected anomaly types. **(RQ4)**
- [ ] **WP6-T10 ★ Cost-sensitive evaluation of reconciliation.** Attach money to the confusion
      matrix: what does a missed `UNBOOKED_USAGE` cost in unbilled revenue? What does a false
      accusation cost in staff trust and dispute handling? Report **expected € recovered per 1,000
      slots** at your chosen operating point, with the assumptions stated. This turns Chapter 7 from
      "we built a feature" into "we quantified the value of the feature", and it is exactly the kind
      of thing that distinguishes a good applied thesis. **(RQ4, RQ6)**
- [ ] **WP6-T11 ★ Reconciliation needs its own ground truth.** Anomaly precision/recall cannot be
      measured without knowing which slots *really* were no-shows. Get the facility to adjudicate a
      set of flagged and non-flagged slots. Without this, RQ4 is unanswerable — which is worth
      noticing now rather than in week 19.

### 6.C Operator surface
- [ ] **WP6-T6 Dashboard.** FastAPI + single-page UI: global meters, live field matrix with
      confidence chips, slot evidence inspector (3 photos, AI reason, one-click override with
      operator name → `is_overridden` audit fields), anomalies view, schedule editor.
      *Accept:* manager daily review flow < 5 min.
- [ ] **WP6-T7 Override → retraining loop.** Overridden slots' evidence frames auto-copy into
      `data/dataset/_incoming/<corrected_class>/`; document the periodic head re-fit (cached
      features → seconds). *Accept:* override produces the file + log row.
- [ ] **WP6-T12 ★ "The system must never bill" design constraint.** Write it into the architecture
      chapter and enforce it in code: output is decision support, a human confirms every anomaly, no
      automated financial action. Cheap to state, and it is the answer to the ethics question you
      *will* be asked about auditing staff.
- [ ] **WP6-T9 M5 gate check** — full pipeline including reconciliation runs end-to-end on mock feeds.
- [ ] **M5 GATE [H]** *By week 19.*

---

## WP7 — Deployment & live validation
*Weeks 18–21 · ~70 h · M6 gate*

- [ ] **WP7-T1 Target-hardware bench.** `benchmark.py --limit` + `run_slot_eval.py` on the actual
      Mini-PC: confirm the < 500 ms/frame budget with 20-camera headroom. *Accept:* efficiency.csv
      row for target hardware. ★ Use the WP0-T10 concurrency method, not 20× single-frame.
- [ ] **WP7-T2 Install & services.** Install doc; systemd (or Windows service) units for engine,
      dashboard, retention; camera VLAN config with client IT; `SOURCE_TYPE=RTSP_LIVE`.
      *Accept:* auto-start after reboot verified.
- [ ] **WP7-T3 Shadow mode (1–2 weeks).** System records verdicts silently; staff keep their normal
      process; compare daily. *Accept:* shadow report with agreement rate.
- [ ] **WP7-T4 48-hour acceptance run.** Every slot verdict adjudicated with facility staff; measure
      slot accuracy, REVIEW rate, reconciliation precision/recall. Doubles as Chapter 7 case-study
      data. *Accept:* signed-off M6 + exported tables.
- [ ] **WP7-T5 ★ Failure-mode runbook.** What happens when a camera dies mid-slot, the network drops,
      the disk fills, or the model's confidence collapses facility-wide? Define the degraded-mode
      behaviour (default to REVIEW, alert, never fabricate a verdict) and test at least the
      camera-offline case. Examiners ask about robustness; clients live it.
- [ ] **WP7-T6 ★ Capture defence artefacts during the live run.** A short screen recording of the
      dashboard, a real anomaly caught end-to-end, before/after evidence frames. You cannot recreate
      these after the hardware goes back, and a live demo clip is worth several slides.
- [ ] **M6 GATE [H]** — 48-h live validation accepted. *By week 21.*

---

## WP8 — Writing & defence
*Ongoing → week 24 · ~140 h · M7 gate*

### 8.A Continuous drafting (do not defer)
- [ ] **WP8-T1 Chapter drafts as each WP closes.** Ch3 after WP2/WP3 · Ch4 after WP5 · Ch5 after WP6 ·
      Ch6 after WP4 · Ch7 after WP7.
- [ ] **WP8-T5 ★ Claims ledger** → `thesis/claims.md`: every quantitative claim in the text mapped to
      the results file, figure, and script that produced it. Update as you write, not at the end.
      This is how you guarantee no unsupported sentence survives to the defence — and it makes the
      final consistency pass an afternoon instead of a week.

### 8.B Figures
- [x] **WP8-T2 Figure set — three done.** `results/figs/`: label-efficiency curve,
      ranking-inversion slope chart, cross-venue per-fold recall. Regenerated from the CSVs,
      validated palette, PNG + PDF.
- [ ] **WP8-T2b Remaining figures.** Label-efficiency curve · leakage comparison bar · ablation tables ·
      reliability diagrams · XAI overlays · reconciliation matrix with real (blurred) evidence.
  - [ ] ★ Risk–coverage / REVIEW-rate curve (WP4-T9).
  - [ ] ★ Trivial-baseline floor chart (WP4-T10).
  - [ ] ★ Accuracy-vs-latency scatter with the CPU budget drawn as a vertical line — one picture that
        answers RQ2 completely.
  - [ ] ★ **Blur all faces in every published figure.** Check this twice before submission.

### 8.C Defence
- [ ] **WP8-T3 Threats to validity + ethics sections** — built from the gotchas in `SUPER_PLAN.md` §2,
      the WP4 findings, and `thesis/ethics.md`.
- [ ] **WP8-T4 Defence deck.** The pilot story (the spec's chosen model lost; the pooler-output
      harness bug and its lesson; the leakage number), the novel modules, a live demo of the
      dashboard and reconciliation.
- [ ] **WP8-T6 ★ Red-team your own defence.** Write down the ten hardest questions and your answers.
      Starting set: *Why CV rather than a €20 motion sensor?* (WP1-T6) · *Is your novelty just
      temporal smoothing?* (WP5-T7) · *Your STAN test set is how many slots?* (WP2-T8) · *Would a
      colour histogram do this?* (WP4-T10) · *How many comparisons did you run before p < 0.05?*
      (WP0-T6, 0.5) · *Is auditing staff by camera ethical?* (WP1-T4, WP6-T12) · *Can anyone
      reproduce this?* (WP0-T11, WP1-T5) · *What is the human ceiling on this task?* (WP2-T9).
- [ ] **WP8-T7 ★ Ask your supervisor to mock-examine you** two weeks before the defence, not two days.
- [ ] **M7 GATE [H]** — thesis submitted, defence ready. *By week 24.*

---

## Milestone gate tracker

| Gate | Week | Exit criterion | Done |
|---|---|---|---|
| M1 | 4 | Protocol, taxonomy, evaluation design approved ★ + ethics/DPIA cleared | [ ] |
| M2 | 9 | Dataset balanced across classes/conditions/venues ★ + ≥30 real slots labelled | [ ] |
| M3 | 13 | Four-model leakage-free benchmark with CIs + significance ★ + trivial-baseline floor | [ ] |
| M4 | 18 | STAN + ≥1 fusion module ablated with significance ★ against strong baselines | [ ] |
| M5 | 19 | End-to-end system + reconciliation on mock feeds | [ ] |
| M6 | 21 | 48-h live validation accepted | [ ] |
| M7 | 24 | Thesis submitted, defence ready | [ ] |

---

## ★ Capacity check — read this before committing to the schedule

The effort plan budgets **~840 h over 24 weeks ≈ 35 h/week, every week, for six months**, and
`SUPER_PLAN.md`'s per-WP numbers sum to ~930 h (≈ 39 h/week). Both assume a **full-time** thesis.

- [ ] **Decide honestly: is this your only commitment for these 24 weeks?**
  - [ ] **Yes, full-time** → the plan stands. Keep weeks 22–24 as the slack they were designed to be
        and do not fill them with new scope.
  - [ ] **No — coursework/job alongside** → at ~20 h/week the plan is a **42-week** project. Choose
        now, in week 1, rather than discovering it in week 15: either extend the timeline with your
        supervisor, or take the 16-week compression below. **Do not attempt the full scope at half
        capacity** — the failure mode is WP5 getting squeezed, which is the one package you cannot
        afford to lose.

**16-week compression (from the effort-plan docx, with additions):** keep WP0–WP4 + STAN only ·
drop the fusion and context heads to the extensions list · data collection at 2–3 venues ·
24-h live validation instead of 48-h. ★ Additionally: keep WP4-T9 (risk–coverage) and WP4-T10
(trivial baselines) even under compression — they are hours of work each and they carry
disproportionate defensive weight. ★ Compress WP6 to reconciliation logic + a minimal dashboard;
polish is not worth thesis marks.

- [ ] **★ Define the minimum viable thesis** and write it in `thesis/mvt.md`. Something like:
      *leakage-free four-model benchmark with honest statistics + the C3 data limitation reported
      openly + a working end-to-end system on recorded slots + STAN as a preliminary result.* That is
      already a complete, defensible thesis. Knowing your floor makes every later scope decision
      calm instead of panicked.

---

## Risk register (act, don't admire)

| Risk | Trigger to watch | Action |
|---|---|---|
| ★ Ethics/DPIA blocks collection | No answer by week 2 | Escalate to supervisor; work WP0/WP1 meanwhile |
| C3 data stays scarce | `coverage.md` cell < 100 by week 6 | Execute the WP2-T11 decision tree |
| ★ Real slots stay < 30 | Week 12 | Demote STAN to preliminary; lead with WP4 as the contribution |
| Only 1 venue accessible | Week 5, no partner venue | Reduce claim to cross-pitch/cross-lighting; state scope explicitly |
| Booking DB access blocked | Week 12, no sample export | Manual booking sheet for the case study |
| Novel module gives no gain | WP5 ablation p > 0.05 | Report as a negative result with analysis — still a contribution |
| Laptop sleep kills runs | Any multi-hour run | Chunk runs, incremental writes, cache-resumable (WP0-T11) |
| ★ Data loss | Any time | 3-2-1 backup verified monthly (WP0-T8) |
| ★ Test-set overfitting | Many ablation cycles | FINAL_TESTSET locked and enforced in code (WP0-T4) |
| ★ Capacity overrun | Behind by week 10 | Switch to the 16-week compression; protect WP5 |

---

## Human-only tasks — surface these early and repeatedly

- [ ] **[H]** Ethics/DPIA answer from supervisor ★ *(blocks WP2)*
- [ ] **[H]** Export more StatBox footage per `thesis/data_requests.md` — especially **maintenance
      windows** (0 frames) and ★ **≥30 complete slots with known verdicts**
- [ ] **[H]** Draw/confirm ROI polygons for every camera (`tools/draw_roi.py`)
- [ ] **[H]** HuggingFace login + accept the Meta DINOv3 licence
- [ ] **[H]** Introduce ≥ 2 partner venues; arrange exports
- [ ] **[H]** Ask the client which booking system they use; obtain a read-only sample export (CSV fine)
      ★ *— ask in the same conversation as the footage request*
- [ ] **[H]** Second annotator for the κ subset ★ *and for the human-ceiling measurement*
- [ ] **[H]** ★ Facility adjudication of flagged/unflagged slots for reconciliation ground truth
- [ ] **[H]** Supervisor sign-offs at M1–M7 ★ *plus a mock examination before M7*

---

## Ideas not yet in the plan

See **[docs/IDEAS.md](docs/IDEAS.md)** - weather detection as an *explanation* for a verdict
(rain turns an apparent no-show into a legitimate cancellation, which is the cheapest
available improvement to anomaly precision), synthetic rain augmentation, conditions
recorded alongside every verdict, transition-aware evidence selection, and a preprocessing
sweep. Each entry carries its own stop condition.

## Extensions backlog (only if ahead of schedule, in this order)

1. DINOv3 integration + quantisation study *(cheap, high visibility)*
2. Cold-start transfer curve as a standalone publishable result *(≈ WP4-T3)*
3. Active learning + pseudo-labelling on the unlabelled pool
4. Self-supervised domain adaptation on facility footage *(flagship stretch; needs GPU time)*
5. NL explanations for REVIEW slots (Florence-2 / Moondream on evidence frames)
6. Camera-health prediction from confidence drift
7. Test-time augmentation + temporal smoothing polish
8. Multi-tenant productisation + mobile manager view
