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
  - [ ] ★ **[H] Reopen two lines of it — not the whole request.** Closing 0.3 was reasonable
        against WP2-T2 *as written*: it asked for maintenance windows, idle-people slots, two more
        pitches, two external venues, rain days, 30–40 complete slots **and** the booking export.
        That is a large ask and it was declined as one. But two of those lines are far cheaper than
        the rest and carry almost all of the blocked value, and they should be asked for separately
        rather than left closed by association:
    - [ ] **(a) Empty-pitch footage at 2–3 other venues.** The cheapest footage a facility owns —
          a pitch is empty most hours of the day, so this is an export of a quiet early-morning
          hour from cameras that are already recording. **No people in frame means it carries
          near-zero privacy exposure**, which also makes it the easiest thing for the operator to
          say yes to. It is the one input behind *every* blocked question: three-class cross-venue
          evaluation (currently impossible), **RQ6 in its entirety** (the risk–coverage curve has
          nothing to trade against on a 99%-single-class test set), the statistical backing for the
          false-play comparison (3–10 effective scenes → dozens), and RQ1's full form rather than
          its active-play-only scope. `EXPERIMENT_LOG.md` reaches the same conclusion
          independently: *"a second venue with genuine downtime … would turn three effective
          observations into dozens."*
    - [ ] **(b) Slot verdicts for footage already held.** No new recording at all — one
          USED/NOTUSED label per hour, read off the facility's own booking sheet for slots they
          have already exported. This is what M4's original criterion and WP5-T8's hard gate need,
          and it is close to free to produce.
        *If both are declined, nothing changes: the re-cut M2/M4 gates below stand and the thesis
        is complete without them. That is exactly why asking costs nothing — there is no plan that
        collapses on a "no".*
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
      as H1's control arm. 24 tests.
  - [ ] ★ **"Materialised, referenced by name, never re-randomised" is claimed but not
        practised.** `write_split` and `read_split` have **no callers outside the tests**, and
        `results/splits/` holds only the lock file. Every experiment calls `grouped_split(...,
        seed=42)` directly, which is deterministic *given the same rows* — and that proviso is
        the whole problem: the row list depends on which feature caches a script filters to.
        `effective_sample_audit.py` (DINOv2 cache only) counts **94** distinct scenes where
        `h4_model_equivalence.py` (all three caches) counts **95**, from the same seed. Harmless
        this time, and precisely the drift materialisation exists to prevent.
        **Decide one of two things** rather than leaving the claim standing: (a) materialise the
        canonical splits once and have every experiment `read_split` them — correct, and a
        retrofit across ~20 scripts; or (b) drop the claim and rely on `seed=42` plus a fixed
        row list, documenting that the row list is part of the split's identity. (a) is what the
        plan intended; (b) is honest and nearly free. Either way the docstring should stop
        asserting a guarantee the code does not provide.
  - [x] ★ **The materialisation machinery works now, whichever way that goes** (2026-09-07):
        the round-trip preserves `group_key` and `seed` (it lost them, leaving a placeholder
        that made `check_split` raise `AttributeError` on *every* split read from disk);
        `read_split` **re-applies the final-venue lock**, since a file written before the lock
        existed can name a locked venue and nothing downstream looks again; `check_split`
        reports an unrunnable group check instead of crashing on one; and `write_split`'s
        default directory is package-relative rather than cwd-relative — the same fail-open
        path bug as the lock, in the same folder as the lock.
  - [x] ★ **`grouped_split(seed=42)` was not reproducible across processes** (2026-09-08).
        It built the test side by iterating `test_names`, a **set** — and `PYTHONHASHSEED` is
        randomised per process, so the same seed returned the same frames in a different
        *order* every run. Membership was right, which is exactly why nothing caught it:
        accuracy and macro-F1 do not depend on row order, so every point estimate reproduced
        exactly while **every bootstrap interval computed on a grouped split was a different
        draw**. Regenerating moved CI bounds by up to 0.0073, `n_distinct` by one, and the
        risk-coverage curve by 0.125 — no point estimate anywhere.
    - [x] `test_grouped_split_is_deterministic` had guarded this since the module was written
          and passed throughout, because it called the function twice **in one process**,
          where set order is fixed for the process's lifetime. Its replacement spawns
          interpreters under three `PYTHONHASHSEED` values; verified failing on the old line
          and passing on the new one. A static sweep of the package found this the only site.
    - [x] It also settles the sub-item above with evidence rather than preference: the 94-vs-95
          drift recorded there was **not** the row-list difference it was attributed to. Both
          scripts now count 95. That does not make materialisation unnecessary — but the
          example that motivated (a) was this defect, and it is worth re-reading before the
          ~20-script retrofit is costed.
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
- [x] **WP0-T11 ★ Reproduction script.** `make reproduce` (or `tools/reproduce_all.py`) that
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
- [x] **WP1-T2 Protocol document — drafted, awaiting sign-off** →
      **[`thesis/labelling_protocol.md`](thesis/labelling_protocol.md)** (2026-09-07). Its own file
      rather than `protocol.md`, which was already taken by the *measurement* protocol — what is
      done to a frame's pixels. This one is what a frame *means*. Consolidates rules that already
      existed but were scattered across `TODO.md`, `SUPER_PLAN.md`, `PLAN.md`, `taxonomy.py` and
      `ethics.md`; anything never decided is marked **OPEN** rather than filled in.
      **All 1,692 labels already rested on these rules with nothing written down** — `thesis/mvt.md`
      calls this the one genuine hole in the minimum viable thesis.
  - [ ] **[H][B] Supervisor sign-off** — this is the M1 gate's labelling half. §6 lists the five
        open items blocking approval; two of them (slot rules, non-sporting booked use) need the
        client.
  - [x] ★ **Slot-level labelling rules** — drafted in §3 with a proposed answer per case (late
        arrival, abandonment, back-to-back boundary, no-show, unbooked usage, camera failure) and
        the rule that matters most: **booking status must never influence the vision label**, since
        that independence *is* the audit. Flagged as the highest-priority gap in the document
        because STAN trains on slot labels and only 2 real slots exist — these must be settled
        before slot labelling starts, not after.
    - [ ] ★ **[H] Client decision:** does a booked slot used for a non-sporting purpose (birthday
          party) count as USED for billing? Vision says C3 and billing may say USED; the
          disagreement is the reconciliation layer's job to surface, not something to hide by
          picking one.
  - [x] ★ **Decision rule for ties/uncertainty** — §2.6, so κ measures genuine disagreement rather
        than missing instructions. Activity beats emptiness · sport beats non-sport · non-sport
        beats emptiness (matching the fusion module's priority order, so labels and code agree) ·
        undecidable frames go to a holding folder rather than being forced, because an honest count
        of ambiguous frames is a result and a forced label is silent noise · and **never label from
        context not visible in the frame** — not the filename, not the timestamp. That last one is
        load-bearing: the day/night confound and the `lighting` mislabelling both came from
        metadata standing in for pixels, and labelling from a clock would bake the confound into
        the ground truth itself.
- [ ] **M1 GATE [H]** — protocol, taxonomy, evaluation design approved by supervisor. *By week 4.*
      ★ **Overdue and now unblocked on our side.** The labelling half is drafted (WP1-T2 above);
      what remains is genuinely external — a supervisor conversation and two client answers. The
      literature half (WP1-T1) is separate and still open.

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
  - [ ] ★ **Empty pitches at *any* venue other than venue_01 — name this one explicitly.**
        All 494 EMPTY frames in the corpus are venue_01, two cameras, two slots. That single
        fact blocks four separate questions at once: RQ1's empty-pitch claim (unanswerable),
        RQ6's calibration, the two confidence thresholds, and the input-path decision
        (WP3-T3), which reverses when those two cameras are swapped and cannot be settled by
        any test on this data. It is cheap to supply — an empty pitch needs no scheduling,
        only a camera pointed at one — and it is the highest-value frame in the request.
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
- [x] **WP2-T4 De-duplication.** Perceptual-hash near-duplicate pass; drop near-identical
      consecutive frames within a class. *Accept:* duplicate rate reported per batch.
  - [x] `data/dedup.py` + `experiments/near_duplicate_audit.py` -> `near_duplicates.csv`.
        1,667 of 1,692 frames (98.5%) have a direct near-duplicate; rates reported per class
        and per venue.
  - [x] **H1's leakage quantified:** the random split puts **37.1%** of near-duplicate pairs
        across the train/test boundary, the grouped split **0.8%** - a 49x reduction.
  - [x] Reported pairwise after single-link chaining gave a meaningless 96.9%.
  - [ ] ★ **Fix two verified label errors** - `2_playing/slot_20260711_1000_camA_t000021_m.jpg`
        and `..._t000027_m.jpg` show an empty pitch. Both are human-labelled. They are two of
        only six daytime ACTIVE_PLAY frames at `venue_01`, so correcting them makes the
        day/night confound *more* absolute. Deferred because moving them invalidates every
        feature cache; do it between search runs, then re-run `reproduce_all.py`.
- [ ] **WP2-T5 Double-labelling & κ.** 10% sample → second annotator, blind → `tools/kappa.py`
      computes Cohen's κ, logs disagreements → resolve, amend `protocol.md`. *Accept:* κ ≥ 0.85.
  - [ ] **WP2-T9 ★ Human ceiling on the test set.** While the second annotator is labelling, have
        them label the **held-out test set** too. Report human-vs-consensus accuracy as the ceiling.
        This is what makes "98.1%" interpretable: if humans hit 97% on the same frames, your model is
        at ceiling and further accuracy chasing is wasted effort — and *that* is a defensible finding.
        Costs almost nothing on top of the κ work you are already doing.
- [ ] **WP2-T6 Screenshot rescue (optional).** Crop/inpaint the red burned-in labels from the 14
      `data/ss data/` screenshots → held-out qualitative set. *Accept:* no legible label text remains.
- [x] **WP2-T10 ★ Camera fingerprinting — built; accept criterion partly met.**
      `vision/fingerprint.py` (median background + gradient-orientation grid + rg-chromaticity,
      watermark masked, 25 tests) and `experiments/camera_fingerprint_audit.py` →
      `results/camera_fingerprint.csv`. Full findings in EXPERIMENT_LOG 2026-09-07; in the reproduction pipeline as `camera-fingerprint`.
  - [x] **Gotcha §2.7 — already solved by `vision/camera_id.py`; this replicates it.** That
        module measured the swap first (0.88 cross-day vs 0.70 own-suffix) and `db/seed.py`
        already encodes it; `fingerprint.py` was written without noticing. Independent
        replication: both cross-day pairings are mutual nearest neighbours with opposite
        suffixes (d=0.111 and 0.277) while same-day camA-vs-camB sit at 0.46–0.57.
  - [x] ★ **Duplication turned into a cross-check** rather than deleted:
        `tests/test_camera_id_agreement.py` (6 tests) asserts both descriptors reach the same
        pairing, that `db/seed.py` still encodes it, and — guarding the premise — that the two
        descriptors have not converged into one measurement. Two independent methods agreeing
        is a stronger write-up claim than one measured twice.
  - [x] ★ **Division of labour recorded:** `camera_id` = which physical camera (runtime,
        wired in); `fingerprint` = how views group into venues (audit tool, not runtime).
  - [x] **Confirms the visual audit's load-bearing claim:** `g` and `h` share no cluster at any
        threshold, so the leave-one-venue-out folds do not leak between them. (`g` splitting
        into 2 view-clusters is correct multi-pitch behaviour, not a contradiction — `venue_01`
        splits into its 2 cameras the same way.)
  - [ ] ★ *Accept only 67/70, not 70/70.* Misses: `cb_…680610`, `cg_…155619`, `ci_…712737` — all
        day/dusk clips landing on another facility. Same-venue and different-venue distances
        **overlap**, so no threshold settles identity alone and unsupervised clustering leaves 4
        clusters mixing two facilities each. Use it to *check* a grouping and to assign new
        footage against known references with a human confirming — not to invent a grouping.
        Revisit when more venues arrive; more references should lift 1-NN.

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
- [x] **WP3-T3 Photometric normalisation audit.** Verify each model's HF processor stats are applied;
      document per model in `protocol.md`. *Accept:* table in protocol.md.
  - [x] Table in `thesis/protocol.md`; 11 tests in `tests/test_normalisation_audit.py`.
  - [x] Photometric half is correct — each model uses its own mean/std.
  - [x] **The geometric half was not, and nobody had looked.** ConvNeXtV2 and DINOv2 both
        resize to 256 and centre-crop 224 *after* `preprocess.py` has produced a 224×224
        letterboxed frame — **23.4% of every frame discarded**, exactly where WP3-T2's
        padding sits. ViT is the only backbone unaffected.
  - [x] Gives a mechanism for the search's `centre_crop=0.5` + `sharpen` = −0.200: a crop
        applied to a crop.
  - [x] ★ **Convention decided by measurement (2026-09-08): keep the processor's geometry.**
        `experiments/geometry_convention_probe.py` → `results/geometry_convention_probe.csv`,
        three arms × three backbones, both axes, in its own cache directory so nothing
        published could be overwritten (main caches md5-checked, byte-identical after).
        `protocol.md` leaned toward *disabling* it; measured, that is wrong here.
    - [x] **ConvNeXtV2**: disabling costs 0.47 of balanced score (0.964 → 0.490), nearly all
          of it false-play (0.021 → 0.482). **DINOv2**: a real trade — false-play improves
          (0.231 → 0.169) but recall falls further (0.960 → 0.869) and the worst fold
          collapses to **0.417**; balanced narrowly favours keeping it. **ViT**: no
          difference at all, and that *confirms* the harness — its two caches are
          bit-identical (max diff 0.0) exactly as `protocol.md` predicted, since its
          processor resizes to 224×224 and crops nothing.
    - [x] **Why keeping it works**, which is not obvious: the letterbox makes a 224×224 frame
          with grey bars, and the processor's resize-to-256-then-crop-224 trims most of that
          padding back off. The pair is aspect-preserved content *with the padding removed*.
          Either step alone is worse than both.
    - [x] ★ **[B] And a much larger finding fell out of it — see WP3-T3 notes 2 and 3
          below. Note 3 tested it and it did not survive; the decision is recorded there.**
  - [x] ★ **Attempted 2026-09-08, and it surfaced something bigger than the convention
        question. Read this before running the comparison.**
    - [x] The flag was **not runnable**. `build_cache` had no `processor_geometry`
          parameter, so the alternative convention could not be cached at all — and the flag
          was not in the fingerprint either, though `thesis/protocol.md` and
          `EXPERIMENT_LOG.md` both stated it was. Both conventions would have collided on
          one cache key *and* one filename. Fixed: the parameter threads through, it is in
          the fingerprint, the non-default convention gets `<backbone>_nogeom.npz`, and six
          tests pin it. Existing caches were verified bit-for-bit as
          `processor_geometry=True` builds before their stale fingerprint was restamped, so
          **no published number moves**.
    - [x] ★ **The real finding: `build_cache` does not apply `preprocess.py` at all.** It
          opens raw frames and hands them to the HF processor, whose resize is the only
          thing making them model-sized. `preproc` is a dict of *labels for the
          fingerprint* — it has never driven a transform. So the letterbox of WP3-T2, ROI
          masking, CLAHE and the other searched switches are **not in the path that produced
          any cached feature the headline experiments read.** The search and ablation caches
          are a separate family precisely because those scripts call `preprocess()`
          themselves.
    - [x] Which is why the flag crashed rather than working: with the processor's geometry
          off and no preprocessing, the model gets a 1080×1920 frame and refuses it
          (`ValueError: Input image size (1080*1920) doesn't match model (224*224)`).
          `build_cache` now requires an explicit `preprocess_fn` when the flag is off, so
          the dependency is legible instead of a crash five frames deep in transformers.
    - [x] ★ **[B] So WP3-T3's experiment is not the one-line run it looks like, and it is a
          decision, not a patch.** Comparing conventions honestly means putting
          `preprocess.py` into the main cache path — which changes the input to *every*
          published number, not just the nogeom arm. Three options, in increasing cost:
          **(a)** run the comparison with `preprocess_fn` supplied for both arms, as a
          self-contained side experiment that touches no existing cache and answers the
          convention question on its own terms; **(b)** adopt `preprocess.py` in
          `build_cache` and regenerate everything, which is the coherent end state and
          invalidates every cached feature; **(c)** leave the default path as it is and state
          plainly in the write-up that the searched preprocessing switches apply to the
          search and ablation caches only. **(a) first** — it is cheap and it tells you
          whether (b) is worth its cost.
      - [x] **(a) run, 2026-09-08** → `geometry_convention_probe.csv`. It settled the
            convention (keep the processor's geometry) and produced note 2 below.
      - [x] ★ **(b) settled — and the answer is do not adopt it, on this data.**
            `experiments/input_path_protocol.py` → `results/input_path_protocol.csv`.
            Note 2's evidence does not survive the protocol; the detail is under note 2.
      - [x] **(c) is therefore what stands**, and the write-up must say so: the searched
            preprocessing switches apply to the search and ablation caches only. Recorded
            in `thesis/protocol.md`, "The pipeline, as intended".
    - [x] ★ Update `protocol.md`'s framing either way. It calls `preprocess.py` "the single
          preprocessing path"; for the main caches it is not a path at all, and that sentence
          should not survive into the thesis unqualified. **Done 2026-09-08** — the section
          now carries the decision and the reason it is provisional.
  - [x] ★ **WP3-T3 note 2 — ConvNeXtV2's false-play was mostly the input path, and this may
        reverse the production recommendation.** *(Tested under the protocol and refuted —
        see note 3. Kept as written, because what it observed is real and only its
        interpretation was wrong.)* `preproc+geom` against the published
        `raw+geom`:

        | backbone | Δrecall | Δfalse-play | Δbalanced |
        |---|---|---|---|
        | convnextv2 | +0.0736 | **−0.9712** | **+1.0448** |
        | dinov2 | +0.0298 | −0.0781 | +0.1079 |
        | vit | +0.0428 | +0.1358 | −0.0930 |

        **ConvNeXtV2's 0.9918 false-play — the number behind "says PLAY to almost everything",
        and part of why the production pick moved to DINOv2 — is 0.0206 when the frame is
        letterboxed instead of handed raw to the processor.** A 1920×1080 frame resized
        shortest-edge to 256 and cropped to 224 keeps about the middle *half* of the pitch:
        the model was shown a central strip and asked whether the pitch was empty.
    - [x] Under preprocessing ConvNeXtV2 leads on **both** axes (0.9841 / 0.0206 vs DINOv2's
          0.9595 / 0.2305) *and* is the fastest. **Do not change the recommendation on this
          evidence** — settle it under the full protocol first (CIs, paired test, effective
          sample). Flagged in `rq_matrix.md` RQ2. **Settled 2026-09-08: the recommendation
          does not change, and the caution was justified.**
  - [x] ★ **WP3-T3 note 3 — the protocol run, and note 2 does not survive it.**
        `experiments/input_path_protocol.py` → `results/input_path_protocol.csv`, 21 tests
        in `tests/test_input_path_protocol.py`. No cache was rebuilt: both arms already
        existed. Full write-up in `results/EXPERIMENT_LOG.md`.
    - [x] ★ **The camera swap breaks it.** Note 2's entire false-play column is *one*
          measurement in one direction — train venue_01 camera A, score camera B. Swapping
          the cameras is the only replication this corpus allows, and it gives:

          | backbone | train A → B | train B → A | |
          |---|---|---|---|
          | convnextv2 | 0.9918 → 0.0206 (−0.9712) | 0.0279 → 0.0159 (−0.0120) | replicates |
          | dinov2 | 0.3086 → 0.2305 (−0.0782) | 0.0000 → 0.9761 (**+0.9761**) | **REVERSES** |
          | vit | 0.8354 → 0.9712 (+0.1358) | 0.0000 → 0.0000 | no effect one side |

          **ConvNeXtV2 has no defect to repair in the other direction** — trained on camera B
          its raw false-play is already 0.0279, so the 0.99 the letterbox "fixes" belongs to
          one training camera, not to the input path. **DINOv2 reverses outright.** The swap
          is a replication, not a mirror, and the asymmetry is recorded: all six C3 frames
          sit on camera A, so training on camera B is a two-class fit on 521 frames against
          camera A's three-class fit on 775.
    - [x] ★ **The recall axis could not have been significant.** Seven venue folds, paired,
          exact sign-flip test. Every CI covers zero — but two folds tie for every backbone,
          so five informative pairs put the **floor at 2/2⁵ = 0.0625**. Reporting "p = 0.31,
          not significant" without that would describe the sample, not the effect — H4's
          error one level down. Six same-signed venues is the minimum that can clear 0.05.
          `evaluation.stats.sign_flip_test` now reports `min_achievable_p` beside every p.
    - [x] ★ **Frame-level significance dissolves when the frames are counted.** The 243 (and
          251) empties come from **two (camera × slot) cells**. Recounted one frame per
          distinct scene: at 6 bits nothing survives in either direction; at 2 bits only the
          ConvNeXtV2 train-A comparison (p = 0.047), which the swap has already localised.
    - [x] **The "one seed" gap was a phantom.** `LinearProbe` passes its seed to
          `LogisticRegression`, which solves with lbfgs — deterministic. Verified across five
          seeds: not one prediction changes. More seeds would have dressed a fixed quantity
          as a robustness check. Another knob that turns nothing, harmless this time.
    - [x] Preprocessing is **not** universally good, and the swap makes that stronger rather
          than weaker: "helps ConvNeXtV2 hugely, DINOv2 modestly, hurts ViT" is itself a
          one-direction statement. Which backbone ships does not determine what to adopt —
          on this data nothing determines it.
    - [ ] ★ **[H] What would settle it is data, not method: empty-pitch footage from a second
          venue.** Every EMPTY frame in the corpus is venue_01, two cameras, two slots, so no
          test of any design can separate the input path from the camera pair. Same blocker
          as C3 (WP2-T11), RQ6's calibration and the confidence thresholds. Add it to
          `thesis/data_requests.md` (WP2-T2) as a named priority, not a general ask.
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
- [x] **WP3-T7 Class balancing.** `class_weight='balanced'` + optional weighted sampling, default ON
      for 3-class runs. *Accept:* C3 recall improves on validation vs unweighted.
  - [x] Default is already ON. `experiments/class_balancing.py` → `class_balancing.csv`.
  - [x] **The acceptance criterion is unmeasurable.** C3's six frames are one moment from one
        camera, so it lands on a single side of *every* leakage-free split — zero folds have
        it on both sides. A random-split number would measure memorisation of 30 seconds.
  - [x] **Kept anyway, for a better reason:** unweighted looks +0.0286 better on cross-venue
        recall, but the folds are 100% ACTIVE_PLAY so that recall is free. On 243 held-out
        EMPTY frames it calls **46.5%** of empty pitches a match, against 23.1% balanced.
  - [x] The first control used 9 EMPTY frames and said 0.000 for both — it would have led to
        the wrong recommendation. Splitting on physical camera gives 243 and reverses it.
- [ ] **WP3-T8 Preprocessing ablation (E-PRE).** *(search done; see the correction below)* Best model + grouped split; toggle
  - [x] ★ **[B] Before quoting any searched result: the search's resolution floor is
        0.0119, and it has adopted switches on margins below it** — measured 2026-09-08 by
        `experiments/search_resolution.py` → `results/search_resolution.csv`, 11 tests, stage
        `search-resolution`. **Neither fix was "re-run the search":** all 88 evaluations still
        have their caches, so this is re-scoring. Both published tables reproduce first — 88
        unweighted recalls and all 52 repaired false-play rates.
    - [x] ★ **RETRACTION: the CLAHE gate is weak, not vacuous.** The diagnostic this bullet
          used to quote said `auto` fires on **99.81%** of frames and differs from `on` on
          **3**. It measured contrast on the letterboxed 224×224 *output*; the gate runs in
          the photometric stage, **before** the resize, so it tests the full-resolution
          frame. Counted by running the switch both ways — which assumes nothing about which
          image is measured — it fires on **78.33%** and the two differ on **342** frames.
          The "0.19% of the input moves the metric 0.02" argument collapses with it.
          `preprocess.py`'s docstring and the log entry both carry the correction.
    - [x] **The floor survives, on independent evidence.** It falls straight out of the fold
          structure: one frame in `f_outdoor_bldg` (n=12) is worth **0.0119** of an
          unweighted mean over seven folds. And the bootstrap interval over folds is
          **0.092** wide at the median configuration — the more honest floor, and wider than
          almost everything that separates these configurations.
    - [x] ★ **(a) Weighting changes what the search adopted, in 3 of 6 rounds.** Re-ranked on
          pooled recall over held-out play frames: ConvNeXtV2 rounds 2 and 3 and DINOv2
          round 3 all pick a different configuration. Two rounds were decided on margins
          **below the floor** (0.0094 and 0.0110). The greedy search compounds it — a round-2
          choice conditions every round after it.
    - [x] ★ **(b) The floor is now quotable beside searched results**, and the CSV carries
          `unweighted`, `weighted`, `ci_low`, `ci_high` and a recomputed `false_play` per
          evaluation, so no configuration can be quoted without its width.
    - [ ] ★ **What follows: treat the search's output as candidates, not a ranking.** A
          configuration is adopted on evidence only when its margin clears the floor *and*
          its false-play control agrees. Re-read WP3-T2's adopted switches against this
          before any of them reaches the thesis as a finding.
    - [x] `clahe_contrast_below = 40.0` joins the hand-picked constants never calibrated
          against this footage, alongside the two confidence thresholds that default to 0.0
          (WP6-T5). Set it from the measured distribution as part of this ablation.
          **Measured:** the distribution the gate tests runs 18.9–56.8, median **30.9**, p90
          **43.2** — so 40.0 sits above the 90th percentile and applies CLAHE almost
          everywhere while claiming to be selective. Calibrated candidates: median 30.9
          ("the darker half") or p10 20.4 ("the worst tenth"). Which one is the ablation's
          decision; 40.0 is not among them.
      {ROI, letterbox-vs-thumbnail, CLAHE, augmentation, balancing} one at a time; deltas with CIs
      → `results/ablation_preprocessing.csv`. *Accept:* table + one-paragraph finding per switch.
  - [x] ★ **Regenerated 2026-09-07, once the run finished.** The search completed at 88
        evaluations (stamp `2026-09-07T20:34`) and released its lock; the viewer is rebuilt from
        it, so the withdrawn 16-evaluation snapshot is no longer what a reader opens.
        *Original note kept below for the record:*
  - [x] ~~**[B] Regenerate `results/preprocess_search.html` — the committed copy is a snapshot of
        the disowned run.**~~ Its embedded payload is stamped `2026-09-06T16:50:02` with **16
        evaluations**: the 500-frame run the project itself renamed
        `preprocess_search_500frame_UNTRUSTWORTHY.json` (stamped 16:55 the same day). The current
        search has **77**. The page states its own provenance in the meta line, but nothing on it
        says that particular run was withdrawn, so anyone opening the committed HTML — a
        supervisor, an examiner — reads retracted numbers as current. **Not regenerated during this
        review on purpose:** a search was in flight and writing `preprocess_search.json`, so
        rebuilding then would only have swapped a withdrawn run for a half-finished one. Do it once
        the run completes: `uv run python experiments/make_search_viewer.py`.
  - [x] ★ **Staleness guard added** → `tests/test_search_viewer_current.py`, in the same shape
        as `tests/test_branch_report.py` and for the same reason: a generated artefact that can
        silently fall behind its source will. Three assertions — the page embeds the current
        run's `generated` stamp, every evaluation in the source appears in the page (a subset
        check, because the `best` block repeats its winners' hashes and an exact count reads 90
        for 88), and the source is not the withdrawn 500-frame run, so re-publishing that one
        needs a deliberate edit. Verified both ways: green on the current page, and it fires when
        the stamp is rolled back.
- [x] ★ **H5 reported (2026-09-08) — one clause unrunnable, one refuted.**
      `experiments/h5_preprocessing_switches.py` → `results/h5_preprocessing_switches.csv`,
      7 tests, stage `h5-preprocessing`, amendment A11. Every arm's published recall
      reproduces first. **All six pre-registered hypotheses are now reported.**
  - [x] **ROI clause: unrunnable, not refuted.** No `configs/cameras.json`, `roi_mask`
        returns the frame untouched without a polygon, all 88 evaluations carry `roi:
        False`, and `SWITCHES` deliberately excludes `roi` so the search cannot record a
        false null. A null result here would be manufactured. **Blocked on WP3-T1.**
  - [x] ★ **CLAHE clause: refuted, and in the opposite direction.** Grouped split, macro-F1,
        paired, Holm over the realised family of 4: **all four significant, all four
        negative.** ConvNeXtV2 −0.008/−0.009; DINOv2 **−0.27**, which collapses it onto
        roughly the score of a model that never recognises an empty pitch. Cohen's g ≈ 0.5,
        so the disagreements are wholly one-sided; 0 of 4 night splits improve.
  - [x] ★ **And "on the night subset specifically" is untestable here.** It needs a day
        column, and venue_01 has exactly two recording days — holding whole slots out fills
        the test side from one of them. Across five seeds four test sets are entirely night
        and the fifth entirely day; **no grouped split holds both**. The day figures come
        from a different partition and are reported as an observation, never a contrast.
        This is a scope limit, and belongs on the preregistration's "not answerable" list.
  - [ ] ★ **What this means for adopting any switch.** CLAHE was a candidate on the search's
        cross-venue metric and is harmful on the grouped one. Read with WP3-T8's floor, the
        rule is now explicit: **a switch is adopted only if it clears the resolution floor on
        the ranking metric *and* does not lose macro-F1 on the honest split.** Re-read
        WP3-T2's adopted switches against both before any reaches the thesis.
- [ ] **WP3-T9 ★ ROI ablation gets its own figure.** Of all preprocessing steps, ROI masking is the
      one with a *visual* story (adjacent pitches firing false ACTIVE_PLAY). Pair the number with
      side-by-side example frames — it will be one of the most quoted figures in your defence.

---

## WP4 — Leakage-free benchmark & statistics
*Weeks 9–13 · ~110 h · M3 gate*

### 4.A The headline experiments
  - [x] ★ **H3 re-reported with a false-play control** (`experiments/h3_with_false_play.py`).
        Recall reproduces the published table exactly, then the second column reframes it:
        ConvNeXtV2 calls **99.2%** of held-out empty pitches a match, ViT 83.5%, DINOv2
        30.9%, and the clock rule — the straw man — just **2.1%**. On `recall − false-play`
        ConvNeXtV2 scores *below* a rule that never looks at the image.
- [x] **WP4-T1 Same-scene vs grouped vs leave-one-venue-out — done, 8 models × 4 protocols.**
      `experiments/benchmark_v2.py` → `results/benchmark_v2.csv` +
      `benchmark_v2_protocols.json`, 14 tests, stage `benchmark-v2`. The seed-42 random and
      grouped rows are checked against `h1_h2_baseline_floor.csv` before anything else runs.
      Full write-up in `results/EXPERIMENT_LOG.md`. **(RQ3)**
  - [x] ★ **The headline is not the delta — it is that a constant predictor wins a
        protocol.** `majority` reads no pixels and scores macro-F1 **1.0000** on
        leave-one-venue-out, ahead of DINOv2's 0.9595, because every held-out clip venue is
        100% ACTIVE_PLAY and the macro average is over the one class present. H3 and the
        false-play control had shown cross-venue *recall* was free on single-class folds;
        this is the same defect reaching the headline metric. `clip_zeroshot` at 0.9907,
        above every trained backbone, is the same artefact from the other side.
    - [x] The floor is now quantified per protocol rather than asserted: trivial-vs-best-
          backbone gaps are +0.023 (random), +0.071 (grouped), **−0.041 (cross-venue)**,
          +0.182 (temporal).
  - [x] ★ **A third of H1's drop is test-set composition, not leakage.** A model that never
        trains cannot leak, so OpenCLIP scored zero-shot on the identical test sets measures
        the composition effect directly: it drops **0.1834** from random to grouped. Raw
        drops of 0.516–0.578 become leakage-attributable drops of **0.332–0.395**.
        `cheap_intensity`, with almost nothing to memorise, has an attributable drop of
        0.0094 — a sanity check pointing the right way. Stated as a **control, not a
        proof**: it assumes the composition effect is additive and similar across models.
    - [x] The control's prompt set is fixed in advance (first descriptor per class, five
          templates) and is deliberately **not** the prompt search's winner, which was
          selected on the folds it reports. A test pins that they differ.
  - [x] ★ Add the **temporal split** as a 4th strategy (from WP0-T4) — measures drift, costs one
        extra run on cached features. **(RQ3)** *Done, and it does not measure drift:* day one
        is 97.6% EMPTY and all daylight, day two 98.9% ACTIVE_PLAY and all floodlit, so it is
        a class-and-lighting flip. Report it as a demonstration that this corpus cannot
        measure drift.
    - [x] Setting it up found `temporal_split` placing **undated frames in train** by string
          comparison (`"" < "2026-07-12"`) — and all 282 undated frames are the seven clip
          venues, none of which appear on the test side. The split called "temporal" was a
          venue-and-time split and said nothing. It now takes an explicit `undated` policy,
          default `exclude`; 5 tests.
  - [x] ★ Report **mean ± std over ≥5 grouped splits**, not one split. A single split's number is a
        sample of size one, and reviewers treat it as such. *Done, and the spread is large:*
        **±0.240** for DINOv2 and ±0.203 for ConvNeXtV2 on a mean of 0.47. Every published
        grouped-split number uses `seed=42` alone; they reproduce exactly, but the
        seed-to-seed variation is the same order as the between-model differences, so **no
        ranking on the grouped split is supported by one split.**
  - [ ] ★ **[B] Quote no benchmark_v2 number without its diagnostics.** Test size, class
        count and majority share ship in the same CSV row for exactly this reason. The four
        columns together say: this corpus can measure occupancy only inside one venue, and
        generalisation only in the direction where the answer is always yes.
- [x] **WP4-T10 ★ Trivial-baseline floor — done, and it fired.** `experiments/h1_h2_baseline_floor.py`.
      Under the random split a **16-bin colour histogram beat ConvNeXtV2 on macro-F1** (0.686 vs
      0.657); under the grouped split the **clock rule came within 0.007** of it using no pixels at
      all. The comparison was indeed measuring the wrong thing — established in week 1, not at the
      defence. **(RQ7)**
  - [x] ★ The complement: `experiments/h3_cross_venue_recall.py` shows a lighting-only rule
        collapsing on unseen venues while ConvNeXtV2 and DINOv2 hold at 0.910 / 0.930. The
        evaluation was uninformative; the models were not the problem. **(RQ3, RQ7)**
        *The measured 0.219 is a lower bound partly produced by label error — quote the
        collapse, not the number, until WP3-T5's relabelling lands.*
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
  - [x] The machinery is done and honest: `risk_coverage_band` reports the curve as a band,
        because on a saturated probe confidence does not order the frames it is being asked
        to order.
  - [x] ★ **The figure is drawn** (2026-09-08) → `results/figs/risk_coverage_band.{png,pdf}`,
        8 tests in `tests/test_figures.py`, declared in the `figures` stage. Two panels:
        accuracy with the band shaded and the **worst case** as the solid edge — the same
        bound `coverage_for_target_accuracy` reads — and beneath it the width the confidences
        leave, because one axis cannot carry both "what is the accuracy" and "how determined
        is that answer".
    - [x] ★ **The figure's own finding: the two best-looking curves belong to the two models
          that never predict EMPTY.** ConvNeXtV2 and ViT sit at 1.000 until 88% coverage
          because they are right on all 898 active-play frames and wrong on all 9 empty ones,
          so confidence ranks the nine last. A plot of curves alone would recommend them.
    - [x] The test renders the figure and asserts the heavy line's y-values are the
          worst-case column — the property, not the source text. Verified both ways.
  - [ ] The **operating point** still needs the class mix WP4-T5 is blocked on; what the
        figure reports now is what this test set can support, which is a band.

### 4.B Making the numbers defensible
- [x] **WP4-T4 Statistical testing.** *(applied to the false-play finding; see below)*
  - [x] `experiments/false_play_significance.py` -> `false_play_significance.csv`. Bootstrap
        CIs, pairwise McNemar, Holm-Bonferroni across the family of six.
  - [x] **It overturned the claim it was meant to support.** All six comparisons look
        significant at p down to 8e-53 - but the 243 held-out empty frames are consecutive
        views of one camera and amount to **3-10 distinct scenes**. De-duplicated, **0 of 6**
        survive. The point estimates stand as observed behaviour; the ranking is not
        statistically established. Underpowered, *not* null - the gap is 0.99 vs 0.31.
  - [x] ★ Same treatment applied to H1/H2 (`experiments/effective_sample_audit.py`). The
        394-frame leaky test set is **62 distinct scenes**, the 907-frame grouped set **94**.
        One comparison dies: on distinct scenes DINOv2 and the colour histogram both score
        **1.0000** under the leaky split, so H1 should say the protocol *cannot tell them
        apart* rather than that the histogram beat the probe - a cleaner form of the same
        finding. H2 survives but is now marginal (p 1.2e-3 -> 2.2e-2).
  - [x] **H3 needs no correction**: its intervals bootstrap over the seven venue folds, not
        over frames. Matching the resampling unit to the thing being generalised over is
        what made it robust.
- [ ] **WP4-T4b Statistical testing for the remaining hypotheses.** Bootstrap CIs + McNemar (frame level) + paired bootstrap (slot
      level) for every headline pair. *Accept:* every claim carries CI + p-value columns.
  - [ ] ★ Apply the Holm–Bonferroni correction from WP0-T6 and say so in the caption.
  - [ ] ★ Report effect sizes next to p-values.
  - [x] ★ **H4 reported** (2026-09-07) → `experiments/h4_model_equivalence.py`,
        `results/h4_model_equivalence.csv`. First of the three pre-registered hypotheses that
        had never been reported. **Refuted as a whole**, and instructively:
    - [x] The accuracy clause is confirmed **degenerately** — ConvNeXtV2 and ViT differ by
          exactly 0.0000 with a *zero-width* interval, because their predictions are
          identical: ACTIVE_PLAY for all 907 frames, EMPTY **zero times**, C1 F1 = 0.000.
          0.4975 is (0.995+0)/2, the score of a model that never recognises an empty pitch.
          Equivalent to each other, and equally equivalent to a constant predictor.
    - [x] The speed clause **fails under the condition the pre-registration names**: 1.83×
          on the 20-camera concurrent median against the ≥2× required. It clears 2× only on
          the single-frame median, which that document declined to rely on. Memory-bandwidth
          contention compresses the gap, as WP0-T10 warned.
    - [x] ★ **The pre-registered decision rule was disowned, not followed.** *"A null result
          confirms H4"* is absence-of-evidence, and on a 99%-single-class test set a null is
          near-guaranteed. Replaced with an interval test against a margin declared in
          advance (±0.02, borrowed from H2's threshold). It matters concretely: under the old
          rule all three pairs "confirm" equivalence, including two where DINOv2 is **8.2
          points** better and the interval reaches −0.23. Those are **inconclusive** — the
          honest third answer the original rule collapses away. Amendment A9.
    - [x] ★ **Closed a statistical gap in the library.** The comparison needed a paired
          bootstrap on *macro-F1* and none existed — `mcnemar` and `paired_bootstrap_diff`
          both work on per-frame correctness, i.e. accuracy. That absence is exactly why H2
          was published with a macro-F1 delta beside an accuracy p-value that for one pair
          pointed the other way. `paired_bootstrap_metric_diff` added, 8 tests, including
          two models with identical accuracy that macro-F1 separates.
  - [x] ★ **H6 reported (2026-09-08) — inconclusive, and the interesting part is why.**
        `experiments/h6_zero_shot_gap.py` → `results/h6_zero_shot_gap.csv`, 9 tests, stage
        `h6-zero-shot`, amendment A10. Both of its blockers had just been removed for other
        reasons: the CLIP cache now covers all 1,578 frames (it held 600), and `benchmark_v2`
        needed a prompt set declared in advance, now `vision.zeroshot.DECLARED_PROMPT_SET`.
    - [x] **Not confirmed.** Zero-shot is significantly worse than **0 of 3** probes and
          significantly **better** than all three (Δ +0.050 to +0.132 macro-F1, Holm
          p ≤ 1.5e−05), winning on 5 of 5 grouped splits. Realised family **3**, not the 4
          anticipated — declared in A10, since OpenCLIP *is* the zero-shot arm.
    - [x] **Not refuted either, and this is the check that earns the experiment.** Scoring
          all **375** prompt sets in the declared space (nothing selected — a distribution,
          not a search): the declared set is above **85%** of them, only **22.9%** beat the
          best trained probe, and the **median** set (0.4967) loses to it. A verdict that
          depends on which prompt was declared is not a verdict.
    - [x] The frame-level significance also fails the effective sample: 907 frames are **95
          distinct scenes**, and recounted on those no comparison survives.
    - [x] ★ **The finding worth keeping is the spread.** Prompt choice moves macro-F1 by
          **0.726** (0.021–0.747); the three backbones span **0.082**. The prompt matters
          roughly nine times more than the model, so the cost of a no-label deployment is not
          a fixed penalty but a wide distribution whose position cannot be known at a new site
          *without* the labels that would make it unnecessary. Sharper than H6 asked, and
          operationally worse. Feeds RQ1.
    - [x] ★ **A6 quantified.** The search's winner leads the declared set by **+0.4474**
          balanced score on the folds it was selected from. A6 called that margin
          "optimistically biased"; this is how much, and it is why H6 does not use it.
- [x] **WP4-T5 Calibration — built, answer blocked.** `evaluation/calibration.py` (ECE,
      reliability bins, temperature scaling, risk-coverage), 25 tests. The run produces
      "99% precision at 0% review", which is an artifact of the 99% single-class test set.
      A boundary warning caught DINOv2's temperature pinning at the grid floor. **No further
      engineering unblocks this** — it needs a test set with a real class mix.
  - [x] ★ **And that boundary warning was pointing at something the curve was hiding**
        (2026-09-08). The pinned temperature sharpens DINOv2's probabilities until **890 of
        907** calibrated confidences are *exactly* 1.0. "The most confident k" is undefined
        inside a tie, so the published curve was one arbitrary ordering of a single tied
        block — which is why 57 of its 180 rows moved, by up to **0.125**, once the split's
        row order was made reproducible. `risk_coverage_band` replaces it: best and worst
        accuracy at each coverage, coinciding wherever confidences are distinct.
        `coverage_for_target_accuracy` now reads the **lower** bound. Reported operating
        points are unchanged (99% precision at 97.9% coverage) but are now identified rather
        than coincidental, and the tie counts ship in both CSVs.
- [ ] ~~WP4-T5 original~~ Reliability diagrams + ECE per model; temperature scaling fitted
      on validation (within training venues only); effect on REVIEW-band volume. *Accept:*
      `results/calibration.csv` + figures; calibrated heads saved with `temperature` in the pkl.
- [x] **WP4-T6 Error taxonomy — done; the explainability half is blocked, not forgotten.**
      `experiments/error_taxonomy.py` → `results/error_taxonomy.csv` +
      `error_taxonomy_summary.csv`, 14 tests, stage `error-taxonomy`. Run across **all four**
      protocols, because on the grouped split alone the table is nearly empty.
  - [x] ★ **The headline is a direct measure of what leaked: 100% of leaky-split errors had
        a near-duplicate on the training side (12 of 12); 0% of honest-split errors did (0 of
        32).** On the leaky split even the frames the models get *wrong* are frames they had
        seen a copy of. The near-duplicate audit said 37.1% of duplicate pairs straddle a
        random split; this says what that means for the evaluation, and it is the cleanest
        single number for H1 so far. It is also a check on the split — a non-zero count on
        the grouped side would be a defect in the partition — and a test pins both halves.
  - [x] ★ **Only one protocol has an error set worth categorising.** Grouped (32 errors, 1
        slot) and temporal (127, 1 slot) are a single failure counted many times; cross-venue
        (50 errors, 16 slots, both lighting conditions) is the only spread.
        `clipvenue_h_teal_pitch` is the hardest venue, taking 25–43% of each model's
        cross-venue errors.
  - [x] **DINOv2 collapses under the day→night shift in a specific way**: 92 errors of 799
        against ConvNeXtV2's 11, and **88% are C2→C3** — it calls active play *maintenance*.
        ViT does the same more mildly; ConvNeXtV2's errors go the other way (82% C1→C2).
  - [x] **The grouped split cannot separate the models.** Two of three make exactly nine
        errors, all the same confusion, all in the same slot — not similar sizes, the same
        failure.
  - [x] The finding sentence per class is **derived from the counts**, not typed beside them:
        a hand-written sentence next to a generated table is how a figure came to contradict
        its own source.
  - [ ] ★ **[B] Attention rollout + Grad-CAM → `results/figs/xai/` — blocked on WP1-T5.**
        It writes frame images, and nine images are already permanent in git history
        including five sheets of unblurred players. Publishing more frames is the open
        data-release decision. Worth doing once that is settled; recorded rather than left
        as a silent gap in the acceptance criterion.
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

> ### ★ Order changed 2026-09-07 — do 5.B before 5.A
>
> **The tier labels were assigned before the data limits were known, and they now point at the
> wrong module first.** STAN (5.A, "Core tier — do this one first") is the *only* WP5 module that
> is data-blocked: it is a **slot** classifier, its real test set is n = 2, and WP5-T8's own hard
> gate forbids reporting it as a headline below 30 real slots. Gated multi-backbone fusion (5.B,
> "Target tier") is **frame**-level and blocked by nothing — the three backbone caches already
> cover all 1,692 frames, the cheap gate statistics are already cached
> (`cheap_histogram.npz`, `cheap_intensity.npz`), and the evaluation protocol, split machinery
> and significance tooling all exist. It can be fitted in seconds on cached features.
>
> **So 5.B carries the M4 gate and 5.A becomes the preliminary result the pre-registration
> already says it must be.** Nothing is dropped and no research question changes; the module that
> can actually be ablated goes first.
>
> **The headroom is measured, not assumed.** From the per-fold table in
> `results/h3_with_false_play.csv`, cross-venue play-recall by held-out venue:
>
> | held-out venue | ConvNeXtV2 | DINOv2 | ViT | best |
> |---|---|---|---|---|
> | a_blue_barrier (n=168) | 0.9345 | 0.9524 | **1.0000** | ViT |
> | f_outdoor_bldg (n=12) | 0.7500 | **1.0000** | 0.5833 | DINOv2 |
> | h_teal_pitch (n=18) | **0.7222** | 0.6667 | 0.5000 | ConvNeXtV2 |
> | i_outdoor_trees (n=18) | **1.0000** | 0.8889 | 1.0000 | ConvNeXtV2 |
> | *(d, e, g: all three at 1.0000)* | | | | tie |
> | **unweighted fold mean** | 0.9105 | **0.9297** | 0.8690 | |
>
> DINOv2 leads overall but is **strictly beaten on three of the seven folds**. A per-fold oracle
> scores **0.9603** against DINOv2's 0.9297 — **+3.1 points of headroom**, which is the
> complementarity a gate would have to capture. There is a real signal here.
>
> **And the two things that must be said in the same breath, or this becomes a fishing trip:**
> 1. **The other axis is dominated.** On held-out empty frames the false-play rates are DINOv2
>    0.309, ViT 0.835, ConvNeXtV2 0.992. Any gate that routes away from DINOv2 on an actually-empty
>    pitch is catastrophic there, so the ablation must report **both** axes — recall *and*
>    false-play — never recall alone. Reporting recall alone is the exact mistake the repaired
>    search control already caught once.
> 2. **The gate's inputs are confounded.** It sees cheap image statistics, and at `venue_01`
>    brightness/contrast is nearly a day/night indicator, which is nearly the class label. A gate
>    trained on those risks re-learning the clock rule and scoring well for the wrong reason —
>    the same confound that has now appeared four times in this project. Check it explicitly.
>
> +3.1 is an *oracle* ceiling that assumes perfect per-venue selection the gate does not have, so
> the realistic gain is well below it and a **clean negative result is a likely and acceptable
> outcome** — which the rules above already accept, provided WP5-T9's logit-average baseline is
> run first so "the ensemble did it, not the gate" is ruled out rather than left open.
>
> ### ★ WP5-T9 has now been run, and it weakens this recommendation. Read it before starting 5.B.
>
> Both caveats above were right, and the second one turned out to be decisive rather than a
> risk to monitor. `results/logit_average_baseline.csv`, full entry in `EXPERIMENT_LOG.md`:
>
> | | play recall | false-play |
> |---|---|---|
> | DINOv2 alone | 0.9297 | **0.3086** |
> | best naive ensemble | 0.9524 | **1.0000** |
> | *oracle ceiling* | *0.9603* | *—* |
>
> **A parameter-free average already captures 74% of the +3.1 headroom**, leaving 0.008 for a
> learned gate — which cannot reach the oracle anyway, lacking venue identity. And **every
> ensemble scores 1.0000 false-play**, worse than every single backbone: blending ConvNeXtV2 in
> destroys the one property that makes DINOv2 worth having. On the balanced view the naive
> ensemble is *strictly worse than DINOv2 alone*.
>
> So **blending is disqualified**, whatever the weights. A gate here can only be a **hard router
> with DINOv2 as the default**, invoking another backbone where it is confident the scene is not
> empty — which is the confound in caveat 2, now as the whole module rather than a footnote.
>
> ### ✔ Resolved 2026-09-08 — the conclusion holds, the reasoning is now the opposite
>
> It rests on ConvNeXtV2's false-play of 0.9918, and the geometry probe indicates that figure
> is largely an artefact of the **input path**. Every cache WP5-T9 used came from *raw* frames
> handed to the HF processor, which keeps roughly the middle half of a 16:9 pitch. Under
> `preprocess.py`'s letterbox ConvNeXtV2 measures **0.0206** false-play with recall **0.9841** —
> better than DINOv2 on *both* axes, which inverts the premise.
>
> **The re-run settles it** (`logit_average_baseline_preproc.csv`). On letterboxed caches the
> two-model ensemble scores **0.0288** false-play at **1.0000** recall — the best balanced
> score measured on this dataset, better than any single backbone. So blending is **not**
> disqualified; that was an artefact of showing ConvNeXtV2 a central strip of the pitch.
>
> **5.B's answer is unchanged and its reasoning is stronger.** The naive average now captures
> **100%** of the oracle headroom (was 74%), so there is nothing left for a gate to learn —
> and that argument no longer rests on the false-play axis or on any claim about routing.
> Report WP5-T2 as a negative result against the naive ensemble.
>
> Caveat that keeps this honest: the recall axis is **saturated** (oracle 1.0000), so the
> headroom was only +0.0159 to begin with, and the ensemble leads ConvNeXtV2 alone by 0.008
> on balanced score — not a difference on three to ten distinct scenes. Under preprocessing
> ConvNeXtV2 and the two-model ensemble are indistinguishable; DINOv2 is behind; ViT is poor.
>
> **Plan 5.B as a negative result and it is worth doing; plan it as a win and it will not
> survive the defence.** Build it to be reported either way, against these numbers as the
> baseline, with the lighting-only gate ablation first. Under the WP5 rules a rigorous negative
> result is a contribution — and this is a far better place to learn it than week 18.
>
> *(The +0.023 recall gain is **not** tested — an unweighted mean over 7 folds, three of them
> 12–18 frames. Do not quote it as an improvement. It does not change the direction of the
> comparison, which the false-play column settles on its own.)*
>
> ### ⚠ Which input path this rests on — 2026-09-08, after `input_path_protocol.csv`
>
> The re-run above reads the **letterboxed** caches, on the premise that they are the truer
> input. That premise did not survive the protocol: the false-play evidence for it reverses
> when the two cameras it is built from are swapped, so neither cache family is established
> as the real one, and the letterboxed ensemble's 0.0288 is a number from one of two paths
> rather than the number.
>
> **5.B's answer survives either way, and that is the point worth carrying.** Raw caches said
> *blending is disqualified, so a gate can only be a hard router*; letterboxed caches say *the
> ensemble is already at the ceiling, so a gate has no room.* Opposite mechanisms, same
> conclusion — plan WP5-T2 as a negative result. **What must not be quoted is either
> mechanism as established**, and in particular not "the two-model ensemble is the best
> configuration measured", which holds only on the unestablished path.

### 5.A STAN — Slot-Temporal Aggregation Network *(preliminary result — data-blocked; see the order note above)*
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

### 5.B Gated multi-backbone fusion *(Core tier — do this one first; carries the M4 gate)*
- [x] **WP5-T9 ★ Sanity baseline for fusion — done, and it answered the question.**
      `experiments/logit_average_baseline.py` → `results/logit_average_baseline.csv`, 10 tests,
      in the reproduction pipeline as `logit-average`. Single-backbone means reproduce the
      published H3 table exactly before anything is added. **(RQ5)**
  - [x] **A parameter-free average captures 74% of the oracle headroom** (0.9524 against a
        0.9603 ceiling, best single 0.9297), leaving 0.008 for a learned gate that cannot reach
        the oracle anyway because it has no venue identity. The plan's own test — *if a naive
        ensemble matches the learned gate, the ensemble is the contribution* — is close to met
        on recall before the gate exists.
  - [x] **And every ensemble scores 1.0000 false-play**, worse than all three backbones alone.
        Blending ConvNeXtV2 (0.9918) in destroys DINOv2's one valuable property (0.3086): the
        ensembles call **100% of 243 held-out empty frames** a match. +0.023 recall for +0.69
        false-play — strictly worse than DINOv2 alone on the balanced view. Third time on this
        dataset that a configuration looked better on a single-class test set while being worse
        at the thing that matters.
  - [x] **Consequence: soft blending is disqualified for WP5-T2**, whatever the weights. See the
        order note at the top of WP5 — the module is now a hard-router-or-negative-result, and
        should be planned as the latter.
  - [ ] ★ **Not established: the +0.023 recall gain is untested** — unweighted mean over 7 folds,
        three of them 12–18 frames. A paired bootstrap over the fold distribution belongs with
        WP4-T4b. Do not quote it as an improvement; it does not change the direction of the
        comparison either way.
- [ ] **WP5-T2 Fusion head.** `engine/fusion_head.py`: features from {ConvNeXtV2, DINOv2} (option
      +ViT); gate = tiny MLP on cheap image statistics (contrast, brightness, edge density) → fusion
      weights → shared linear head. **(RQ5)**
  - [ ] ★ **Report both axes, never recall alone.** Cross-venue play-recall *and* false-play on
        held-out empty frames, in the same table. On the second axis DINOv2 (0.309) dominates ViT
        (0.835) and ConvNeXtV2 (0.992) outright, so a gain in recall bought by routing away from
        DINOv2 on empty pitches is not a gain. Recall-only on a single-class test set is the
        precise mistake the repaired search control already caught.
  - [ ] ★ **Test the gate for the confound before believing it.** The gate reads cheap image
        statistics; at `venue_01` brightness and contrast are close to a day/night indicator, and
        day/night is close to the class label. Ablate the gate against one fed *only* lighting —
        if a lighting-only gate matches it, the gate has re-learned the clock rule and the
        confound has appeared for the fifth time. Log the comparison either way.
- [ ] **WP5-T2b Conditional-compute variant.** Run ConvNeXt first; invoke DINOv2 only when confidence
      < τ. Report accuracy **and** average ms/frame vs always-both. *Accept:* ablation table with CIs,
      p-values, latency; adopt/reject decision logged.
  - [ ] ★ **State the honest motivation.** WP0-T10 measured 10–24× headroom in the 60 s cycle, so
        this variant does **not** buy needed speed on the dev machine and must not be sold as if it
        did. Its real justification is the *target Mini-PC* (WP7-T1, unmeasured) and the
        scaling claim beyond 20 cameras. Frame it that way or drop it to the extensions list.

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
  - [ ] ★ **Correction: "low-confidence slots are downgraded" is a wired code path, not an
        active protection.** Both confidence guards default to **0.0**, which is off —
        `Thresholds.review_below_confidence` and `reconcile(min_confidence=...)`. `conf < 0.0`
        is never true, no non-test caller raises either, and `experiments/end_to_end_slots.py`
        passes `review_below_confidence=0.0` explicitly. Only the tests set 0.6.
        **Third time in this project a documented guard turned out not to be guarding** — the
        final-test-set lock's cwd-relative path and the processor-geometry fingerprint were the
        others. Worth naming as a pattern: a guard with a permissive default reads, in code
        review and in prose, exactly like a guard.
    - [ ] **Do not fix it by picking a number.** That is the "hyper-parameters, not
          constants" mistake `aggregate.py`'s own header warns about. Calibrate it against
          the confidence distribution — `evaluation/calibration.py` already has the
          risk–coverage machinery, and this *is* RQ6's question in operational form: what
          REVIEW rate buys what verdict reliability. So it is blocked by the same degenerate
          test set, and should be set as part of answering RQ6 rather than before it.
    - [x] **The ethics commitment does not rest on this**, and that is worth stating so the
          finding is not read as worse than it is: a human confirms every anomaly and no
          automated financial action is taken (WP6-T12). This guard is defence-in-depth on
          top of that, not the thing standing between the audit and a false accusation.
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
- [x] **WP8-T5 ★ Claims ledger — built and executable.** `thesis/claims.toml` (the ledger),
      `experiments/verify_claims.py` (the verifier), `thesis/claims.md` (generated), 14 tests,
      stage `claims-ledger` — last, because it checks what every other stage writes.
      **18 claims, all verified, none unsupported.**
  - [x] A claim fails in three distinguishable ways and the output separates them, because
        they need different work: **stale result** (source moved, prose did not), **stale
        prose** (prose moved, ledger did not), **unsupported** (nothing checks it).
  - [x] ★ **It earned its keep on the first run.** Two claims quoted in live documents could
        not be re-derived from any artefact — the search's resolution floor lived only in a
        script's printed output, and H6's prompt-space span was absent from the CSV. Both
        experiments now store them. Two `where` fields were also wrong, which is the second
        thing it is for.
  - [x] `where` lists **live** documents, never `EXPERIMENT_LOG.md`: the log is append-only
        history, and requiring it to match current values would forbid keeping retractions.
  - [x] `reproduce_all.py` derives the stage's requirements **from the ledger**, because a
        hand-written copy would be the copy that goes stale.
  - [ ] ★ **Add a claim when you write the sentence, not at the end.** The ledger only covers
        what has been written down so far; every new quantitative sentence in the thesis needs
        an entry, and `--check` in CI would make that automatic.

### 8.B Figures
- [x] ★ **The README said the implementation had not started** (2026-09-08). It read
      "Planning complete. Implementation starting." through the whole period in which the
      implementation was written, six hypotheses were reported and four evaluation defects
      were found — the same failure as the stale branch tip and the stale export, with the
      widest audience of the three.
  - [x] The countable half is generated: `scripts/branch_report.py` now maintains a status
        block in `README.md` as well as the branch block in `docs/CODEBASE.md`, and
        `--check` reports either stale.
  - [x] The judgement half is written by hand and guarded: three tests — block current,
        counts match git, and the README does not claim the work has not started while
        results exist. Each verified by breaking it.
  - [x] ★ **A findings section added.** The README had none: a reader got the pilot's
        superseded accuracy table and nothing about what the project found. It now leads with
        the leakage decomposition, the constant predictor that wins cross-venue, the prompt
        variance and the risk–coverage band, each linked to its artefact — and names the
        defects, which are a result of this project rather than something to keep off the
        front page.
- [x] ★ **The standalone site had stopped covering the thesis, silently** (2026-09-08).
      `make_site.py` reads a fixed list of eight result files while its docstring claimed
      "the whole thesis"; the eleven experiments added since it was written were omitted with
      no signal — regenerating produced a byte-identical page. Found because H6's 79 log lines
      changed nothing. The **served** front end was never affected: `api/thesis_site.py`
      renders `EXPERIMENT_LOG.md` directly.
  - [x] WP4-T1 added to the page, where it overtakes the existing headline ("the protocol
        reverses the ranking" — on one protocol there is no ranking to reverse). Both new
        tables read from `benchmark_v2.csv`, including the figure quoted in the prose.
  - [x] `tests/test_site_coverage.py` makes omission a **decision**: every committed result
        file is either rendered or named in `OUT_OF_SCOPE` with a reason. 21 are out of scope.
        A companion test asserts the sections reach the **generated HTML**, not just the
        builder — the distinction the search-viewer safeguard failed on.
  - [ ] ★ Decide what else belongs on the export before the defence. The list is now
        explicit, so this is a review rather than an archaeology exercise; `rq6_risk_coverage`
        in particular needs its band drawn as a band (WP4-T9) before it can go on.
- [x] **WP8-T2 Figure set — three done.** `results/figs/`: label-efficiency curve,
      ranking-inversion slope chart, cross-venue per-fold recall. Regenerated from the CSVs,
      validated palette, PNG + PDF.
- [ ] **WP8-T2b Remaining figures.** Label-efficiency curve · leakage comparison bar · ablation tables ·
      reliability diagrams · XAI overlays · reconciliation matrix with real (blurred) evidence.
  - [x] ★ Risk–coverage / REVIEW-rate curve (WP4-T9). **Done** → `figs/risk_coverage_band`.
        Drawn as a band, with a second panel for the width the confidences leave.
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

> ### ★ Re-cut 2026-09-07, because two gates had become unpassable
>
> **Decision 0.3 closed the footage request. Nobody re-cut the gates that depended on it.**
> As originally written, M2 required "dataset balanced across classes/conditions/venues +
> ≥30 real slots labelled" and M4 required STAN "ablated with significance". Two real slots
> exist, C3 holds 6 frames and maintenance 0, EMPTY exists at exactly one venue, and the 66
> clips are 10–14 s highlights with no slot structure. No further footage is coming. So both
> criteria were, from the moment 0.3 was taken, **impossible** — and WP5-T8's own hard gate
> already says not to report STAN as a headline below 30 real slots.
>
> A gate that cannot be passed is not a gate. It is a permanent red row that trains everyone
> reading the table to ignore it, and it hides the fact that the *rest* of the milestone was
> met. The criteria below are re-cut against the data that actually exists. **The originals
> are kept, struck through, so the reduction is visible rather than quietly absorbed** — and
> each re-cut says what would restore the original.
>
> This changes no research question and drops no contribution. It changes what counts as
> done, to match what the evidence can support.

| Gate | Week | Exit criterion | Done |
|---|---|---|---|
| M1 | 4 | Protocol, taxonomy, evaluation design approved ★ + ethics/DPIA cleared | [ ] |
| M2 | 9 | ~~Dataset balanced across classes/conditions/venues + ≥30 real slots~~ → **Dataset characterised and its limits quantified**: coverage matrix published, concentration table showing EMPTY 100% venue_01, near-duplicate rate and per-split leakage measured, effective sample size (~150 distinct scenes) reported, and every unanswerable question recorded in `preregistration.md` §"Not answerable" | [ ] |
| M3 | 13 | Four-model leakage-free benchmark with CIs + significance ★ + trivial-baseline floor | [ ] |
| M4 | 18 | ~~STAN + ≥1 fusion module ablated with significance~~ → **≥1 frame-level novel module (gated multi-backbone fusion, WP5-T2) fully ablated with significance against a strong baseline — including plain logit averaging (WP5-T9)** — plus STAN reported as a preliminary result on synthesised sequences with its synthetic-to-real gap stated as unquantifiable | [ ] |
| M5 | 19 | End-to-end system + reconciliation on mock feeds | [ ] |
| M6 | 21 | 48-h live validation accepted | [ ] |
| M7 | 24 | Thesis submitted, defence ready | [ ] |

**What would restore the originals.** Both re-cuts are reversed by one thing: more footage.
M2's original needs empty-pitch and maintenance footage at **more than one venue**; M4's needs
**≥30 complete slots with known verdicts**. Both are cheap for the operator to supply and are
still listed under *Human-only tasks* — see the note under 0.3 about the one request worth
reopening. Until then, these are the gates.

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

- [x] **★ Minimum viable thesis defined** → **[`thesis/mvt.md`](thesis/mvt.md)** (2026-09-07).
      Four load-bearing items — leakage-free benchmark with honest statistics · the
      *decision-reversal* methodological result · the data limits reported as findings · a working
      end-to-end system on recorded slots — and **all four are already in hand**. The document also
      lists what the floor deliberately does *not* require (STAN as a headline, 30 real slots, a
      non-degenerate RQ6 test set, live deployment, a released dataset), so a failure in any of them
      is a scoping note rather than a crisis.
  - [ ] ★ **The one genuine hole in the floor: the labelling protocol (WP1-T2) is unwritten.**
        All 1,692 labels rest on a definition that exists only in one person's head, and it is the
        M1 artefact that was due week 4. It is a writing task, it is cheap, and it should come
        **before anything in WP5**. Everything else in the floor is done.

---

## Risk register (act, don't admire)

> ★ **Recalibrated 2026-09-07.** Most triggers were date-locked on week numbers, and several
> waited on answers to a request that decision 0.3 closed — so they could never fire, which
> makes a register decorative. Risks that have already *resolved* are marked as such (a
> register that never closes anything is not being used), and the live ones are re-triggered on
> something observable rather than on a calendar.

| Risk | Trigger to watch | Action |
|---|---|---|
| ~~Ethics/DPIA blocks collection~~ | — | **Resolved.** Operator holds the approval; `thesis/ethics.md`. One supervisor question remains, and it blocks nothing. |
| ~~C3 data stays scarce~~ | — | **Resolved as fact, not as risk.** 6 frames, 0 maintenance, no more footage coming. Act on it: take the WP2-T11 decision (Option C, 2-class + flagged exception, is the recommendation) rather than watching a cell that cannot change. |
| ~~Real slots stay < 30~~ | — | **Resolved as fact.** 2 exist. STAN is already demoted to preliminary and M4 re-cut onto 5.B. Nothing left to watch. |
| ~~Only 1 venue accessible~~ | — | **Resolved better than feared.** 9 venues for ACTIVE_PLAY. *But* still exactly 1 venue for EMPTY — which is the live risk below, and is not the same thing. |
| **EMPTY stays single-venue** | Now — it already is | **The live scientific risk, and the one worth acting on.** It is what makes RQ6 unanswerable, keeps cross-venue evaluation recall-only, and leaves the false-play comparison on 3–10 effective scenes. Action: the 0.3(a) request. If declined, all three stay scoped as unanswerable in `preregistration.md` — which is already written, so the cost is bounded. |
| **The labelling protocol stays unwritten** | Now — it is | Highest-value open item in the plan (see `thesis/mvt.md`). 1,692 labels rest on it and it is the overdue M1 artefact. Write WP1-T2 before starting WP5. |
| Booking DB access blocked | No sample export by the time WP6-T4 starts | Manual booking sheet for the case study; RQ4 reported as design + fixtures, not measured precision |
| Novel module gives no gain | WP5 ablation p > 0.05 | Report as a negative result with analysis — still a contribution, **provided** WP5-T9's logit-average baseline was run first so the null is about the gate and not the ensemble |
| Laptop sleep kills runs | Any multi-hour run | Chunk runs, incremental writes, cache-resumable (WP0-T11). ★ Note the feature-cache stage is *not* in fact resumable — see the reproduction audit |
| ★ Data loss | **Now — 0.2 is still open** | **The one unmitigated risk in the register.** 4.2 GB of irreplaceable footage in one place. Every result above is regenerable; the footage is not. WP0-T8. |
| ★ Test-set overfitting | Many ablation cycles | FINAL_TESTSET locked and enforced in code (WP0-T4). ★ The lock was found to be cwd-relative on 2026-09-07 and is now package-relative with a raising guard — see amendment A7 |
| ★ Capacity overrun | Behind by week 10 | Switch to the 16-week compression; protect WP5 — now meaning **5.B**, the module that is not data-blocked |
| ★ **Pixels are already permanent in git history** | Now | 9 image files are committed, incl. 5 venue-audit sheets showing unblurred players. This forecloses WP1-T5 option (a), "features-only, no pixels", because history cannot be quietly rewritten across 85 commits and 50 branches. **Decide WP1-T5 before the repo is shared with anyone.** |

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
