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
- [x] **0.2 ★ [B] Back up `data/` — done (confirmed 2026-09-09).** The 4.2 GB lives on **S3
      and on Google Drive**, so the 3-2-1 rule's "two places, not one" is satisfied and the
      register's one unmitigated risk is closed.
  - [ ] **[H] Two follow-ups the policy asks for and this does not yet cover.** Open one file
        from each copy to confirm a restore actually works, and record the date here —
        `docs/backup.md` says a restore counts as verified when `reproduce_all --check`
        reports nothing missing. An unverified backup is a backup you find out about later.
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
- [x] **WP0-T8 ★ Backup policy documented.** → `docs/backup.md`. 3-2-1, encrypted off-site,
      `data/cache/` excluded as derived, deliberately manual rather than a sync (a two-way sync
      propagates a deletion as faithfully as a file). Includes the restore procedure, which is
      checkable: a restore succeeds when `reproduce_all --check` reports nothing missing.
      **The policy is written and, as of 2026-09-09, the backup is made** — S3 plus Google
      Drive. What remains is verifying a restore, which is the part the policy says counts.
- [ ] ~~WP0-T8 original~~ Formalise 0.2 into `docs/backup.md`: what is backed up,
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
  - [x] ★ **Decided 2026-09-10: (b), and the claim is gone from the docstring.** The plan
        said splits were "materialised, referenced by name, never re-randomised".
        `write_split` and `read_split` still have no callers outside the tests, every
        experiment calls `grouped_split(..., seed=42)` directly, and the row list — which the
        seed does not determine — depends on which feature caches the script filtered to.
        That is how `effective_sample_audit.py` counted **94** distinct scenes and
        `h4_model_equivalence.py` **95**, both correctly, from one seed.
    - [x] **Why not (a).** Retrofitting `read_split` across ~20 scripts would change which
          rows several *published* experiments were fitted on — invalidating results in
          order to protect them, in write-up week. The machinery stays available and works;
          the claim is what was dropped.
    - [x] ★ **What replaces it: a split is identified by its rows, not by its seed.**
          `split_identity()` fingerprints strategy, group key, seed **and the ordered row
          list**, so two runs that disagree are detectable instead of surfacing later as an
          unexplained difference in a count. Five tests, including the case a seed cannot
          distinguish (one row fewer, same seed) and the ordering case that made every
          bootstrap interval on a grouped split a different draw.
    - [x] ★ **And the materialisation path now verifies itself.** `write_split` records a
          partition digest and `read_split` checks it: a split file is a CSV, and flipping
          one row from test to train yields a file that reads back cleanly and describes a
          different experiment. The digest excludes the *name*, so copying a split to a new
          filename stays legal, and files written before the column carry none and still
          read — a missing digest is a real state, not a failure.
    - [x] ★ **And it is quoted where it matters**, because a fingerprint nobody writes down
          detects nothing. Both scripts that disagreed now record their split identity, and
          `tests/test_splits.py` compares the two CSVs — a cross-script check neither script
          could make alone, which fails the moment they drift apart again. **They agree
          today** (`f73f5a29b425`): the divergence was healed by the 2026-09-08
          reproducibility repair, and nothing had recorded either the divergence or the
          return. Adding the column also caught `h4_model_equivalence.py` writing with
          `extrasaction="ignore"`, which attached the field to every record and wrote it on
          none — the test skipped instead of passing, which is the only reason it showed.
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
      cycle with 14-31x headroom**, so latency is *not* the binding constraint and the model
      choice falls to accuracy. Concurrency proved *faster* than naive 20x extrapolation, the
      opposite of the expectation the harness was built to test.
- [ ] **WP0-T10b Repeat on the target Mini-PC** (WP7-T1). Nothing above settles deployment.
- [ ] ~~WP0-T10 original~~ `tools/bench_latency.py`: discard warm-up runs,
      N≥50 reps, report **median and p95** (not mean), declare thread count, pin CPU affinity,
      measure with nothing else running. The pilot's ms/frame numbers were probably measured
      casually — the whole "20–30 cameras on one Mini-PC" claim rests on them.
  - [x] ★ **Concurrency test, not multiplication — built, and it earned its keep twice.**
        `evaluation/latency.py:measure_concurrent` runs 20 streams and the CSV carries
        `round_wall_s` against `naive_extrapolation_s`, so the measured and the multiplied
        numbers sit side by side (1.9 s against 2.0 s for ConvNeXtV2).
    - [x] ★ **And the second use was not the one it was built for.** The concurrent column
          is what showed that H4's speed clause fails: 1.63x under load against the >= 2x
          required, where the single-frame median alone read 2.01x and looked like a pass.
    - [x] ★ **[2026-09-10] The whole table had been measured under load, as its own
          docstring warned.** ConvNeXtV2 150.9 → **100.5 ms**, DINOv2 418.3 → **219.4**, ViT
          303.3 → **169.4**, re-measured on an idle machine. Contention costs the heavier
          model more, so it *inflates* a ratio between models of different weight — the
          direction that flatters the compact backbone. H4's speed clause now fails on both
          readings (1.69x single, 1.63x concurrent) and headroom rises to 14–31x. See
          amendment A13.
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

> **Author's position, 2026-09-09: deferred, and not expected to be contentious.** Recorded
> here so the deferral is a decision with a date rather than an item that quietly stopped
> being mentioned. Two things follow and neither is a formality. The M1 gate criterion stays
> **unmet** — "no objection anticipated" is not a documented outcome, and the gate check will
> keep reporting it as waiting on a person, correctly. And the *technical* work has already
> been built to the strict reading rather than the permissive one: retention is enforced in
> code (WP6-T8), the system is advisory-only and cannot bill (WP6-T12), reconciliation reports
> per field and never per staff member (WP1-T4's recommendation), and every frame image written
> since 2026-09-09 is person-pixelated and blurred. So a later answer that turns out stricter
> than expected costs a document, not a rebuild.

- [ ] **WP1-T3 ★ [H][B] Ethics/DPIA outcome documented** → `thesis/ethics.md`: legal basis for
      recording, retention periods, signage/notification of players and staff, whether a DPIA is
      required and (if so) its completion, works-council/HR position on the staff-audit feature.
      *Deferred by the author (see above); still blocks M1 and still needs the supervisor's
      answer before submission.*
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
- [x] **WP1-T1 Related-work skeleton — scaffolded** → `thesis/ch2_related_work.md`. **Nine**
      strands, not seven: §2.8 prior art and §2.9 selective prediction were added because the
      work produced results that need them. Each strand states *what it must establish* and
      *why this project needs it*, with the project's own findings placed against it, so the
      literature search has a target rather than a topic.
  - [x] ★ **No references are invented, and the document says why in bold at the top.** A
        fabricated citation is the one error in a thesis that cannot be defended at all, and
        it is the specific failure an LLM-assisted draft is most likely to introduce. Every
        `[CITE]` is a placeholder for a paper the author has opened.
  - [ ] **[H] Fill the placeholders.** §2.8 first if time is short — it is the only strand
        that can change what the thesis claims.
- [ ] ~~WP1-T1 original~~ Related-work skeleton → `thesis/ch2_related_work.md`, the 7 strands from the thesis
      doc §2.1–2.7, 3–5 bullet claims + candidate citations each, placeholders marked `[CITE]`.
  - [ ] ★ **Strand 8 — prior art / competitors.** Search for existing commercial and academic
        sports-facility occupancy systems. The §2.7 "research gap" claim is exposed until you have
        actually looked. If a product already does this, your gap statement must narrow to the parts
        it does not do (CPU-only, leakage-free evaluation, reconciliation) — which is fine, but you
        must say it deliberately rather than be told it at the defence.
  - [ ] ★ **Strand 9 — selective prediction / learning-to-defer.** Your REVIEW band *is*
        selective prediction with a human fallback. There is a literature for it, and citing it
        upgrades REVIEW from an engineering hack to a principled design choice (supports RQ6).
- [x] **WP1-T6 ★ Alternative-solutions analysis** → `thesis/alternatives.md`. PIR, turnstile,
      floodlight draw, app check-in and manual logging on cost, accuracy, failure modes and
      retrofit. Four arguments that survive scrutiny, four concessions where the alternatives
      genuinely win, and a section naming where the argument is weakest — the comparison is
      argued rather than measured, and "the cameras are already there" is a property of this
      client rather than of the problem.
- [ ] ~~WP1-T6 original~~ → one page in Chapter 1. Why computer vision
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
- [x] **WP2-T2 [H] Collection request doc — drafted** → `thesis/data_requests.md`. Seven items in
      priority order, with what each unblocks; a §7 saying what is *not* being asked for (no
      names, no payment data, no staff identifiers, no continuous recording), because an
      over-broad request is harder to approve; and an "if only one thing" closing. **Needs a
      human to send it.** Original checklist kept below.
- [ ] ~~WP2-T2 original~~ [H] Collection request doc → `thesis/data_requests.md`. Ask for:
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
  - [ ] ★ **Fix two verified label errors** — `2_playing/slot_20260711_1000_camA_t000021_m.jpg`
        and `..._t000027_m.jpg` show an empty pitch. Both are human-labelled. They are two of
        only six daytime ACTIVE_PLAY frames at `venue_01`, so correcting them makes the
        day/night confound *more* absolute.
    - [x] ★ **Both re-verified by eye, 2026-09-10, and the target class checked against the
          protocol.** The playing surface is empty in both; the only people are off-pitch, by
          the sideline shelter behind the barrier. §2.1 says people outside the pitch do not
          count and §2.3 defines EMPTY as no people *within the ROI* — so the correct folder
          is **`1_empty`** (C1), not `3_people_not_playing`. The error is real and the
          destination is not ambiguous.
    - [ ] ★ **[H] The cost is measured, and it is a scope decision rather than a task.**
          Moving the two files changes their cache keys and their labels, so: rebuild the
          feature caches, then re-run the **24 stages that read a cache — 203 minutes** —
          and re-verify the ledger, of which **32 of 34 claims would move**. Every one of
          those numbers is also quoted in prose somewhere. Call it half a day, most of it
          editing rather than computing.
          *The pixels are not the expensive part:* the images themselves are unchanged, so
          the cached vectors could be re-keyed rather than recomputed. It is the two labels
          that move everything downstream.
    - [ ] ★ **Recommendation: correct the record, not the pixels — unless there is time.**
          The correction buys no scientific gain: it makes a known confound slightly worse
          and moves 32 numbers by amounts nobody would notice (2 frames of 1,692, 0.12%).
          What it costs is a day in write-up week and a re-check of every quoted figure. The
          honest alternative is free and arguably better: report the two frames as a
          **measured label-noise floor of 0.12%** in `threats_to_validity.md`, name them, and
          say why they were left. A thesis that states its label errors is stronger than one
          that quietly fixed two and cannot say how many remain — the 396 `bulk` clip frames
          have never been spot-checked one by one (WP2-T3), so 0.12% is a floor either way.
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
- [x] **WP2-T11 ★ C3 fallback decision — taken 2026-09-10, and it is none of the three.**
      The tree offered Option A, B or C if C3 stayed under ~100 real frames. It has 6. The
      measurement refutes the recommended option and produces a fourth answer:
      **the class stays, and the claim shrinks instead.** Written up in `thesis/protocol.md`.
  - [x] ★ **Measured rather than argued (2026-09-09, `empty_recognition.py`).** The six-frame
        class *is* doing damage, but not the damage assumed: with `class_weight="balanced"` it
        absorbs 168 of 243 out-of-distribution frames, and removing it moves those into
        ACTIVE_PLAY instead.
  - [x] ★ **Option C is refuted for the production model, not merely unhelpful.** On the 243
        held-out empty frames, DINOv2's false-play rate is **0.309** at 3-class balanced and
        **1.000** at 2-class. Dropping C3 does not leave the false-play axis alone; it takes
        the model from calling 31% of unfamiliar empty pitches "play" to calling all of them
        that.
  - [x] ★ **And the reason is operational, which is what makes it worth writing.**
        `empty_accuracy` is 0.000 in every configuration — no arrangement of classes makes the
        model recognise an empty pitch here. What moves is *where the errors land*: C3 maps to
        NOTUSED/REVIEW and ACTIVE_PLAY to USED, so the starved class converts 69% of would-be
        **billing errors** into correct-or-reviewable verdicts. It earns its place as a "none
        of the above" sink — not as the maintenance detector it was defined to be.
  - [x] **Option A (copy-paste augmentation) is not worth building yet.** The blocker is not
        C3's size: no configuration of the class recovers empty-pitch accuracy, and what does
        is one labelled frame of the target camera. Synthesising maintenance crops would be
        answering a question the data says is not the binding one.
  - [x] **Option B (detector-based C3) stays available and is now better motivated** — the
        sink works, and a detector would make it deliberate rather than incidental. It is a
        new module in write-up week and nothing rests on it, so it is not being built.
  - [x] **The consequence for every table.** Three classes stay; **C3 is not claimable**.
        Macro-over-three must name C3's support beside it, and no sentence may say the system
        detects maintenance. It does not — it has a sink, and the sink is load-bearing.
  - [ ] **[H] Tell the supervisor at the next check-in.** The decision is documented and
        defended; this is the half a person owns. Real maintenance footage (WP2-T8) is what
        would turn C3 back into a claimable class.
- [x] **WP2-T7 M2 gate check — generated, and M2 is met.** `experiments/gate_check.py` →
      `results/gate_status.md`. All six re-cut criteria check out against artefacts:
      coverage matrix, EMPTY in exactly one venue (494 frames), near-duplicate rate,
      per-split leakage (100% vs 0%), effective sample, and the "not answerable" list.
- [ ] ~~WP2-T7 original~~ M2 gate check. Every class ≥ target in ≥ 2 lighting regimes and ≥ 2 venues; grouped
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
  - [x] ★ **Run where it answers something — `experiments/augmentation_transfer.py`**, 29
        tests. A generic benchmark flag still needs its own extraction budget (`IDEAS.md` #2);
        this asks one question on one boundary: **can augmentation buy what five labelled
        frames of a new camera buy?**
  - [x] ★ **RETRACTED, same day, by the check this list was already asking for.** The bullet
        below reported `light` taking empty-pitch recall from 0.000 to **0.687** at 0.8550
        macro-F1, and the next bullet noted that only one draw had been taken. That note was
        the finding. Four further draws — same rows, same preset, same probe seed, same test
        set, only the random draw differs — score **0.3479, 0.3501, 0.3510 and 0.4136**.
        Median 0.3510, sd 0.2206, and **four of the five land below the 0.4406 the probe
        reaches with no augmentation at all** (`augmentation-light-is-draw-dependent`). The
        published number is the maximum of five.
  - [x] ★ **The repeated 0.3479 was the tell, and `pred_empty` made it legible.** Presets
        and draws sharing nothing else kept scoring exactly 0.3479 — `colour`, `full`, and
        three of the five `light` draws. It is the macro-F1 of answering ACTIVE_PLAY to all
        521 frames (`augmentation-collapse-is-one-class`). What varies between draws is
        whether the fitted boundary reaches camera B's empty pitch at all, so the result does
        not degrade gracefully; it is a working classifier or the trivial one.
  - [x] ~~**`light` (brightness, gamma, noise) takes empty-pitch recall on an unseen camera
        from 0.000 to 0.687** with no target labels, closing **75%** of the gap one labelled
        frame closes~~ — **true of one draw in five** (`augmentation-light-recovers-empty`,
        restated as a claim about that draw). The duplicate-rows control still rules out row
        count *in that draw*, which is worth keeping: what the draw was not is a separate
        question from what it was, and only the second answer was wrong.
  - [x] ★ ~~**`full` — every effect at once — scores 0.3479 … turning more on erased the
        gain. Match the augmentation to the shift.**~~ — one draw of `full` against one draw
        of `light`, and `light`'s own number moves by 0.5 between draws, so the comparison
        cannot carry that reading. What the 0.3479 does say is that this configuration
        produced the trivial predictor.
  - [x] Neither clean story is true, and the third one was not either: 0.855 is not the
        0.9895 a labelled frame reaches, "it does not help across cameras" was too strong,
        and "the right preset recovers most of the distance" was one draw.
  - [x] ★ **Five draws taken, and the run is now built for it**: `--seeds`, `--presets`,
        resume from the CSV, a `pred_empty` column, an `augmentation_transfer_spread.csv`
        with the range and sd, and a lock file — resuming made two concurrent runs erase
        each other's rows, which nearly happened during development.
  - [ ] ★ **What would make augmentation quotable here.** Not more presets: the variance is
        in the fit, not the recipe. Either anchor the EMPTY boundary with target-camera
        frames (which is the five-frame recipe, and it already works), or get empty-pitch
        footage from a second venue so the class is not one morning at one site — 0.3(a)
        again, from a different direction.
  - [ ] Flag in the full benchmark — still deferred, still for the extraction-budget reason,
        and now also because a flag whose effect moves 0.5 between draws would need every
        cell of that table replicated to mean anything.
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
      Under the grouped split the **clock rule came within 0.0068 of ConvNeXtV2** using no pixels
      at all. The comparison was indeed measuring the wrong thing — established in week 1, not at
      the defence. **(RQ7)**
  - [x] ★ **Corrected 2026-09-08 (A12): the random-split half is withdrawn.** This entry read
        "a 16-bin colour histogram beat ConvNeXtV2 (0.686 vs 0.657)". Both numbers were
        superseded by A8's estimand fix and the comparison **reverses** — ConvNeXtV2 0.9879
        against the histogram's 0.9616. A8 corrected the numbers and nobody re-read the claim
        resting on them. What survives is A2's scene-level form: on the 62 distinct scenes in
        that test set both score 1.0000, so *the leaky protocol cannot tell them apart*.
        Both figures are in the claims ledger now, which is why it went unnoticed — they
        were not.
  - [x] ★ The complement: `experiments/h3_cross_venue_recall.py` shows a lighting-only rule
        collapsing on unseen venues while ConvNeXtV2 and DINOv2 hold at 0.910 / 0.930. The
        evaluation was uninformative; the models were not the problem. **(RQ3, RQ7)**
        *The measured 0.219 is a lower bound partly produced by label error — quote the
        collapse, not the number, until WP3-T5's relabelling lands.*
- [ ] **WP4-T2 Label-efficiency curves.** Training sizes {10, 25, 50, 100, 300, 1000, all} × 5 seeds
      × 4 models; zero-shot OpenCLIP as the 0-label horizontal line. *Accept:*
      `results/label_efficiency.csv` + `results/figs/label_curve.png`. **(RQ1, RQ2)**
- [x] **WP4-T3 Onboarding cost — answered one level down, and the control is the finding.**
      `experiments/onboarding_cost.py` → `onboarding_cost.csv`, 14 tests.
  - [x] ★ **As written it is not runnable, and that is measured rather than asserted.**
        Exactly **one of eight venues carries more than one class** — `venue_01` has all three,
        the seven clip venues are ACTIVE_PLAY only. Holding out `venue_01` leaves nothing to
        fit; holding out any other leaves a single-class test set where adaptation is
        unmeasurable. A test asserts the count is still 1, so if footage ever arrives the
        proxy gets *replaced* rather than quietly kept.
  - [x] **The runnable form is camera onboarding**, which is also the deployment reality — a
        facility adds a camera, not a venue. Both of `venue_01`'s cameras carry EMPTY and
        ACTIVE_PLAY, and a probe trained on A has never seen B.
  - [x] **Macro-F1 on the new camera: 0.441 / 0.357 / 0.508 at k=0 → ~0.99 at k=1**, for
        DINOv2 / ConvNeXtV2 / ViT. Five seeds per budget, because at k=1 *which* frame you draw
        matters more than anything else (`onboarding-one-frame`).
  - [x] ★ **From k=5 the source camera stops contributing.** Training on five target frames
        *alone* matches five plus **775** source frames, on all three backbones
        (`onboarding-source-stops-helping`). So what buys the accuracy is having *any* labels
        from the new camera, not a large corpus elsewhere — a weaker claim than "the model
        generalises", and the one the numbers support. Operationally it is good news: five
        labels is cheap, a corpus per site is not.
  - [x] Reported as an **optimistic bound**: a camera on the same pitch is an easier target
        than a new site. Every row carries its near-duplicate crossing count and camera B's
        **12 distinct scenes**, which is why small budgets go so far.
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
  - [x] ★ **XAI overlays built — unblocked by the author (2026-09-09), redacted anyway.**
        `vision/explain.py` + `experiments/make_xai_figures.py` → `results/figs/xai/`
        (27 sheets), 16 tests. WP1-T5 is still open, so every frame is person-pixelated by
        YOLO **and** blurred whole-frame before writing: these images are permanent in git,
        and redacting now costs nothing while un-redacting later is impossible.
  - [x] **The decomposition is exact, not a saliency heuristic.** The probes are logistic
        regressions on *mean-pooled* frozen features, so the score is exactly the mean of a
        per-position contribution — no gradient to approximate, no smoothing to tune. The run
        aborts if the map fails to reconstruct the score; worst error over 27 explanations is
        **8.65e-07**. Grad-CAM was not needed and would have been the weaker instrument.
  - [x] ★ **Turned into a measurement, and it found something.** Share of positive evidence
        inside YOLO person boxes against those boxes' area: ConvNeXtV2 **2.18×**, DINOv2
        **1.61×**, **ViT 1.02× — the null exactly.** ViT classifies ACTIVE_PLAY correctly
        while putting no more evidence on the players than on the turf, which is consistent
        with it being the weakest cross-venue model and with it taking the camera shortcut in
        WP4-T13. Nine frames: a direction, not a rate.
  - [ ] **[H] Decide whether the ViT result goes in the thesis as a finding.** It is the
        clearest visual evidence for the confound argument, and it is n=9.
- [ ] **WP4-T11 ★ Efficiency table with honest methodology.** Using WP0-T10: median and p95 ms/frame,
      peak RAM, concurrent-20-camera throughput, PyTorch vs OpenVINO, on both the dev laptop and the
      target Mini-PC. **(RQ2)**
- [ ] **WP4-T12 ★ Measure actual power draw.** A €15 plug meter on the Mini-PC gives W under load →
      kWh/year → €/year and gCO₂/year. Trivial effort, and it substantiates the "low-bandwidth,
      low-power edge" framing with a real measurement instead of a datasheet quote.
- [ ] **WP4-T7 DINOv3** (if licence access granted) — add as a 5th row to every WP4 table, same
      protocol, or log explicitly as blocked. Requires [H] HuggingFace login + Meta licence accept.
- [x] **WP4-T8 M3 gate check — generated, and M3 is met.** Eight models across four
      protocols, intervals on every row, Holm-corrected families in three reports, four
      trivial baselines, figures exported, and 26 claims re-deriving from their artefacts.
- [ ] ~~WP4-T8 original~~ M3 gate check. Benchmark v2 complete, stats attached, figures exported.
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
- [x] **WP5-T1 STAN implementation — built, and preliminary by its own gate.**
      `src/pitch_occupancy/slots/stan.py`: a dilated 1-D TCN over per-minute class
      probabilities, 1,651 parameters against the 100k ceiling, masked pooling so slot length
      is not read as evidence. `experiments/stan_preliminary.py` → `stan_preliminary.csv`.
  - [x] **On 200 held-out composed slots STAN scores 1.0000**, beating a tuned HMM by 0.105
        and summary-statistic logistic regression by 0.105 (both p < 0.0001). **Read that as a
        saturated benchmark, not a win.** The composed label is a deterministic function of
        the template, the five templates stay separable under jitter, and a model that reads
        *contiguity* — one long block of play versus scattered short runs, exactly what a play
        ratio discards — recovers the generating process. Necessary, not sufficient.
  - [x] **The WP5-T8 gate is enforced in code, not promised.** `MIN_REAL_SLOTS = 30`, and
        `assert_preliminary()` raises on the two slots that exist. The experiment calls it,
        catches the refusal, and prints the caveat beside every table; the M4 gate criterion
        checks the caveat is in the CSV rather than that the file exists.
  - [x] ★ **A structural finding: no clean real test set exists, and cannot be made.** The two
        recorded slots hold **1,296 of 1,692 frames — every EMPTY and every MAINTENANCE frame
        in the dataset.** The other 396 are highlight clips, all ACTIVE_PLAY. So "train on
        synthetic, test on real slots only" cannot be made frame-disjoint here. This is the
        sharpest statement of why **WP2-T8 blocks WP5-T1**, and it is measured, not asserted.
  - [x] ★ **[2026-09-10] STAN rested on one synthetic draw. Now it rests on five, and the
        answer split.** `stan_preliminary.py` composed its train and test sequences from a
        single seeded draw, and the project replicates wherever the draw is the object of
        study — five split replicates per benchmark protocol, five seeds per label-efficiency
        point, `n_seeds` on the onboarding curve — and skipped it wherever the draw was a
        means to an end. There were exactly two of those; the augmentation one moved
        **0.855 → 0.350** between draws and took a published headline with it, so this one
        was run rather than noted. `--seeds`, `stan_draw_spread.csv`, 4 new tests.
    - [x] **The ordering survives, and that is now a stronger claim than the entry could
          make.** STAN is first in **five draws of five**, mean 0.9420, and no baseline
          matches it in any (`test_stan_is_first_in_every_draw`). Still preliminary by the
          WP5-T8 gate — that is about the two real slots and is untouched.
    - [x] ★ **The ceiling does not.** 1.0000 in three draws, 0.9100 and 0.8000 in the other
          two. So *"the composed test set is exhausted"* is a property of a composition, not
          of the design: it **can** be exhausted, often enough that one draw is likely to
          show it. `threats_to_validity.md` §1.3 and defence slide 13 both said the stronger
          version and now say this one.
    - [x] ★ **The 0.105 margin over the HMM is one draw's margin.** `summary_logistic` runs
          0.7350–0.9550 (sd 0.0863) and is the best baseline in one draw and the worst in
          another. The five-draw mean gap is +0.116, and that is the number with a spread
          behind it.
  - [ ] **WP5-T6 remains open**: STAN consumes the *fused* sequence, not both camera halves.
- [ ] **WP5-T6 ★ Feed both camera halves separately instead of pre-fusing them.** Current design
      fuses camera A/B with a max-activity rule *before* aggregation, which throws away information:
      **disagreement between the two halves is itself a signal** (occlusion, dirty lens, play
      confined to one half). Let STAN consume both sequences and learn the fusion. This gives you a
      genuinely stronger architecture *and* a clean extra ablation (fuse-then-aggregate vs
      aggregate-jointly) — one of the cheapest real novelty gains available in this plan.
- [x] **WP5-T7 ★ STAN baselines strengthened — all four built.** They live in `slots/stan.py`
      beside STAN and share its `fit`/`predict` interface, so the table is a loop rather than
      four special cases. On 200 held-out composed slots: tuned thresholds **0.825**, median
      smoothing (window tuned, not guessed) **0.835**, HMM **0.895**, summary-statistic
      logistic regression **0.895**, STAN **1.000**.
  - [x] **STAN does beat the tuned HMM** (+0.105, p < 0.0001, 21 informative pairs of 200) —
        the comparison this task says is the real one. But see WP5-T1: the composed set is
        saturated, so this establishes that STAN recovers the generator, not that it would win
        on real slots. The point of attack survives; only real labelled slots close it.
  - [x] The HMM gets the *same* information STAN gets — the classifier's own per-minute
        probabilities as emissions, transitions counted from the training slots — rather than
        being handed argmax states, which would have made it a straw man.
- [x] **WP5-T8 ★ Slot dataset built, with the boundary enforced rather than described.**
      `src/pitch_occupancy/slots/synthetic.py` composes slots from manifest frames under all
      five templates. Every composed slot is marked `synthetic=True` by the constructor — it
      is not settable by argument — and the gate counts real ones by reading that field.
  - [x] **Labels come from the template, not from `aggregate_slot`.** A composed full match is
        USED because it is a full match. Labelling by the aggregation rule would make the
        threshold baseline correct by construction and every comparison against it circular.
  - [x] **Template boundaries are jittered.** With fixed fractions each template is one
        stereotyped shape, the label is recoverable by recognising which of five shapes a slot
        is, and every shape-reading model scores 1.000 — which is what the first run did.
  - [x] ★ **Hard gate written down as code**: `MIN_REAL_SLOTS = 30` and `assert_preliminary()`
        in `slots/stan.py`, checked by a test that asks it to pass on 30 and refuse on 2, and
        by a test that composed slots never count towards it.
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
- [x] **WP5-T2 Fusion head — built, ablated, and it does not earn its place.**
      `src/pitch_occupancy/slots/fusion_head.py` implements the specified architecture: cheap
      image statistics → tiny MLP → softmax weights → shared linear head over the weighted
      concatenation. `experiments/fusion_head_ablation.py` → `fusion_head_ablation.csv`,
      `fusion_head_comparisons.csv`, `fusion_head_gate.csv`.
  - [x] **Routing is worth −0.0238 cross-venue recall** against the same head with the gate
        off, on one informative fold of seven, p = 1.000. **(RQ5 answered negatively.)**
  - [x] ★ **The ablation is a three-rung ladder, not an on/off switch**, and that was the
        correction that mattered. `uniform` → `constant` → `mlp`, each rung adding one
        capability with head, trainer, seed and regularisation held identical, so *learned
        mixing* and *routing* are separated. On/off would have credited routing with an effect
        that is a learned constant: the gate puts ~0.70 of its weight on DINOv2 in **every**
        fold and moves it by only 0.086 between frames.
  - [x] The implementation **can** route — checked, not assumed. On a fixture where the useful
        backbone flips with the gate statistic, `uniform` and `constant` score 0.671 and `mlp`
        scores 1.000; a gate fed zeros collapses to 0.663. The null is about the data.
  - [x] ★ **Both axes reported** — and neither is sufficient alone; see WP4-T13 below.
  - [x] ★ **Confound tested; the answer is *partly*.** The gate's DINOv2 weight correlates
        **−0.653** with a night indicator across all seven folds, so lighting is much of what
        those three statistics carry. But a gate fed *only* lighting is 0.1015 worse on recall,
        so it is not all of it. Coincidence worth recording: the lighting-only gate's false-play
        rate is 0.0206 — the published clock rule's rate to four decimals, both 5 of 243.
- [ ] **WP4-T13 ★ [NEW] The false-play rate is not a specificity measure, and the write-up
      must stop reading it as one.** Found while checking whether the gated head's false-play
      rate of 0.000 was real. Asking what each model answers *instead* of ACTIVE_PLAY on the 243
      held-out empty frames:
  - [x] **DINOv2 answers ACTIVE_PLAY 75 times and MAINTENANCE 168 times — it is correct 0
        times.** The gated head answers MAINTENANCE on all 243: false-play 0.000, accuracy
        0.000. ConvNeXtV2 gets 2 right, ViT 40, and **no model exceeds 0.165**.
  - [x] The complement of the rate is not correctness; the third class absorbs the difference.
        DINOv2's **0.309**, quoted in `README.md`, `rq_matrix.md` and throughout the log as
        dominating ConvNeXtV2's 0.992, is the rate at which it makes *one kind* of mistake
        rather than another. Both published numbers remain correct as stated and both ledger
        claims re-derive; the *inference* drawn from them does not follow.
  - [x] **Mechanism, and it is the dataset not the model:** this protocol trains on 518
        ACTIVE_PLAY, 251 EMPTY and **6** MAINTENANCE frames, and `class_weight="balanced"`
        gives that six-frame class a weight of **43**. Out-of-distribution frames land in it.
  - [x] An `empty_accuracy` column now sits beside the rate; the joint summary is built from
        accuracy, not from `1 - false_play`. Built the other way it ranked the gated head first
        on the strength of 243 wrong answers. Ledger claim `false-play-is-not-accuracy` added.
  - [x] One sentence in the H4 log entry drew exactly the wrong inference and is corrected in
        place.
  - [x] ★ **Followed up, and my own mechanism was half wrong — `empty_recognition.py`.**
        Dropping C3 entirely does **not** rescue EMPTY accuracy: it stays at 0.0000 and
        false-play goes 0.309 → 1.000, because all 243 frames move into ACTIVE_PLAY instead.
        The class weighting decides *which* wrong answer appears, not whether it is wrong.
  - [x] ★ **The control is a camera-transfer test.** Train A → B's empties gives 0.0000;
        add camera B's *play* frames and DINOv2 reaches 0.7325 — but that condition puts
        camera B in one class only, so "camera B implies play" is a shortcut, and **ViT takes
        it completely, falling 0.165 → 0.004**. On that axis the model ranking is partly a
        ranking of shortcut resistance. Sixth appearance of this confound.
  - [x] ★ **One labelled empty frame of the held-out camera takes every backbone to 0.9793**,
        flat to k=25. Adding a camera needs labels *from that camera*, not a bigger dataset.
        Reported with the near-duplicate crossing count (1,400 already at k=1) and the
        3-distinct-scene caveat, which is *why* one frame suffices.
  - [ ] **[H] Decide what the thesis says.** The honest reading is that on this protocol *no
        model can recognise an empty pitch*, which is a stronger and more uncomfortable claim
        than "ConvNeXtV2 is worse than DINOv2". It belongs in RQ2 and in the red-team chapter.
  - [ ] Re-word the README and `rq_matrix.md` sentences so the rate is quoted as a rate. Not
        done unilaterally: they are currently true, and the fix is a rewrite the author owns.
- [ ] **WP5-T2b Conditional-compute variant.** Run ConvNeXt first; invoke DINOv2 only when confidence
      < τ. Report accuracy **and** average ms/frame vs always-both. *Accept:* ablation table with CIs,
      p-values, latency; adopt/reject decision logged.
  - [ ] ★ **State the honest motivation.** WP0-T10 measured 14–31× headroom in the 60 s cycle, so
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
- [x] **WP6-T1 Simulator API — built, and it fails closed.** `src/pitch_occupancy/api/
      simulator.py`, mounted on the existing app, 17 tests. `GET /api/v1/cameras/{id}/snapshot
      ?mode=simulation` returns a real JPEG (SOI/EOI checked, and decoded back to an image in
      a test) with metadata on `X-` headers so the body stays a plain image; `format=json`
      returns the metadata alone.
  - [x] ★ **An unset `PITCH_SIMULATOR_TOKEN` means *disabled*, not *open*.** That is the
        whole point of the default being empty: a secret that degrades to "no authentication
        required" is the same defect class as a threshold set where it can never fire, and
        this endpoint serves frames of identifiable people. Unconfigured → **503**, missing or
        wrong token → 401, compared with `hmac.compare_digest`. Tested, including a
        one-character-short token.
  - [x] **`mode=live` returns 501 and does not fall back to the recording.** An endpoint that
        quietly served footage when asked for a camera would make a deployment check pass
        against a file — the worst failure this endpoint could have.
  - [x] **A gap is a 404, never a placeholder image.** A substituted frame here would be a
        fabricated observation served with a 200, which is the failure `frame_source` returns
        `None` to avoid; it must stay a gap through every layer.
  - [ ] Wire it into a running deployment (WP7-T2) and point an `RTSPSource` at it end to end.
- [x] **WP6-T2 Scheduler service — built, and M5 passes with it.**
      `src/pitch_occupancy/scheduler.py` + `configs/slots_schedule.json` + `pitch schedule`,
      18 tests. `run_slot` had done the sampling for some time; what was missing was the part
      that decides *when*.
  - [x] ★ **The clock is a parameter and deciding is separated from doing.** `due()` is pure
        — a schedule and a timestamp in, a list out — so *what would run* is answerable
        without running anything, the same shape as `pitch retention`. `run_forever` takes an
        injectable sleep and an iteration bound, because a loop whose only observable
        behaviour is that it does not return has no tests.
  - [x] ★ **`venue_id` and `field_id` are different things**, and conflating them was a
        foreign-key failure that the first version hit: the manifest keys a slot as
        `<venue>_<date>_<HHMM>` while `rental_slots.field_id` references a *pitch*. Fixed in
        the model rather than papered over, and `db/store.ensure_slot` now declares the
        venue, field, cameras and rental slot before any sample is written — the thing that
        knows a slot is starting is the thing that should declare it.
  - [x] **One dead camera does not stop the other pitches**: a slot that raises is reported
        and skipped, with a test.
  - [x] Schedule validation is strict — a missing field, a slot with no cameras, two slots at
        the same time on one venue, an unknown weekday all raise. A silently-dropped entry is
        a slot with no footage and no explanation.
  - [ ] ★ **Not a daemon and not a deployment.** No supervision, no restart policy, no
        systemd unit (WP7-T2), and it has never run against a camera — the tests hand it
        recorded video. WP7-T3's shadow run is where that changes.
- [x] ★ **WP6-T2 The classifier the whole pipeline was missing** (2026-09-10).
      `src/pitch_occupancy/vision/classifier.py` + a runnable
      `python -m pitch_occupancy.worker`, 14 new tests. `run_slot` and `run_due` have taken a
      `classify` callable since they were written and **nothing outside a test ever supplied
      one**: every scheduler test hands the seam a stub, `end_to_end_slots.py` rebuilds each
      slot's per-minute sequence from the *labels*, and `worker.main` raised
      `NotImplementedError` naming exactly this. Sampling, fusion, aggregation, evidence
      selection and reconciliation had all been exercised end to end without a single frame
      ever having been classified by the system itself.
  - [x] **The acceptance criterion is now met with a model in the loop.** Both exported
        slots agree with the label-derived verdicts in `end_to_end_slots.csv`:
        `2026-07-11_1000` NOTUSED (model empty 0.947 against the labels' 0.932) and
        `2026-07-12_2030` USED (play 1.000 either way), at 100% capture. The other two slot
        instances the derived schedule names were never exported and are reported SKIPPED.
        That the model wrote the rows is checkable: the seeded ones carried a constant
        `mean_confidence` of 0.95, these carry 0.9932 and 0.9999, and the 469 frame samples
        behind them hold 233 distinct confidences. **This is a wiring check, not an accuracy result** — both slots'
        frames are in the probe's training set, so what it shows is that the pipeline is
        connected, and nothing about generalisation. The honest numbers are WP4's.
  - [x] ★ **The input path is the risk, so it is tested against the cache itself.** A
        deployed classifier that embeds frames even slightly differently from the way the
        feature cache was built is a model evaluated on one distribution and run on another
        — this project has already been bitten by that once (the input-path challenge,
        `docs/PM_REVIEW_2026-09-08.md`). So the module applies no preprocessing of its own,
        calls the same `embed_batch` under the same processor geometry, and a slow test
        embeds real frames through the deployment path (OpenCV decode, BGR→RGB) and asserts
        they reproduce the stored vectors. They agree exactly.
  - [x] ★ **The venue lock holds in production too.** The probe is fitted on
        `development_rows` only — 1,578 frames — so a deployment cannot quietly train on the
        held-out venues and make the one honest number in the thesis unquotable. There is a
        second explicit check for locked venues behind that, which fires only if
        `development_rows` ever changes.
  - [x] ★ **Fitted at construction, not loaded from a pickle.** A serialised sklearn
        pipeline is an artefact that can drift from the manifest it claims to come from,
        breaks on a library upgrade, and that nothing here regenerates. The fit is a second
        on cached features, so refitting means the deployed model is always the one the
        current cache and manifest imply.
  - [x] **Three defects found by running it, which is the point of running it.** `run_due`
        hands `on_slot` the **exception** when a slot fails — one dead camera must not stop
        the other pitches — and the first version printed `run.verdict` unconditionally, so
        it died inside the handler for the failure it was reporting. The schedule derived
        from the recordings names four slot instances and only two were exported, so this
        fired immediately. Overlapping slot windows would also have run a slot twice, since
        `run_due` runs everything due at that moment.
  - [x] ★ **The third is the one worth keeping: `run_due` persists only when it is given a
        connection.** `main` did not pass one, so the first version classified both slots in
        full, wrote nothing, and printed "2 slot(s) written to the database". Found by
        looking in the database rather than by reading the output — the rows there were
        dated 2026-09-06 and had come from `pitch seed`. Not a crash; a confident sentence
        about something that did not happen. Fixed, and a test now asserts the connection
        reaches `run_due` with the schema initialised.
  - [ ] ★ **The confidence is not calibrated.** It is the probe's maximum class probability,
        which `fuse` uses to weigh two cameras and `aggregate_slot` never reads. Anything
        that wants a *threshold* on it — a REVIEW band, RQ6's operating point — has to go
        through `evaluation/calibration.py` and state the temperature it fitted. Blocked on
        the same test-set degeneracy as WP4-T9.
  - [x] ★ **M5's end-to-end criterion was weaker than it reads, and now has a committed
        artefact behind it.** It was satisfied by `results/end_to_end_slots.csv`, whose
        per-minute sequences are rebuilt from the **label column** — so "end-to-end run on
        real slots" had never included the model, and the run that does writes to a
        `.gitignore`d database no gate criterion can read.
        `experiments/end_to_end_model.py` → `end_to_end_model_slots.csv`, a `reproduce_all`
        stage, 8 tests: **2/2 verdicts and 106/106 comparable minutes agree** with the
        label-derived answers. Two caveats travel with that number and are in the module's
        first paragraph — it is **in-sample**, and the minutes are compared scene-to-scene
        rather than frame-aligned, because the worker reads minute x 60 s and the labelled
        frames sit at irregular instants.
    - [x] ★ **The M5 criterion now points at it.** `gate_check.py` read the label-derived
          table for months — "end-to-end run on real slots" was true of the decision layer
          and of no model. It now reads `end_to_end_model_slots.csv` and checks the verdicts
          **agree**, since a run that classified both slots wrongly would satisfy "a file
          exists" just as well. Tested by breaking it both ways: flip a verdict and it must
          fail with the file present; hide the file and it must name the command to run
          rather than falling back to the label-derived table.
  - [ ] ★ **Still not a deployment.** This replays recordings through the scheduler; the
        live path (`scheduler.live_sources`) has still never been pointed at a camera, and
        there is no supervision or restart policy. WP7-T2 and WP7-T3.
- [x] ~~WP6-T2 original~~ `main.py`: reads `config/slots_schedule.json`; during active
      slots pulls 1 frame/min per camera (VIDEO_SIM | API_SIM | RTSP_LIVE), classifies, fuses, writes
      DB; at slot end runs the aggregator (threshold or STAN per config) + evidence selection.
      *Accept:* runs continuously; DB fills; verdicts correct on recorded slots.
- [x] **WP6-T3 RTSP source — it existed, and it could not run. Now it can, with 17 tests.**
      `frame_source.RTSPSource` had been written and never exercised, and both defects were
      the kind that only appear when something drives it.
  - [x] ★ **It was structurally incompatible with its only caller.** `n_minutes` raised
        `NotImplementedError` saying "the scheduler decides", but the scheduler calls
        `worker.run_slot`, whose *first statement* is `for minute in range(source.n_minutes)`.
        The one class whose whole purpose is live operation could not get past line one. The
        slot length is now passed in by `scheduler.live_sources` — which is what "the
        scheduler decides" should have meant — and is **required**, never defaulted.
  - [x] **It had no pacing.** `run_slot` loops over minutes without waiting, which is right for
        a recording (minute *k* is a seek) and wrong for a stream: it would have taken sixty
        snapshots in under a second and called it an hour. `read` now blocks until the minute
        has arrived, catching up rather than stretching if a run has fallen behind — missed
        minutes belong in the capture rate, not hidden by a longer slot.
  - [x] The clock, the sleep **and the socket** are all parameters, so an hour-long class is
        tested in milliseconds: retry, gap-on-failure, capture release, and the whole path
        through the real `run_slot`.
  - [x] `scheduler.live_sources()` refuses a venue with no URLs and a slot whose camera has no
        URL — half a pitch reported as the whole pitch is what `slots/fusion.py` exists to
        prevent, and dropping a camera here would reintroduce it upstream of the fusion.
  - [ ] **Still never run against a camera.** That is WP7-T3.
      *Accept:* tested against a reachable RTSP or a local ffmpeg loop of the mp4s.
- [x] **WP6-T8 Retention worker — built, and the refusals are the point.**
      `src/pitch_occupancy/retention.py` + `pitch retention` (dry run by default), 12 tests.
      The acceptance criterion — *a dry run prints the correct purge set* — is met by making
      the **plan** the artefact: `plan()` never deletes, `apply()` needs an explicit
      `confirm=True`, and the CLI reports unless given `--apply`.
  - [x] ★ **A retention worker in this repository can delete the irreplaceable thing.**
        `data/raw/` is 4.2 GB from a client facility with one copy; `data/processed/` is the
        hand-labelled corpus. Both are in `PROTECTED_ROOTS`, planning against one **raises**,
        and `apply()` aborts on a plan containing a protected path rather than skipping it —
        a plan with one in it was built wrongly and the rest is not to be trusted either.
        Three tests, including one that puts a file in `data/raw/` and asserts it survives.
  - [x] "Raw frames" in the ethics commitment means frames sampled by the **running system**
        (`data/interim/`), not the research corpus. Conflating the two would be the most
        expensive bug this project could ship, so the distinction is written down rather
        than understood.
  - [x] A file whose age cannot be read is **kept and reported**, never swept up — deleting
        on a failed `stat()` is how a retention worker becomes a data-loss incident. Writing
        that test found a real bug: `is_file()` stats too and sat outside the guard.
  - [x] ★ Periods match `thesis/ethics.md` (7 days sampled, 365 evidence) and a test checks
        them **against the document**, so the two cannot drift. The document said *"Code
        enforces this (WP6-T8)"* while no such code existed — a claim about code, in the
        ethics chapter, with nothing behind it.
  - [x] The clock is injectable, which is the only way a 365-day rule gets a test before the
        year is up.
- [ ] ~~WP6-T8 original~~ Purge raw frames > 7 days, keep evidence 365 days; disk usage
      bounded and logged. *Accept:* dry-run prints the correct purge set.
  - [ ] ★ Make retention periods match whatever `thesis/ethics.md` (WP1-T3) actually committed to.

### 6.B Reconciliation — the operational contribution
- [x] **WP6-T4 Booking schema + connector — code done; the real export is still [H].**
      `src/pitch_occupancy/bookings.py`, `configs/bookings_example.csv`, `pitch bookings`,
      21 tests. Normalised exactly as specified.
  - [x] **READ-ONLY by interface, not by flag.** `BookingSource` has one method, `read`.
        There is no write path to disable and no `dry_run` default to get wrong, and a test
        asserts the public surface is exactly `{read, path, source}` — same reasoning as
        `slots/authority.py`: a booking system is the facility's financial record.
  - [x] **Every problem raises; nothing is skipped.** A dropped booking reconciles to
        *unbooked usage*, which is the anomaly that accuses a customer, so a typo must not be
        able to manufacture one. Unknown status, duplicate slot, end-before-start, empty
        field_id and unparseable date all raise — and the date error names the MM/DD/YYYY
        trap explicitly, because no importer can detect it.
  - [x] **The example is tagged as an example by the importer**, not by the file, so a table
        built on 8 hand-written rows cannot read as one built on a client export. It covers
        every status and keys to the two really-recorded slots.
  - [ ] **[H] The actual export.** Ask the facility for a read-only CSV in these columns —
        showing them `configs/bookings_example.csv` is cheaper than describing it.
  - [ ] SQL/REST connectors behind the same interface, once the client's system is known.
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
- [x] **WP6-T10 ★ Cost-sensitive evaluation — done, as a break-even rather than a figure.**
      `experiments/reconciliation_value.py` → `results/reconciliation_value.csv`, 13 tests,
      stage `reconciliation-value`.
  - [x] ★ **A single € figure would be invented.** Of the three inputs it needs, the flag
        *rate* is known (the rule table is deterministic), prices are **assumed** (parameters
        with a sensitivity sweep), and flag **precision** is unknowable here — WP6-T11, it
        needs adjudicated slots and there are two. So the reported answer is the question a
        facility actually has to answer: *how often must a flag be right before this pays?*
  - [x] ★ **Break-even precision: 94.7%** on the stated defaults. 200 flags per 1,000 slots,
        of which only 50 recover money; €600 of investigations regardless, €120 per wrong
        flag, against at most €2,000 recoverable. **The feature's viability hinges almost
        entirely on precision** — the one quantity this corpus cannot supply.
  - [x] It gives the human-in-the-loop design a number rather than a principle: at a 95%
        break-even no automated action could be justified on this cost structure.
  - [x] **Below roughly a €20 slot price there is no bar at all** — the feature cannot pay at
        any precision. The sign of the result depends on a number the facility supplies,
        which is the clearest argument for a break-even over a euro figure.
  - [x] A modelling error caught by its own test: the first version multiplied by precision
        twice, halving recovery at p=0.5. Recovery is per anomaly *type* now — an unbooked
        slot is revenue never invoiced, a no-show is a correction to a record.
- [ ] ~~WP6-T10 original~~ Attach money to the confusion
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
- [ ] **WP6-T6 Dashboard — the evidence inspector now shows the evidence.** Meters, anomalies,
      slot list, evidence inspector and one-click override were already built; what was missing
      was the photos, and the reason was upstream.
  - [x] ★ **`run_slot` never saved an evidence frame.** `EvidenceFrame.image_path` was set to a
        literal `None` on every minute, so the selection machinery ran, chose three good
        minutes, and recorded three paths to nothing. The inspector said "no evidence images
        bound" and was telling the truth, and the WP6-T7 harvest would have found every frame
        missing against the real database. **A verdict whose evidence cannot be seen is the one
        thing this system must not produce**, since a human confirming every anomaly is what
        `slots/authority.py` rests on.
  - [x] `run_slot(evidence_dir=...)` writes the **fusion-winning** camera's frame for every
        observed minute and deletes the unselected ones once the choice is made — which three
        matter is not knowable until the slot has been seen, and holding sixty 1080p frames is
        most of a gigabyte. Off by default: writing frames of identifiable people to disk is a
        caller's decision. A failed write degrades to "no picture", never to a lost verdict.
  - [x] `GET /api/v1/slots/{id}/evidence/{index}` serves them **addressed by position, so the
        URL carries no path and traversal is not expressible** — stronger than sanitising one.
        The stored path is still resolved and confined, because the database is not a trust
        boundary either. A retention-deleted frame is a 404 and renders as *gone*, never as a
        blank image an operator might read as an empty pitch.
  - [x] ★ **Live field matrix with confidence chips** — `GET /api/v1/fields/day/{day}` plus a
        grid on the dashboard, 10 tests. Every pitch, hour by hour, for one day.
  - [x] **A scheduled slot with no verdict is drawn hollow**, dashed and captioned *no
        verdict*. An unobserved hour and an hour observed to be empty are different claims,
        and a filled neutral chip would read as "we looked and it was quiet" — the opposite
        of the truth, about exactly the case reconciliation exists to catch. The endpoint
        returns those slots rather than filtering them out, for the same reason.
  - [x] **The evidence endpoint stopped leaking server paths.** It returned whatever was
        stored, which is an absolute path on the deployment, into a browser page — for no
        benefit, since images are fetched by index and never by path. It sends file *names*
        now; the name is kept because an operator disputing a verdict may need to quote which
        frame they were shown.
  - [x] ★ **Schedule editor — the API's only write path, and the docstring argues why that
        is defensible here.** `api/schedule_editor.py`, 22 tests. `bookings.py` refuses to
        have a write path at all because a booking sheet is the *facility's* financial
        record; the schedule is this system's own configuration and an operator who adds a
        pitch has to say so somewhere. The asymmetry is deliberate, not an inconsistency.
  - [x] **It is still the most dangerous endpoint here**, because the schedule decides
        whether an hour is *observed at all* — a dropped entry is no verdict and no footage,
        found weeks later when someone disputes a booking. Three protections:
        **validation runs `load_schedule` itself** over a temp copy rather than a second copy
        of its rules (two validators agree until they do not, and the one that matters is the
        one running at 10:00); **`os.replace` from a staging file beside the target**, so a
        crash mid-write leaves the old schedule intact rather than truncated JSON; and **the
        previous version is copied to `configs/schedule_history/` first**, pruned to 20.
  - [x] Validation is its own endpoint and writes nothing — deciding and doing stay separate,
        the same shape as `pitch retention` and `pitch schedule`. A rejected write leaves no
        trace at all, not even a history entry, since the check precedes the backup.
  - [x] A schedule that no longer loads is **still returned** with its problem: an operator
        cannot repair a file the editor refuses to display.
  - [ ] *Accept:* manager daily review flow < 5 min — needs a manager (WP7-T4).
- [x] **WP6-T7 Override → retraining loop — built, with one deliberate departure.**
      `src/pitch_occupancy/retraining.py`, `pitch retraining`, 13 tests.
  - [x] ★ **Frames are staged UNFILED, not under `<corrected_class>/`, and that is a
        correctness argument rather than a scope cut.** *A slot override is not a frame
        label.* An operator overriding an hour to USED is not saying each evidence frame shows
        active play — a slot is USED at 35% play, so most of its minutes may be empty. Filing
        by the slot verdict would inject confidently wrong labels from the one source this
        project treats as ground truth. They land in `_incoming/_unfiled/<slot_status>/` with
        a sidecar carrying slot, operator, time, note and both verdicts, plus an empty
        `frame_label` for a human to fill.
  - [x] **The underscore prefix is load-bearing and tested, not trusted.** `build_manifest`
        globs `[0-9]_*`, so nothing staged can reach a training set by accident.
  - [x] Copies, never moves — the evidence still justifies a billing decision and the audit
        trail points at it. `apply(confirm=False)` is inert; an existing destination is
        skipped, so a re-run cannot undo a human's annotation. A deleted evidence frame
        (retention runs on a schedule) is *reported*, not silently dropped.
  - [x] **The feedback loop is named in the module docstring.** Operators override what looks
        wrong, so this harvests the model's own errors: valuable for training, and a biased
        sample. Any accuracy measured on harvested frames is meaningless and evaluation stays
        on the held-out sets. Written down because the temptation will be strong.
- [x] **WP6-T12 ★ "The system must never bill" — enforced, not stated.**
      `src/pitch_occupancy/slots/authority.py` + 13 tests. `Advisory` is the only output a
      discrepancy can produce, carries no monetary field, and
      `requires_human_confirmation` is a **property rather than a parameter** — there is no way
      to construct one that does not need a person. The permitted vocabulary is three actions;
      `FORBIDDEN` names what was left out so the omission is legible rather than an oversight
      waiting to be filled in.
  - [x] The boundary is **walked, not asserted**: tests parse every FastAPI decorator and fail
        if a financial route, a booking write, or an unexpected mutating route appears, and
        scan the db layer for a monetary column. Verified by adding a `POST /charge` route —
        two guards fire.
  - [x] It is a statement about what the software offers, kept true by tests — **not** a
        safeguard against a determined operator, and the docstring says so.
- [ ] ~~WP6-T12 original~~ Write it into the architecture
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
- [x] **WP7-T5 ★ Failure-mode runbook — written, and one row of it made real.**
      `docs/runbook.md`: an eight-row degraded-mode ladder, each marked **enforced** or **not
      implemented**, because a runbook that does not distinguish those is a wish list.
  - [x] ★ **Row 3 was a gap and is now closed.** `capture_rate` was computed and reported but
        **never gated the verdict**: a slot that lost 48 of 60 minutes produced a confident
        `NOTUSED` from the fragment that survived. `Thresholds.review_below_capture` (0.5)
        downgrades it to REVIEW, and `worker.py` passes the slot's real length through —
        without which the ratios are taken over whatever arrived and the check is vacuous.
        4 tests, including that one.
  - [x] **It ships on, unlike `review_below_confidence`, and the runbook says why they
        differ.** Capture rate is not a model quantity — "we saw eleven minutes of this hour"
        is not a statement about a pitch — so a floor can be set from first principles.
        A confidence threshold is a hyper-parameter, is blocked on RQ6, and a fourth silently
        zero threshold would be the same defect this project has already met three times.
        0.5 is recorded as a judgement, not a calibration.
  - [x] The two real slots are unaffected (59–60 of 60 minutes captured), so no published
        verdict moves.
  - [ ] ★ **The rows still marked "not implemented"** — disk-full, facility-wide confidence
        collapse, stale booking export, camera re-aimed — and the honest caveat that none of
        this has met a real outage. WP7-T3's shadow run is where the ladder stops being a
        prediction.
- [ ] ~~WP7-T5 original~~ What happens when a camera dies mid-slot, the network drops,
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
  - [x] **Leakage comparison bar** → `figs/leakage_decomposition`. One stacked bar per model —
        the question is how a single quantity divides, not how two compare — with the
        zero-shot control drawn as its own bar so the subtraction is visible. The title's
        "about two thirds" is **derived from the data**, not typed.
  - [x] ★ **Reliability diagrams: deliberately not drawn, and the reason is recorded.**
        `rq6_reliability.csv` now persists the bins (only the worst was printed before), and
        they show 907 of 907 frames in a single bin for two models and 905 of 907 for the
        third. A ten-bin plot would be one dot on the right-hand edge. The risk–coverage band
        already reports the same fact from the other side.
  - [x] **Onboarding curve** → `figs/onboarding_cost`, and on the served site. Deliberately
        not a plain learning curve: two series per backbone, because the dashed *target-only*
        line is the finding — where it meets the solid one, transfer has stopped mattering.
        The crossover marker and the title's numbers are read from the CSV, not typed.
  - [x] **Direct labels moved to k=0**, where the series are separated. The first version
        labelled the right-hand end, which is exactly where all three converge, and stacked
        three words on one point.
  - [x] **XAI overlays** — no longer blocked; see WP4-T5. 27 sheets, redacted.
  - [ ] Remaining: ablation tables; reconciliation matrix
        with blurred evidence (same block).
  - [x] ★ Risk–coverage / REVIEW-rate curve (WP4-T9). **Done** → `figs/risk_coverage_band`.
        Drawn as a band, with a second panel for the width the confidences leave.
  - [x] ★ Trivial-baseline floor chart (WP4-T10). **Done** → `figs/baseline_floor`. Four
        protocols on one axis, best trivial against best backbone; the cross-venue column is
        marked **"the floor is not cleared"** because a constant predictor scores 1.000
        there. Drawing it surfaced A12 — see WP4-T10 above.
  - [x] ★ Accuracy-vs-latency scatter with the CPU budget drawn as a vertical line. **Done**
        → `figs/accuracy_vs_latency`. The answer turns on a negative — 20 cameras take
        2.5–5.7 s of a 60 s cycle, so every candidate sits in the leftmost tenth and latency
        cannot discriminate — which is only visible because the budget line is on the axis.
        The vertical axis is **recall minus false play**, not recall: the model with the best
        recall has the worst balanced score, and a plot of recall would recommend it.
  - [ ] ~~original~~ Accuracy-vs-latency scatter with the CPU budget drawn as a vertical line — one picture that
        answers RQ2 completely.
  - [ ] ★ **Blur all faces in every published figure.** Check this twice before submission.

### 8.C Defence
- [x] **WP8-T3 Threats to validity — drafted.** `thesis/threats_to_validity.md`, 10 tests.
      Organised by the four kinds of validity rather than by work package, because that is how
      an examiner asks. Every figure cites a claim id, and a test asserts each cited id exists
      **and carries a source artefact** — so a renamed or failing claim breaks the chapter
      instead of leaving a dangling reference that still reads as evidence.
  - [x] ★ **Three labels, and the third is the point:** *mitigated* (something in the repo
        fails if the mitigation is removed), *quantified* (still present, size measured), and
        **unquantifiable** (size not knowable from this data, and no analysis will change it).
        A chapter with only mitigated threats has stopped looking. A test asserts all three
        labels appear.
  - [x] Leads with the sharpest finding rather than burying it: the false-play rate is a
        **camera-transfer test**, not a specificity measure, and DINOv2 gets 0 of 243 right.
  - [x] Closes with what would actually change the limits, ordered by value not effort — and
        the honest summary that **items 1-3 are cheap**: the limitations are mostly a
        data-access problem, not a methodological one.
  - [ ] **Ethics section** still to write — it depends on WP1-T3's answers, which are deferred.
  - [ ] **[H] Rewrite the arguments in your own words** before the mock exam. The numbers are
        checked; the arguments are not, and an argument you have not made yourself falls apart
        on the first follow-up.
- [x] **WP8-T4 Defence deck — drafted.** `thesis/defence_deck.md`, 16 tests. A slide plan
      rather than slides: per slide, what is on it, what you say, and where it matters what
      **not** to say. Twenty minutes of talk, ten of demo, thirty of questions.
  - [x] ★ **It opens by forcing a framing decision.** There is an apologetic version of this
        defence (*the fusion head did not work, STAN's test set was too small*) and a true and
        stronger one: **this is a thesis about how to evaluate this problem, and the evaluation
        kept measuring something other than what it claimed.** Every negative result becomes
        evidence for the thesis rather than a gap in it. The two need different opening
        sentences, so the choice is made on the first page.
  - [x] §3 is the heart: four independent cases in sequence — the clock rule, the protocol a
        constant predictor wins, the false-play rate measuring camera transfer, and the
        onboarding cost — then "guards that were not guarding" as the practice that found
        half of them.
  - [x] **§5 is the demo and is marked un-cuttable**, with the run order, the exact commands,
        and a fallback for when it fails (`project_site.html` opens offline). A test asserts
        both the marking and the fallback survive an edit.
  - [x] Every figure cites a claim id and is named by file so it is regenerated, not redrawn;
        a test asserts each cited id exists, re-derives, and that each named figure has been
        generated. Another asserts the deck **never hardcodes a claim count** — it did once,
        saying 28, which is exactly the drift the ledger exists to prevent.
  - [x] Four sentences are pinned by test against being softened in an edit, including *"it is
        correct zero times"* and *"no inter-annotator figure exists"*. Each is stronger stated
        plainly than discovered by an examiner.
  - [ ] **[H] Build the actual slides, and rewrite every argument in your own words** before
        the mock exam (WP8-T7). The numbers are ledger-checked; the phrasing is not.
- [x] **WP8-T6 ★ Red-team your own defence — drafted** → `thesis/defence_redteam.md`. Ten
      questions with the evidence and a pointer for each, plus four more worth having ready and
      a rehearsal note. Every figure in it is in the claims ledger, so the document cannot
      drift from the results. **The answers are arguments and are yours to make your own** —
      an argument in someone else's phrasing collapses on the first follow-up.
- [ ] ~~WP8-T6 original~~ Write down the ten hardest questions and your answers.
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

> **The `Done` column is generated.** `uv run python -m experiments.gate_check` checks each
> criterion against the repository and writes `results/gate_status.md`; a test asserts the two
> agree. Every row here read `[ ]` while two gates had been met for some time, which is the
> same drift the README and the site export both had. Three outcomes rather than two, because
> "needs a person" cannot be moved by anything in the repository.
>
> **As of 2026-09-09: M2, M3, M4 and M5 met on artefacts; M1, M6 and M7 wait on a person.
> Nothing in the repository is now blocking a gate.** M4 closed with the gated fusion head
> (WP5-T2) and STAN (WP5-T1), both reported as the negative and preliminary results they are —
> routing is worth −0.024 recall, and STAN's real test set is two slots. Its two criteria were
> also strengthened at the same time: they now read the ablation and the preliminary caveat
> rather than checking that two files exist, and each is tested by removing the work and
> confirming the criterion fails.

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
| ~~C3 data stays scarce~~ | — | **Resolved as fact, and the recommendation it carried was wrong.** 6 frames, 0 maintenance, no more footage coming — and WP2-T11 was decided on 2026-09-10 *against* the Option C this row recommended: dropping to 2 classes takes DINOv2 from 0.309 false-play on held-out empty frames to **1.000**. The class stays as a load-bearing "none of the above" sink and the *claim* shrinks instead. See `thesis/protocol.md`. |
| ~~Real slots stay < 30~~ | — | **Resolved as fact.** 2 exist. STAN is already demoted to preliminary and M4 re-cut onto 5.B. Nothing left to watch. |
| ~~Only 1 venue accessible~~ | — | **Resolved better than feared.** 9 venues for ACTIVE_PLAY. *But* still exactly 1 venue for EMPTY — which is the live risk below, and is not the same thing. |
| **EMPTY stays single-venue** | Now — it already is | **The live scientific risk, and the one worth acting on.** It is what makes RQ6 unanswerable, keeps cross-venue evaluation recall-only, and leaves the false-play comparison on 3–10 effective scenes. Action: the 0.3(a) request. If declined, all three stay scoped as unanswerable in `preregistration.md` — which is already written, so the cost is bounded. |
| **The labelling protocol stays unwritten** | Now — it is | Highest-value open item in the plan (see `thesis/mvt.md`). 1,692 labels rest on it and it is the overdue M1 artefact. Write WP1-T2 before starting WP5. |
| Booking DB access blocked | No sample export by the time WP6-T4 starts | Manual booking sheet for the case study; RQ4 reported as design + fixtures, not measured precision |
| Novel module gives no gain | WP5 ablation p > 0.05 | Report as a negative result with analysis — still a contribution, **provided** WP5-T9's logit-average baseline was run first so the null is about the gate and not the ensemble |
| Laptop sleep kills runs | Any multi-hour run | Chunk runs, incremental writes, cache-resumable (WP0-T11). ★ Note the feature-cache stage is *not* in fact resumable — see the reproduction audit |
| ★ Data loss | **Mitigated 2026-09-09** | Was the register's one unmitigated risk. The 4.2 GB is now on **S3 and Google Drive** — two places, neither of them the working copy. Residual: no restore has been *verified* yet (0.2 follow-up), so this is mitigated rather than closed. WP0-T8. |
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
