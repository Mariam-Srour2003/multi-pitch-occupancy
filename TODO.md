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

- [ ] **0.1 ★ [H][B] Ethics / data-protection clearance check.** Ask your supervisor *this week*
      whether the programme requires ethics-board approval and/or a GDPR DPIA before further
      footage collection. Two triggers make this likely: (a) systematic monitoring of a publicly
      accessible area, (b) the reconciliation layer produces **per-staff-member discrepancy rates**
      — that is workplace monitoring, and in many jurisdictions it needs a legal basis, worker
      notification, and sometimes works-council consultation. **This can block WP2 entirely, so
      resolve it before you collect one more frame.** Record the answer in `thesis/ethics.md`.
- [ ] **0.2 ★ [B] Backup the data you already have.** 2.6 GB of video + 1,296 hand-labelled frames
      exist in exactly one place right now. One disk failure ends the thesis. Do 3-2-1: local copy,
      external drive, cloud. Verify the restore actually works, don't just trust the upload.
- [ ] **0.3 [H][B] Send the footage request to the client** (WP2-T2). Longest external lead time of
      anything in the plan. ★ **Critically: ask for a number of *distinct slots*, not hours of
      video** — see the reasoning in WP2-T2 below. Every week you delay this delays M2 and M4.
- [ ] **0.4 ★ [B] Lock a final test set now, before you look at it.** Choose a hold-out (ideally one
      whole venue, or failing that one whole lighting regime × day) and write it to
      `results/splits/FINAL_TESTSET.csv`. Do not evaluate on it during WP3/WP4/WP5 iteration — you
      touch it once, at the end. Without this, dozens of ablation cycles quietly overfit your
      reported numbers and you have no clean claim left.
- [ ] **0.5 ★ Pre-register the analysis plan.** Before running any WP4/WP5 experiment, write
      `thesis/preregistration.md`: hypotheses, primary metric per RQ, the statistical test for each,
      the multiple-comparison correction, and the stopping rule. Commit it to git so it is
      timestamped. Costs an hour; permanently kills the "did you p-hack this?" question at defence.

---

## 1. Research questions & traceability

The RQs exist in the thesis-plan docx but were dropped from `SUPER_PLAN.md`, and no experiment is
tagged with the question it answers. Fix that first — it is what turns a build log into a thesis.

- [ ] **1.1 Copy RQ1–RQ5 into `SUPER_PLAN.md` §0.5** verbatim from the thesis-plan docx §1.3.
- [ ] **1.2 ★ Build an RQ traceability matrix** → `thesis/rq_matrix.md`: one row per experiment
      (`WPx-Ty`), columns = RQ answered · hypothesis · primary metric · test · result file · figure ·
      thesis section. Every experiment must map to at least one RQ. **Anything that maps to no RQ is
      either a missing RQ or scope you should cut** — this matrix is the cheapest scope-control tool
      you have.
- [ ] **1.3 ★ Add RQ6 (recommended).** The current five RQs never ask the *operational* question a
      facility manager actually cares about: *"How much human review must be budgeted to reach a
      given verdict reliability?"* Phrase as: **RQ6 — What is the achievable trade-off between
      automated-verdict precision and REVIEW rate, and where is the operating point that maximises
      recovered revenue per hour of human review?** This is answered by WP4-T9 and WP6-T10 below and
      is one of the more publishable angles in the whole project.
- [ ] **1.4 ★ Add RQ7 (recommended).** *Do deep frozen backbones actually earn their cost over
      trivial baselines on this task?* Answered by WP4-T10. Sounds like a small question; it is the
      first thing a sharp examiner will ask.

---

## WP0 — Repo & rigour hardening
*Weeks 1–2 · ~40 h · feeds M1*

### 0.A Version control & safety
- [ ] **WP0-T1 Git init.** Commit everything except `data/*.mp4`, `data/dataset/`, HF caches, `*.pkl`
      (use `.gitignore`; reference large data by manifest). *Accept:* clean `git status`, first commit.
- [ ] **WP0-T8 ★ Backup policy documented.** Formalise 0.2 into `docs/backup.md`: what is backed up,
      where, how often, and the last verified restore date. Re-verify monthly.
- [ ] **WP0-T9 ★ Environment pinning.** Freeze exact versions (`pip freeze` → `requirements.lock.txt`),
      record Python version, OS, CPU model. *Accept:* a fresh machine can reproduce the environment.

### 0.B Data plumbing
- [ ] **WP0-T2 Dataset manifest.** `tools/build_manifest.py` → `data/dataset/manifest.csv`, columns:
      `file, class4, class3, venue, camera, slot_date, slot_time, t_s, source(regular|motion),
      labeled_by(human|assisted), lighting(day|night)`. *Accept:* 1,296 rows, zero missing fields.
  - [ ] ★ Add columns `slot_id` (venue×date×start — the grouping key used everywhere) and
        `quality` (set by WP3-T5) and `split_role` (train/val/test/FINAL, set once by WP0-T4).
- [ ] **WP0-T3 Taxonomy layer.** `engine/taxonomy.py`: `to_class3(label4)`, class lists, display
      names. Wire into `benchmark.py` (`--classes 3|4`, default 3) and `slot_aggregator.py`.
      *Accept:* pilot benchmark reruns with `--classes 3` and reproduces ≥ pilot accuracy.
- [ ] **WP0-T4 Split module.** `engine/splits.py`: `grouped_split(group=venue×date×slot)`,
      `leave_one_group_out`, legacy random split (for the leakage-comparison experiment only).
      Splits **materialised to CSV** in `results/splits/`, referenced by name, never re-randomised
      silently. *Accept:* unit test proves no group appears on both sides.
  - [ ] ★ Add `temporal_split` (train on earlier dates, test on later) — the plan lists concept
        drift as a risk but never measures it. This makes it measurable for free.
  - [ ] ★ Emit `FINAL_TESTSET.csv` here (see 0.4) and have every other split function *refuse* to
        include those rows. Enforce it in code, not in discipline.
- [ ] **WP0-T5 Feature cache.** `engine/feature_cache.py`: embed every manifest image once per
      backbone → `data/cache/<backbone>.npz`, keyed by file path + preprocessing hash.
      *Accept:* second run of the same backbone takes seconds, identical metrics.

### 0.C Measurement & bookkeeping
- [ ] **WP0-T6 Stats utilities.** `engine/stats.py`: bootstrap 95% CI (accuracy & macro-F1,
      n=10,000), paired McNemar, slot-level paired bootstrap. *Accept:* unit tests vs hand-computed
      toy examples.
  - [ ] ★ Add **multiple-comparison correction** (Holm–Bonferroni). WP4+WP5 run dozens of pairwise
        tests; uncorrected `p < 0.05` across that many comparisons is not a real claim, and an
        examiner who knows statistics will say so.
  - [ ] ★ Add **effect size** alongside every p-value (accuracy delta with CI, Cohen's g for
        McNemar). "Significant but 0.3% better" is a finding, and you want to be the one who says it.
- [ ] **WP0-T10 ★ Latency-measurement harness.** `tools/bench_latency.py`: discard warm-up runs,
      N≥50 reps, report **median and p95** (not mean), declare thread count, pin CPU affinity,
      measure with nothing else running. The pilot's ms/frame numbers were probably measured
      casually — the whole "20–30 cameras on one Mini-PC" claim rests on them.
  - [ ] ★ **Concurrency test, not multiplication.** Measure 20 cameras *actually running
        concurrently*, not `20 × single-frame latency`. Memory-bandwidth contention on a Mini-PC
        makes those two numbers different, and the honest one is the one you must report.
- [ ] **WP0-T7 Experiment log.** Create `results/EXPERIMENT_LOG.md`; backfill pilot entries.
      Every experiment: date, task id, command, seed, data version, result file, one-line finding.
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
- [ ] **WP2-T1 Coverage tracker.** `tools/coverage_report.py`: per-(class × lighting × venue) tally
      → `results/coverage.md`, empty cells highlighted. *Accept:* `C3 × night = 0` visibly.
  - [ ] ★ Add a **slot-count row** to the report, not only frame counts (see 2.B).

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
- [ ] **WP2-T3 Ingest loop (per batch).** `extract_frames.py` (extend with `--venue`, `--pitch`) →
      contact sheets → label → update manifest → `coverage_report.py`.
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
- [ ] **WP3-T2 Letterbox resize.** Aspect-preserving, replaces naive thumbnail. *Accept:* unit test —
      224×224, no distortion, grey padding.
- [ ] **WP3-T3 Photometric normalisation audit.** Verify each model's HF processor stats are applied;
      document per model in `protocol.md`. *Accept:* table in protocol.md.
- [ ] **WP3-T4 Low-light / fog branch.** RMS contrast on ROI; below threshold → CLAHE/gamma variant;
      expose `clahe on|off|auto`. *Accept:* toggleable; before/after visuals saved.
- [ ] **WP3-T5 Quality filter & camera health.** Over/under-exposure, Laplacian-variance blur,
      lens-dirt proxy (persistent contrast drop vs the camera's own 7-day baseline) → `quality=bad`
      in manifest; emit `camera_health.csv`. *Accept:* known-bad frames flagged.
- [ ] **WP3-T6 Train-time augmentation.** Photometric jitter, synthetic fog (gaussian haze),
      night-gamma, horizontal flip. **No rotations/warps** — cameras are fixed. *Accept:* flag in
      benchmark; visual grid saved.
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
- [ ] **WP4-T10 ★ Trivial-baseline floor.** Before claiming deep backbones are needed, run:
      majority class · mean ROI pixel intensity + logistic regression · colour histogram + HOG ·
      frame-differencing motion energy. Cheap (hours, on cached data). **If a colour histogram gets
      95%, your entire model comparison is measuring the wrong thing and you need to know that in
      week 10, not at the defence.** Whatever the result, it becomes a strong first table. **(RQ7)**
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
- [ ] **WP4-T5 Calibration study.** Reliability diagrams + ECE per model; temperature scaling fitted
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
- [ ] **WP6-T5 Reconciliation job.** `engine/reconcile.py`: join slot_evaluations × bookings on
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
- [ ] **WP8-T2 Figure set.** Label-efficiency curve · leakage comparison bar · ablation tables ·
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

## Extensions backlog (only if ahead of schedule, in this order)

1. DINOv3 integration + quantisation study *(cheap, high visibility)*
2. Cold-start transfer curve as a standalone publishable result *(≈ WP4-T3)*
3. Active learning + pseudo-labelling on the unlabelled pool
4. Self-supervised domain adaptation on facility footage *(flagship stretch; needs GPU time)*
5. NL explanations for REVIEW slots (Florence-2 / Moondream on evidence frames)
6. Camera-health prediction from confidence drift
7. Test-time augmentation + temporal smoothing polish
8. Multi-tenant productisation + mobile manager view
