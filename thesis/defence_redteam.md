# Red-teaming the defence (WP8-T6)

The ten hardest questions, asked as an examiner would ask them, with the evidence for each
answer and a pointer to where it lives.

**Read this as a draft you have to make your own.** The evidence is checked — every figure
here is in `thesis/claims.md` and re-derived from its artefact on every run — but the answers
are arguments, and an argument you have not made yourself falls apart on the first follow-up.
Rewrite each in your own words before the mock examination (WP8-T7).

Two things worth knowing before reading. **Most of these have a good answer**, and it is
usually "we measured that, and here is what it showed". And the ones that do not have a good
answer are the ones to prepare hardest, because saying *"that is a real limitation and here is
exactly how large it is"* is a strong answer, while being surprised by it is not.

---

## 1 · "Why computer vision, and not a €20 motion sensor?"

**The most dangerous question**, because the naive answer sounds sufficient.

*Short answer:* a sensor answers *was something moving*; the audit needs *was this booking
used, and here is the picture* — and it has to tell a five-a-side match from a mower. The
cameras are already installed, so the marginal hardware cost is zero and the bandwidth is
under 1 Mbps against 60–90 for continuous decoding.

*Do not skip the concession:* a PIR is genuinely more robust — fog, darkness, a dirty lens.
Say so first; it makes the rest credible.

→ `thesis/alternatives.md` (full comparison, and where the argument is weakest)

## 2 · "Would a colour histogram have done this?"

*On the leaky protocol, effectively yes — and that is the finding, not an embarrassment.*

Under a random split a 16-bin colour histogram scores **0.9616** macro-F1 against
ConvNeXtV2's 0.9879, and on the **62 distinct scenes** in that test set the two are
**indistinguishable — both 1.0000**. A rule that reads only the clock and never looks at the
image scores **0.4907** on the honest grouped split, within **0.0068** of ConvNeXtV2.

Across venues the picture reverses completely: the trivial baselines collapse while the frozen
features hold above 0.86.

*The answer to give:* "Inside a single confounded venue, no — the deep features do not earn
their cost, and we established that in week one rather than at this examination. Across
venues, emphatically yes."

**Be ready for the follow-up** — *"so why use a backbone at all?"* — the cross-venue column,
which is the deployment case.

→ `results/figs/baseline_floor.png`, `thesis/rq_matrix.md` RQ7, amendments A1/A2/**A12**

## 3 · "How many comparisons did you run before you found p < 0.05?"

*A fair question, and the answer is disclosed rather than defended.*

**451 evaluations across two searches** — 76 preprocessing configurations and 375 prompt sets
— neither declared as a multiple-comparison family in advance. That is recorded as **amendment
A6**, and the position is: both were *selection* procedures, not hypothesis tests, so no
p-value was attached to either and no searched result is quoted as a significance claim.

The optimism is quantified rather than asserted: the search's winning prompt set leads a
pre-declared one by **+0.4474** balanced score on the folds it was selected from.

Everywhere a test *is* reported, Holm–Bonferroni corrects the family and the realised family
size is declared — including where it shrank (H4: 3 pairs, not 6; H6: 3, not 4; H5: 4, not 5).

→ `thesis/preregistration.md` A6, A9, A10, A11

## 4 · "Is your slot-level test set really two slots?"

**Yes.** Say it plainly and immediately; hedging on this is worse than the fact.

The slot-level decision layer is the operational contribution and it has **n = 2** real
labelled slots. No statistical claim is made from it, any result is labelled preliminary and
trained on synthesised sequences, and the synthetic-to-real gap is explicitly unquantifiable.
It is item 4 in the pre-registration's *"not answerable with the available data"* list, written
before the analysis rather than after.

*What makes this survivable:* it is a **data** limitation with a named, cheap fix — 30–40
slots from the facility's own booking sheet — and that request is written and prioritised.

→ `thesis/preregistration.md` §not answerable, `thesis/data_requests.md` §2

## 5 · "Can anyone reproduce this?"

*Yes, and it is checked rather than claimed.*

`experiments/reproduce_all.py` regenerates every committed result from the footage in
dependency order, verifies each stage's outputs exist rather than trusting exit 0, and reports
blocked stages instead of silently succeeding. `uv.lock` and `.python-version` are committed,
torch is pinned to the CPU wheel index. **Every quantitative claim in the write-up is re-derived from its
artefact on every run** and checked against the documents that state it.

*Be ready to volunteer the caveat:* the footage itself cannot be distributed — it shows
identifiable people at a client facility — so reproduction requires the data, and nine frames
are already permanent in git history (WP1-T5), which forecloses a features-only release.

→ `experiments/reproduce_all.py`, `thesis/claims.md`, `docs/CODEBASE.md`

## 6 · "Is auditing staff with cameras ethical?"

*The design answer is that the system never audits a person.*

Three constraints, and all three are enforced rather than promised:

1. **Anomalies are per field, never per person.** `entered_by` is carried for the operator and
   deliberately never aggregated; a test asserts no per-person field reaches the output.
2. **A human confirms every anomaly.** `Advisory.requires_human_confirmation` is a property,
   not a setting — there is no way to construct one that does not need a person.
3. **No automated financial action, ever.** The permitted vocabulary is three actions and money
   is not among them; tests walk the HTTP surface and the database layer and fail if either
   grows a way to charge, invoice, penalise or amend a booking.

Also: scene state, not identities. No face recognition, no re-identification, no headcount.

*The honest edge:* the operator can do whatever they like with the database. This is a
statement about what the software offers, kept true by tests — not a safeguard against a
determined employer. Say that; it is more convincing than claiming otherwise.

→ `src/pitch_occupancy/slots/authority.py`, `tests/test_authority.py`, `thesis/ethics.md`

## 7 · "Your evaluation says the models barely work. What is the contribution?"

**The evaluation is the contribution.** This is the frame to establish early, not to retreat
into.

- Leakage does not shave points off, it **halves** the score — and a zero-shot control shows
  about **a third** of that fall is test-set composition rather than leakage, so the honest
  figure is 0.33–0.40, not 0.52–0.58.
- On the cross-venue protocol **a constant predictor scores macro-F1 1.000**, ahead of every
  backbone, because every held-out venue is one class. That protocol cannot rank models at all.
- **100% of leaky-split errors had a near-duplicate in training; 0% of honest-split errors
  did.** A direct measurement of what the protocol leaked.
- Six pre-registered hypotheses, all reported: one confirmed and corrected upward, one refuted,
  one inconclusive, one with a clause that is unrunnable, two confirmed.

*The sentence to have ready:* "Three of the six hypotheses needed an amendment, and every
amendment is recorded. That is what following a protocol looks like from the inside."

→ `results/EXPERIMENT_LOG.md`, `results/benchmark_v2.csv`

## 8 · "Is your novelty just temporal smoothing?"

*Partly, and the honest answer is to say where the line is.*

Aggregating per-minute predictions into a slot verdict **is** smoothing, and a
majority-vote-with-threshold baseline is most of it. The claim is not that the aggregation is
novel in the abstract; it is that a **learned, calibrated** decision layer beats hand-set ratio
thresholds — and that comparison is the experiment, not the assumption.

*What weakens it, and say so first:* beating hand-tuned thresholds proves very little unless
the baselines are strong, and the slot test set is n=2. The fusion module's own sanity
baseline already came back as a **negative result** — a parameter-free average captures the
whole of the oracle headroom, so a learned gate has no room — and it is reported as one.

→ `TODO.md` WP5-T7/T9, `results/logit_average_baseline.csv`

## 9 · "What is the human ceiling on this task?"

*Not measured, and that is a real gap.*

No second annotator, so no inter-annotator κ and no human ceiling. A reported accuracy of 0.99
means nothing without knowing whether two people agree at 0.99 or 0.85 — and on the ambiguous
cases (a single person crossing an empty pitch; a warm-up) they may well not.

*What to say:* "We do not know, it is the right question, and the protocol specifies the
10% double-labelling pass that would answer it." Do not improvise a number.

→ `TODO.md` WP2-T5, WP2-T9, `thesis/labelling_protocol.md`

## 10 · "You found bugs in your own evaluation. Why should I trust any of it?"

*The question behind the question is whether the errors were found by luck or by method.*

Several results moved because a guard existed and did not operate: a final-test-set lock that
failed open from any other directory; a seeded split that returned a different row **order**
every process, so every bootstrap interval was a different draw; a false-play control scored
against training data; a figure with its ranks hardcoded contradicting its own CSV; a
diagnostic quoted for a day before a second measurement **retracted** it.

*The answer:* every one was found by a check that was added because a previous one was missed,
and each is now covered by a test that was verified to fail on the old code. The claims ledger
exists because a number can go stale silently — and **it caught a real one on its first run**:
an amendment corrected a figure without following it through to the claim resting on it (A12).

*The strongest form:* "A project where nothing was ever retracted has either done very little
or looked very lightly. Every retraction here is in the log with the measurement that forced
it."

→ `results/EXPERIMENT_LOG.md`, `thesis/claims.md`, `thesis/preregistration.md` A8/A12

---

## Four more worth having ready

**"Why frozen backbones rather than fine-tuning?"** — CPU-only is a stated design constraint,
not a limitation worked around; a frozen backbone plus a logistic head retrains in seconds on
cached features, which is what makes the whole ablation programme affordable. The cost is
accepted and named.

**"Your best model changed three times. Which is it?"** — DINOv2, on cross-venue evidence.
Latency stopped being the binding constraint (10–24× headroom), an input-path finding
challenged the pick, and that challenge was **tested and did not survive** — it reverses when
the two cameras it rests on are swapped. The intervals overlap, so the lead is not established
by them, and the page that makes the recommendation says so.

**"Why is macro-F1 sometimes over two classes and sometimes three?"** — because C3 has six
frames in the entire corpus and support 1 in one test set. The rule is fixed once from the
full test set, stated in every row, and the change that introduced it corrected an interval
that had been ~15× too wide (A8).

**"What would you do with another six months?"** — not more models. Empty-pitch footage from a
second venue and 30–40 labelled slots, in that order. Both are in the data request, both are
cheap for the facility, and between them they unblock RQ1, RQ6, the confidence thresholds, the
input-path decision and the entire slot-level evaluation.

---

## How to rehearse this

1. Rewrite every answer above in your own words. An argument in someone else's phrasing
   collapses on the first follow-up.
2. For each, prepare **one artefact you can point at** — a figure, a table, a log entry.
3. Have the mock examination two weeks before the defence, not two days (WP8-T7).
4. Practise the concessions out loud. *"That is a real limitation and here is exactly how
   large it is"* has to sound like a finding, because it is one.
