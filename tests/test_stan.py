"""STAN, its baselines, and the composed slot dataset (WP5-T1, WP5-T7, WP5-T8).

The thing most worth testing here is not the network. It is the **gate**: WP5-T8 says no
headline result below 30 real labelled slots, this project has two, and a rule of that kind
is worthless unless something refuses. So the first tests are about the refusal, and they
check it by asking it to pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.slots.stan import (
    CLASS_ORDER,
    MIN_REAL_SLOTS,
    PREDICTORS,
    STAN,
    HMMSmoothing,
    MedianSmoothing,
    NotEnoughRealSlots,
    SummaryStatsLogistic,
    TunedThresholds,
    assert_preliminary,
    preliminary_caveat,
)
from pitch_occupancy.slots.synthetic import (
    TEMPLATES,
    SlotSequence,
    compose_dataset,
    compose_slot,
    template_states,
)

ROOT = Path(__file__).resolve().parents[1]

INDEX = {c: i for i, c in enumerate(CLASS_ORDER)}


def sequence(states, *, slot_id="s", truth=SlotStatus.USED, synthetic=True, noise=0.0, seed=0):
    """A slot whose probabilities are one-hot on the given states, optionally blurred."""
    rng = np.random.default_rng(seed)
    p = np.full((len(states), 3), noise / 3)
    for i, s in enumerate(states):
        p[i, INDEX[s]] += 1.0 - noise
    if noise:
        p = np.clip(p + rng.normal(0, noise / 4, p.shape), 1e-6, None)
    p /= p.sum(1, keepdims=True)
    return SlotSequence(slot_id, p, truth, synthetic, "test", tuple(states))


PLAY, EMPTY, MAINT = Class3.ACTIVE_PLAY, Class3.EMPTY, Class3.MAINTENANCE_NON_SPORTING


# --- the WP5-T8 gate ------------------------------------------------------------------


def test_the_gate_is_the_number_the_plan_names() -> None:
    assert MIN_REAL_SLOTS == 30


def test_two_real_slots_are_refused() -> None:
    """The project's actual situation. If this ever stops raising without 28 more labelled
    slots appearing, the gate has been removed rather than met."""
    real = [sequence([PLAY] * 5, synthetic=False) for _ in range(2)]
    with pytest.raises(NotEnoughRealSlots, match="below the WP5-T8 gate"):
        assert_preliminary(real)


def test_synthetic_slots_do_not_count_towards_the_gate() -> None:
    """Thirty composed slots are not thirty observations, and a gate that could be satisfied
    by generating more data would be worse than no gate at all."""
    with pytest.raises(NotEnoughRealSlots):
        assert_preliminary([sequence([PLAY] * 5) for _ in range(100)])


def test_the_gate_passes_once_enough_real_slots_exist() -> None:
    """Proves the refusal is about the count and not unconditional."""
    real = [sequence([PLAY] * 5, synthetic=False) for _ in range(MIN_REAL_SLOTS)]
    assert_preliminary(real)
    assert "PRELIMINARY" not in preliminary_caveat(real)


def test_the_caveat_counts_rather_than_hardcodes() -> None:
    real = [sequence([PLAY] * 5, synthetic=False) for _ in range(3)]
    caveat = preliminary_caveat(real)
    assert "3 real labelled slot(s)" in caveat
    assert "PRELIMINARY" in caveat


# --- the composed dataset --------------------------------------------------------------


def test_every_template_has_a_verdict_and_produces_states() -> None:
    rng = np.random.default_rng(0)
    for name in TEMPLATES:
        states = template_states(name, 60, rng)
        assert len(states) == 60
        assert isinstance(TEMPLATES[name][1], SlotStatus)


def test_an_unknown_template_is_refused_rather_than_defaulted() -> None:
    """A silent default would file a differently-shaped slot under someone else's label."""
    with pytest.raises(KeyError, match="unknown template"):
        template_states("abandoned", 60, np.random.default_rng(0))


def test_the_templates_mean_what_they_say() -> None:
    rng = np.random.default_rng(1)
    assert set(template_states("no_show", 60, rng)) == {EMPTY}
    assert MAINT in template_states("maintenance_window", 60, rng)
    late = template_states("late_start", 60, rng)
    assert late[0] is EMPTY and late[-1] is PLAY


def test_the_boundaries_are_jittered() -> None:
    """With fixed fractions each template is one stereotyped shape and the label is
    recoverable by recognising which of five shapes a slot is - which is how the first run of
    `stan_preliminary.py` got 1.000 from every shape-reading model."""
    rng = np.random.default_rng(2)
    starts = {template_states("late_start", 60, rng).index(PLAY) for _ in range(20)}
    assert len(starts) > 1, "the late start begins at the same minute every time"


def test_intermittent_play_is_scattered_and_a_late_start_is_contiguous() -> None:
    """The distinction the ratio rule cannot see and STAN can. If both templates produced the
    same shape, the comparison between them would be measuring noise."""
    rng = np.random.default_rng(3)
    def longest_run(states):
        best = current = 0
        for s in states:
            current = current + 1 if s is PLAY else 0
            best = max(best, current)
        return best

    late = [longest_run(template_states("late_start", 60, rng)) for _ in range(20)]
    inter = [longest_run(template_states("intermittent", 60, rng)) for _ in range(20)]
    assert min(late) > max(inter)


def test_a_composed_slot_is_always_marked_synthetic() -> None:
    """`synthetic` is what the gate counts, so it must not be settable by argument."""
    pool = {c.value: np.arange(5) for c in CLASS_ORDER}
    seq, chosen = compose_slot("x", "full_match", 10, pool, np.random.default_rng(0))
    assert seq.synthetic is True
    assert len(chosen) == 10


def test_composing_without_the_needed_class_is_refused() -> None:
    """Substituting another class would relabel the minute and quietly change the template."""
    pool = {Class3.ACTIVE_PLAY.value: np.arange(5)}
    with pytest.raises(ValueError, match="no frames available"):
        compose_slot("x", "no_show", 10, pool, np.random.default_rng(0))


def test_compose_dataset_checks_the_scorer_shape() -> None:
    """A scorer returning the wrong shape would otherwise be broadcast into place."""
    pool = {c.value: np.arange(20) for c in CLASS_ORDER}
    with pytest.raises(ValueError, match="expected"):
        compose_dataset(pool, lambda idx: np.zeros((len(idx), 2)), n_per_template=1)


def test_a_probability_matrix_of_the_wrong_width_is_refused() -> None:
    with pytest.raises(ValueError, match="expected"):
        SlotSequence("x", np.zeros((10, 4)), SlotStatus.USED, True)


# --- the predictors ---------------------------------------------------------------------


def training_set(seed: int = 0):
    """Slots whose verdict turns on contiguity, not only on the play ratio.

    A late start and an intermittent slot are given the *same* play ratio and different
    shapes. A model reading ratios alone cannot separate them, which is what makes this a
    fixture that distinguishes the predictors rather than ranking them all equal.
    """
    slots = []
    for i in range(30):
        slots.append(sequence([PLAY] * 60, slot_id=f"m{i}", truth=SlotStatus.USED, seed=i))
        slots.append(sequence([EMPTY] * 60, slot_id=f"n{i}", truth=SlotStatus.NOTUSED, seed=i))
        slots.append(sequence(
            [EMPTY] * 40 + [PLAY] * 20, slot_id=f"l{i}", truth=SlotStatus.USED, seed=i))
        scattered = [EMPTY] * 60
        for k in range(0, 60, 3):
            scattered[k] = PLAY
        slots.append(sequence(scattered, slot_id=f"i{i}", truth=SlotStatus.REVIEW, seed=i))
    return slots


@pytest.mark.parametrize("factory", PREDICTORS)
def test_every_predictor_fits_and_predicts_one_status_per_slot(factory) -> None:
    slots = training_set()
    model = factory().fit(slots)
    predicted = model.predict(slots[:8])
    assert len(predicted) == 8
    assert all(isinstance(p, SlotStatus) for p in predicted)


def test_the_ratio_rule_cannot_separate_shape_and_stan_can() -> None:
    """The claim the whole of WP5-T1 rests on, as a test. A late start and a scattered slot
    with the same play ratio mean different things, and a rule reading only ratios must give
    them the same verdict."""
    slots = training_set()
    tuned = TunedThresholds().fit(slots)
    stan = STAN(epochs=200).fit(slots)
    late = [s for s in slots if s.slot_id.startswith("l")]
    scattered = [s for s in slots if s.slot_id.startswith("i")]
    assert tuned.predict(late[:1]) == tuned.predict(scattered[:1]), (
        "the fixture no longer poses the problem: the ratio rule can already tell these apart"
    )
    assert stan.predict(late[:1]) == [SlotStatus.USED]
    assert stan.predict(scattered[:1]) == [SlotStatus.REVIEW]


def test_the_hmm_decodes_a_noisy_sequence_back_to_its_true_states() -> None:
    """The baseline that matters is only a real opponent if its smoothing works. A single
    stray PLAY minute inside an empty hour should be decoded away.

    The stray minute is a *near-tie* (0.55/0.45), not a one-hot spike, and that distinction
    is the whole test. A one-hot spike gives the true state a likelihood of 1e-12, and the
    HMM correctly prefers two transitions costing e^-6.4 each over an emission costing
    e^-27.6 - so an over-confident fixture makes a working decoder look broken. Real
    classifier output is never one-hot, and this is what it looks like.
    """
    clean = [EMPTY] * 60
    hmm = HMMSmoothing().fit([
        sequence(clean, slot_id=f"n{i}", truth=SlotStatus.NOTUSED, seed=i) for i in range(10)
    ] + [
        sequence([PLAY] * 60, slot_id=f"m{i}", truth=SlotStatus.USED, seed=i) for i in range(10)
    ])
    probabilities = sequence(clean, noise=0.3).probabilities.copy()
    probabilities[30] = [0.45, 0.55, 0.0]  # EMPTY, PLAY, MAINTENANCE - a marginal misread
    decoded = hmm._viterbi(probabilities)
    assert decoded.count(PLAY) == 0, "the stray minute survived decoding"
    assert probabilities[30].argmax() == INDEX[PLAY], (
        "the fixture no longer poses the problem: minute 30 is not misread at all"
    )


def test_the_viterbi_runs_in_log_space() -> None:
    """A 90-minute product of probabilities underflows to zero and every path ties at -inf,
    which shows up as a decoder that always returns the same state."""
    hmm = HMMSmoothing().fit([
        sequence([EMPTY] * 90, truth=SlotStatus.NOTUSED),
        sequence([PLAY] * 90, truth=SlotStatus.USED),
    ])
    decoded = hmm._viterbi(sequence([EMPTY] * 45 + [PLAY] * 45).probabilities)
    assert set(decoded) == {EMPTY, PLAY}


def test_median_smoothing_tunes_its_window_rather_than_guessing_one() -> None:
    model = MedianSmoothing().fit(training_set())
    assert model.window in MedianSmoothing.WINDOWS


def test_the_summary_features_describe_the_sequence() -> None:
    late = sequence([EMPTY] * 40 + [PLAY] * 20)
    features = SummaryStatsLogistic.features(late)
    assert features[0] == pytest.approx(20 / 60)
    assert features[3] == pytest.approx(20 / 60), "longest run"
    assert features[4] == pytest.approx(40 / 60), "first active minute"


# --- STAN itself --------------------------------------------------------------------------


def test_stan_stays_under_the_hundred_thousand_parameter_ceiling() -> None:
    stan = STAN(epochs=2).fit(training_set())
    assert stan.n_parameters < 100_000


def test_slot_length_is_masked_rather_than_read_as_evidence() -> None:
    """Padding a short slot to the batch's longest and pooling over the whole width would
    scale its ratios by length, so the model would learn that short slots mean something.
    The same slot must score identically whatever it is batched with.
    """
    slots = training_set()
    stan = STAN(epochs=100).fit(slots)
    short = sequence([PLAY] * 30, slot_id="short")
    alone = stan.predict_proba([short])[0]
    with_long = stan.predict_proba([short, sequence([EMPTY] * 90, slot_id="long")])[0]
    assert np.allclose(alone, with_long, atol=1e-5)


def test_a_single_verdict_training_set_is_refused() -> None:
    slots = [sequence([PLAY] * 60, slot_id=str(i), truth=SlotStatus.USED) for i in range(5)]
    with pytest.raises(ValueError, match="one verdict"):
        STAN(epochs=2).fit(slots)


def test_predicting_before_fitting_says_so() -> None:
    with pytest.raises(RuntimeError, match="was not fitted"):
        STAN().predict([sequence([PLAY] * 5)])


def test_the_same_seed_reproduces_the_fit_exactly() -> None:
    slots = training_set()
    a = STAN(epochs=30, seed=3).fit(slots).predict_proba(slots[:5])
    b = STAN(epochs=30, seed=3).fit(slots).predict_proba(slots[:5])
    assert np.array_equal(a, b)


def test_the_loss_falls() -> None:
    losses = STAN(epochs=150).fit(training_set()).training_loss
    assert losses[-1] < losses[0] / 2


# --- the construction draws (2026-09-10) ---------------------------------------------------
#
# `stan_preliminary.py` composed its slots from one seeded draw. So did the augmentation
# experiment, whose headline did not survive being drawn again - so the same check was run
# here. It came back differently: the ordering replicates and the ceiling does not.


def _spread() -> dict[str, dict]:
    import csv

    path = ROOT / "results" / "stan_draw_spread.csv"
    if not path.exists():
        pytest.skip("draw replication not run")
    return {r["model"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}


def test_stan_is_first_in_every_draw() -> None:
    """What replication confirmed. If a baseline ever matches it in some draw, the ordering
    is no longer the part that survives and the write-up has to change with it."""
    rows = _spread()
    stan = rows["stan"]
    seeds = [k for k in stan if k.startswith("seed_") and stan[k]]
    assert len(seeds) >= 4, "the claim rests on the draws; do not thin them out"
    for seed in seeds:
        best_baseline = max(float(r[seed]) for name, r in rows.items()
                            if name != "stan" and r.get(seed))
        assert float(stan[seed]) > best_baseline, f"a baseline caught stan in {seed}"


def test_the_ceiling_is_a_property_of_the_draw_not_the_benchmark() -> None:
    """What replication corrected. The published 1.0000 was reported as an exhausted test
    set; it is exhausted in some compositions and not others, and both halves of that have
    to stay visible or the caveat drifts back to being unconditional."""
    stan = _spread()["stan"]
    assert float(stan["max"]) >= 1.0, "no draw reaches the ceiling any more"
    assert float(stan["min"]) < 1.0, "every draw saturates - the correction is stale"


def test_the_baselines_move_enough_to_forbid_quoting_one_margin() -> None:
    """`summary_logistic` is the best baseline in one draw and the worst in another, so the
    published 0.105 margin over the HMM is one draw's margin."""
    rows = _spread()
    assert float(rows["summary_logistic"]["sd"]) > 0.05


def test_a_draw_cannot_be_built_from_the_next_draw_s_test_pool() -> None:
    """`seed + 1` composes the test slots, so seeds one apart would share a construction -
    correlated draws reported as independent, which is the opposite of replication."""
    import argparse
    import importlib
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    exp = importlib.import_module("experiments.stan_preliminary")
    assert exp.parse_seeds("42,7,13") == [42, 7, 13]
    with pytest.raises(argparse.ArgumentTypeError):
        exp.parse_seeds("42,43")
