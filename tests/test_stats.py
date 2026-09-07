"""Statistics utilities checked against hand-computable cases.

Every significance claim in the thesis passes through this module, so its failures would
be invisible in the results and only surface under questioning at the defence."""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.evaluation.metrics import confusion_matrix, evaluate
from pitch_occupancy.evaluation.stats import (
    bootstrap_ci,
    bootstrap_metric_ci,
    cohens_g,
    holm_bonferroni,
    mcnemar,
    paired_bootstrap_diff,
    paired_bootstrap_metric_diff,
)

# --- metrics ----------------------------------------------------------------

A, B, C = "C1_EMPTY", "C2_ACTIVE_PLAY", "C3_MAINTENANCE_NON_SPORTING"


def test_perfect_prediction() -> None:
    r = evaluate([A, B, A, B], [A, B, A, B])
    assert r.accuracy == 1.0
    assert r.macro_f1 == 1.0


def test_metrics_match_hand_computation() -> None:
    #        pred A  pred B
    # true A    2       1     -> recall 2/3
    # true B    1       2     -> recall 2/3
    y_true = [A, A, A, B, B, B]
    y_pred = [A, A, B, A, B, B]
    r = evaluate(y_true, y_pred)
    assert r.accuracy == pytest.approx(4 / 6)
    for s in r.per_class:
        assert s.precision == pytest.approx(2 / 3)
        assert s.recall == pytest.approx(2 / 3)
        assert s.f1 == pytest.approx(2 / 3)
    assert r.macro_f1 == pytest.approx(2 / 3)


def test_absent_class_is_named_not_averaged_as_zero() -> None:
    """C3 has 6 frames in the whole dataset; folding an undefined F1 into the macro
    average as 0.0 would depress the headline for a reason unrelated to the model."""
    r = evaluate([A, A, B, B], [A, A, B, B])
    assert C in r.absent_classes
    assert {s.label for s in r.per_class} == {A, B}
    assert r.macro_f1 == 1.0


def test_confusion_matrix_orientation() -> None:
    cm = confusion_matrix([A, A, B], [A, B, B], classes=[A, B])
    assert cm.tolist() == [[1, 1], [0, 1]]  # rows true, cols predicted


def test_balanced_accuracy_ignores_class_imbalance() -> None:
    """1000 of one class, 2 of another, majority-class predictor."""
    y_true = [A] * 1000 + [B] * 2
    y_pred = [A] * 1002
    r = evaluate(y_true, y_pred)
    assert r.accuracy > 0.99
    assert r.balanced_accuracy == pytest.approx(0.5)


# --- bootstrap --------------------------------------------------------------


def test_bootstrap_ci_brackets_the_estimate() -> None:
    ci = bootstrap_ci([1.0] * 80 + [0.0] * 20, resamples=2000)
    assert ci.estimate == pytest.approx(0.8)
    assert ci.low < 0.8 < ci.high


def test_bootstrap_ci_is_deterministic_for_a_seed() -> None:
    v = [1.0] * 50 + [0.0] * 50
    assert bootstrap_ci(v, resamples=500, seed=7) == bootstrap_ci(v, resamples=500, seed=7)


def test_ci_narrows_as_the_sample_grows() -> None:
    small = bootstrap_ci([1.0] * 8 + [0.0] * 2, resamples=2000)
    large = bootstrap_ci([1.0] * 800 + [0.0] * 200, resamples=2000)
    assert (large.high - large.low) < (small.high - small.low)


def test_bootstrap_on_empty_sample_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])


def test_bootstrap_metric_ci_works_for_macro_f1() -> None:
    y_true = [A] * 50 + [B] * 50
    y_pred = [A] * 45 + [B] * 5 + [B] * 50
    ci = bootstrap_metric_ci(y_true, y_pred, lambda t, p: evaluate(t, p).macro_f1, resamples=400)
    assert 0.0 < ci.low <= ci.estimate <= ci.high <= 1.0


def test_paired_bootstrap_detects_a_consistent_advantage() -> None:
    a = np.array([True] * 60 + [False] * 40)
    b = np.array([True] * 40 + [False] * 60)
    ci = paired_bootstrap_diff(a, b, resamples=2000)
    assert ci.estimate == pytest.approx(0.2)
    assert ci.low > 0  # excludes zero -> a real difference


def test_paired_bootstrap_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        paired_bootstrap_diff([True, False], [True])


# --- mcnemar ----------------------------------------------------------------


def test_mcnemar_identical_models_cannot_be_distinguished() -> None:
    r = mcnemar([True, False, True], [True, False, True])
    assert r.n_discordant == 0
    assert r.p_value == 1.0


def test_mcnemar_counts_only_discordant_pairs() -> None:
    a = [True, True, False, False]
    b = [True, False, True, False]
    r = mcnemar(a, b)
    assert (r.n_only_a_correct, r.n_only_b_correct) == (1, 1)
    assert r.n_discordant == 2


def test_mcnemar_significant_when_one_model_dominates() -> None:
    a = np.array([True] * 30 + [False] * 30)
    b = np.array([False] * 30 + [False] * 30)
    r = mcnemar(a, b)
    assert r.p_value < 0.001
    assert r.effect_size == pytest.approx(0.5)  # maximally lopsided


def test_mcnemar_exact_and_approximate_regimes_agree() -> None:
    a = np.array([True] * 18 + [False] * 6 + [True] * 100)
    b = np.array([False] * 18 + [True] * 6 + [True] * 100)
    exact = mcnemar(a, b, exact_below=100)
    approx = mcnemar(a, b, exact_below=1)
    assert exact.p_value == pytest.approx(approx.p_value, abs=0.05)


def test_mcnemar_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        mcnemar([True], [True, False])


def test_cohens_g_is_zero_when_errors_are_symmetric() -> None:
    assert cohens_g(10, 10) == 0.0
    assert cohens_g(0, 0) == 0.0


# --- multiple comparisons ---------------------------------------------------


def test_holm_leaves_a_single_test_untouched() -> None:
    adj, rej = holm_bonferroni([0.04])
    assert adj == [pytest.approx(0.04)]
    assert rej == [True]


def test_holm_preserves_input_order() -> None:
    adj, _ = holm_bonferroni([0.5, 0.001, 0.2])
    assert adj[1] < adj[2] < adj[0]


def test_holm_kills_the_borderline_result_in_a_large_family() -> None:
    """Thirty comparisons with one 'significant' p=0.04 is the p-hacking pattern
    the pre-registration commits to correcting for."""
    p = [0.04] + [0.6] * 29
    adj, rejected = holm_bonferroni(p)
    assert adj[0] == pytest.approx(1.0)
    assert not any(rejected)


def test_holm_keeps_a_genuinely_strong_result() -> None:
    adj, rejected = holm_bonferroni([1e-6] + [0.6] * 29)
    assert rejected[0]
    assert adj[0] < 0.05


def test_holm_is_monotone() -> None:
    p = sorted([0.001, 0.01, 0.02, 0.04, 0.3])
    adj, _ = holm_bonferroni(p)
    assert all(x <= y + 1e-12 for x, y in zip(adj, adj[1:], strict=False))


def test_holm_never_exceeds_one() -> None:
    adj, _ = holm_bonferroni([0.9] * 50)
    assert max(adj) <= 1.0


def test_holm_on_empty_family() -> None:
    assert holm_bonferroni([]) == ([], [])


# --- paired_bootstrap_metric_diff: testing the metric that is actually reported ----


def _macro_f1(y_true, y_pred) -> float:
    from pitch_occupancy.evaluation.metrics import evaluate

    return evaluate(list(y_true), list(y_pred)).macro_f1


_Y = ["C1_EMPTY"] * 50 + ["C2_ACTIVE_PLAY"] * 50


def test_identical_predictions_give_a_zero_difference_with_a_zero_width_interval() -> None:
    ci = paired_bootstrap_metric_diff(_Y, _Y, _Y, _macro_f1, resamples=300)
    assert ci.estimate == 0.0
    assert (ci.low, ci.high) == (0.0, 0.0)


def test_the_difference_is_antisymmetric_in_its_arguments() -> None:
    """Swapping the models must flip the sign and mirror the interval.

    Cheap to check and it would catch the single most likely implementation slip - scoring
    the two models on different resamples, which breaks the pairing and makes the interval
    a statement about sampling noise rather than about the models.
    """
    worse = ["C2_ACTIVE_PLAY"] * 10 + _Y[10:]
    ab = paired_bootstrap_metric_diff(_Y, _Y, worse, _macro_f1, resamples=300)
    ba = paired_bootstrap_metric_diff(_Y, worse, _Y, _macro_f1, resamples=300)
    assert ab.estimate == pytest.approx(-ba.estimate)
    assert ab.low == pytest.approx(-ba.high)
    assert ab.high == pytest.approx(-ba.low)


def test_a_real_difference_gives_an_interval_clear_of_zero() -> None:
    worse = ["C2_ACTIVE_PLAY"] * 25 + _Y[25:]
    ci = paired_bootstrap_metric_diff(_Y, _Y, worse, _macro_f1, resamples=500)
    assert ci.estimate > 0
    assert ci.low > 0, "an interval spanning zero would call a clear difference inconclusive"


def test_the_observed_estimate_is_the_unresampled_difference() -> None:
    """The point estimate must come from the data, not from the bootstrap mean.

    A bootstrap distribution of a bounded metric is skewed near the boundary, so reporting
    its mean would quietly bias the headline number.
    """
    worse = ["C2_ACTIVE_PLAY"] * 10 + _Y[10:]
    ci = paired_bootstrap_metric_diff(_Y, _Y, worse, _macro_f1, resamples=200)
    assert ci.estimate == pytest.approx(_macro_f1(_Y, _Y) - _macro_f1(_Y, worse))


def test_misaligned_inputs_raise_rather_than_broadcast() -> None:
    with pytest.raises(ValueError, match="align"):
        paired_bootstrap_metric_diff(_Y, _Y, _Y[:-1], _macro_f1, resamples=10)


def test_an_empty_sample_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_metric_diff([], [], [], _macro_f1, resamples=10)


def test_it_is_reproducible_for_a_fixed_seed_and_varies_without_one() -> None:
    worse = ["C2_ACTIVE_PLAY"] * 10 + _Y[10:]
    a = paired_bootstrap_metric_diff(_Y, _Y, worse, _macro_f1, resamples=200, seed=1)
    b = paired_bootstrap_metric_diff(_Y, _Y, worse, _macro_f1, resamples=200, seed=1)
    c = paired_bootstrap_metric_diff(_Y, _Y, worse, _macro_f1, resamples=200, seed=2)
    assert (a.low, a.high) == (b.low, b.high)
    assert (a.low, a.high) != (c.low, c.high)


def test_it_answers_a_different_question_from_mcnemar() -> None:
    """The reason this function exists.

    Two models with the *same* accuracy can have very different macro-F1 when one of them
    concentrates its errors on the minority class. McNemar sees no difference; the reported
    metric does. H2 was published with a macro-F1 delta beside an accuracy p-value, and for
    one pair they disagreed on direction.
    """
    y = ["C1_EMPTY"] * 10 + ["C2_ACTIVE_PLAY"] * 90
    # a errs on 5 majority frames; b errs on 5 minority frames. Same accuracy, 95%.
    a = ["C1_EMPTY"] * 10 + ["C1_EMPTY"] * 5 + ["C2_ACTIVE_PLAY"] * 85
    b = ["C2_ACTIVE_PLAY"] * 5 + ["C1_EMPTY"] * 5 + ["C2_ACTIVE_PLAY"] * 90

    acc = lambda p: sum(x == t for x, t in zip(p, y, strict=True)) / len(y)  # noqa: E731
    assert acc(a) == acc(b)

    ci = paired_bootstrap_metric_diff(y, a, b, _macro_f1, resamples=400)
    assert abs(ci.estimate) > 0.05, "macro-F1 should separate them where accuracy cannot"

