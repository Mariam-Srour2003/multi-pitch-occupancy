"""Calibration decides how much human review the facility is billed for, so a
miscalibrated score is an operational error, not a cosmetic one."""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.evaluation.calibration import (
    apply_temperature,
    coverage_for_target_accuracy,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    risk_coverage_curve,
)


def test_perfect_calibration_has_zero_error() -> None:
    """90% confident and right 90% of the time, in every bin."""
    conf, ok = [], []
    for c in (0.55, 0.65, 0.75, 0.85, 0.95):
        n = 200
        conf += [c] * n
        ok += [True] * round(c * n) + [False] * (n - round(c * n))
    assert expected_calibration_error(np.array(conf), np.array(ok)) < 0.01


def test_overconfidence_is_detected() -> None:
    conf = np.full(200, 0.99)
    ok = np.array([True] * 100 + [False] * 100)  # 99% confident, 50% right
    assert expected_calibration_error(conf, ok) == pytest.approx(0.49, abs=0.01)


def test_bins_report_signed_gap() -> None:
    bins = reliability_bins(np.full(100, 0.95), np.array([True] * 50 + [False] * 50))
    assert len(bins) == 1
    assert bins[0].gap > 0  # positive == over-confident


def test_empty_bins_are_omitted_not_reported_as_zero() -> None:
    bins = reliability_bins(np.full(50, 0.95), np.ones(50, bool), n_bins=10)
    assert len(bins) == 1  # only the 0.9-1.0 bin exists


def test_misaligned_inputs_raise() -> None:
    with pytest.raises(ValueError, match="align"):
        reliability_bins(np.zeros(5), np.zeros(3, bool))


# --- temperature ------------------------------------------------------------


def test_temperature_preserves_predictions() -> None:
    """The whole point: it changes confidence, never the decision."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 3)) * 3
    before = logits.argmax(1)
    for t in (0.3, 1.0, 2.5, 8.0):
        assert np.array_equal(apply_temperature(logits, t).argmax(1), before)


def test_temperature_above_one_softens() -> None:
    logits = np.array([[5.0, 0.0, 0.0]])
    assert apply_temperature(logits, 4.0).max() < apply_temperature(logits, 1.0).max()


def test_temperature_below_one_sharpens() -> None:
    logits = np.array([[2.0, 1.0, 0.0]])
    assert apply_temperature(logits, 0.4).max() > apply_temperature(logits, 1.0).max()


def test_rows_remain_probability_distributions() -> None:
    rng = np.random.default_rng(1)
    p = apply_temperature(rng.normal(size=(50, 4)), 2.0)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def test_non_positive_temperature_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(np.zeros((2, 2)), 0.0)


def test_fit_temperature_cools_an_overconfident_model() -> None:
    """Wildly separated logits that are often wrong should be softened, T > 1."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, size=400)
    logits = np.zeros((400, 3))
    logits[np.arange(400), y] = 5.0
    wrong = rng.random(400) < 0.30
    logits[wrong] = np.roll(logits[wrong], 1, axis=1)
    assert fit_temperature(logits, y) > 1.0


def test_fit_temperature_leaves_a_calibrated_model_alone() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=600)
    logits = rng.normal(size=(600, 3))
    logits[np.arange(600), y] += 1.4
    assert 0.5 < fit_temperature(logits, y) < 2.5


# --- risk / coverage --------------------------------------------------------


def test_full_coverage_equals_overall_accuracy() -> None:
    conf = np.linspace(0.5, 1.0, 100)
    ok = np.array([True] * 80 + [False] * 20)
    cov, acc, review = risk_coverage_curve(conf, ok)[-1]
    assert cov == pytest.approx(1.0)
    assert acc == pytest.approx(0.8)
    assert review == pytest.approx(0.0)


def test_accuracy_improves_as_coverage_shrinks() -> None:
    """The defining property: deferring the least confident cases should help."""
    rng = np.random.default_rng(4)
    conf = rng.random(500)
    ok = rng.random(500) < conf  # confidence genuinely predicts correctness
    curve = risk_coverage_curve(conf, ok)
    assert curve[0][1] > curve[-1][1]


def test_review_rate_is_the_complement_of_coverage() -> None:
    conf = np.linspace(0, 1, 50)
    ok = np.ones(50, bool)
    assert all(abs(cov + rev - 1.0) < 1e-9 for cov, _, rev in risk_coverage_curve(conf, ok))


def test_coverage_for_target_finds_an_operating_point() -> None:
    conf = np.linspace(0, 1, 200)
    ok = conf > 0.25  # everything above 0.25 is correct
    result = coverage_for_target_accuracy(conf, ok, target=0.99)
    assert result is not None
    cov, review = result
    assert 0.6 < cov < 0.8
    assert review == pytest.approx(1 - cov)


def test_unreachable_target_returns_none_rather_than_lying() -> None:
    conf = np.linspace(0, 1, 100)
    ok = np.zeros(100, bool)  # always wrong
    assert coverage_for_target_accuracy(conf, ok, target=0.99) is None


def test_boundary_temperature_warns() -> None:
    """A temperature pinned to the grid edge means the optimum is outside it - i.e. the
    calibration slice does not resemble the evaluation data. It must not pass silently."""
    logits = np.zeros((50, 3))
    logits[np.arange(50), np.zeros(50, int)] = 30.0  # extreme, always correct
    y = np.zeros(50, dtype=int)
    with pytest.warns(RuntimeWarning, match="grid boundary"):
        fit_temperature(logits, y)
