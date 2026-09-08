"""WP6-T10's cost model.

Its output is a break-even, not a euro figure, because two of the three inputs a euro figure
needs are assumptions and the third — flag precision — cannot be measured on this corpus at
all. So what is tested is the arithmetic and the honesty of the framing: that the model is
linear in precision, that only the anomaly types which recover money do, and that the
assumptions are parameters rather than constants baked into a conclusion.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
rv = importlib.import_module("experiments.reconciliation_value")


@pytest.fixture
def base() -> "rv.Assumptions":
    return rv.Assumptions()


# --- the arithmetic ---------------------------------------------------------------


def test_recovery_is_linear_in_precision(base) -> None:
    """The first version multiplied by precision twice - once to count the true flags and
    again against the total recoverable - which halved the recovery at p=0.5 and moved the
    break-even by several points. Doubling precision must double recovery."""
    quarter = rv.value_per_1000(base, 0.25)["recovered_eur"]
    half = rv.value_per_1000(base, 0.5)["recovered_eur"]
    assert half == pytest.approx(2 * quarter)


def test_only_the_types_that_recover_money_do(base) -> None:
    """An unbooked slot is revenue never invoiced; a no-show is a correction to a record.
    Flattening them to one number would credit the system with money it did not find."""
    rates = rv.flag_rate(base)
    assert rates["UNBOOKED_USAGE"][1] == pytest.approx(base.slot_price_eur)
    assert rates["NO_SHOW_OR_OVERRECORDED"][1] == 0.0
    assert rates["PLAYED_NOT_RECORDED"][1] == 0.0


def test_a_perfect_detector_costs_only_the_investigations(base) -> None:
    got = rv.value_per_1000(base, 1.0)
    assert got["false_accusation_eur"] == 0.0
    assert got["investigation_eur"] > 0
    assert got["net_eur"] == pytest.approx(
        got["recovered_eur"] - got["investigation_eur"]
    )


def test_a_useless_detector_recovers_nothing_and_still_costs(base) -> None:
    got = rv.value_per_1000(base, 0.0)
    assert got["recovered_eur"] == 0.0
    assert got["net_eur"] < 0


def test_the_flag_rate_comes_from_the_rule_table_not_a_model(base) -> None:
    """`reconcile` is deterministic given (booking, record, verdict), so this part of the
    estimate carries no model uncertainty - only the assumed mix does."""
    rates = rv.flag_rate(base)
    assert rates["UNBOOKED_USAGE"][0] == pytest.approx(base.mix_unbooked_usage * 1000)
    assert rates["NO_SHOW_OR_OVERRECORDED"][0] == pytest.approx(
        base.mix_booked_no_show * 1000
    )


def test_a_consistent_case_is_never_flagged(base) -> None:
    """80% of slots are booked and used. If those produced flags the whole model would be
    describing a different system."""
    assert "CONSISTENT" not in rv.flag_rate(base)


# --- the break-even ---------------------------------------------------------------


def test_the_break_even_is_where_the_net_turns_positive(base) -> None:
    be = rv.break_even_precision(base)
    assert be is not None
    assert rv.value_per_1000(base, be)["net_eur"] > 0
    assert rv.value_per_1000(base, be - 0.01)["net_eur"] <= 0


def test_a_costlier_false_accusation_raises_the_bar(base) -> None:
    """The asymmetry that makes the human-in-the-loop design necessary: the more a wrong
    flag costs, the more nearly perfect the detector has to be."""
    cheap = rv.break_even_precision(replace(base, false_accusation_eur=20.0))
    dear = rv.break_even_precision(replace(base, false_accusation_eur=400.0))
    assert cheap < dear


def test_a_more_valuable_slot_lowers_the_bar(base) -> None:
    """And below some price there is no bar to clear at all.

    At a EUR 10 slot price the feature cannot pay at *any* precision: 200 flags cost more to
    investigate than 50 recoverable slots are worth even when every flag is right. That is a
    finding rather than an edge case, and it is why the answer is a break-even rather than a
    euro figure - the sign of the result depends on a number the facility supplies.
    """
    cheap = rv.break_even_precision(replace(base, slot_price_eur=10.0))
    dear = rv.break_even_precision(replace(base, slot_price_eur=200.0))
    assert cheap is None, "a EUR 10 slot should not be worth chasing"
    assert dear is not None and dear < rv.break_even_precision(base)


def test_a_feature_that_can_never_pay_reports_that_rather_than_a_number(base) -> None:
    """If no precision makes it worthwhile, the answer is "never", not the last grid point."""
    hopeless = replace(base, slot_price_eur=0.01, false_accusation_eur=10_000.0)
    assert rv.break_even_precision(hopeless) is None


# --- the framing ------------------------------------------------------------------


def test_the_mix_must_sum_to_one(base) -> None:
    with pytest.raises(SystemExit, match="does not|not 1.0"):
        replace(base, mix_booked_and_used=0.5).check()


def test_every_cost_is_a_parameter_rather_than_a_literal() -> None:
    """The output is reported as a function of the assumptions so a reader substitutes
    their own. A cost hardcoded in the arithmetic could not be substituted."""
    import inspect

    source = inspect.getsource(rv.value_per_1000)
    assert "40" not in source and "120" not in source


def test_the_module_says_it_is_an_evaluation_not_an_action() -> None:
    """Attaching money to the confusion matrix and taking a financial action are opposites,
    and they are easy to conflate. WP6-T12 forbids the second."""
    assert "evaluation, not an action" in rv.__doc__
