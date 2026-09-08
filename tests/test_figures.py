"""The thesis figures, checked for the ways a plot can lie without erroring.

A figure that renders is not a figure that is right. The most-quoted plot in this project
once hardcoded its own ranks and printed scores, so it contradicted the CSV it was generated
from while the reproduction stage reported success. These tests go after that class: does
the figure read its data, and does it say the thing the data supports?
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
mf = importlib.import_module("experiments.make_figures")

pd = pytest.importorskip("pandas")


def test_every_figure_function_reads_a_result_file() -> None:
    """A figure that reads nothing cannot disagree with the data, and cannot agree either."""
    import inspect

    for name, fn in vars(mf).items():
        if not name.startswith("fig_"):
            continue
        source = inspect.getsource(fn)
        assert "read_csv" in source, f"{name} draws without reading a result file"


def test_the_risk_coverage_figure_draws_a_band_where_the_confidences_are_tied() -> None:
    """The band is the finding. If the figure ever plots only one line per model it has
    quietly become the misleading plot it was written to replace."""
    import inspect

    source = inspect.getsource(mf.fig_risk_coverage_band)
    assert "fill_between" in source
    assert "accuracy_worst" in source and "accuracy_best" in source


def test_the_operating_bound_drawn_is_the_worst_case(monkeypatch) -> None:
    """`coverage_for_target_accuracy` reads the lower bound, so the solid line - the one a
    reader traces to an operating point - has to be that same edge.

    Checked on the rendered artists rather than on the source text. Reading the source only
    proves a string is present; this proves the heaviest line actually drawn for the banded
    model carries the worst-case values, which is what a reader's eye follows.
    """
    path = ROOT / "results" / "rq6_risk_coverage.csv"
    if not path.exists():
        pytest.skip("risk-coverage results not present")

    captured = {}
    monkeypatch.setattr(mf, "save", lambda fig, name: captured.setdefault("fig", fig))
    mf.fig_risk_coverage_band()
    fig = captured["fig"]

    df = pd.read_csv(path)
    banded = df[df.n_tied > 0].sort_values("coverage")
    if banded.empty:
        pytest.skip("no banded model to check")
    worst = set(banded.accuracy_worst.round(4))

    ax = fig.axes[0]
    heavy = [ln for ln in ax.get_lines() if ln.get_linewidth() >= 2.0 and len(ln.get_ydata()) > 5]
    assert heavy, "no solid series drawn"
    assert any(
        set(pd.Series(ln.get_ydata()).round(4)) == worst for ln in heavy
    ), "the heavy line for the banded model is not its worst-case edge"


@pytest.mark.parametrize(
    "name",
    ["label_efficiency", "ranking_inversion", "cross_venue_recall", "risk_coverage_band",
     "baseline_floor", "accuracy_vs_latency"],
)
def test_each_figure_is_written_in_both_formats(name: str) -> None:
    """PNG for the site, PDF for the thesis. A missing PDF is only noticed at submission."""
    figs = ROOT / "results" / "figs"
    if not (figs / f"{name}.png").exists():
        pytest.skip(f"{name} not generated")
    assert (figs / f"{name}.pdf").exists(), f"{name} has no vector version"


def test_the_band_figure_matches_the_data_it_claims() -> None:
    """The caption asserts DINOv2's confidences are tied and the other two models' are not.
    That is a claim about the CSV, so it is checked against the CSV rather than trusted."""
    path = ROOT / "results" / "rq6_risk_coverage.csv"
    if not path.exists():
        pytest.skip("risk-coverage results not present")
    df = pd.read_csv(path)
    tied = df.groupby("model").n_tied.max()
    assert tied.get("dinov2", 0) > 0, "the band's subject has no ties; the figure would mislead"
    for model in ("convnextv2", "vit"):
        if model in tied:
            assert tied[model] == 0
            sub = df[df.model == model]
            assert (sub.accuracy_best == sub.accuracy_worst).all(), (
                f"{model} has no ties but a non-zero band width"
            )


def test_the_floor_chart_marks_where_the_backbone_loses() -> None:
    """The cross-venue column is the one worth drawing: a constant predictor scores a
    perfect macro-F1 there, so the trivial bar is *above* the backbone bar. A chart that
    did not say so would read as four columns where the backbone wins."""
    import inspect

    source = inspect.getsource(mf.fig_baseline_floor)
    assert "the floor is not cleared" in source
    assert 'tv"] >= r["dv"]' in source or "beaten" in source


def test_the_floor_chart_reads_all_four_protocols(monkeypatch) -> None:
    """One protocol would answer "would something trivial have done this?" with a number;
    four answer it with "it depends entirely on which protocol you ask", which is the
    finding."""
    path = ROOT / "results" / "benchmark_v2.csv"
    if not path.exists():
        pytest.skip("benchmark not present")
    captured = {}
    monkeypatch.setattr(mf, "save", lambda fig, name: captured.setdefault("fig", fig))
    mf.fig_baseline_floor()
    ax = captured["fig"].axes[0]
    assert len(ax.get_xticks()) == 4, "the chart does not cover all four protocols"


def test_the_rq2_figure_plots_the_balanced_score_not_recall() -> None:
    """Cross-venue folds contain no empty pitch, so recall alone is earned by answering
    "playing" more often - and the model with the best recall here has the worst balanced
    score. A figure of recall would recommend it."""
    import inspect

    source = inspect.getsource(mf.fig_accuracy_vs_latency)
    assert "false_play_rate" in source
    assert "recall - float" in source or "balanced = recall" in source


def test_the_rq2_figure_draws_the_budget_line_beyond_every_model(monkeypatch) -> None:
    """The answer turns on a negative - the budget is not binding - which is only visible if
    the line is on the axis. An axis that stopped at the slowest model would imply the
    opposite."""
    if not (ROOT / "results" / "efficiency_latency.csv").exists():
        pytest.skip("latency results not present")
    captured = {}
    monkeypatch.setattr(mf, "save", lambda fig, name: captured.setdefault("fig", fig))
    mf.fig_accuracy_vs_latency()
    ax = captured["fig"].axes[0]
    slowest = pd.read_csv(ROOT / "results" / "efficiency_latency.csv").round_wall_s.max()
    assert ax.get_xlim()[1] > 60.0 > slowest, "the 60 s budget is not on the axis"
