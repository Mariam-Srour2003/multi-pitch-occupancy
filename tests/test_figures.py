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
    "name", ["label_efficiency", "ranking_inversion", "cross_venue_recall", "risk_coverage_band"]
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
