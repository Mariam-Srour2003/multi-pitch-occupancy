"""H5's clause-by-clause report.

Its two clauses fail in different ways and the value of the experiment is keeping them
apart: one switch is *unrunnable* and the other is *refuted*, and collapsing those into a
single verdict would either invent a null result or bury a real one. What is checked here is
the machinery that keeps the distinction honest.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from pitch_occupancy.data.manifest import ManifestRow
from pitch_occupancy.data.splits import Split
from pitch_occupancy.vision.preprocess import SWITCHES

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
h5 = importlib.import_module("experiments.h5_preprocessing_switches")

PLAY, EMPTY, C3 = "C2_ACTIVE_PLAY", "C1_EMPTY", "C3_MAINTENANCE_NON_SPORTING"


def row(name: str, *, venue="venue_01", lighting="night", cls=PLAY) -> ManifestRow:
    return ManifestRow(
        file=name, class4="2_active_play", class3=cls, venue=venue, camera="c",
        slot_date="2026-07-12", slot_time="20:30", slot_id="s1", t_s=0, source="regular",
        labeled_by="human", lighting=lighting, quality="unknown", split_role="",
    )


# --- clause 1: unrunnable, and provably so -------------------------------------


def test_roi_is_kept_out_of_the_search_space() -> None:
    """A switch that cannot change the image must never enter a search.

    With no polygon, `roi_mask` returns the frame untouched - so an ROI arm would record
    "ROI masking does not help" from a transform that never ran. H5's first clause is
    reported unrunnable on exactly this basis, so the basis is pinned.
    """
    assert "roi" not in SWITCHES


def test_the_roi_clause_reports_why_rather_than_asserting_it() -> None:
    """The claim "ROI was never evaluated" is checked against the search record, not
    written into the output - a hardcoded justification is the failure mode this project
    has already met in a figure."""
    if not (ROOT / "results" / "preprocess_search.json").exists():
        pytest.skip("search results not present")
    got = h5.report_roi_clause()
    assert got["roi_in_search_space"] is False
    assert got["roi_values_evaluated"] == ["False"]
    assert got["n_evaluations"] > 0
    assert got["polygon_file_exists"] == (ROOT / "configs" / "cameras.json").exists()


# --- clause 2: the lighting stratification -------------------------------------


def test_lighting_subsets_are_restricted_to_venue_01() -> None:
    """`lighting` elsewhere is a brightness proxy WP3-T5 found wrong for three venues.
    Within venue_01 it is the recording day, which is why the restriction exists."""
    split = Split(
        name="s", group_key="slot_id",
        train=(),
        test=(row("a.jpg"), row("b.jpg", lighting="day"),
              row("c.jpg", venue="clipvenue_g_netting", lighting="day")),
    )
    got = h5.lighting_subsets(split)
    assert got == {"night": [0], "day": [1]}


def test_an_absent_lighting_condition_is_omitted_not_reported_empty() -> None:
    """Four of five grouped splits have no day frames at all. An empty list would be
    scored as a subset with n=0 and produce a nan that reads as a number."""
    split = Split(name="s", group_key="slot_id", train=(),
                  test=(row("a.jpg"), row("b.jpg")))
    assert set(h5.lighting_subsets(split)) == {"night"}


def test_the_paired_restriction_cuts_both_predictions_with_the_truth() -> None:
    """A paired comparison needs both arms scored on identical frames; restricting one
    vector to the evaluable classes would silently unpair it."""
    truth = [PLAY, PLAY, EMPTY, EMPTY, C3]
    a = [PLAY, EMPTY, EMPTY, PLAY, C3]
    b = [PLAY, PLAY, EMPTY, EMPTY, PLAY]
    yt, ya, yb = h5._evaluable_triple(truth, a, b)
    assert len(yt) == len(ya) == len(yb) == 4
    assert C3 not in yt


def test_the_arms_are_read_from_the_search_record() -> None:
    """Hardcoding a cache hash would point at the wrong features the moment the search is
    re-run, and nothing would say so."""
    if not (ROOT / "results" / "preprocess_search.json").exists():
        pytest.skip("search results not present")
    found = h5.arms()
    assert found, "no CLAHE arm found"
    for model, by_value in found.items():
        assert "off" in by_value, f"{model} has no baseline to compare against"
        assert set(by_value) <= {"off", "on", "auto"}


# --- the committed result -------------------------------------------------------


def test_the_report_keeps_the_two_clauses_apart() -> None:
    """The point of the experiment. If the CSV ever carried a single H5 verdict it would
    be claiming either a null result for a switch that never ran, or a pass for a clause
    whose direction is refuted."""
    import csv

    path = ROOT / "results" / "h5_preprocessing_switches.csv"
    if not path.exists():
        pytest.skip("H5 results not present")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    switches = {r["switch"] for r in rows if r["switch"]}
    assert switches, "no switch was compared"
    assert all(s.startswith("clahe=") for s in switches), (
        f"a switch other than CLAHE is reported as tested: {switches}"
    )
    scored = [r for r in rows if r["delta"]]
    assert scored, "no comparison carries a delta"
    assert all(r["ci_low"] and r["ci_high"] for r in scored), "a delta lacks its interval"
