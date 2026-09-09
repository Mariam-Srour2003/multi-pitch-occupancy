"""What onboarding a new camera costs in labels (WP4-T3).

The experiment's headline is that five labelled frames of a new camera match five plus 775
from another one. That is a flattering result, so the tests are mostly about the machinery
that keeps it honest: a sample that spreads across classes, a test set that never contains a
frame that was trained on, and a leakage count reported beside every row.
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
oc = importlib.import_module("experiments.onboarding_cost")

OUT = ROOT / "results" / "onboarding_cost.csv"


class Row:
    def __init__(self, file: str, class3: str, venue: str = "v") -> None:
        self.file, self.class3, self.venue = file, class3, venue


def pool(n_a: int = 20, n_b: int = 20) -> list[Row]:
    return (
        [Row(f"a{i}.jpg", "C1_EMPTY") for i in range(n_a)]
        + [Row(f"b{i}.jpg", "C2_ACTIVE_PLAY") for i in range(n_b)]
    )


# --- the sample ------------------------------------------------------------------------


def test_a_budget_of_zero_takes_nothing() -> None:
    assert oc.stratified_sample(pool(), 0, np.random.default_rng(0)) == set()


def test_the_sample_is_the_size_asked_for() -> None:
    for k in (1, 2, 5, 10):
        assert len(oc.stratified_sample(pool(), k, np.random.default_rng(1))) == k


def test_the_sample_spreads_across_classes_rather_than_taking_the_biggest() -> None:
    """An operator onboarding a camera labels a few frames of each thing they care about.
    Drawing uniformly would spend a budget of two on the majority class and measure nothing
    about the other."""
    rows = pool(n_a=200, n_b=5)
    picked = oc.stratified_sample(rows, 2, np.random.default_rng(2))
    classes = {r.class3 for r in rows if r.file in picked}
    assert len(classes) == 2, classes


def test_the_sample_never_repeats_a_frame() -> None:
    """A duplicate would make a budget of k mean fewer than k distinct labels."""
    rows = pool(n_a=3, n_b=3)
    picked = oc.stratified_sample(rows, 6, np.random.default_rng(3))
    assert len(picked) == 6


def test_a_budget_larger_than_the_pool_takes_what_exists() -> None:
    rows = pool(n_a=2, n_b=2)
    assert len(oc.stratified_sample(rows, 10, np.random.default_rng(4))) == 4


def test_different_seeds_draw_different_frames() -> None:
    """At k=1 which frame you get matters more than anything else, which is why the
    experiment runs five seeds rather than reporting one draw as a result."""
    rows = pool(n_a=50, n_b=50)
    draws = {frozenset(oc.stratified_sample(rows, 2, np.random.default_rng(s)))
             for s in range(5)}
    assert len(draws) > 1


# --- feasibility ---------------------------------------------------------------------------


def test_a_single_class_venue_is_not_usable_as_a_target() -> None:
    """The whole reason WP4-T3 cannot be run as written: a held-out venue with one class has
    a test set on which adaptation is unmeasurable."""
    rows = [Row("x.jpg", "C2_ACTIVE_PLAY", venue="clip_a")]
    assert oc.venue_feasibility(rows)[0]["usable_as_target"] is False


def test_a_multi_class_venue_is_usable() -> None:
    rows = [Row("x.jpg", "C2_ACTIVE_PLAY", venue="v1"),
            Row("y.jpg", "C1_EMPTY", venue="v1")]
    assert oc.venue_feasibility(rows)[0]["usable_as_target"] is True


def test_the_real_dataset_has_exactly_one_usable_venue() -> None:
    """If this ever changes, WP4-T3 becomes runnable as written and the proxy should be
    replaced rather than kept. That is the point of asserting it."""
    if not (ROOT / "data" / "processed" / "manifest.csv").exists():
        pytest.skip("manifest not present")
    from pitch_occupancy.config import settings
    from pitch_occupancy.data.manifest import read_manifest
    from pitch_occupancy.data.splits import development_rows

    rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    usable = [v for v in oc.venue_feasibility(rows) if v["usable_as_target"]]
    assert len(usable) == 1, [v["venue"] for v in usable]
    assert usable[0]["venue"] == "venue_01"


# --- the published curve --------------------------------------------------------------------


def published() -> list[dict]:
    if not OUT.exists():
        pytest.skip("onboarding curve not generated")
    return list(csv.DictReader(OUT.open(encoding="utf-8")))


def test_every_row_carries_its_leakage_count_and_scene_count() -> None:
    """0.99 without the 1,082 crossing pairs beside it is the flattering half of the result."""
    for row in published():
        assert row["crossing_pairs"] != ""
        assert int(row["n_distinct_scenes"]) > 0


def test_the_test_set_shrinks_as_the_budget_grows() -> None:
    """Frames used for adaptation must leave the test set. If n_test stayed constant, the
    model would be scored on frames it had just been trained on."""
    for backbone in {r["backbone"] for r in published()}:
        mine = sorted((r for r in published() if r["backbone"] == backbone),
                      key=lambda r: int(r["k"]))
        sizes = [int(r["n_test"]) for r in mine]
        assert sizes == sorted(sizes, reverse=True), (backbone, sizes)
        assert sizes[0] - sizes[-1] == int(mine[-1]["k"])


def test_one_labelled_frame_beats_none_for_every_backbone() -> None:
    """The headline, as a direction rather than a threshold, so it survives the numbers
    moving."""
    for backbone in {r["backbone"] for r in published()}:
        mine = {int(r["k"]): float(r["adapted_mean"]) for r in published()
                if r["backbone"] == backbone}
        assert mine[1] > mine[0] + 0.3, (backbone, mine[0], mine[1])


def test_the_source_only_baseline_is_recorded_as_the_k_zero_row() -> None:
    for row in published():
        if int(row["k"]) == 0:
            assert row["crossing_pairs"] == "0", "the published protocol must have no leak"


def test_every_budget_ran_all_of_its_seeds() -> None:
    """A budget that silently ran fewer seeds would report a narrower spread than it earned."""
    for row in published():
        assert int(row["n_seeds"]) == len(oc.SEEDS), row
