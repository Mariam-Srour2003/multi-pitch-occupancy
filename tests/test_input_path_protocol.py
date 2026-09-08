"""The input-path experiment's guards and its statistical primitive.

The experiment itself needs feature caches and a few minutes, so what is checked here is
everything that decides whether its output means anything: the sign-flip test's exactness
and its resolution floor, the guard that refuses to compare a cache against itself, and the
reproduce-before-extending check that aborts when the harness disagrees with the table it
is extending.

That last one is not hypothetical. A previous fix to `bootstrap_metric_ci` moved published
accuracies by 0.0025 and was caught only because a guard of this shape existed in an
unrelated file, with no test covering the interaction. This is the test that would have
covered it.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from pitch_occupancy.evaluation.stats import EXACT_SIGN_FLIP_LIMIT, sign_flip_test

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "input_path_protocol", ROOT / "experiments" / "input_path_protocol.py"
)
assert spec and spec.loader
ipp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ipp
spec.loader.exec_module(ipp)


# --- the sign-flip test -----------------------------------------------------------


def test_all_differences_in_one_direction_hits_the_floor() -> None:
    """Seven venues all favouring one arm is the most extreme result available, so its p
    must be the smallest the design can return - not something smaller."""
    r = sign_flip_test([0.10, 0.20, 0.05, 0.30, 0.15, 0.02, 0.08])
    assert r.p_value == pytest.approx(2 / 2**7)
    assert r.p_value == pytest.approx(r.min_achievable_p)
    assert r.exact


def test_four_pairs_cannot_reach_significance_at_all() -> None:
    """The property the floor exists to expose: with four clusters, 'not significant'
    describes the sample size and nothing about the effect."""
    r = sign_flip_test([1.0, 1.0, 1.0, 1.0])
    assert r.min_achievable_p == pytest.approx(0.125)
    assert not r.can_reach(0.05)
    assert r.can_reach(0.2)


def test_seven_pairs_can_reach_five_percent_and_only_just() -> None:
    r = sign_flip_test([1.0] * 7)
    assert r.can_reach(0.05)
    assert not r.can_reach(0.01)


def test_p_matches_brute_force_enumeration() -> None:
    """Checked against the definition rather than against itself."""
    d = np.array([0.4, -0.1, 0.25, 0.05, -0.3])
    observed = d.mean()
    hits = sum(
        abs(np.dot(signs, d) / d.size) >= abs(observed) - 1e-12
        for signs in product((-1, 1), repeat=d.size)
    )
    assert sign_flip_test(d).p_value == pytest.approx(hits / 2**d.size)


def test_the_estimate_is_the_plain_mean_difference() -> None:
    d = [0.1, -0.4, 0.2]
    assert sign_flip_test(d).estimate == pytest.approx(np.mean(d))


def test_it_is_symmetric_under_negating_every_difference() -> None:
    """Two-sided means the arms are interchangeable; a p that moved when they were
    swapped would be a one-sided test wearing the wrong name."""
    d = [0.3, -0.1, 0.2, 0.05, 0.4, -0.02, 0.11]
    assert sign_flip_test(d).p_value == pytest.approx(sign_flip_test([-x for x in d]).p_value)


def test_ties_lower_the_floor_they_do_not_raise_it() -> None:
    """Zero differences carry no sign to flip, so they cost power. The floor must be
    computed from the informative pairs, not from the count supplied."""
    r = sign_flip_test([0.5, 0.5, 0.5, 0.0, 0.0])
    assert r.n_pairs == 5
    assert r.n_informative == 3
    assert r.min_achievable_p == pytest.approx(2 / 2**3)


def test_every_pair_tied_is_reported_as_no_evidence_rather_than_a_perfect_result() -> None:
    r = sign_flip_test([0.0, 0.0, 0.0, 0.0])
    assert r.p_value == 1.0
    assert r.min_achievable_p == 1.0
    assert r.estimate == 0.0


def test_a_large_sample_falls_back_to_sampling_and_says_so() -> None:
    d = np.linspace(0.01, 1.0, EXACT_SIGN_FLIP_LIMIT + 1)
    r = sign_flip_test(d, resamples=2000)
    assert not r.exact
    assert r.min_achievable_p == pytest.approx(1 / 2000)


def test_an_empty_set_of_pairs_raises() -> None:
    with pytest.raises(ValueError):
        sign_flip_test([])


def test_a_null_effect_is_not_significant() -> None:
    rng = np.random.default_rng(0)
    r = sign_flip_test(rng.normal(0.0, 1.0, size=12))
    assert r.p_value > 0.05


# --- the experiment's guards --------------------------------------------------------


def test_identical_arms_abort_rather_than_reporting_no_effect() -> None:
    """The failure this experiment would be worst at noticing: pointing both arms at one
    cache and concluding the input path does not matter."""
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    with pytest.raises(SystemExit, match="bit-identical"):
        ipp.assert_arms_differ("vit", {"raw": X, "preproc": X.copy()})


def test_differing_arms_return_how_far_apart_they_are() -> None:
    X = np.zeros((3, 4), dtype=np.float32)
    Y = X.copy()
    Y[1, 2] = 2.5
    assert ipp.assert_arms_differ("dinov2", {"raw": X, "preproc": Y}) == pytest.approx(2.5)


def test_misaligned_arms_abort() -> None:
    with pytest.raises(SystemExit, match="different shapes"):
        ipp.assert_arms_differ("dinov2", {"raw": np.zeros((3, 4)), "preproc": np.zeros((2, 4))})


def test_reproduction_guard_passes_within_tolerance() -> None:
    published = {("dinov2", "a"): 0.9524, ("dinov2", "b"): 1.0}
    computed = {("dinov2", "a"): 0.95241, ("dinov2", "b"): 1.0}
    assert ipp.check_reproduces("x", computed, published) == 2


def test_reproduction_guard_aborts_on_a_number_that_moved() -> None:
    """0.0025 is the size of the drift that a guard like this actually caught once."""
    published = {("dinov2", "a"): 0.9524}
    computed = {("dinov2", "a"): 0.9549}
    with pytest.raises(SystemExit, match="did not reproduce"):
        ipp.check_reproduces("x", computed, published)


def test_reproduction_guard_only_compares_keys_present_in_both() -> None:
    """A published table that gains or loses rows must not fail the run; a row present in
    both and disagreeing must."""
    assert ipp.check_reproduces("x", {("a", "1"): 0.5}, {("b", "2"): 0.9}) == 0
    assert ipp.check_reproduces("x", {("a", "1"): 0.5, ("b", "2"): 0.1}, {("a", "1"): 0.5}) == 1


def test_published_reader_skips_blank_values() -> None:
    """The published CSVs leave `false_play_rate` empty on per-fold rows and fill it only
    on the mean row; parsing a blank as 0.0 would compare against a number nobody wrote."""
    path = ROOT / "results" / "geometry_convention_probe.csv"
    if not path.exists():
        pytest.skip("probe results not present")
    got = ipp._published(path, ("backbone", "arm", "held_out_venue"), "false_play_rate")
    assert got, "no false-play values parsed"
    assert all(k[2] == "MEAN_ACROSS_FOLDS" for k in got)
    assert all(math.isfinite(v) for v in got.values())


def test_physical_camera_resolves_the_mislabelled_tag() -> None:
    """`slot_20260712_2030_camB` is physically camera A. Using the tag would put one view
    on both sides of the false-play control, which is the whole point of the control."""
    from pitch_occupancy.data.manifest import ManifestRow

    row = ManifestRow(
        file="f.jpg", class4="1_empty", class3="C1_EMPTY", venue="venue_01",
        camera="slot_20260712_2030_camB", slot_date="2026-07-12", slot_time="20:30",
        slot_id="venue_01_2026-07-12_2030", t_s=0, source="regular", labeled_by="human",
        lighting="night", quality="unknown", split_role="",
    )
    assert ipp.physical(row) == "camera_A"


def test_camera_slot_cells_counts_camera_and_slot_together() -> None:
    from pitch_occupancy.data.manifest import ManifestRow

    def row(camera: str, slot: str) -> ManifestRow:
        return ManifestRow(
            file=f"{camera}_{slot}.jpg", class4="1_empty", class3="C1_EMPTY",
            venue="venue_01", camera=camera, slot_date="2026-07-11", slot_time="10:00",
            slot_id=slot, t_s=0, source="regular", labeled_by="human", lighting="day",
            quality="unknown", split_role="",
        )

    rows = [
        row("slot_20260711_1000_camA", "s1"),
        row("slot_20260711_1000_camA", "s1"),
        row("slot_20260711_1000_camA", "s2"),
        row("slot_20260711_1000_camB", "s1"),
    ]
    assert ipp.camera_slot_cells(rows) == 3
