"""WP3-T8's re-scoring of the preprocessing search.

Its job is to say when a searched margin means nothing, so its own arithmetic has to be
right: the two fold means, the leverage that defines the resolution floor, and the guard
that refuses to re-rank a search it cannot first reproduce.

The gate calibration gets its own attention. The claim it replaced - that `clahe='auto'`
fires on 99.81% of frames, making `auto` and `on` differ on three - came from measuring
contrast on the letterboxed *output* while the gate tests the full-resolution frame. It
fires on 78.33%, and the two switch values differ on 342 frames. The lesson is in the
method, so that is what is pinned: the headline is **counted** by running the switch both
ways, which needs no assumption about which image is measured.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sr = importlib.import_module("experiments.search_resolution")

from pitch_occupancy.vision.preprocess import PreprocessConfig, preprocess


# --- the two means -------------------------------------------------------------


def test_the_weighted_mean_is_pooled_recall_over_frames() -> None:
    """Written as a weighted average of fold recalls, it must equal correct/total."""
    folds = [("small", 0.5, 12), ("large", 1.0, 168)]     # 6 + 168 correct of 180
    got = sr.summarise(folds)
    assert got["weighted"] == pytest.approx((6 + 168) / 180)
    assert got["unweighted"] == pytest.approx(0.75)


def test_the_two_means_agree_when_the_folds_are_the_same_size() -> None:
    folds = [("a", 0.5, 20), ("b", 1.0, 20), ("c", 0.75, 20)]
    got = sr.summarise(folds)
    assert got["weighted"] == pytest.approx(got["unweighted"])


def test_the_unweighted_mean_over_weights_the_small_fold() -> None:
    """The defect being measured: a fold of 12 counts as much as one of 168."""
    folds = [("small", 0.0, 12), ("large", 1.0, 168)]
    got = sr.summarise(folds)
    assert got["unweighted"] == pytest.approx(0.5)
    assert got["weighted"] == pytest.approx(168 / 180)
    assert got["unweighted"] < got["weighted"]


def test_the_interval_brackets_the_unweighted_estimate() -> None:
    folds = [("a", 0.9, 20), ("b", 0.8, 20), ("c", 1.0, 20), ("d", 0.7, 20)]
    got = sr.summarise(folds)
    assert got["ci_low"] <= got["unweighted"] <= got["ci_high"]
    assert got["worst_fold"] == pytest.approx(0.7)


# --- the resolution floor ------------------------------------------------------


def test_leverage_is_one_frame_over_folds_times_fold_size() -> None:
    """The floor is derived, not estimated: the smallest change the reported statistic can
    register is one frame flipping in the smallest fold."""
    folds = [("small", 1.0, 12), ("mid", 1.0, 30), ("big", 1.0, 168)]
    lev = sr.fold_leverage(folds)
    assert lev[0][0] == "small"
    assert lev[0][2] == pytest.approx(1 / (3 * 12))
    assert [name for name, _, _ in lev] == ["small", "mid", "big"]


def test_leverage_skips_a_fold_with_no_play_frames() -> None:
    """Dividing by zero would produce an infinite floor and reject every margin."""
    lev = sr.fold_leverage([("empty", float("nan"), 0), ("real", 1.0, 20)])
    assert [name for name, _, _ in lev] == ["real"]


# --- the reproduction guard ----------------------------------------------------


def test_the_guard_aborts_when_a_rescored_number_moved() -> None:
    records = [{"model": "convnextv2", "describe": "baseline", "unweighted": 0.9841}]
    sr.check_reproduces(records, {("convnextv2", "baseline"): 0.9841}, "unweighted", "x")
    with pytest.raises(SystemExit, match="did not reproduce"):
        sr.check_reproduces(records, {("convnextv2", "baseline"): 0.9500}, "unweighted", "x")


def test_the_guard_ignores_configurations_the_published_table_lacks() -> None:
    """`false_play_rescored.csv` repaired 52 of the 88 evaluations; the other 36 have
    nothing to check against, and that is not a failure."""
    records = [{"model": "dinov2", "describe": "never_published", "false_play": 0.3}]
    sr.check_reproduces(records, {}, "false_play", "x")


# --- the gate, counted rather than predicted -----------------------------------


def test_auto_and_on_differ_exactly_where_the_gate_does_not_fire() -> None:
    """The property the counted measurement rests on.

    Above the threshold `auto` leaves the frame alone and `on` applies CLAHE, so the two
    outputs differ; below it they are identical. Nothing here depends on knowing which
    image the gate measures, which is precisely why the count is trustworthy where the
    derivation was not.
    """
    base = PreprocessConfig()
    auto = dataclasses.replace(base, clahe="auto")
    on = dataclasses.replace(base, clahe="on")
    off = dataclasses.replace(base, clahe="off")

    rng = np.random.default_rng(0)
    flat = np.full((120, 200, 3), 128, np.uint8)          # zero contrast: gate fires
    noisy = rng.integers(0, 256, (120, 200, 3), dtype=np.uint8)  # high contrast: it does not

    assert np.array_equal(preprocess(flat, auto), preprocess(flat, on))
    assert not np.array_equal(preprocess(flat, auto), preprocess(flat, off))

    assert np.array_equal(preprocess(noisy, auto), preprocess(noisy, off))
    assert not np.array_equal(preprocess(noisy, auto), preprocess(noisy, on))


def test_the_gate_reads_the_frame_before_the_resize() -> None:
    """The mistake that produced 99.81%.

    The gate sits in the photometric stage, so it tests the full-resolution frame - not the
    224x224 output whose contrast is a different number. Pinned by constructing a frame
    whose contrast straddles the threshold only after letterboxing: if the gate read the
    output, the two configs would agree here.
    """
    base = dataclasses.replace(PreprocessConfig(), clahe_contrast_below=40.0)
    auto = dataclasses.replace(base, clahe="auto")
    off = dataclasses.replace(base, clahe="off")

    # A wide frame of pure black and white: raw contrast is ~127, far above the threshold,
    # so the gate must not fire whatever the resized version looks like.
    frame = np.zeros((200, 800, 3), np.uint8)
    frame[:, 400:] = 255
    from pitch_occupancy.vision.preprocess import rms_contrast

    assert rms_contrast(frame) > 40.0
    assert np.array_equal(preprocess(frame, auto), preprocess(frame, off)), (
        "the gate fired on a high-contrast frame, so it is not reading the frame it tests"
    )


def test_the_committed_gate_figures_are_counted_not_predicted() -> None:
    """The recorded fire rate must be the counted one.

    If a future change makes the contrast distribution disagree with the count again, the
    count is the number to keep - and the experiment reports both so the disagreement is
    visible rather than silent.
    """
    import json

    path = ROOT / "results" / "search_resolution_gate.json"
    if not path.exists():
        pytest.skip("gate calibration not present")
    gate = json.loads(path.read_text(encoding="utf-8"))
    n, differ = gate["n_frames"], gate["auto_differs_from_on"]
    assert gate["fires_at_current"] == pytest.approx((n - differ) / n)
    assert differ > 3, "the retracted diagnostic claimed 3; the counted answer is 342"
