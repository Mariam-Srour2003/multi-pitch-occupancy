"""Baseline floors and the linear probe.

The floors exist to be beaten. If they are not, the benchmark is measuring the dataset
rather than the models — which is the finding H2 is pre-registered to test."""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.data.manifest import ManifestRow
from pitch_occupancy.vision.cheap_features import CHEAP_FEATURES, colour_histogram, mean_intensity
from pitch_occupancy.vision.heads import ClockRule, LinearProbe, MajorityClass, make_predictor

EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"


def row(cls: str, lighting: str, i: int = 0) -> ManifestRow:
    return ManifestRow(
        file=f"{cls}/{lighting}{i}.jpg", class4="1_empty", class3=cls, venue="v1",
        camera="c", slot_date="2026-07-11", slot_time="10:00", slot_id="s1", t_s=i,
        source="regular", labeled_by="human", lighting=lighting, quality="unknown",
        split_role="",
    )


@pytest.fixture
def confounded() -> list[ManifestRow]:
    """The shape of the real dataset: empty by day, play by night."""
    return (
        [row(EMPTY, "day", i) for i in range(40)]
        + [row(PLAY, "night", i) for i in range(40)]
        + [row(EMPTY, "night", 99)]
    )


# --- floors -----------------------------------------------------------------


def test_majority_predicts_the_commonest_training_label(confounded) -> None:
    m = MajorityClass().fit(np.zeros((len(confounded), 1)), confounded)
    preds = m.predict(np.zeros((3, 1)), confounded[:3])
    assert set(preds) == {EMPTY}  # 41 empty vs 40 play


def test_clock_rule_ignores_the_pixels_entirely(confounded) -> None:
    """It must give identical answers on random noise and on real features."""
    r = ClockRule().fit(np.zeros((len(confounded), 5)), confounded)
    a = r.predict(np.random.RandomState(0).rand(len(confounded), 5), confounded)
    b = r.predict(np.zeros((len(confounded), 5)), confounded)
    assert a == b


def test_clock_rule_recovers_the_confound(confounded) -> None:
    """On confounded data it should score near-perfectly using no image content."""
    r = ClockRule().fit(np.zeros((len(confounded), 1)), confounded)
    preds = r.predict(np.zeros((len(confounded), 1)), confounded)
    acc = np.mean([p == x.class3 for p, x in zip(preds, confounded, strict=True)])
    assert acc > 0.95


def test_clock_rule_falls_back_for_unseen_lighting(confounded) -> None:
    r = ClockRule().fit(np.zeros((len(confounded), 1)), confounded)
    unseen = [row(PLAY, "fog", 1)]
    assert r.predict(np.zeros((1, 1)), unseen) == [EMPTY]


# --- probe ------------------------------------------------------------------


def test_probe_learns_a_separable_signal() -> None:
    rows = [row(EMPTY, "day", i) for i in range(30)] + [row(PLAY, "night", i) for i in range(30)]
    X = np.vstack([np.zeros((30, 4)), np.ones((30, 4))])
    p = LinearProbe("test").fit(X, rows)
    assert p.predict(X, rows) == [r.class3 for r in rows]


def test_probe_raises_before_being_fitted() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        LinearProbe("x").predict(np.zeros((1, 2)), [row(EMPTY, "day")])


def test_balanced_weighting_lets_a_rare_class_be_predicted() -> None:
    """C3 has 6 frames in the whole dataset; unweighted, it is never predicted."""
    rows = [row(EMPTY, "day", i) for i in range(100)] + [row(PLAY, "day", i) for i in range(3)]
    X = np.vstack([np.zeros((100, 2)), np.ones((3, 2))])
    balanced = LinearProbe("b", balanced=True).fit(X, rows)
    assert PLAY in balanced.predict(np.ones((3, 2)), rows[-3:])


def test_probe_is_deterministic() -> None:
    rows = [row(EMPTY, "day", i) for i in range(20)] + [row(PLAY, "night", i) for i in range(20)]
    X = np.random.RandomState(0).rand(40, 6)
    a = LinearProbe("a", seed=1).fit(X, rows).predict(X, rows)
    b = LinearProbe("b", seed=1).fit(X, rows).predict(X, rows)
    assert a == b


def test_make_predictor_dispatches() -> None:
    assert isinstance(make_predictor("majority"), MajorityClass)
    assert isinstance(make_predictor("clock_rule"), ClockRule)
    assert isinstance(make_predictor("dinov2"), LinearProbe)


# --- cheap features ---------------------------------------------------------


def test_mean_intensity_is_one_number() -> None:
    img = np.full((16, 16, 3), 128, np.uint8)
    v = mean_intensity(img)
    assert v.shape == (1,)
    assert v[0] == pytest.approx(128, abs=1)


def test_mean_intensity_separates_day_from_night() -> None:
    assert mean_intensity(np.full((8, 8, 3), 200, np.uint8))[0] > mean_intensity(
        np.full((8, 8, 3), 20, np.uint8)
    )[0]


def test_colour_histogram_shape_and_normalisation() -> None:
    h = colour_histogram(np.full((8, 8, 3), 100, np.uint8), bins=16)
    assert h.shape == (48,)
    assert h.sum() == pytest.approx(3.0)  # one normalised histogram per channel


def test_colour_histogram_distinguishes_colours() -> None:
    red = np.zeros((8, 8, 3), np.uint8); red[:, :, 2] = 255
    green = np.zeros((8, 8, 3), np.uint8); green[:, :, 1] = 255
    assert not np.allclose(colour_histogram(red), colour_histogram(green))


def test_every_cheap_feature_is_callable_and_documented() -> None:
    img = np.full((8, 8, 3), 90, np.uint8)
    for name, (fn, doc) in CHEAP_FEATURES.items():
        assert fn(img).ndim == 1, name
        assert doc
