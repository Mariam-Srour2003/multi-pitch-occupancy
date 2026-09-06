"""The 4-class -> 3-class mapping is load-bearing: every thesis metric is reported
through it, so a silent change here would quietly rewrite the results."""

from __future__ import annotations

import pytest

from pitch_occupancy.data.taxonomy import (
    CLASS3_ORDER,
    CLASS4_ORDER,
    Class3,
    Class4,
    to_class3,
)


def test_every_label_maps() -> None:
    for label in CLASS4_ORDER:
        assert to_class3(label) in CLASS3_ORDER


def test_mapping_is_the_documented_one() -> None:
    assert to_class3(Class4.EMPTY) is Class3.EMPTY
    assert to_class3(Class4.PLAYING) is Class3.ACTIVE_PLAY
    # both non-sporting occupancy folders collapse into C3
    assert to_class3(Class4.PEOPLE_NOT_PLAYING) is Class3.MAINTENANCE_NON_SPORTING
    assert to_class3(Class4.MAINTENANCE) is Class3.MAINTENANCE_NON_SPORTING


def test_accepts_raw_folder_names() -> None:
    assert to_class3("2_playing") is Class3.ACTIVE_PLAY


def test_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        to_class3("5_fireworks")


def test_c3_is_reachable_from_two_folders() -> None:
    """If this ever becomes one-to-one, the 4-class ablation has been lost."""
    sources = [c for c in CLASS4_ORDER if to_class3(c) is Class3.MAINTENANCE_NON_SPORTING]
    assert len(sources) == 2
