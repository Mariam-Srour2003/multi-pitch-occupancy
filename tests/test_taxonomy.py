"""The labelling -> reporting mapping is load-bearing: every thesis metric is reported
through it, so a silent change here would quietly rewrite the results.

It was four folders onto three classes until 2026-09-21 and is now one-to-one. The test that
used to guard the four-class ablation (`test_c3_is_reachable_from_two_folders`) is replaced
below by the one that matters now: **the retired folder names still read.** Every CSV, backup
and log entry this project wrote before the collapse says `3_people_not_playing` or
`4_maintenance`, and an artefact that needs editing to stay readable is not a record.
"""

from __future__ import annotations

import pytest

from pitch_occupancy.data.taxonomy import (
    CLASS3_ORDER,
    LABEL_ORDER,
    LEGACY_FOLDERS,
    Class3,
    Label,
    to_class3,
)


def test_every_label_maps() -> None:
    for label in LABEL_ORDER:
        assert to_class3(label) in CLASS3_ORDER


def test_the_folders_and_the_classes_are_now_the_same_three() -> None:
    assert to_class3(Label.EMPTY) is Class3.EMPTY
    assert to_class3(Label.PLAYING) is Class3.ACTIVE_PLAY
    assert to_class3(Label.MAINTENANCE_NON_SPORTING) is Class3.MAINTENANCE_NON_SPORTING
    assert len(LABEL_ORDER) == len(CLASS3_ORDER) == 3


def test_accepts_raw_folder_names() -> None:
    assert to_class3("2_playing") is Class3.ACTIVE_PLAY
    assert to_class3("3_maintenance_non_sporting") is Class3.MAINTENANCE_NON_SPORTING


def test_the_retired_folder_names_still_read_and_still_mean_c3() -> None:
    """`3_people_not_playing` and `4_maintenance` mapped to C3 before the collapse and map
    to C3 after it, so no historical row changes meaning - only the folder it names."""
    assert set(LEGACY_FOLDERS) == {"3_people_not_playing", "4_maintenance"}
    for old in LEGACY_FOLDERS:
        assert to_class3(old) is Class3.MAINTENANCE_NON_SPORTING
        assert Label.parse(old) is Label.MAINTENANCE_NON_SPORTING
    # and they are not offered as somewhere to file a new frame
    assert old not in {label.value for label in LABEL_ORDER}


def test_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        to_class3("5_fireworks")
    with pytest.raises(ValueError, match="unknown label"):
        Label.parse("5_fireworks")
