"""Augmentation across the camera boundary (WP3-T6).

The experiment's headline is a large gain from one preset, which is exactly the shape of
result that deserves suspicion: augmentation multiplies the training set, and more rows can
flatter a model on their own. So the tests are mostly about the control that separates variety
from row count, and about the second finding - that turning everything on erases the gain -
since that one runs against instinct and will be the first thing an editor "corrects".
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
at = importlib.import_module("experiments.augmentation_transfer")

OUT = ROOT / "results" / "augmentation_transfer.csv"


def published() -> dict[str, dict]:
    if not OUT.exists():
        pytest.skip("augmentation transfer not run")
    return {r["preset"]: r for r in csv.DictReader(OUT.open(encoding="utf-8"))}


# --- the control ------------------------------------------------------------------------


def test_the_duplicate_rows_control_is_run() -> None:
    """Without it, a gain from augmentation cannot be told from a gain from having more
    rows. It is the first entry in ORDER so the table reads control-first."""
    assert at.ORDER[0] == "none"
    assert "none" in published()


def test_the_control_adds_the_same_number_of_rows_as_a_real_preset() -> None:
    """It is only a control if it differs in variety alone."""
    rows = published()
    assert rows["none"]["n_train"] == rows["light"]["n_train"]


def test_the_gain_is_not_explained_by_row_count() -> None:
    """The claim the headline rests on. If duplication alone reproduced it, the finding would
    be about training-set size and not about augmentation."""
    rows = published()
    base = float(rows["baseline"]["macro_f1"])
    assert float(rows["none"]["macro_f1"]) <= base
    assert float(rows["light"]["macro_f1"]) > base + 0.2


# --- the two findings -----------------------------------------------------------------------


def test_light_recovers_empty_recall_from_zero() -> None:
    """The failure that mattered: a probe that never sees the target camera calls none of its
    empty frames empty."""
    rows = published()
    assert float(rows["baseline"]["empty_recall"]) == 0.0
    assert float(rows["light"]["empty_recall"]) > 0.5


def test_turning_everything_on_does_not_beat_the_one_preset_that_works() -> None:
    """`full` contains every effect `light` has. If it ever overtakes it, the finding that
    augmentation must be matched to the shift has changed and the write-up is stale."""
    rows = published()
    assert float(rows["full"]["macro_f1"]) < float(rows["light"]["macro_f1"])


def test_it_does_not_reach_what_one_labelled_frame_reaches() -> None:
    """Augmentation is not a substitute for the five-frame recipe, and the deck says so. The
    comparison is against the onboarding table rather than a number typed here."""
    onboarding = ROOT / "results" / "onboarding_cost.csv"
    if not onboarding.exists():
        pytest.skip("onboarding curve not generated")
    one_frame = next(
        float(r["adapted_mean"]) for r in csv.DictReader(onboarding.open(encoding="utf-8"))
        if r["backbone"] == "dinov2" and r["k"] == "1"
    )
    assert float(published()["light"]["macro_f1"]) < one_frame


# --- the harness ------------------------------------------------------------------------------


def test_every_preset_in_the_order_is_a_real_one() -> None:
    from pitch_occupancy.vision.augment import AUGMENTATIONS

    assert set(at.ORDER) <= set(AUGMENTATIONS)


def test_the_table_is_written_after_every_preset(tmp_path, monkeypatch) -> None:
    """A run that keeps nothing until it finishes is one crash away from having done nothing,
    which is exactly what happened on the first attempt: it died during the fourth preset and
    lost the three already complete."""
    monkeypatch.setattr(at, "OUT", tmp_path / "partial.csv")
    at.save([{"preset": "none", "views": 4, "n_train": 10, "macro_f1": 0.5,
              "empty_recall": 0.1, "play_recall": 0.9}])
    written = list(csv.DictReader((tmp_path / "partial.csv").open(encoding="utf-8")))
    assert written and written[0]["preset"] == "none"


def test_the_stand_in_row_carries_a_label_into_the_probe() -> None:
    """Augmented views are not manifest rows; they only need to answer `class3`."""
    row = at.Row("aug_light_3", "C1_EMPTY")
    assert row.class3 == "C1_EMPTY"


def test_augmented_views_differ_from_each_other() -> None:
    """A preset that silently did nothing would produce identical views and a null result
    that looked like a measurement."""
    from pitch_occupancy.vision.augment import AUGMENTATIONS, augment

    rng = np.random.default_rng(0)
    image = rng.integers(40, 200, (64, 64, 3), dtype=np.uint8)
    views = [augment(image, AUGMENTATIONS["light"], rng=rng) for _ in range(6)]
    assert any(not np.array_equal(views[0], v) for v in views[1:])


def test_the_none_preset_really_changes_nothing() -> None:
    """Otherwise the control is not a control."""
    from pitch_occupancy.vision.augment import AUGMENTATIONS, augment

    rng = np.random.default_rng(1)
    image = rng.integers(40, 200, (32, 32, 3), dtype=np.uint8)
    assert np.array_equal(augment(image, AUGMENTATIONS["none"], rng=rng), image)
