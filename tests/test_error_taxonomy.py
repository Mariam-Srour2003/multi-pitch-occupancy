"""WP4-T6's error taxonomy.

The table's value is not the counts - it is the columns that say whether the counts mean
anything. Nine errors from one slot is a single failure counted nine times, and a taxonomy
that reported only "9 errors, all C1→C2" would read as a pattern. So what is tested is the
concentration measure, the derived finding sentence, and the near-duplicate column, which is
a check on the split rather than on the model.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from pitch_occupancy.data.manifest import ManifestRow

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
et = importlib.import_module("experiments.error_taxonomy")


def err(**kw) -> dict:
    base = {"pair": "C1->C2", "slot_id": "s1", "camera": "camera_A", "lighting": "night",
            "near_duplicate_in_train": ""}
    return {**base, **kw}


# --- concentration: does the error set have anything to categorise? --------------


def test_concentration_reports_the_spread_and_the_commonest_share() -> None:
    errs = [err(slot_id="s1"), err(slot_id="s1"), err(slot_id="s2")]
    assert et.concentration(errs, "slot_id") == "2 (s1 67%)"


def test_a_single_slot_is_visible_as_a_single_slot() -> None:
    """The finding the grouped split produced: nine errors, one scene."""
    errs = [err() for _ in range(9)]
    assert et.concentration(errs, "slot_id").startswith("1 (")
    assert "100%" in et.concentration(errs, "slot_id")


def test_an_empty_error_set_does_not_divide_by_zero() -> None:
    assert et.concentration([], "slot_id") == "—"


# --- the derived finding sentence ------------------------------------------------


def test_the_finding_sentence_is_derived_not_written() -> None:
    """A sentence typed by hand beside a generated table is the arrangement that let a
    figure contradict its own source."""
    errs = [err() for _ in range(9)]
    got = et.finding("grouped_slot", "vit", errs, 907)
    assert "9 errors of 907" in got
    assert "all C1->C2" in got
    assert "from 1 slot" in got
    assert "all night" in got


def test_a_mixed_error_set_reports_the_dominant_pair_as_a_share() -> None:
    errs = [err(pair="C1->C2") for _ in range(8)] + [err(pair="C2->C3") for _ in range(2)]
    got = et.finding("grouped_slot", "dinov2", errs, 907)
    assert "80% of them across 2 pairs" in got
    assert "all C1->C2" not in got


def test_no_errors_says_so_rather_than_producing_an_empty_sentence() -> None:
    assert et.finding("random", "vit", [], 394) == "no errors on 394 test frames"


def test_mixed_lighting_is_not_claimed_as_a_condition() -> None:
    """"all night" is a real pattern; asserting it when half the errors are daylight is
    the kind of sentence a hand-written taxonomy produces."""
    errs = [err(lighting="night"), err(lighting="day")]
    assert "all night" not in et.finding("lo_venue_out", "vit", errs, 30)


def test_the_near_duplicate_share_appears_only_when_it_was_measured() -> None:
    plain = et.finding("lo_venue_out", "vit", [err() for _ in range(4)], 30)
    assert "near-duplicate" not in plain
    measured = et.finding("random", "vit", [err(near_duplicate_in_train=True)] * 4, 394)
    assert "100% had a near-duplicate in training" in measured


# --- the near-duplicate column ----------------------------------------------------


def test_a_frame_matching_a_training_hash_is_flagged() -> None:
    cache = {"a.jpg": 0b1010, "t.jpg": 0b1010}
    assert et.has_near_duplicate("a.jpg", [cache["t.jpg"]], cache) is True


def test_a_distant_frame_is_not_flagged() -> None:
    cache = {"a.jpg": 0, "t.jpg": (1 << 40) - 1}   # 40 bits apart, far past the threshold
    assert et.has_near_duplicate("a.jpg", [cache["t.jpg"]], cache) is False


def test_an_unreadable_frame_is_unknown_rather_than_false() -> None:
    """`False` would read as "checked, and it had none"; the frame was never hashed."""
    assert et.has_near_duplicate("missing.jpg", [0], {}) is None


def test_physical_camera_resolves_the_mislabelled_tag() -> None:
    row = ManifestRow(
        file="f.jpg", class4="1_empty", class3="C1_EMPTY", venue="venue_01",
        camera="slot_20260712_2030_camB", slot_date="2026-07-12", slot_time="20:30",
        slot_id="s", t_s=0, source="regular", labeled_by="human", lighting="night",
        quality="unknown", split_role="",
    )
    assert et.physical(row) == "camera_A"


# --- the committed result ---------------------------------------------------------


def test_the_honest_split_has_no_error_with_a_training_near_duplicate() -> None:
    """The column is a check on the protocol, so it is checked.

    On the grouped split a near-duplicate across the boundary would be a leak the split
    exists to prevent - a non-zero count here is a defect in the split, not a property of
    the model.
    """
    import csv

    path = ROOT / "results" / "error_taxonomy.csv"
    if not path.exists():
        pytest.skip("taxonomy not generated")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    grouped = [r for r in rows if r["protocol"] == "grouped_slot"
               and r["near_duplicate_in_train"] != ""]
    assert grouped, "the near-duplicate column was never computed on the grouped split"
    leaked = [r for r in grouped if r["near_duplicate_in_train"] == "True"]
    assert not leaked, f"{len(leaked)} grouped-split errors have a near-duplicate in training"


def test_the_leaky_split_is_where_the_near_duplicates_are() -> None:
    """The complement, and the reason the column is worth having: on the random split every
    error is a frame the model had seen a copy of."""
    import csv

    path = ROOT / "results" / "error_taxonomy.csv"
    if not path.exists():
        pytest.skip("taxonomy not generated")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    leaky = [r for r in rows if r["protocol"] == "random"
             and r["near_duplicate_in_train"] != ""]
    assert leaky, "the near-duplicate column was never computed on the random split"
    share = sum(r["near_duplicate_in_train"] == "True" for r in leaky) / len(leaky)
    assert share > 0.5, f"only {share:.0%} of leaky-split errors had a near-duplicate"
