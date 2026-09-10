"""The recorded slots through the model, beside the same slots through the labels.

The artefact exists because M5's "end-to-end run on real slots" was satisfied by a table
whose per-minute sequences came from the **label column** — true of the pipeline, and true
of no model. These tests are about the two places the comparison could quietly flatter
itself: the slot-id translation, and what happens to a minute whose own labels disagree.
"""

from __future__ import annotations

import csv
import importlib
import sys
from datetime import date, time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
e2e = importlib.import_module("experiments.end_to_end_model")

from pitch_occupancy.data.taxonomy import Class3  # noqa: E402

OUT = ROOT / "results" / "end_to_end_model_slots.csv"


class Row:
    def __init__(self, t_s: int, class3: str, camera: str) -> None:
        self.t_s, self.class3, self.camera = t_s, class3, camera


# --- the two key conventions ------------------------------------------------------------


def test_the_slot_id_translation_is_the_inverse_of_the_scheduler_s() -> None:
    """Recordings are keyed `slot_YYYYMMDD_HHMM` and the manifest keys the same slot
    `<venue>_YYYY-MM-DD_HHMM`. Two conventions for one thing is how a comparison ends up
    silently comparing a slot with nothing, so both directions are named functions and this
    holds them to each other."""
    from pitch_occupancy.scheduler import ScheduledSlot, _recording_key

    slot = ScheduledSlot("venue_01", time(10, 0), 60, ("camA", "camB"))
    day = date(2026, 7, 11)
    assert e2e.manifest_slot_id(_recording_key(slot, day)) == slot.slot_id(day)


def test_a_minute_whose_labels_disagree_is_not_scored() -> None:
    """It cannot make a single prediction right or wrong. Folding it in either direction
    would move the headline agreement without measuring anything."""
    fused = {0: Class3.EMPTY, 1: Class3.ACTIVE_PLAY, 2: Class3.EMPTY}
    truth = {0: {"C1_EMPTY"}, 1: {"C1_EMPTY"}, 2: {"C1_EMPTY", "C2_ACTIVE_PLAY"}}
    compared, agreeing, ambiguous = e2e.compare(fused, truth)
    assert (compared, agreeing, ambiguous) == (2, 1, 1)


def test_a_minute_the_labels_do_not_cover_is_skipped_not_counted_wrong() -> None:
    """The video runs to 60 minutes and the labels cover 59. Counting the extra minute as a
    disagreement would penalise the model for a minute nobody labelled."""
    compared, agreeing, ambiguous = e2e.compare({7: Class3.EMPTY}, {})
    assert (compared, agreeing, ambiguous) == (0, 0, 0)


def test_the_label_sequence_fuses_the_two_cameras() -> None:
    """The same construction `end_to_end_slots.py` uses, so the two artefacts describe the
    same thing. Strongest activity wins, so a minute where one camera saw play is play."""
    rows = [Row(0, "C1_EMPTY", "camA"), Row(30, "C2_ACTIVE_PLAY", "camB"),
            Row(70, "C1_EMPTY", "camA")]
    assert e2e.label_sequence(rows) == [Class3.ACTIVE_PLAY, Class3.EMPTY]


def test_labels_are_grouped_by_minute_across_both_cameras() -> None:
    rows = [Row(0, "C1_EMPTY", "camA"), Row(30, "C2_ACTIVE_PLAY", "camB")]
    assert e2e.labels_by_minute(rows) == {0: {"C1_EMPTY", "C2_ACTIVE_PLAY"}}


# --- the artefact, when it has been generated --------------------------------------------


def published() -> list[dict]:
    if not OUT.exists():
        pytest.skip("end-to-end model run not generated")
    return list(csv.DictReader(OUT.open(encoding="utf-8")))


def test_the_model_verdicts_match_the_label_derived_ones() -> None:
    """The acceptance criterion WP6-T2 was written around, with a model in the loop. If this
    fails, either the classifier or the decision layer has changed and the write-up is
    stale — it is not a threshold to be relaxed."""
    rows = published()
    assert rows, "the run produced no slots"
    for row in rows:
        assert row["verdict_model"] == row["verdict_labels"], row["slot"]


def test_the_comparison_is_declared_in_sample_where_anyone_would_read_it() -> None:
    """Both slots' frames are in the probe's training set. Every place this number is
    quoted has to say so, and the module docstring is the first of them — an agreement
    figure that travels without that caveat becomes an accuracy claim in one retelling."""
    assert "In-sample" in e2e.__doc__
    assert "accuracy claim" in e2e.__doc__


def test_ambiguous_minutes_are_reported_rather_than_hidden() -> None:
    """The count is in the table, so a reader can see how much of the slot the agreement
    rate actually covers."""
    assert all("minutes_ambiguous" in row for row in published())
