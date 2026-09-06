"""Database schema and evidence store.

The property that matters: a verdict never exists without the samples that produced it,
because an unsupported verdict is exactly what an operator would dispute."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pitch_occupancy.data.taxonomy import SlotStatus
from pitch_occupancy.db.schema import connect, initialise
from pitch_occupancy.db.store import (
    Sample,
    load_verdict,
    overridden_slots,
    override_verdict,
    record_slot,
)
from pitch_occupancy.slots.aggregate import SlotVerdict


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    initialise(c)
    c.execute("INSERT INTO venues VALUES ('v1', 'Venue One')")
    c.execute("INSERT INTO fields VALUES ('f1', 'v1', 'Pitch 1')")
    c.execute("INSERT INTO cameras VALUES ('camA', 'f1', 'A', NULL, NULL)")
    c.execute("INSERT INTO cameras VALUES ('camB', 'f1', 'B', NULL, NULL)")
    c.execute("INSERT INTO rental_slots VALUES ('s1', 'f1', '2026-07-11', '10:00', '11:00')")
    c.commit()
    return c


def verdict(status: SlotStatus = SlotStatus.USED) -> SlotVerdict:
    return SlotVerdict(status, "test", 0.8, 0.2, 0.0, 60, 0.9)


def samples(n: int = 3) -> list[Sample]:
    return [
        Sample("camA", i, "C2_ACTIVE_PLAY", 0.9, "2026-07-11T10:0%d:00" % i) for i in range(n)
    ]


# --- schema -----------------------------------------------------------------


def test_initialise_is_idempotent(tmp_path: Path) -> None:
    c = connect(tmp_path / "x.db")
    initialise(c)
    initialise(c)  # must not raise
    assert load_verdict(c, "nope") is None


def test_a_field_cannot_have_two_cameras_on_the_same_side(conn) -> None:
    """Exactly two cameras per pitch, one per half - the blueprint correction."""
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO cameras VALUES ('camC', 'f1', 'A', NULL, NULL)")


def test_camera_side_is_constrained(conn) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO cameras VALUES ('camZ', 'f1', 'C', NULL, NULL)")


def test_foreign_keys_are_enforced(conn) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO cameras VALUES ('camX', 'ghost_field', 'A', NULL, NULL)")


def test_slot_status_is_constrained(conn) -> None:
    record_slot(conn, "s1", verdict(), samples(), model_key="m")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE slot_evaluations SET status = 'MAYBE' WHERE slot_id = 's1'")


# --- evidence ---------------------------------------------------------------


def test_verdict_and_samples_are_written_together(conn) -> None:
    record_slot(conn, "s1", verdict(), samples(5), model_key="convnextv2")
    row = load_verdict(conn, "s1")
    assert row["status"] == "USED"
    assert row["model_key"] == "convnextv2"
    n = conn.execute("SELECT COUNT(*) c FROM frame_samples WHERE slot_id='s1'").fetchone()["c"]
    assert n == 5


def test_a_failed_write_leaves_no_partial_evidence(conn) -> None:
    """A verdict with half its samples missing is worse than no verdict."""
    bad = [Sample("ghost_camera", 0, "C1_EMPTY", 0.9, "2026-07-11T10:00:00")]
    with pytest.raises(sqlite3.IntegrityError):
        record_slot(conn, "s1", verdict(), bad, model_key="m")
    assert load_verdict(conn, "s1") is None
    assert conn.execute("SELECT COUNT(*) c FROM frame_samples").fetchone()["c"] == 0


def test_re_evaluating_a_slot_replaces_rather_than_duplicates(conn) -> None:
    record_slot(conn, "s1", verdict(SlotStatus.USED), samples(3), model_key="m")
    record_slot(conn, "s1", verdict(SlotStatus.NOTUSED), samples(3), model_key="m")
    assert load_verdict(conn, "s1")["status"] == "NOTUSED"
    assert conn.execute("SELECT COUNT(*) c FROM frame_samples").fetchone()["c"] == 3


def test_one_sample_per_camera_per_minute(conn) -> None:
    record_slot(conn, "s1", verdict(), samples(3), model_key="m")
    rows = conn.execute(
        "SELECT minute_index FROM frame_samples WHERE camera_id='camA'"
    ).fetchall()
    assert len({r["minute_index"] for r in rows}) == len(rows)


# --- overrides --------------------------------------------------------------


def test_override_keeps_the_original_verdict(conn) -> None:
    """Both are kept: the disagreement is the signal."""
    record_slot(conn, "s1", verdict(SlotStatus.USED), samples(), model_key="m")
    override_verdict(conn, "s1", SlotStatus.NOTUSED, operator="maria", note="rain, no play")
    row = load_verdict(conn, "s1")
    assert row["status"] == "USED"          # what the model said
    assert row["override_status"] == "NOTUSED"  # what the human said
    assert row["is_overridden"] == 1
    assert row["override_by"] == "maria"


def test_overriding_a_missing_slot_raises(conn) -> None:
    with pytest.raises(KeyError, match="no evaluation exists"):
        override_verdict(conn, "ghost", SlotStatus.USED, operator="maria")


def test_overridden_slots_are_listable_as_free_labels(conn) -> None:
    """Corrected slots are the cheapest real slot labels the project can get."""
    record_slot(conn, "s1", verdict(), samples(), model_key="m")
    assert overridden_slots(conn) == []
    override_verdict(conn, "s1", SlotStatus.NOTUSED, operator="maria")
    got = overridden_slots(conn)
    assert len(got) == 1
    assert got[0]["override_status"] == "NOTUSED"


def test_evidence_paths_round_trip(conn) -> None:
    record_slot(
        conn, "s1", verdict(), samples(), model_key="m",
        evidence_paths=["a.jpg", "b.jpg", "c.jpg"],
    )
    import json

    assert json.loads(load_verdict(conn, "s1")["evidence_paths"]) == ["a.jpg", "b.jpg", "c.jpg"]
