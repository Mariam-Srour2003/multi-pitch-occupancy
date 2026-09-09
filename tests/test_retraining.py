"""Overrides back into training data (WP6-T7).

The dangerous failure here is not a crash. It is a frame reaching the training set with a
label nobody assigned, from the one source the project treats as ground truth. So most of
these tests are about the harvest *not* doing something: not labelling, not overwriting, not
copying without confirmation, and not putting anything where the manifest builder will find
it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.db.schema import connect, initialise
from pitch_occupancy.db.store import Sample, override_verdict, record_slot
from pitch_occupancy.retraining import STAGING_ROOT, apply, plan
from pitch_occupancy.slots.aggregate import aggregate_slot

SLOT = "venue_01_2026-07-11_1000"


@pytest.fixture()
def staged(tmp_path):
    """A database with one overridden slot whose evidence frames exist on disk."""
    import cv2

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    paths = []
    for i in range(3):
        p = dataset / f"evidence_{i}.jpg"
        cv2.imwrite(str(p), np.zeros((8, 8, 3), np.uint8))
        paths.append(str(p))

    conn = connect(":memory:")
    initialise(conn)
    from pitch_occupancy.db.store import ensure_slot

    ensure_slot(conn, slot_id=SLOT, venue_id="venue_01", field_id="field_01",
                cameras=["camera_A"], slot_date="2026-07-11", start_time="10:00",
                end_time="11:00")
    verdict = aggregate_slot([Class3.EMPTY] * 10)
    record_slot(
        conn, SLOT, verdict,
        samples=[Sample("camera_A", 0, Class3.EMPTY.value, 0.9, "2026-07-11T10:00:00", None)],
        evidence_paths=paths, model_key="test",
    )
    override_verdict(conn, SLOT, status=SlotStatus.USED, operator="a.demir",
                     note="the pitch was clearly in use")
    return conn, dataset, paths


# --- the plan is pure ---------------------------------------------------------------------


def test_planning_copies_nothing(staged) -> None:
    conn, dataset, _ = staged
    harvest = plan(conn, dataset_dir=dataset)
    assert len(harvest) == 3
    assert not (dataset / STAGING_ROOT).exists()


def test_apply_is_inert_without_confirmation(staged) -> None:
    """Same shape as `retention.apply`. A default that acts is a default that acts by
    accident."""
    conn, dataset, _ = staged
    harvest = plan(conn, dataset_dir=dataset)
    assert apply(harvest) == 0
    assert not (dataset / STAGING_ROOT).exists()


def test_confirmed_apply_copies_the_frames(staged) -> None:
    conn, dataset, _ = staged
    harvest = plan(conn, dataset_dir=dataset)
    assert apply(harvest, confirm=True) == 3
    staged_files = sorted((dataset / STAGING_ROOT / "USED").glob("*.jpg"))
    assert len(staged_files) == 3


def test_the_source_frames_are_copied_not_moved(staged) -> None:
    """Evidence justifies a billing decision and the audit trail points at it. Moving it to
    feed the model would break the record."""
    conn, dataset, paths = staged
    apply(plan(conn, dataset_dir=dataset), confirm=True)
    assert all(Path(p).exists() for p in paths)


# --- the thing that must not happen ---------------------------------------------------------


def test_frames_are_staged_unfiled_rather_than_under_a_frame_class(staged) -> None:
    """A slot overridden to USED does not mean each evidence frame shows active play - a slot
    is USED at 35% play, so most of its minutes may be empty. Filing them by the slot verdict
    would inject confidently wrong labels."""
    conn, dataset, _ = staged
    apply(plan(conn, dataset_dir=dataset), confirm=True)
    for class_folder in ("1_empty", "2_playing", "3_maintenance"):
        assert not (dataset / class_folder).exists(), (
            f"a frame was filed into {class_folder} on a slot-level verdict"
        )


def test_nothing_staged_is_picked_up_by_the_manifest_builder(staged) -> None:
    """The underscore prefix is load-bearing, so it is tested rather than trusted:
    `build_manifest` globs `[0-9]_*`, and staging lives outside that."""
    from pitch_occupancy.data.manifest import build_manifest

    conn, dataset, _ = staged
    apply(plan(conn, dataset_dir=dataset), confirm=True)
    rows, _ = build_manifest(dataset, venue="venue_01")
    assert rows == []


def test_each_staged_frame_carries_its_provenance(staged) -> None:
    conn, dataset, _ = staged
    apply(plan(conn, dataset_dir=dataset), confirm=True)
    sidecars = sorted((dataset / STAGING_ROOT / "USED").glob("*.json"))
    assert len(sidecars) == 3
    payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert payload["slot_id"] == SLOT
    assert payload["operator"] == "a.demir"
    assert payload["operator_slot_verdict"] == "USED"
    assert payload["model_slot_verdict"] == "NOTUSED"
    assert payload["note"] == "the pitch was clearly in use"


def test_the_sidecar_leaves_the_frame_label_empty_and_says_why(staged) -> None:
    """An empty field a human must fill, not a guess they might not notice."""
    conn, dataset, _ = staged
    apply(plan(conn, dataset_dir=dataset), confirm=True)
    payload = json.loads(
        next((dataset / STAGING_ROOT / "USED").glob("*.json")).read_text(encoding="utf-8")
    )
    assert payload["frame_label"] == ""
    assert "not a label for this frame" in payload["warning"]


# --- idempotence and missing files ------------------------------------------------------------


def test_a_second_run_stages_nothing_new(staged) -> None:
    conn, dataset, _ = staged
    apply(plan(conn, dataset_dir=dataset), confirm=True)
    second = plan(conn, dataset_dir=dataset)
    assert len(second) == 0
    assert len(second.already_staged) == 3


def test_an_existing_destination_is_never_overwritten(staged) -> None:
    """A human may already have annotated the sidecar. A re-run must not undo that."""
    conn, dataset, _ = staged
    harvest = plan(conn, dataset_dir=dataset)
    apply(harvest, confirm=True)
    sidecar = next((dataset / STAGING_ROOT / "USED").glob("*.json"))
    sidecar.write_text('{"frame_label": "C1_EMPTY"}', encoding="utf-8")
    apply(harvest, confirm=True)  # the same plan again, destinations now present
    assert json.loads(sidecar.read_text(encoding="utf-8"))["frame_label"] == "C1_EMPTY"


def test_a_deleted_evidence_frame_is_reported_not_skipped_silently(staged) -> None:
    """Retention deletes frames on a schedule, so this happens in normal operation and the
    operator should see that the label they gave has no image behind it any more."""
    conn, dataset, paths = staged
    Path(paths[0]).unlink()
    harvest = plan(conn, dataset_dir=dataset)
    assert len(harvest) == 2
    assert len(harvest.missing) == 1
    assert SLOT in harvest.missing[0]


def test_slots_that_were_not_overridden_are_left_alone(tmp_path) -> None:
    conn = connect(":memory:")
    initialise(conn)
    from pitch_occupancy.db.store import ensure_slot

    ensure_slot(conn, slot_id=SLOT, venue_id="venue_01", field_id="field_01",
                cameras=["camera_A"], slot_date="2026-07-11", start_time="10:00",
                end_time="11:00")
    record_slot(conn, SLOT, aggregate_slot([Class3.EMPTY] * 5), samples=[], model_key="t")
    assert len(plan(conn, dataset_dir=tmp_path)) == 0


def test_the_description_says_the_frames_are_unfiled(staged) -> None:
    """The command's output is where an operator learns they still have work to do."""
    conn, dataset, _ = staged
    text = plan(conn, dataset_dir=dataset).describe()
    assert "UNFILED" in text
    assert "USED" in text
