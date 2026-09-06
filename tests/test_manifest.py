"""The manifest is what splits, coverage and every experiment read from, so a
regression here silently changes results rather than raising."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pitch_occupancy.data.manifest import (
    ManifestRow,
    build_manifest,
    confound_warnings,
    cross_tab,
    read_manifest,
    write_manifest,
)

FRAMES = {
    "1_empty": [
        "slot_20260711_1000_camA_t000000.jpg",
        "slot_20260711_1000_camB_t000030_m.jpg",
    ],
    "2_playing": [
        "slot_20260712_2030_camA_t000015.jpg",
        "slot_20260712_2030_camB_t000045.jpg",
    ],
    "4_maintenance": ["slot_20260711_1000_camA_t000600.jpg"],
}


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "dataset"
    labelled = []
    for cls, names in FRAMES.items():
        (ds / cls).mkdir(parents=True)
        for n in names:
            (ds / cls / n).write_bytes(b"")
            # deliberately leave the motion frame out of labels.csv -> "bulk"
            if not n.endswith("_m.jpg"):
                labelled.append({"file": f"{cls}/{n}", "label": cls, "labeled_at": "2026-09-01"})
    with (ds / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "label", "labeled_at"])
        w.writeheader()
        w.writerows(labelled)
    return ds


def test_indexes_every_frame(dataset: Path) -> None:
    rows, problems = build_manifest(dataset)
    assert len(rows) == 5
    assert problems == []


def test_parses_filename_metadata(dataset: Path) -> None:
    rows, _ = build_manifest(dataset)
    row = next(r for r in rows if r.file.endswith("camA_t000000.jpg"))
    assert row.slot_date == "2026-07-11"
    assert row.slot_time == "10:00"
    assert row.t_s == 0
    assert row.source == "regular"
    assert row.lighting == "day"
    assert row.camera == "slot_20260711_1000_camA"


def test_motion_frames_are_marked(dataset: Path) -> None:
    rows, _ = build_manifest(dataset)
    motion = [r for r in rows if r.source == "motion"]
    assert len(motion) == 1
    assert motion[0].t_s == 30


def test_evening_slot_is_night(dataset: Path) -> None:
    rows, _ = build_manifest(dataset)
    assert {r.lighting for r in rows if r.slot_time == "20:30"} == {"night"}


def test_both_cameras_of_a_pitch_share_one_slot_id(dataset: Path) -> None:
    """The split grouping key must not let camera A and camera B of the same moment
    land on opposite sides of a train/test boundary - they show the same scene."""
    rows, _ = build_manifest(dataset)
    morning = {r.slot_id for r in rows if r.slot_date == "2026-07-11"}
    assert len(morning) == 1
    cameras = {r.camera for r in rows if r.slot_date == "2026-07-11"}
    assert len(cameras) == 2


def test_frames_absent_from_labels_csv_are_marked_bulk(dataset: Path) -> None:
    rows, _ = build_manifest(dataset)
    by_prov = {r.labeled_by for r in rows if r.source == "motion"}
    assert by_prov == {"bulk"}
    assert all(r.labeled_by == "human" for r in rows if r.source == "regular")


def test_four_classes_collapse_to_three(dataset: Path) -> None:
    rows, _ = build_manifest(dataset)
    maintenance = next(r for r in rows if r.class4 == "4_maintenance")
    assert maintenance.class3 == "C3_MAINTENANCE_NON_SPORTING"


def test_unparseable_filename_is_reported_not_silently_dropped(dataset: Path) -> None:
    (dataset / "1_empty" / "IMG_1234.jpg").write_bytes(b"")
    rows, problems = build_manifest(dataset)
    assert len(rows) == 5
    assert any("IMG_1234" in p for p in problems)


def test_folder_wins_over_labels_csv_but_mismatch_is_reported(dataset: Path) -> None:
    with (dataset / "labels.csv").open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["1_empty/slot_20260711_1000_camA_t000000.jpg", "2_playing", "x"])
    rows, problems = build_manifest(dataset)
    row = next(r for r in rows if r.file.endswith("camA_t000000.jpg"))
    assert row.class4 == "1_empty"  # the folder is authoritative
    assert any("label mismatch" in p for p in problems)


def test_round_trips_through_csv(dataset: Path, tmp_path: Path) -> None:
    rows, _ = build_manifest(dataset)
    path = write_manifest(rows, tmp_path / "manifest.csv")
    assert read_manifest(path) == rows


def test_confound_warning_fires_when_class_sits_in_one_slot(dataset: Path) -> None:
    rows, _ = build_manifest(dataset)
    warnings = confound_warnings(rows)
    assert any("ACTIVE_PLAY" in w and "single slot" in w for w in warnings)


def test_no_confound_warning_when_class_is_spread(tmp_path: Path) -> None:
    ds = tmp_path / "dataset"
    (ds / "2_playing").mkdir(parents=True)
    for slot in ("20260711_1000", "20260712_2030"):
        for sec in range(5):
            (ds / "2_playing" / f"slot_{slot}_camA_t{sec:06d}.jpg").write_bytes(b"")
    rows, _ = build_manifest(ds)
    assert confound_warnings(rows) == []


def test_cross_tab_totals_match_row_count(dataset: Path) -> None:
    rows, _ = build_manifest(dataset)
    table = cross_tab(rows, "class3", "lighting")
    assert table.splitlines()[-1].split()[-1] == str(len(rows))


def test_manifest_row_is_immutable(dataset: Path) -> None:
    """Rows are frozen so a downstream step cannot quietly relabel the dataset
    in memory and have later stages disagree with the CSV on disk."""
    rows, _ = build_manifest(dataset)
    with pytest.raises(AttributeError):
        rows[0].class4 = "tampered"  # type: ignore[misc]
