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
    "3_maintenance_non_sporting": ["slot_20260711_1000_camA_t000600.jpg"],
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


def test_the_folders_are_the_reporting_classes(dataset: Path) -> None:
    rows, _ = build_manifest(dataset)
    maintenance = next(r for r in rows if r.label == "3_maintenance_non_sporting")
    assert maintenance.class3 == "C3_MAINTENANCE_NON_SPORTING"


def test_a_manifest_written_before_the_collapse_still_reads(tmp_path: Path) -> None:
    """Every manifest and backup older than 2026-09-21 has a `class4` column holding one of
    four folder names. They are read as labels - both C3 folders becoming the one that
    exists now - and the file on disk is not touched."""
    old = tmp_path / "manifest_2026-09-01.csv"
    tail = ",v,c,,,s,0,clip,human,day,unknown,"
    old.write_text("".join(line + chr(10) for line in [
        ("file,class4,class3,venue,camera,slot_date,slot_time,slot_id,t_s,source,"
         "labeled_by,lighting,quality,split_role"),
        "4_maintenance/a.jpg,4_maintenance,C3_MAINTENANCE_NON_SPORTING" + tail,
        ("3_people_not_playing/b.jpg,3_people_not_playing,"
         "C3_MAINTENANCE_NON_SPORTING" + tail),
        "1_empty/c.jpg,1_empty,C1_EMPTY" + tail,
    ]), encoding="utf-8")
    rows = read_manifest(old)
    assert [r.label for r in rows] == ["3_maintenance_non_sporting",
                                       "3_maintenance_non_sporting", "1_empty"]
    assert [r.class3 for r in rows] == ["C3_MAINTENANCE_NON_SPORTING",
                                        "C3_MAINTENANCE_NON_SPORTING", "C1_EMPTY"]
    assert "class4" in old.read_text(encoding="utf-8"), "the artefact is read, never rewritten"


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
    assert row.label == "1_empty"  # the folder is authoritative
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
        rows[0].label = "tampered"  # type: ignore[misc]

def test_a_rebuild_carries_rows_it_cannot_name_instead_of_destroying_them(
        dataset: Path, tmp_path: Path) -> None:
    """Regenerating the manifest used to delete every generated frame from it.

    `ingest_synthetic.py` appends rows for frames named `syn_<batch>_<n>.jpg`, which no
    pattern in this module matches. A rebuild reported each as "unparseable filename" and
    wrote a manifest without them - **200 rows, silently, with a zero exit code and a
    success line printed.** It happened on 2026-09-21 during the folder collapse, and a
    comment in `scripts/assign_scene_ids.py` had warned for weeks that it would. This is
    that warning as a guard.
    """
    generated = dataset / "1_empty" / "syn_batch7_001.jpg"
    generated.write_bytes(b"")

    rows, problems = build_manifest(dataset)
    assert generated.name not in {Path(r.file).name for r in rows}
    assert any("unparseable" in p for p in problems), "still reported, never silent"

    previous = tmp_path / "manifest.csv"
    write_manifest(list(rows) + [ManifestRow(
        file="1_empty/syn_batch7_001.jpg", label="1_empty", class3="C1_EMPTY", venue="v",
        camera="synthetic_batch7", slot_date="", slot_time="", slot_id="synthetic_batch7",
        t_s=1, source="synthetic", labeled_by="synthetic", lighting="unknown",
        quality="synthetic:ok", split_role="train")], previous)

    rebuilt, problems = build_manifest(dataset, carry_unparseable=previous)
    carried = next(r for r in rebuilt if r.file.endswith("syn_batch7_001.jpg"))
    assert carried.source == "synthetic" and carried.quality == "synthetic:ok"
    assert any("carried over" in p for p in problems), "carried, and said so"


def test_carrying_only_rescues_frames_that_are_still_on_disk(
        dataset: Path, tmp_path: Path) -> None:
    """A row for a deleted frame must not come back from the old manifest."""
    previous = tmp_path / "manifest.csv"
    rows, _ = build_manifest(dataset)
    write_manifest(list(rows) + [ManifestRow(
        file="1_empty/syn_gone_001.jpg", label="1_empty", class3="C1_EMPTY", venue="v",
        camera="synthetic_gone", slot_date="", slot_time="", slot_id="synthetic_gone",
        t_s=1, source="synthetic", labeled_by="synthetic", lighting="unknown",
        quality="synthetic:ok", split_role="train")], previous)
    rebuilt, _ = build_manifest(dataset, carry_unparseable=previous)
    assert not any("syn_gone" in r.file for r in rebuilt)
