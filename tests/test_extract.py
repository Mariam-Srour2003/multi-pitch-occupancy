"""Clip frames carry their metadata in a sidecar rather than in the filename, so the
join between the two is the thing that can silently break."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from pitch_occupancy.data.extract import (
    NIGHT_BRIGHTNESS_BELOW,
    extract_clip_frames,
    load_clip_venues,
    read_sidecar,
    write_sidecar,
)
from pitch_occupancy.data.manifest import build_manifest


def make_clip(path: Path, *, value: int, n: int = 60, size=(64, 48)) -> None:
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for _ in range(n):
        vw.write(np.full((h, w, 3), value, np.uint8))
    vw.release()


@pytest.fixture
def clips(tmp_path: Path) -> tuple[Path, dict[str, tuple[str, str]]]:
    d = tmp_path / "clips"
    d.mkdir()
    make_clip(d / "statbox-watermarked-111.mp4", value=200)  # bright -> day
    make_clip(d / "statbox-watermarked-222.mp4", value=20)  # dark   -> night
    venues = {
        "statbox-watermarked-111.mp4": ("clipvenue_a_test", "ca"),
        "statbox-watermarked-222.mp4": ("clipvenue_b_test", "cb"),
    }
    return d, venues


def test_extracts_requested_number_per_clip(clips, tmp_path: Path) -> None:
    d, venues = clips
    rows = extract_clip_frames(d, tmp_path / "out", venues=venues, per_clip=4)
    assert len(rows) == 8
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 8


def test_filename_encodes_venue_clip_and_offset(clips, tmp_path: Path) -> None:
    d, venues = clips
    rows = extract_clip_frames(d, tmp_path / "out", venues=venues, per_clip=2)
    names = sorted(r.file for r in rows)
    assert names[0].startswith("clip_ca_111_t")
    assert all(n.endswith(".jpg") for n in names)
    assert len({r.file for r in rows}) == len(rows), "frame names collide"


def test_lighting_is_measured_not_assumed(clips, tmp_path: Path) -> None:
    """Clips have no timestamp, so day/night can only come from the pixels."""
    d, venues = clips
    rows = extract_clip_frames(d, tmp_path / "out", venues=venues, per_clip=2)
    bright = [r for r in rows if r.venue_code == "ca"]
    dark = [r for r in rows if r.venue_code == "cb"]
    assert all(r.lighting == "day" for r in bright)
    assert all(r.lighting == "night" for r in dark)
    assert all(r.brightness > NIGHT_BRIGHTNESS_BELOW for r in bright)


def test_unassigned_clip_is_refused(clips, tmp_path: Path) -> None:
    """A clip with no venue cannot be grouped, so extraction must not guess."""
    d, venues = clips
    del venues["statbox-watermarked-222.mp4"]
    with pytest.raises(KeyError, match="no venue assignment"):
        extract_clip_frames(d, tmp_path / "out", venues=venues, per_clip=2)


def test_sidecar_round_trips(clips, tmp_path: Path) -> None:
    d, venues = clips
    rows = extract_clip_frames(d, tmp_path / "out", venues=venues, per_clip=3)
    back = read_sidecar(write_sidecar(rows, tmp_path / "sidecar.csv"))
    assert len(back) == len(rows)
    assert back[rows[0].file] == rows[0]


def test_read_sidecar_absent_is_empty_not_an_error(tmp_path: Path) -> None:
    assert read_sidecar(tmp_path / "nope.csv") == {}


def test_load_clip_venues_reads_the_real_config() -> None:
    venues = load_clip_venues()
    assert len(venues) == 66
    assert len({v for v, _ in venues.values()}) == 9


# --- the join into the manifest --------------------------------------------


@pytest.fixture
def dataset_with_clips(clips, tmp_path: Path) -> tuple[Path, Path]:
    d, venues = clips
    ds = tmp_path / "dataset"
    (ds / "2_playing").mkdir(parents=True)
    rows = extract_clip_frames(d, ds / "2_playing", venues=venues, per_clip=3)
    sidecar = write_sidecar(rows, tmp_path / "clip_frames.csv")
    with (ds / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "label", "labeled_at"])
        w.writeheader()
    return ds, sidecar


def test_manifest_indexes_clip_frames(dataset_with_clips) -> None:
    ds, sidecar = dataset_with_clips
    rows, problems = build_manifest(ds, sidecar=sidecar)
    assert len(rows) == 6
    assert problems == []
    assert {r.venue for r in rows} == {"clipvenue_a_test", "clipvenue_b_test"}
    assert all(r.source == "clip" for r in rows)


def test_each_clip_is_its_own_group(dataset_with_clips) -> None:
    """Frames from one 12-second clip are the same scene and must move together."""
    ds, sidecar = dataset_with_clips
    rows, _ = build_manifest(ds, sidecar=sidecar)
    assert len({r.slot_id for r in rows}) == 2
    for r in rows:
        assert r.slot_id.startswith(r.venue)


def test_clip_frames_have_no_invented_date(dataset_with_clips) -> None:
    """Clip filenames encode the export session, not the recording - leave it blank
    rather than fabricating a date that a temporal split would then trust."""
    ds, sidecar = dataset_with_clips
    rows, _ = build_manifest(ds, sidecar=sidecar)
    assert all(r.slot_date == "" and r.slot_time == "" for r in rows)


def test_clip_frame_without_sidecar_entry_is_reported(dataset_with_clips) -> None:
    ds, sidecar = dataset_with_clips
    orphan = ds / "2_playing" / "clip_cz_999_t000000.jpg"
    cv2.imwrite(str(orphan), np.zeros((8, 8, 3), np.uint8))
    rows, problems = build_manifest(ds, sidecar=sidecar)
    assert len(rows) == 6
    assert any("no sidecar entry" in p for p in problems)
