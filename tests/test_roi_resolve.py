"""Resolving the camera ids production actually uses (A36).

`configs/cameras.json` calls a camera `camera_A` under `venue_01`; the worker replaying a
recording calls it `file0`; the derived store calls it `slot_20260711_1000_camA`. `roi.get`
knew only the last, so in production the boundary machinery applied to nothing. These pin
`roi.resolve`, the `_aliases` block, the per-recording `file0 -> camA` mapping, and the store
hygiene the fix needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pitch_occupancy.vision import roi

MIDDLE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
OTHER = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    """Both stores isolated, as `test_roi.py` does: the real ones are committed config."""
    monkeypatch.setattr(roi, "STORE", tmp_path / "roi.json")
    monkeypatch.setattr(roi, "DERIVED_STORE", tmp_path / "roi_derived.json")


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_get_does_not_know_production_ids_and_resolve_does() -> None:
    """`_read` skipped every underscored key, so an `_aliases` block was always legal in
    the file and never read. `resolve` reads it, one hop."""
    _write(roi.DERIVED_STORE, {"slot_20260711_1000_camA": MIDDLE})
    _write(roi.STORE, {"_aliases": {"venue_01/camera_A": "slot_20260711_1000_camA"}})
    assert roi.get("camera_A") is None
    assert roi.get("venue_01/camera_A") is None
    assert roi.resolve("camera_A", venue="venue_01") == MIDDLE
    assert roi.resolve_with_key("camera_A", venue="venue_01") == (
        MIDDLE, "slot_20260711_1000_camA")
    assert roi.resolve("camera_A") is None, "the bare id has no alias in this store"
    assert roi.resolve_with_key("nobody", venue="venue_01") == (None, None)


def test_resolve_maps_the_exports_file_keys_onto_the_recordings_cameras() -> None:
    """`file0` is `camA` and `file1` is `camB` within one recording - measured by IoU on all
    four recordings (`roi.FILE_KEY_CAMERA`) - and only within one, because the physical
    camera behind each flips between days. So the mapping needs the recording key."""
    _write(roi.DERIVED_STORE, {"slot_20260711_1000_camA": MIDDLE,
                               "slot_20260711_1000_camB": OTHER})
    assert roi.resolve("file0", slot_key="slot_20260711_1000") == MIDDLE
    assert roi.resolve("file1", slot_key="slot_20260711_1000") == OTHER
    assert roi.resolve("file0") is None
    assert roi.resolve("file0", slot_key="slot_20260712_2030") is None


def test_resolve_prefers_the_most_specific_key() -> None:
    _write(roi.STORE, {"slot_x/file0": MIDDLE, "file0": OTHER})
    assert roi.resolve("file0", slot_key="slot_x") == MIDDLE
    assert roi.resolve("file0") == OTHER


def test_malformed_aliases_are_dropped_not_raised() -> None:
    """Read on the live path, like the boundaries themselves."""
    _write(roi.STORE, {"_aliases": ["not", "a", "dict"], "cam": MIDDLE})
    assert roi.load_aliases() == {}
    _write(roi.STORE, {"_aliases": {"a": 3, "b": "", "c": "cam"}, "cam": MIDDLE})
    assert roi.load_aliases() == {"c": "cam"}
    assert roi.resolve("c") == MIDDLE


def test_saving_a_hand_drawn_boundary_keeps_the_aliases_and_the_stores_apart() -> None:
    """Until 2026-09-19 `save` merged the derived store into the hand-drawn one - the first
    outline drawn in the editor would have copied 99 derived boundaries into `roi.json` and
    lost the provenance split. And `_write` kept no underscored key but its own comment,
    which would have deleted the aliases on the first save."""
    _write(roi.STORE, {"_aliases": {"camera_A": "x"}})
    _write(roi.DERIVED_STORE, {"d1": MIDDLE, "d2": OTHER})
    roi.save("camera_Z", MIDDLE)
    raw = json.loads(roi.STORE.read_text(encoding="utf-8"))
    assert raw["_aliases"] == {"camera_A": "x"}
    assert set(roi._read(roi.STORE)) == {"camera_Z"}
    assert set(roi._read(roi.DERIVED_STORE)) == {"d1", "d2"}
    assert set(roi.load_all()) == {"camera_Z", "d1", "d2"}


def test_save_derived_writes_the_derived_store_and_a_hand_drawn_one_still_wins() -> None:
    roi.save_derived("cam", OTHER)
    assert roi.get("cam") == OTHER
    assert set(roi._read(roi.STORE)) == set()
    roi.save("cam", MIDDLE)
    assert roi.get("cam") == MIDDLE, "a person's correction wins over a measurement"
    assert roi.remove("cam")
    assert roi.get("cam") == OTHER, "forgetting the correction reveals the measurement"
    assert not roi.remove("cam"), "there is no hand-drawn one left to forget"


def test_the_real_stores_resolve_every_production_camera_id(monkeypatch) -> None:
    """The ids the worker and the camera config actually use, against the committed stores.
    This is the regression for the defect A36 closes: every one of these was None."""
    configs = Path(__file__).resolve().parents[1] / "configs"
    monkeypatch.setattr(roi, "STORE", configs / "roi.json")
    monkeypatch.setattr(roi, "DERIVED_STORE", configs / "roi_derived.json")
    for camera in ("camera_A", "camera_B"):
        assert roi.resolve(camera, venue="venue_01") is not None, camera
        assert roi.resolve(camera) is not None, camera
    for slot_key in ("slot_20260711_1000", "slot_20260712_2030"):
        for file_key in ("file0", "file1"):
            assert roi.resolve(file_key, slot_key=slot_key) is not None, (slot_key, file_key)
    assert roi.resolve("file0") is None, "without a recording, file0 names nothing"


def test_derive_from_frames_finds_a_green_pitch_and_nothing_in_the_dark() -> None:
    """The routine that measures a boundary, on a frame it must find and one it must not."""
    import numpy as np

    frame = np.zeros((180, 320, 3), np.uint8)
    frame[40:140, 60:260] = (40, 180, 40)  # a green rectangle on black, BGR
    polygon = roi.derive_from_frames([frame, frame, frame])
    assert polygon is not None
    assert 0.25 < roi.coverage(polygon) < 0.45, roi.coverage(polygon)
    assert roi.derive_from_frames([]) is None
