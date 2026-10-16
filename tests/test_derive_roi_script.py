"""scripts/derive_roi.py shares `configs/roi_derived.json` with the worker's
`--derive-roi`, and rewrote it from its own run alone - deleting the production entries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("derive_roi", ROOT / "scripts" / "derive_roi.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
TRIANGLE = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]


def test_entries_this_run_did_not_derive_are_kept(script, tmp_path) -> None:
    out = tmp_path / "roi_derived.json"
    out.write_text(json.dumps({"venue_02/camera_A": SQUARE, "old_cam": SQUARE}),
                   encoding="utf-8")
    script.write_merged(out, {"old_cam": TRIANGLE, "new_cam": TRIANGLE})
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert stored["venue_02/camera_A"] == SQUARE
    assert stored["old_cam"] == TRIANGLE and stored["new_cam"] == TRIANGLE
    assert not list(tmp_path.glob("*.new"))


def test_an_unreadable_store_is_refused_not_overwritten(script, tmp_path) -> None:
    out = tmp_path / "roi_derived.json"
    out.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        script.write_merged(out, {"cam": TRIANGLE})
    assert out.read_text(encoding="utf-8") == "{not json"
