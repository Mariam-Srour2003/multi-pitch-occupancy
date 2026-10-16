"""scripts/ingest_davinci.py writes the only record of where each DaVinci frame came from.

`provenance.csv` lives in `data/raw/`, which has no backup. The script opened it with "w"
before reading its first row, so a run that found no clips (any machine but the original,
whose Downloads folders it looked in) emptied it and then crashed.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ingest():
    spec = importlib.util.spec_from_file_location(
        "ingest_davinci", ROOT / "scripts" / "ingest_davinci.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(name: str, clip: str = "a.mp4") -> dict:
    return {"file": name, "venue": "venue_01", "venue_code": "dv01", "clip_id": "1",
            "t_ms": 0, "brightness": 1.0, "lighting": "day", "source_clip": clip}


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_no_rows_leaves_both_files_untouched(ingest, tmp_path) -> None:
    sidecar, provenance = tmp_path / "clip_frames.csv", tmp_path / "provenance.csv"
    ingest.write_outputs([row("x.jpg")], sidecar=sidecar, provenance=provenance)
    before = provenance.read_bytes(), sidecar.read_bytes()
    ingest.write_outputs([], sidecar=sidecar, provenance=provenance)
    assert (provenance.read_bytes(), sidecar.read_bytes()) == before


def test_a_re_run_merges_rather_than_replacing_earlier_runs(ingest, tmp_path) -> None:
    sidecar, provenance = tmp_path / "clip_frames.csv", tmp_path / "provenance.csv"
    ingest.write_outputs([row("x.jpg"), row("y.jpg")], sidecar=sidecar, provenance=provenance)
    ingest.write_outputs([row("y.jpg", clip="b.mp4"), row("z.jpg")],
                         sidecar=sidecar, provenance=provenance)
    kept = {r["file"]: r["source_clip"] for r in read(provenance)}
    assert kept == {"x.jpg": "a.mp4", "y.jpg": "b.mp4", "z.jpg": "a.mp4"}
    assert [r["file"] for r in read(sidecar)] == ["x.jpg", "y.jpg", "z.jpg"]
    assert not list(tmp_path.glob("*.new"))


def test_the_ingested_copies_in_raw_count_as_a_source(ingest) -> None:
    clip = next(iter(ingest.RAW.glob("*.mp4")), None)
    if clip is None:
        pytest.skip("no ingested DaVinci clips on this machine")
    assert ingest.find_source(clip.name) is not None
