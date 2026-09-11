"""Evidence images: saving them, keeping the right ones, and serving them (WP6-T6).

The selection machinery had worked for weeks and chose three good minutes every time. What it
recorded for each was a literal ``None``: `run_slot` never saved a frame, so the dashboard's
inspector said "no evidence images bound" and was telling the truth. A verdict whose evidence
cannot be seen is the one thing this system must not produce, because a human confirming every
anomaly is what the advisory-only guarantee rests on.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.frame_source import Frame, FrameSource
from pitch_occupancy.worker import run_slot

SLOT = "venue_01_2026-07-11_1000"


class TwoCameraSlot(FrameSource):
    """Twenty minutes; play starts at minute 10 on camera A."""

    def __init__(self, minutes: int = 20) -> None:
        self._minutes = minutes

    def cameras(self) -> list[str]:
        return ["camera_A", "camera_B"]

    @property
    def n_minutes(self) -> int:
        return self._minutes

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        value = 200 if (camera_id == "camera_A" and minute_index >= 10) else 40
        return Frame(camera_id, minute_index,
                     np.full((32, 32, 3), value, np.uint8), f"{camera_id}.mp4")


def classify(image: np.ndarray) -> tuple[Class3, float]:
    if image[0, 0, 0] > 100:
        return Class3.ACTIVE_PLAY, 0.95
    return Class3.EMPTY, 0.70


# --- saving ---------------------------------------------------------------------------------


def test_no_directory_means_no_images_and_no_paths(tmp_path) -> None:
    """Off by default. Writing frames of identifiable people to disk is a caller's decision,
    not something that happens because a function was called."""
    run = run_slot(SLOT, TwoCameraSlot(), classify)
    assert [e.image_path for e in run.evidence] == [None, None, None]
    assert not list(tmp_path.iterdir())


def test_the_selected_evidence_is_written_and_bound(tmp_path) -> None:
    run = run_slot(SLOT, TwoCameraSlot(), classify, evidence_dir=tmp_path)
    bound = [e.image_path for e in run.evidence]
    assert len(bound) == 3
    assert all(p and Path(p).is_file() for p in bound)


def test_only_the_selected_minutes_survive(tmp_path) -> None:
    """Every observed minute is written because which three matter is not knowable until the
    slot has been seen; the rest are removed once it has."""
    run = run_slot(SLOT, TwoCameraSlot(minutes=20), classify, evidence_dir=tmp_path)
    on_disk = sorted((tmp_path / SLOT).glob("*.jpg"))
    assert len(on_disk) == 3
    assert {str(p) for p in on_disk} == {e.image_path for e in run.evidence}


def test_the_saved_frame_is_the_camera_that_won_the_fusion(tmp_path) -> None:
    """Saving an arbitrary camera would hand an operator a picture of an empty half to justify
    a verdict of "there was play on the other one"."""
    import cv2

    run = run_slot(SLOT, TwoCameraSlot(), classify, evidence_dir=tmp_path)
    playing = [e for e in run.evidence if e.predicted is Class3.ACTIVE_PLAY]
    assert playing, "the fixture should produce at least one ACTIVE_PLAY minute"
    for frame in playing:
        assert "camera_A" in frame.image_path
        image = cv2.imread(frame.image_path)
        assert image is not None and image[0, 0, 0] > 100


def test_the_images_decode(tmp_path) -> None:
    """A file of the right name that is not a readable JPEG would pass every check above."""
    import cv2

    run = run_slot(SLOT, TwoCameraSlot(), classify, evidence_dir=tmp_path)
    for frame in run.evidence:
        assert cv2.imread(frame.image_path) is not None


def test_two_slots_do_not_share_a_directory(tmp_path) -> None:
    run_slot(SLOT, TwoCameraSlot(), classify, evidence_dir=tmp_path)
    run_slot("venue_01_2026-07-12_2030", TwoCameraSlot(), classify, evidence_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        SLOT, "venue_01_2026-07-12_2030"
    ]


def test_a_slot_with_no_readable_frames_writes_nothing(tmp_path) -> None:
    class Dead(TwoCameraSlot):
        def read(self, camera_id, minute_index):
            return None

    run = run_slot(SLOT, Dead(), classify, evidence_dir=tmp_path)
    assert run.evidence == []
    assert not list((tmp_path / SLOT).glob("*.jpg"))


def test_an_unwritable_directory_does_not_take_the_slot_down(tmp_path, monkeypatch) -> None:
    """The verdict is still valid without a picture. Losing an hour's classification because
    a disk was full would be far worse than losing the illustration of it."""
    import pitch_occupancy.worker as worker

    monkeypatch.setattr(worker, "_write_evidence", lambda *a, **k: None)
    run = run_slot(SLOT, TwoCameraSlot(), classify, evidence_dir=tmp_path)
    assert run.verdict.status.value == "USED"
    assert [e.image_path for e in run.evidence] == [None, None, None]


# --- serving ---------------------------------------------------------------------------------


@pytest.fixture()
def served(tmp_path, monkeypatch):
    """A database holding one slot whose evidence images are on disk."""
    from pitch_occupancy.api import routes
    from pitch_occupancy.api.app import app
    from pitch_occupancy.config import settings
    from pitch_occupancy.db.schema import connect, initialise
    from pitch_occupancy.db.store import ensure_slot, record_slot

    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(settings, "interim_dir", tmp_path)
    run = run_slot(SLOT, TwoCameraSlot(), classify, evidence_dir=evidence_root)

    # same_thread=False because TestClient serves the app on another thread; sqlite3 binds a
    # connection to its creating thread otherwise.
    conn = connect(":memory:", same_thread=False)
    initialise(conn)
    ensure_slot(conn, slot_id=SLOT, venue_id="venue_01", field_id="field_01",
                cameras=["camera_A", "camera_B"], slot_date="2026-07-11",
                start_time="10:00", end_time="11:00")
    record_slot(conn, SLOT, run.verdict, run.samples, model_key="test",
                evidence_paths=[e.image_path for e in run.evidence if e.image_path])
    app.dependency_overrides[routes.get_conn] = lambda: conn
    yield TestClient(app), run
    app.dependency_overrides.clear()


def test_an_evidence_image_is_served_as_a_jpeg(served) -> None:
    client, _ = served
    response = client.get(f"/api/v1/slots/{SLOT}/evidence/0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


def test_all_three_are_addressable_by_position(served) -> None:
    client, _ = served
    for i in range(3):
        assert client.get(f"/api/v1/slots/{SLOT}/evidence/{i}").status_code == 200


def test_an_index_past_the_end_is_a_404_that_says_how_many_there_are(served) -> None:
    client, _ = served
    response = client.get(f"/api/v1/slots/{SLOT}/evidence/9")
    assert response.status_code == 404
    assert "3 evidence image(s)" in response.json()["detail"]


def test_the_url_carries_no_path_so_traversal_is_not_expressible(served) -> None:
    """The endpoint takes a slot id and an integer. A traversal attempt cannot even be routed,
    which is a stronger property than sanitising a filename."""
    client, _ = served
    for attempt in ("../../etc/passwd", "..%2f..%2fsecret", "0/../../x"):
        assert client.get(f"/api/v1/slots/{SLOT}/evidence/{attempt}").status_code in (404, 422)


def test_a_path_outside_the_permitted_roots_is_refused(served, tmp_path) -> None:
    """The database is not a trust boundary either: a row could have been written by an older
    version, or edited."""
    import json

    from pitch_occupancy.api import routes
    from pitch_occupancy.api.app import app

    client, _ = served
    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"\xff\xd8\xff\xd9")
    conn = app.dependency_overrides[routes.get_conn]()
    conn.execute("UPDATE slot_evaluations SET evidence_paths = ? WHERE slot_id = ?",
                 (json.dumps([str(outside)]), SLOT))
    conn.commit()
    response = client.get(f"/api/v1/slots/{SLOT}/evidence/0")
    assert response.status_code == 404
    assert "outside the permitted directories" in response.json()["detail"]


def test_a_deleted_image_reads_as_gone_not_as_a_blank_frame(served) -> None:
    """Retention removes evidence on a schedule, so this is normal operation. It must not
    return an empty image an operator could mistake for an empty pitch."""
    client, run = served
    Path(run.evidence[0].image_path).unlink()
    response = client.get(f"/api/v1/slots/{SLOT}/evidence/0")
    assert response.status_code == 404
    assert "not on disk" in response.json()["detail"]


def test_the_inspector_renders_images_rather_than_listing_paths(served) -> None:
    """What the dashboard actually shows is the point of all of the above."""
    client, _ = served
    page = client.get("/client").text
    assert "/evidence/${i}" in page or "/evidence/" in page
    assert "<img" in page
    assert "no longer on disk" in page, "a deleted frame must read as gone in the UI too"


# --- the configured evidence directory (2026-09-11) ----------------------------------------


def test_the_configured_evidence_dir_is_a_permitted_root() -> None:
    """`settings.evidence_dir` is where `run_slot` writes, `retention.py` sweeps and
    `pitch info` prints - and it was missing from the API's allowlist.

    A real run with `--evidence-dir data/evidence` produced three images on disk that the
    inspector refused to serve, reporting *"retention may have removed it"* while the files
    sat there. The allowlist was the right shape; it simply did not contain the one place the
    system puts evidence.
    """
    from pitch_occupancy.api.routes import _evidence_roots
    from pitch_occupancy.config import settings

    assert settings.evidence_dir.resolve() in _evidence_roots()


def test_a_path_relative_to_the_project_root_resolves(tmp_path, monkeypatch) -> None:
    """The worker stores what it was given: `--evidence-dir data/evidence` reaches the
    database as `data/evidence/...`. Resolving that against `dataset_dir` produced
    `data/processed/data/evidence/...`, which is nowhere - and the handler then blamed
    retention for a file that existed."""
    import json

    from fastapi.testclient import TestClient

    from pitch_occupancy.api.app import app
    from pitch_occupancy.config import PROJECT_ROOT, settings
    from pitch_occupancy.db.schema import connect, initialise
    from pitch_occupancy.db.store import ensure_slot

    relative = Path("data") / "evidence" / "pytest_slot" / "minute_000_camA.jpg"
    target = PROJECT_ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    cv2.imwrite(str(target), np.full((16, 16, 3), 120, np.uint8))

    conn = connect(settings.db_path)
    initialise(conn)
    try:
        # the schema's foreign keys want the venue, field, camera and rental slot declared
        # first - `ensure_slot` is the function that exists for exactly that, and using it
        # here keeps the test honest about the shape a real row has
        ensure_slot(conn, slot_id="pytest_slot", venue_id="venue_pytest",
                    field_id="field_pytest", cameras=("camA",),
                    slot_date="2026-09-11", start_time="10:00", end_time="11:00")
        conn.execute(
            """INSERT OR REPLACE INTO slot_evaluations
               (slot_id, status, reason, play_ratio, empty_ratio, maintenance_ratio,
                n_samples, mean_confidence, evidence_paths, evaluated_at, model_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("pytest_slot", "USED", "test", 1.0, 0.0, 0.0, 1, 0.9,
             json.dumps([str(relative)]), "2026-09-11T00:00:00+00:00", "dinov2"),
        )
        conn.commit()
        r = TestClient(app).get("/api/v1/slots/pytest_slot/evidence/0")
        assert r.status_code == 200, r.json()
        assert r.headers["content-type"] == "image/jpeg"
    finally:
        conn.execute("DELETE FROM slot_evaluations WHERE slot_id = 'pytest_slot'")
        conn.commit()
        conn.close()
        target.unlink(missing_ok=True)
        with contextlib.suppress(OSError):  # the directory is shared if two tests overlap
            target.parent.rmdir()
