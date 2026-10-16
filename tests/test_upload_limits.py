"""The bounds the upload routes share: `api/limits.py`, and the routes that use it.

Each property here was once missing and was measured missing:

* model-heavy work ran on the event loop, so one clip analysis stalled every other request;
* a streamed response deleted its upload only when the garbage collector got round to the
  generator a disconnected client left suspended;
* malformed input reached code that raised AttributeError or OverflowError - a 500 - and a
  malformed outline was noticed only after the model had loaded.

No test here touches the real stores, the real model, or `data/`.
"""

from __future__ import annotations

import base64
import os
import threading
import time

import anyio
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from pitch_occupancy.api import limits
from pitch_occupancy.api.app import app
from pitch_occupancy.vision import roi

MIDDLE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    """Both boundary stores in tmp_path: several routes here resolve or list boundaries."""
    monkeypatch.setattr(roi, "STORE", tmp_path / "roi.json")
    monkeypatch.setattr(roi, "DERIVED_STORE", tmp_path / "roi_derived.json")


def _png_claiming(width: int, height: int) -> bytes:
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(b"\x00" * 64)) + chunk(b"IEND", b""))


# --- UP-4: CPU-heavy work off the event loop, and bounded ------------------------------


def test_heavy_work_runs_at_most_heavy_concurrency_at_a_time(monkeypatch) -> None:
    monkeypatch.setattr(limits, "HEAVY_CONCURRENCY", 1)
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def work() -> None:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1

    async def main() -> None:
        async with anyio.create_task_group() as group:
            for _ in range(4):
                group.start_soon(limits.run_heavy, work)

    anyio.run(main)
    assert state["peak"] == 1


def test_a_clip_analysis_does_not_hold_the_event_loop(monkeypatch) -> None:
    """With a stub model at 0.4 s a frame, `GET /api/v1/roi` took 4.1 s instead of 0.03 s
    while one clip was analysed. Here the analysis blocks until released: if it were on the
    loop, the GET could not be answered until then."""
    from pitch_occupancy.api import clip_review as cr
    from pitch_occupancy.api import roi_editor

    started, release = threading.Event(), threading.Event()

    def blocking(path, **_kw):
        started.set()
        release.wait(30)
        raise HTTPException(422, "stub analysis")

    monkeypatch.setattr(cr, "_analyse_file", blocking)
    mini = FastAPI()
    mini.include_router(cr.router)
    mini.include_router(roi_editor.router)

    result: dict = {}
    with TestClient(mini) as client:  # one event loop shared by both requests
        worker = threading.Thread(target=lambda: result.update(
            response=client.post("/api/v1/clip/analyse", content=b"not really a video")))
        worker.start()
        try:
            assert started.wait(30)
            began = time.monotonic()
            assert client.get("/api/v1/roi").status_code == 200
            elapsed = time.monotonic() - began
        finally:
            release.set()
            worker.join(30)

    assert elapsed < 5, f"the listing waited {elapsed:.1f}s behind the analysis"
    assert result["response"].status_code == 422


# --- UP-5: a stream's files go when the response ends, however it ends -----------------


def _tracked_stream(n: int = 50):
    events: list[str] = []

    def generate():
        try:
            for i in range(n):
                events.append(f"step {i}")
                yield f"{i}\n"
        finally:
            events.append("closed")

    response = limits.CleanupStreamingResponse(
        generate(), cleanup=lambda: events.append("cleaned"))
    return response, events


def test_a_client_that_disconnects_mid_stream_still_gets_its_upload_deleted() -> None:
    """ASGI 2.4 servers report a disconnect as an OSError from `send`. The plain response
    left the generator suspended - its `finally` waited for a GC cycle, measured at 3-6 of 30
    uploads still on disk 8 s later - and skipped its background task."""
    response, events = _tracked_stream()

    async def send(message) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            raise OSError("client went away")

    async def receive():
        await anyio.sleep_forever()

    with pytest.raises(ClientDisconnect):
        anyio.run(response, {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

    assert events[-2:] == ["closed", "cleaned"]
    assert "step 49" not in events


def test_an_older_server_reporting_a_disconnect_message_also_cleans_up() -> None:
    """Before ASGI 2.4 the disconnect arrives through `receive` and the stream is cancelled."""
    response, events = _tracked_stream(10_000)

    async def send(message) -> None:
        await anyio.sleep(0.001)

    async def receive():
        await anyio.sleep(0.05)
        return {"type": "http.disconnect"}

    anyio.run(response, {"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send)

    assert events[-2:] == ["closed", "cleaned"]
    assert "step 9999" not in events


def test_a_completed_stream_cleans_up_exactly_once() -> None:
    response, events = _tracked_stream(3)
    sent: list[dict] = []

    async def send(message) -> None:
        sent.append(message)

    async def receive():
        await anyio.sleep_forever()

    anyio.run(response, {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    response.release()

    assert events.count("cleaned") == 1 and events.count("closed") == 1
    assert b"".join(m.get("body", b"") for m in sent) == b"0\n1\n2\n"


def test_leftovers_of_a_killed_process_are_swept_and_nothing_else(tmp_path) -> None:
    old = time.time() - limits.STALE_AFTER_S - 60
    stale_file = tmp_path / f"{limits.UPLOAD_PREFIX}abc.upload"
    stale_dir = tmp_path / f"{limits.WORKSPACE_PREFIX}abc"
    fresh = tmp_path / f"{limits.UPLOAD_PREFIX}new.upload"
    unrelated = tmp_path / "someone-elses.upload"
    for path in (stale_file, fresh, unrelated):
        path.write_bytes(b"x")
    stale_dir.mkdir()
    (stale_dir / "000.img").write_bytes(b"x")
    for path in (stale_file, stale_dir, unrelated):
        os.utime(path, (old, old))

    assert limits.sweep_stale(tmp_path) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([fresh.name, unrelated.name])


def test_a_clip_upload_is_named_so_the_sweep_recognises_it() -> None:
    fd, path = limits.upload_file()
    try:
        os.close(fd)
        assert path.name.startswith(limits.UPLOAD_PREFIX)
    finally:
        path.unlink(missing_ok=True)


# --- UP-7: pixel caps from the header ---------------------------------------------------


def test_the_pixel_check_reads_the_header_and_lets_unknown_bytes_through() -> None:
    limits.check_image_pixels(_png_claiming(4000, 3000))  # 12 MP: a phone photo
    with pytest.raises(HTTPException) as caught:
        limits.check_image_pixels(_png_claiming(12000, 12000))
    assert caught.value.status_code == 413
    with pytest.raises(HTTPException):  # past PIL's own bomb threshold as well
        limits.check_image_pixels(_png_claiming(30000, 30000))
    # Not an image PIL knows: whether it is an image at all stays OpenCV's decision.
    limits.check_image_pixels(b"definitely not an image")


def test_a_video_with_oversized_frames_is_refused(tmp_path, monkeypatch) -> None:
    import cv2

    class _Capture:
        def __init__(self, _path):
            pass

        def isOpened(self):  # noqa: N802 - mirrors cv2.VideoCapture
            return True

        def get(self, prop):
            return 16000 if prop == cv2.CAP_PROP_FRAME_WIDTH else 9000

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", _Capture)
    with pytest.raises(HTTPException) as caught:
        limits.check_video_pixels(tmp_path / "x.mp4")
    assert caught.value.status_code == 413


def test_the_boundary_preview_refuses_a_decompression_bomb() -> None:
    """The measured case: 17 KiB of PNG, 1 GiB of RSS once decoded."""
    image = base64.b64encode(_png_claiming(12000, 12000)).decode()
    response = TestClient(app).post("/api/v1/roi/preview",
                                    json={"image": image, "explain": False})
    assert response.status_code == 413


def test_the_boundary_suggestion_refuses_a_decompression_bomb() -> None:
    image = base64.b64encode(_png_claiming(12000, 12000)).decode()
    response = TestClient(app).post("/api/v1/roi/suggest", json={"images": [{"data": image}]})
    assert response.status_code == 413


# --- UP-8: bodies bounded before they are buffered ---------------------------------------


def test_the_preview_body_is_bounded(monkeypatch) -> None:
    from pitch_occupancy.api import roi_editor

    monkeypatch.setattr(roi_editor, "MAX_PREVIEW_BODY", 1024)
    response = TestClient(app).post("/api/v1/roi/preview", json={"image": "A" * 4096})
    assert response.status_code == 413


def test_the_suggestion_body_is_bounded(monkeypatch) -> None:
    from pitch_occupancy.api import roi_editor

    monkeypatch.setattr(roi_editor, "MAX_SUGGEST_BODY", 1024)
    response = TestClient(app).post("/api/v1/roi/suggest",
                                    json={"images": [{"data": "A" * 4096}]})
    assert response.status_code == 413


def test_a_booking_file_over_the_limit_is_refused_rather_than_truncated(monkeypatch) -> None:
    """It was buffered whole and then cut at 20 MB - mid-row, silently."""
    from pitch_occupancy.api import testing

    monkeypatch.setattr(testing, "_load_run", lambda _id: {"field_id": "f", "day": "2026-07-13",
                                                            "slots": []})
    monkeypatch.setattr(testing, "_MAX_BOOKING_BYTES", 1024)
    response = TestClient(app).post("/api/v1/testing/runs/20260713-120000-abcdef/compare",
                                    content=b"x" * 4096)
    assert response.status_code == 413


# --- UP-15: malformed input is a 4xx, and the model is not loaded first -----------------


@pytest.mark.parametrize("body", [
    b"not json", b"[]", b'"a string"', b'{"images": "abc"}', b'{"images": [{"data": 5}]}',
    b'{"images": [{"data": ""}]}', b'{"images": [5]}', b"",
])
def test_a_malformed_suggestion_request_is_a_422(body) -> None:
    response = TestClient(app).post("/api/v1/roi/suggest", content=body,
                                    headers={"Content-Type": "application/json"})
    assert response.status_code == 422


@pytest.mark.parametrize("route", ["/api/v1/clip/analyse", "/api/v1/clip/walkthrough"])
@pytest.mark.parametrize("polygon", [
    "[[1" + "0" * 400 + ", 0], [0, 1], [1, 1]]",  # OverflowError in float(), once a 500
    "[" * 2000 + "]" * 2000,                       # RecursionError in json.loads
    "[[0.1, 0.1], [0.2, 0.1]]",
])
def test_a_malformed_outline_is_refused_before_the_upload_or_the_model(
        monkeypatch, route, polygon) -> None:
    from pitch_occupancy.api import clip_review as cr
    from pitch_occupancy.api import clip_walkthrough as cw

    def _no(*_a, **_kw):
        raise AssertionError("the model was loaded for a request that was going to fail")

    monkeypatch.setattr(cr, "_classifier", _no)
    monkeypatch.setattr(cw, "_classifier", _no)
    uploads = []
    real = limits.upload_file
    monkeypatch.setattr(limits, "upload_file", lambda: uploads.append(1) or real())

    response = TestClient(app).post(route, params={"polygon": polygon}, content=b"video")

    assert response.status_code == 422
    assert uploads == [], "the upload was received before the outline was checked"


def test_a_malformed_pair_outline_is_refused_before_the_upload(monkeypatch) -> None:
    from pitch_occupancy.api import clip_pair as cp

    def _no():
        raise AssertionError("the detector was loaded for a request that was going to fail")

    monkeypatch.setattr(cp, "_pipeline", _no)
    response = TestClient(app).post(
        "/api/v1/clip/pair/analyse",
        params={"size_a": 3, "polygon_b": "[[1" + "0" * 400 + ", 0], [0, 1], [1, 1]]"},
        content=b"abcdef")
    assert response.status_code == 422
