"""The camera registry, masking and the staged connection test.

No test here opens a socket or a stream: the TCP connect and the capture opener are injected,
which is the only way the "port closed" and "no picture arrived" branches - the ones an
installer will actually meet - can be exercised at all.
"""

from __future__ import annotations

import json
import socket

import numpy as np
import pytest

from pitch_occupancy import cameras

SECRET = "s3cretPW"
URL = f"rtsp://admin:{SECRET}@10.0.0.5:554/Streaming/Channels/102"


@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / "configs" / "cameras.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "_comment": ["keep me"],
        "venue_01": {"_note": "also keep me", "camera_A": URL},
    }), encoding="utf-8")
    monkeypatch.setattr(cameras, "CONFIG_PATH", path)
    return path


class FakeCapture:
    def __init__(self, opened: bool = True, frame: bool = True) -> None:
        self.opened, self.frame, self.released = opened, frame, 0

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV's name
        return self.opened

    def read(self):
        if not self.frame:
            return False, None
        return True, np.zeros((360, 640, 3), np.uint8)

    def release(self) -> None:
        self.released += 1


# --- the registry -------------------------------------------------------------------------

def test_round_trip_keeps_comments_and_masks(config) -> None:
    saved = cameras.save_camera("venue_01", "camera_B", "rtsp://u:pw2@10.0.0.6/stream1")
    assert saved["url"] == "rtsp://u:***@10.0.0.6/stream1"
    raw = json.loads(config.read_text(encoding="utf-8"))
    assert raw["_comment"] == ["keep me"]
    assert raw["venue_01"]["_note"] == "also keep me"
    assert raw["venue_01"]["camera_B"] == "rtsp://u:pw2@10.0.0.6/stream1"

    listed = cameras.list_cameras()
    assert [(c["venue"], c["camera_id"]) for c in listed] == [
        ("venue_01", "camera_A"), ("venue_01", "camera_B")]
    assert SECRET not in json.dumps(listed)

    cameras.delete_camera("venue_01", "camera_B")
    cameras.delete_camera("venue_01", "camera_A")
    raw = json.loads(config.read_text(encoding="utf-8"))
    assert raw["_comment"] == ["keep me"]
    assert raw["venue_01"] == {"_note": "also keep me"}  # comment keeps the venue
    with pytest.raises(KeyError):
        cameras.delete_camera("venue_01", "camera_A")


def test_writes_keep_a_history_copy_and_leave_no_staging_file(config) -> None:
    cameras.save_camera("venue_01", "camera_B", "usb:0")
    history = list(cameras.history_dir().glob("cameras.*.json"))
    assert len(history) == 1
    assert URL in history[0].read_text(encoding="utf-8")
    assert not config.with_suffix(".json.new").exists()


def test_new_file_is_created_with_a_comment(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cameras.json"
    monkeypatch.setattr(cameras, "CONFIG_PATH", path)
    cameras.save_camera("v1", "cam1", "usb:1")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "_comment" in raw and raw["v1"] == {"cam1": "usb:1"}


def test_corrupt_file_is_refused_not_overwritten(config) -> None:
    config.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        cameras.save_camera("venue_01", "camera_B", "usb:0")
    assert config.read_text(encoding="utf-8") == "{not json"


@pytest.mark.parametrize("bad", ["", "_hidden", "a b", "../x", "x/y", "a" * 65, "-x"])
def test_ids_are_validated(config, bad) -> None:
    with pytest.raises(ValueError):
        cameras.save_camera(bad, "camera_A", "usb:0")
    with pytest.raises(ValueError):
        cameras.save_camera("venue_01", bad, "usb:0")


@pytest.mark.parametrize("url", [
    "rtsp://10.0.0.5/live", "rtsps://h:322/x", "http://10.0.0.5/snap.jpg",
    "https://cam.local/mjpeg", "usb:0", "USB:3",
])
def test_good_urls(url) -> None:
    assert cameras.validate_url(url)


@pytest.mark.parametrize("url", [
    "ftp://10.0.0.5/x", "10.0.0.5:554", "file:///etc/passwd", "rtsp://", "usb:x", "usb0",
    "rtsp://h:99999/x", "rtsp://a b@h/x", "rtsp://admin:pa/ss@h/x",
])
def test_bad_urls(url) -> None:
    with pytest.raises(ValueError):
        cameras.validate_url(url)


# --- masking ------------------------------------------------------------------------------

@pytest.mark.parametrize(("url", "masked"), [
    (URL, "rtsp://admin:***@10.0.0.5:554/Streaming/Channels/102"),
    ("rtsp://admin:p@ss@10.0.0.5/x", "rtsp://admin:***@10.0.0.5/x"),
    ("rtsp://admin@10.0.0.5/x", "rtsp://admin@10.0.0.5/x"),
    ("rtsp://10.0.0.5/x", "rtsp://10.0.0.5/x"),
    ("http://h/snap.cgi?user=admin&pwd=xx&chn=1", "http://h/snap.cgi?user=admin&pwd=***&chn=1"),
    ("usb:0", "usb:0"),
    ("nonsense", "***"),
])
def test_mask(url, masked) -> None:
    assert cameras.mask(url) == masked


def test_blank_or_masked_password_keeps_the_stored_one(config) -> None:
    cameras.save_camera("venue_01", "camera_A", "rtsp://admin:***@10.0.0.9:554/new/path")
    assert cameras.get_url("venue_01", "camera_A") == f"rtsp://admin:{SECRET}@10.0.0.9:554/new/path"
    cameras.save_camera("venue_01", "camera_A", "rtsp://admin@10.0.0.9:554/other")
    assert cameras.get_url("venue_01", "camera_A") == f"rtsp://admin:{SECRET}@10.0.0.9:554/other"


def test_masked_password_with_nothing_stored_is_refused(config) -> None:
    with pytest.raises(ValueError, match="hidden"):
        cameras.save_camera("venue_01", "camera_new", "rtsp://admin:***@10.0.0.9/x")
    # A changed user name does not inherit the old user's password.
    with pytest.raises(ValueError, match="hidden"):
        cameras.save_camera("venue_01", "camera_A", "rtsp://viewer:***@10.0.0.5/x")


def test_query_secret_is_kept_too() -> None:
    merged = cameras.merge_password("http://h/s?user=a&pwd=***", "http://h/s?user=a&pwd=zz")
    assert merged == "http://h/s?user=a&pwd=zz"


# --- the staged test ----------------------------------------------------------------------

def _connect_ok(host, port, timeout):
    assert (host, port) == ("10.0.0.5", 554)


def _raising(exc):
    def connect(host, port, timeout):
        raise exc
    return connect


def test_unreachable() -> None:
    r = cameras.probe(URL, connect=_raising(TimeoutError()),
                      open_capture=lambda u, t: pytest.fail("must not open"))
    assert (r.ok, r.stage) == (False, "unreachable")
    assert r.hint and SECRET not in json.dumps(r.to_dict())


def test_no_route_is_unreachable() -> None:
    r = cameras.probe(URL, connect=_raising(OSError(113, "No route to host")))
    assert r.stage == "unreachable"


def test_port_closed() -> None:
    r = cameras.probe(URL, connect=_raising(ConnectionRefusedError()))
    assert r.stage == "port_closed" and "RTSP" in r.hint


def test_name_not_found_is_an_address_problem() -> None:
    r = cameras.probe("rtsp://u:p@nosuchcam.invalid/x", connect=_raising(socket.gaierror()))
    assert r.stage == "address"


def test_stream_will_not_open() -> None:
    cap = FakeCapture(opened=False)
    r = cameras.probe(URL, connect=_connect_ok, open_capture=lambda u, t: cap)
    assert r.stage == "stream" and "password" in r.hint
    assert cap.released == 1


def test_opener_exception_is_reported_by_type_only() -> None:
    def boom(url, timeout):
        raise RuntimeError(f"cannot open {url}")
    r = cameras.probe(URL, connect=_connect_ok, open_capture=boom)
    assert r.stage == "stream"
    assert SECRET not in json.dumps(r.to_dict())


def test_opened_but_no_frame() -> None:
    cap = FakeCapture(frame=False)
    r = cameras.probe(URL, connect=_connect_ok, open_capture=lambda u, t: cap)
    assert r.stage == "frame" and "H.264" in r.hint
    assert cap.released == 1


def test_ok_reports_resolution_and_frame() -> None:
    seen = []
    r = cameras.probe(URL, connect=_connect_ok,
                      open_capture=lambda u, t: seen.append(u) or FakeCapture())
    assert (r.ok, r.stage, r.width, r.height) == (True, "ok", 640, 360)
    assert r.frame is not None and "frame" not in r.to_dict()
    assert seen == [URL]  # the raw URL is what is opened...
    assert r.url == cameras.mask(URL)  # ...and only the masked one is reported
    assert cameras.jpeg_base64(r.frame)


def test_default_ports() -> None:
    ports = []
    def connect(host, port, timeout):
        ports.append(port)
    for url in ("rtsp://h/x", "http://h/x", "https://h/x", "rtsp://h:8554/x"):
        cameras.probe(url, connect=connect, open_capture=lambda u, t: FakeCapture())
    assert ports == [554, 80, 443, 8554]


def test_usb_skips_the_network_step() -> None:
    r = cameras.probe("usb:0", connect=lambda *a: pytest.fail("no TCP for a webcam"),
                      open_capture=lambda u, t: FakeCapture())
    assert r.ok and r.host is None


def test_open_capture_maps_usb_to_a_device_index(monkeypatch) -> None:
    calls = []

    class Recorder:
        def __init__(self, *args) -> None:
            calls.append(args)

        def set(self, *args) -> None:
            calls.append(("set", *args))

    monkeypatch.setattr(cameras.cv2, "VideoCapture", Recorder)
    cameras.open_capture("usb:2")
    assert calls == [(2,)]
    calls.clear()
    cameras.open_capture(URL, 1234)
    assert calls[0] == (URL, cameras.cv2.CAP_FFMPEG)
    assert ("set", cameras.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1234) in calls


def test_rtsp_source_opens_through_the_shared_opener(monkeypatch) -> None:
    from pitch_occupancy.frame_source import RTSPSource

    seen = []
    monkeypatch.setattr(cameras, "open_capture",
                        lambda url, timeout_ms: seen.append((url, timeout_ms)) or FakeCapture())
    source = RTSPSource({"cam": "usb:1"}, minutes=1, timeout_ms=777)
    assert source.read("cam", 0) is not None
    assert seen == [("usb:1", 777)]
