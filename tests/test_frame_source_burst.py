"""Bursts from the frame sources (A36).

The detector-first path reads three frames about a second apart per camera-minute. A source
that knows nothing about bursts gives one frame; a recording seeks to each offset; a stream
paces itself on the camera's own frame rate rather than a sleep.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from pitch_occupancy.frame_source import Frame, FrameSource, RTSPSource, VideoSlotSource

START = datetime(2026, 7, 11, 10, 0)


class Single(FrameSource):
    def cameras(self) -> list[str]:
        return ["cam"]

    @property
    def n_minutes(self) -> int:
        return 2

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        if minute_index == 1:
            return None
        return Frame(camera_id, minute_index, np.zeros((4, 4, 3), np.uint8), "x")


def test_a_source_without_bursts_gives_one_frame_and_a_gap_is_an_empty_list() -> None:
    src = Single()
    assert len(src.read_burst("cam", 0, n=3)) == 1
    assert src.read_burst("cam", 1, n=3) == []


def _video(path: Path, *, seconds: int, fps: int = 10) -> Path:
    """A clip whose brightness is the second it belongs to, so a frame says when it was."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for second in range(seconds):
        for _ in range(fps):
            writer.write(np.full((48, 64, 3), 20 + 20 * second, np.uint8))
    writer.release()
    return path


def test_a_recording_seeks_to_each_burst_offset(tmp_path) -> None:
    video = _video(tmp_path / "slot.mp4", seconds=8)
    with VideoSlotSource({"cam": video}, seconds_per_minute=4) as src:
        assert src.n_minutes == 2
        burst = src.read_burst("cam", 1, n=3, spacing_s=1.0)
        assert len(burst) == 3
        seconds = [int(round((float(f.image_bgr.mean()) - 20) / 20)) for f in burst]
        assert seconds == [4, 5, 6], seconds
        assert all(f.minute_index == 1 for f in burst)
        tail = src.read_burst("cam", 1, n=6, spacing_s=1.0)
        assert 3 <= len(tail) < 6, "the burst ran off the end and kept what it read"


class FakeStream:
    """A connection that hands out numbered frames at a stated rate."""

    def __init__(self, fps: float = 10.0, frames: int = 1000) -> None:
        self.fps, self.frames, self.position, self.grabs, self.released = fps, frames, 0, 0, 0

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def grab(self) -> bool:
        self.grabs += 1
        self.position += 1
        return self.position < self.frames

    def read(self):
        if self.position >= self.frames:
            return False, None
        image = np.full((4, 4, 3), self.position % 256, np.uint8)
        self.position += 1
        return True, image

    def release(self) -> None:
        self.released += 1


class Clock:
    def __init__(self) -> None:
        self.now = START
        self.naps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.naps.append(seconds)
        self.now += timedelta(seconds=seconds)


def test_a_stream_burst_uses_one_connection_and_paces_on_the_frame_rate() -> None:
    streams: list[FakeStream] = []

    def open_capture(url):
        streams.append(FakeStream(fps=10.0))
        return streams[-1]

    clock = Clock()
    src = RTSPSource({"cam": "rtsp://h/1"}, minutes=3, open_capture=open_capture,
                     clock=clock, sleep=clock.sleep, started_at=START)
    burst = src.read_burst("cam", 0, n=3, spacing_s=1.0)
    assert len(burst) == 3
    assert len(streams) == 1 and streams[0].released == 1, "one connection per burst"
    assert streams[0].grabs == 20, "ten frames grabbed per second of spacing, twice"
    positions = [int(f.image_bgr[0, 0, 0]) for f in burst]
    assert positions == [0, 11, 22], positions
    assert clock.naps == [], "the pacing came from the stream, not a sleep"


def test_a_stream_burst_waits_for_its_minute_and_records_a_gap_when_every_retry_fails():
    class Broken(FakeStream):
        def read(self):
            return False, None

    clock = Clock()
    src = RTSPSource({"cam": "rtsp://h/1"}, minutes=3, retries=1,
                     open_capture=lambda url: Broken(), clock=clock, sleep=clock.sleep,
                     started_at=START)
    assert src.read_burst("cam", 2, n=3) == []
    assert clock.naps == [120.0], "minute 2 arrives two minutes after the start"


def test_the_video_source_refuses_an_unknown_camera_for_bursts_too(tmp_path) -> None:
    video = _video(tmp_path / "slot.mp4", seconds=2)
    with VideoSlotSource({"cam": video}, seconds_per_minute=1) as src, pytest.raises(KeyError):
        src.read_burst("other", 0)
