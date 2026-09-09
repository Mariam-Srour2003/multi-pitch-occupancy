"""Where frames come from (WP6-T1, WP6-T3).

Three backends behind one interface, so the pipeline is identical whether it is replaying
a recording, polling the simulator, or pulling from real cameras. That equivalence is the
point: a bug that only appears under RTSP is a bug nobody finds until deployment.

* :class:`VideoSlotSource` - replays recorded footage as if it were live, one frame per
  simulated minute. This is what makes a 60-minute slot testable in seconds.
* :class:`RTSPSource` - one snapshot per camera per minute from a live stream.
* the simulator API backend serves recorded frames over HTTP (WP6-T1).

**A read failure returns ``None`` rather than raising or substituting a frame.** A camera
that drops out for a minute must produce a gap in the record, not a fabricated observation:
downstream, a missing minute weakens a verdict, while an invented EMPTY would silently
bill a used slot as unused.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

__all__ = ["Frame", "FrameSource", "VideoSlotSource", "RTSPSource", "discover_slots"]

#: `StatBox_Replay_<venue>_2026-07-11_10-00 (1).mp4` -> date, time, and which camera
_VIDEO_RE = re.compile(r"(?P<d>\d{4}-\d{2}-\d{2})_(?P<t>\d{2}-\d{2})(?P<second>\s*\(1\))?")


@dataclass(frozen=True, slots=True)
class Frame:
    camera_id: str
    minute_index: int
    image_bgr: np.ndarray
    source_path: str | None = None


class FrameSource(ABC):
    """One frame per camera per sampled minute."""

    @abstractmethod
    def cameras(self) -> list[str]: ...

    @abstractmethod
    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        """Return the frame for that minute, or ``None`` if it could not be read."""

    @property
    @abstractmethod
    def n_minutes(self) -> int: ...


class VideoSlotSource(FrameSource):
    """Replays a slot's recordings, one frame per simulated minute.

    ``videos`` maps camera id -> file. Every camera of one slot must come from the same
    recording session; the minute index is an offset into all of them.
    """

    def __init__(self, videos: dict[str, Path], *, seconds_per_minute: int = 60) -> None:
        if not videos:
            raise ValueError("a slot needs at least one camera recording")
        self._videos = dict(videos)
        self._step = seconds_per_minute
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._minutes = min(self._duration_minutes(p) for p in videos.values())

    def _duration_minutes(self, path: Path) -> int:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise OSError(f"cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(int(frames / fps // self._step), 0)

    def _capture(self, camera_id: str) -> cv2.VideoCapture:
        if camera_id not in self._caps:
            self._caps[camera_id] = cv2.VideoCapture(str(self._videos[camera_id]))
        return self._caps[camera_id]

    def cameras(self) -> list[str]:
        return sorted(self._videos)

    @property
    def n_minutes(self) -> int:
        return self._minutes

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        if camera_id not in self._videos:
            raise KeyError(f"unknown camera {camera_id!r}; have {self.cameras()}")
        cap = self._capture(camera_id)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(minute_index * self._step * fps))
        ok, image = cap.read()
        if not ok:
            return None  # a gap, not a fabricated frame
        return Frame(camera_id, minute_index, image, str(self._videos[camera_id]))

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()

    def __enter__(self) -> VideoSlotSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RTSPSource(FrameSource):
    """One snapshot per camera per minute from live streams (WP6-T3).

    Opens and closes per read rather than holding streams: at one frame per minute a
    persistent connection buys nothing and turns every network blip into stale buffered
    frames, which is worse than a clean reconnect.

    **Two things about this class were wrong until 2026-09-09 and neither raised until it was
    run.** It is worth stating both, because the shape of the mistake recurs.

    *It could not run at all.* ``n_minutes`` raised ``NotImplementedError`` with the comment
    "the scheduler decides" - but the scheduler calls ``worker.run_slot``, whose first
    statement is ``for minute in range(source.n_minutes)``. So the one class whose entire
    purpose is live operation was structurally incompatible with the only code that would ever
    drive it, and nothing noticed because nothing tested it. A live slot's length is now
    supplied at construction, which is what "the scheduler decides" should have meant: the
    scheduler knows the slot's ``duration_minutes`` and passes it in. It is **required**, not
    defaulted, because a default would make the slot's own length a guess.

    *It would not have paced itself.* ``run_slot`` loops over minutes without waiting, which
    is correct for a recording - minute *k* is a seek - and wrong for a stream, where it would
    have taken sixty snapshots back to back in under a second and called it an hour. So
    :meth:`read` blocks until minute *k* has actually arrived. The clock and the sleep are
    parameters for the same reason the scheduler's are: a class whose only observable
    behaviour is that it waits an hour is a class with no tests.

    ``open_capture`` is injectable too, so the retry and gap behaviour can be tested without a
    camera. That is the part most likely to matter in the field and was the part with no test.
    """

    def __init__(
        self,
        urls: dict[str, str],
        *,
        minutes: int,
        started_at: datetime | None = None,
        timeout_ms: int = 5000,
        retries: int = 2,
        open_capture: Callable[[str], object] | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not urls:
            raise ValueError("no camera URLs configured")
        if minutes <= 0:
            raise ValueError(
                f"minutes={minutes}: a live source must be told how long the slot is. The "
                f"scheduler has this as ScheduledSlot.duration_minutes"
            )
        self._urls = dict(urls)
        self._minutes = int(minutes)
        self._timeout_ms = timeout_ms
        self._retries = retries
        self._open = open_capture or self._open_stream
        self._clock = clock or (lambda: datetime.now())
        self._sleep = sleep or time.sleep
        self._started_at = started_at or self._clock()

    def _open_stream(self, url: str):
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._timeout_ms)
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._timeout_ms)
        return capture

    def cameras(self) -> list[str]:
        return sorted(self._urls)

    @property
    def n_minutes(self) -> int:
        return self._minutes

    @property
    def started_at(self) -> datetime:
        return self._started_at

    def wait_for(self, minute_index: int) -> None:
        """Block until ``minute_index`` minutes have elapsed since the slot began.

        Returns immediately when that moment has passed, so the second camera of a minute does
        not wait again and a run that has fallen behind catches up rather than stretching. A
        slot that overruns is a slot with missed minutes, which the capture rate already
        reports; sleeping the shortfall would hide it.
        """
        due_at = self._started_at + timedelta(minutes=minute_index)
        remaining = (due_at - self._clock()).total_seconds()
        if remaining > 0:
            self._sleep(remaining)

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        if camera_id not in self._urls:
            raise KeyError(f"unknown camera {camera_id!r}; have {self.cameras()}")
        self.wait_for(minute_index)
        url = self._urls[camera_id]
        for _ in range(self._retries + 1):
            capture = self._open(url)
            try:
                ok, image = capture.read()
            finally:
                capture.release()
            if ok and image is not None:
                return Frame(camera_id, minute_index, image, url)
        return None  # every retry failed: record the gap


def discover_slots(directory: Path) -> dict[str, dict[str, Path]]:
    """Group recordings into ``slot_id -> {file_key: path}``.

    Keys are ``file0`` / ``file1``, **not** ``camA`` / ``camB``. The export marks a slot's
    second recording with a ``(1)`` suffix, but that suffix does not identify a physical
    camera: measured on the venue_01 recordings, ``file0`` of the morning slot matches
    ``file1`` of the evening slot (view similarity 0.88) and not its own-suffix counterpart
    (0.70). The mapping flips between days.

    Naming these ``camA``/``camB`` would therefore be wrong on one day in two - and wrong
    invisibly, because fusion still yields a plausible verdict from swapped halves. To
    resolve real camera identity, describe each recording with
    :func:`pitch_occupancy.vision.camera_id.describe_view` and match it against references.
    """
    slots: dict[str, dict[str, Path]] = {}
    for path in sorted(directory.glob("*.mp4")):
        m = _VIDEO_RE.search(path.stem)
        if not m:
            continue
        slot_id = f"slot_{m['d'].replace('-', '')}_{m['t'].replace('-', '')}"
        file_key = "file1" if "(1)" in path.stem else "file0"
        slots.setdefault(slot_id, {})[file_key] = path
    return slots
