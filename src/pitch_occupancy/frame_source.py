"""Where frames come from.

Three backends behind one interface, so the pipeline is identical whether it is replaying
a recording, polling the simulator, or pulling from real cameras. That equivalence is the
point: a bug that only appears under RTSP is a bug nobody finds until deployment.

* :class:`VideoSlotSource` - replays recorded footage as if it were live, one frame per
  simulated minute. This is what makes a 60-minute slot testable in seconds.
* :class:`RTSPSource` - one snapshot per camera per minute from a live stream.
* the simulator API backend serves recorded frames over HTTP.

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

__all__ = ["Frame", "FrameSource", "VideoSlotSource", "RTSPSource", "discover_slots",
           "is_incomplete", "read_whole", "WHOLE_FRAME_WAIT_S"]

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

    def read_burst(self, camera_id: str, minute_index: int, *, n: int = 3,
                   spacing_s: float = 1.0) -> list[Frame]:
        """Up to ``n`` frames about ``spacing_s`` apart at the start of the minute.

        The detector-first path reads a short burst per camera per minute rather than one
        frame: a ball seen in any of three counts, the burst supplies a motion cue without
        waiting for the previous minute, and a detection has to persist in two of three
        frames to be counted. The default is a single `read`, so a source that knows nothing
        about bursts still works and simply gives the rule one frame - which the rule
        handles, with motion unmeasured. An empty list is a gap, never a fabricated frame.
        """
        frame = self.read(camera_id, minute_index)
        return [frame] if frame is not None else []

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

    def read_burst(self, camera_id: str, minute_index: int, *, n: int = 3,
                   spacing_s: float = 1.0) -> list[Frame]:
        """``n`` frames at ``minute*step + k*spacing_s`` seconds, by seeking to each."""
        if camera_id not in self._videos:
            raise KeyError(f"unknown camera {camera_id!r}; have {self.cameras()}")
        cap = self._capture(camera_id)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out: list[Frame] = []
        for k in range(max(1, n)):
            position = (minute_index * self._step + k * spacing_s) * fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, image = cap.read()
            if not ok:
                break  # the burst ran off the end of the recording; keep what was read
            out.append(Frame(camera_id, minute_index, image, str(self._videos[camera_id])))
        return out

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()

    def __enter__(self) -> VideoSlotSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _shown(url: str) -> str:
    """A camera URL as it may be recorded: password masked.

    A frame's `source_path` travels on into `frame_samples.image_path` in the database, logs
    and the evidence trail. Until 2026-10-10 a live frame carried the real RTSP URL there, so
    every recorded minute stored the camera's password. The masked form still says which
    camera and stream the frame came from, which is all a reader of the record needs.
    """
    from pitch_occupancy.cameras import mask

    return mask(url)


def is_incomplete(image: np.ndarray) -> bool:
    """Whether a decoded frame is only partly drawn.

    When a stream stalls mid-frame, FFmpeg still returns the frame with everything after the
    stall filled in. Counted as it is, the filled part hides the people in it. Three fills
    are seen from these cameras, all reaching the bottom edge (checked on 64x48 rows):

    - **a flat pale band**: rows almost uniform (std < 8), very bright (V > 225), colourless
      (S < 40); two such rows at the bottom;
    - **a flat magenta or green band** (missing colour data): rows almost uniform with a
      strongly coloured mean (S > 90, hue magenta or green); two such rows at the bottom;
    - **a smear**: the last good row copied downwards, so the rows have detail across but
      none down - neighbouring rows differ by under 1.5 grey levels while each row's own
      std is above 8; six such rows at the bottom. A dark or empty night frame has no detail
      across, so it is not taken for a smear.
    """
    if image is None or getattr(image, "ndim", 0) != 3 or image.shape[0] < 48:
        return False
    small = cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    std = small.astype(np.float32).std(axis=1).max(axis=1)
    sat, val = hsv[:, :, 1].mean(axis=1), hsv[:, :, 2].mean(axis=1)
    hue = np.median(hsv[:, :, 0], axis=1)
    flat = std < 8
    pale = flat & (val > 225) & (sat < 40)
    coloured = flat & (sat > 90) & (((hue > 135) & (hue < 170)) | ((hue > 45) & (hue < 75)))
    if (pale[-1] and pale[-2]) or (coloured[-1] and coloured[-2]):
        return True
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    still = np.abs(np.diff(grey, axis=0)).mean(axis=1) < 1.5  # row i vs row i+1
    smear = still & (std[1:] > 8)
    return bool(smear[-6:].all())


#: How long to keep reading for a whole frame when the stream gives a half-drawn one: past the
#: cameras' keyframe interval (4 s measured) so a fresh complete picture can arrive.
WHOLE_FRAME_WAIT_S = 5.0


def read_whole(capture, fps: float, *, deadline: float | None = None):
    """The next frame that is fully drawn, or ``(False, None)`` if none comes in time.

    Reads up to ``fps * whole_frame_wait_s`` frames, skipping half-drawn ones
    (`is_incomplete`), and stops at ``deadline`` (a `time.monotonic` value) whatever is left:
    over a slow link each read can take seconds, and a frame count alone let one camera hold
    a grab slot for ten minutes.
    """
    import time

    from pitch_occupancy.config import settings

    wait_s = max(WHOLE_FRAME_WAIT_S, settings.whole_frame_wait_s)
    for _ in range(max(1, int(fps * wait_s))):
        if deadline is not None and time.monotonic() > deadline:
            return False, None
        ok, image = capture.read()
        if not ok or image is None:
            return False, None
        if not is_incomplete(image):
            return True, image
    return False, None


class RTSPSource(FrameSource):
    """One snapshot per camera per minute from live streams.

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

    **A minute whose moment has gone is a gap, not a frame taken now** (2026-10-10). Minute *k*
    belongs to ``[started_at + k, started_at + k + 1)``. Until this date a read that arrived
    after that window still opened the stream and labelled whatever it saw "minute k" - so a
    worker started at 09:00 for a slot that began at 08:00 would have read sixty frames of
    09:00 back to back and filed them as 08:00-08:59, the fabricated observation this module
    exists to refuse. ``max_late_s`` (default one minute) is how late a read may be before it
    is skipped; the scheduler now passes the slot's real start, so the skipped minutes are the
    ones that truly were not watched, and the capture floor in `slots/aggregate.py` decides
    what that means for the verdict.
    """

    def __init__(
        self,
        urls: dict[str, str],
        *,
        minutes: int,
        started_at: datetime | None = None,
        timeout_ms: int = 5000,
        retries: int = 2,
        max_late_s: float = 60.0,
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
        self._max_late_s = float(max_late_s)
        self._open = open_capture or self._open_stream
        self._clock = clock or (lambda: datetime.now())
        self._sleep = sleep or time.sleep
        self._started_at = started_at or self._clock()

    def _open_stream(self, url: str):
        # The same opener the /cameras page tests with, so a URL that passed there opens the
        # same way here - and ``usb:N`` (a local webcam) works in the live worker too.
        from pitch_occupancy.cameras import open_capture

        return open_capture(url, self._timeout_ms)

    def cameras(self) -> list[str]:
        return sorted(self._urls)

    @property
    def n_minutes(self) -> int:
        return self._minutes

    @property
    def started_at(self) -> datetime:
        return self._started_at

    def wait_for(self, minute_index: int) -> bool:
        """Block until minute ``minute_index`` has arrived; say whether it can still be read.

        Returns immediately when that moment has passed, so the second camera of a minute does
        not wait again and a run that has fallen behind catches up rather than stretching. A
        slot that overruns is a slot with missed minutes, which the capture rate already
        reports; sleeping the shortfall would hide it.

        Returns ``False`` when the minute is ``max_late_s`` or more in the past - the worker
        started after it, or the previous minute's work overran by a whole minute. The caller
        then records a gap: reading the stream now and calling it minute *k* would be a frame
        from the wrong time with the right label, which is worse than no frame.
        """
        due_at = self._started_at + timedelta(minutes=minute_index)
        remaining = (due_at - self._clock()).total_seconds()
        if remaining > 0:
            self._sleep(remaining)
            return True
        return -remaining < self._max_late_s

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        if camera_id not in self._urls:
            raise KeyError(f"unknown camera {camera_id!r}; have {self.cameras()}")
        if not self.wait_for(minute_index):
            return None  # the minute has gone: a gap, never "now" relabelled as then
        url = self._urls[camera_id]
        for _ in range(self._retries + 1):
            capture = self._open(url)
            try:
                fps = float(getattr(capture, "get", lambda _p: 0.0)(cv2.CAP_PROP_FPS) or 25.0)
                ok, image = read_whole(capture, fps)
            finally:
                capture.release()
            if ok and image is not None:
                return Frame(camera_id, minute_index, image, _shown(url))
        return None  # every retry failed: record the gap

    def read_burst(self, camera_id: str, minute_index: int, *, n: int = 3,
                   spacing_s: float = 1.0) -> list[Frame]:
        """``n`` frames from one connection, ``spacing_s`` apart, paced by the stream itself.

        One open per burst rather than one per frame: reconnecting three times a minute is
        three chances for a blip. Between frames the stream is *grabbed* (decoded and
        discarded) for ``spacing_s`` worth of frames at the stream's reported rate, so the
        spacing comes from the camera's clock and a stale buffered frame is never returned
        as "a second later". A burst that loses the connection part-way keeps what it read;
        a burst that read nothing is retried like a single frame, and an empty list is the gap.
        """
        if camera_id not in self._urls:
            raise KeyError(f"unknown camera {camera_id!r}; have {self.cameras()}")
        if not self.wait_for(minute_index):
            return []  # the minute has gone: a gap, never "now" relabelled as then
        url = self._urls[camera_id]
        for _ in range(self._retries + 1):
            capture = self._open(url)
            out: list[Frame] = []
            try:
                fps = float(getattr(capture, "get", lambda _p: 0.0)(cv2.CAP_PROP_FPS) or 25.0)
                skip = max(1, int(round(fps * spacing_s)))
                for k in range(max(1, n)):
                    if k:
                        for _ in range(skip):
                            if not capture.grab():
                                break
                    # A half-drawn frame is skipped, not counted: see `is_incomplete`.
                    ok, image = read_whole(capture, fps)
                    if not ok or image is None:
                        break
                    out.append(Frame(camera_id, minute_index, image, _shown(url)))
            finally:
                capture.release()
            if out:
                return out
        return []


def discover_slots(directory: Path) -> dict[str, dict[str, Path]]:
    """Group recordings into ``slot_id -> {file_key: path}``.

    Keys are ``file0`` / ``file1``, **not** ``camA`` / ``camB``. The export marks a slot's
    second recording with a ``(1)`` suffix, but that suffix does not identify a physical
    camera: measured on the venue_01 recordings, ``file0`` of the morning slot matches
    ``file1`` of the evening slot (view similarity 0.88) and not its own-suffix counterpart
    (0.70). The mapping flips between days.

    Naming these ``camA``/``camB`` would therefore be wrong on one day in two - and wrong
    invisibly, because fusion still yields a plausible verdict from swapped halves. Real
    camera identity has to come from matching each recording's view against a reference,
    never from the filename.
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
