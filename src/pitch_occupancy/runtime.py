"""The live step both live paths share: grab a whole picture, count one camera, decide a pitch.

The slot worker (`worker.run_slot`) reads its frames through a `frame_source` source and the
playground checks (`playgrounds/check.py`) through :func:`grab_whole`; both read with
`frame_source.read_whole`, count with `Pipeline.observe_minute` and decide a pitch whose
cameras are added with `slots/fusion.fuse_pitch`, so a camera is counted and a pitch decided
the same way on either path.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping

from pitch_occupancy.config import settings
from pitch_occupancy.slots.fusion import fuse_pitch
from pitch_occupancy.vision.rules import MinuteState, RuleConfig

__all__ = ["COUNTING_MODES", "detect_lock", "load_rules", "play_min", "shared_pipeline",
           "grab_whole", "observe_camera", "count_camera", "combine_counts", "combine_pitch"]

#: How a playground's cameras are combined - a user setting (`playgrounds/__init__.py`).
COUNTING_MODES: tuple[str, ...] = ("add", "largest", "average", "vote")

#: Held while the detector counts one picture: the detector is the part that uses the
#: processor, so pictures from concurrent checks are counted one at a time.
detect_lock = threading.Lock()


def load_rules() -> RuleConfig:
    """The decision rule (`configs/rules.json`). Raises when it cannot be read, as `assemble`
    does: a count with no rule behind it is not a verdict."""
    return RuleConfig.load(settings.rules_path)


def play_min() -> int:
    """People at or above which a pitch can count as playing - the rule's own number."""
    return int(load_rules().play_min)


def shared_pipeline():
    """The process-wide interactive pipeline (`pipeline.shared`)."""
    from pitch_occupancy.pipeline import shared

    return shared()


def grab_whole(url: str, timeout_s: float):
    """One whole picture from a camera URL: ``(frame | None, problem | None, wall_ms)``.

    ``problem`` is a sentence for a person; half-drawn frames are skipped
    (`frame_source.read_whole`).
    """
    import cv2

    from pitch_occupancy import cameras
    from pitch_occupancy.frame_source import read_whole

    started = time.monotonic()
    capture = cameras.open_capture(url, int(timeout_s * 1000))
    try:
        if not bool(getattr(capture, "isOpened", lambda: True)()):
            return None, "The camera did not answer.", int((time.monotonic() - started) * 1000)
        get = getattr(capture, "get", None)
        fps = float(get(cv2.CAP_PROP_FPS) or 25.0) if get is not None else 25.0
        ok, frame = read_whole(capture, fps)
    finally:
        capture.release()
    wall = int((time.monotonic() - started) * 1000)
    if not ok or frame is None:
        return None, "No whole picture arrived in time.", wall
    return frame, None, wall


def observe_camera(pipeline, camera: str, frames, previous=None, *, polygon=None,
                   venue: str | None = None):
    """One camera's frames (a burst, or one picture) counted by the pipeline.

    ``polygon`` wins over the stored boundary; ``venue`` narrows the lookup and is passed on
    only when given.
    """
    extra = {} if venue is None else {"venue": venue}
    return pipeline.observe_minute(camera, list(frames), previous, polygon=polygon, **extra)


def count_camera(pipeline, venue: str, camera: str, frame) -> dict:
    """One camera's picture counted inside its drawn area (`Pipeline.observe_minute`).

    Returns ``{people, state, detect_ms, instances, verdict}``; ``verdict`` is the camera's
    `FrameVerdict`, which :func:`combine_pitch` fuses. Serialised on :data:`detect_lock`.
    """
    with detect_lock:
        started = time.perf_counter()
        obs = observe_camera(pipeline, camera, [frame], venue=venue)
        took = time.perf_counter() - started
    v = obs.verdict
    return {"people": v.people, "state": str(getattr(v.state, "value", v.state)),
            "detect_ms": int(took * 1000),
            "instances": getattr(obs, "instances", ()), "verdict": v}


def combine_counts(counting: str, counts: list[int], threshold: int,
                   ) -> tuple[int | None, int | None]:
    """People and playing (1/0) from the cameras' counts alone; ``None`` without any."""
    if not counts:
        return None, None
    if counting == "largest":
        total = max(counts)
    elif counting == "average":
        total = round(sum(counts) / len(counts))
    elif counting == "vote":
        yes = sum(1 for n in counts if n >= threshold)
        return max(counts), int(2 * yes >= len(counts))
    else:
        total = sum(counts)
    return total, int(total >= threshold)


def combine_pitch(counting: str, verdicts: Mapping[str, object], cfg: RuleConfig,
                  ) -> tuple[int | None, int | None]:
    """People and playing (1/0/``None``) for one pitch from its counted cameras' verdicts.

    ``add`` (cameras see different parts) decides with `fuse_pitch`, as the slot worker does:
    the counts are summed and the rule applied once, so a dark pitch is not sure (``None``)
    and a standing crowd is not playing. The other modes compare counts with ``play_min``.
    """
    counts = [v.people for v in verdicts.values() if v.people is not None]
    total, playing = combine_counts(counting, counts, cfg.play_min)
    if counting == "add" and total is not None:
        pitch = fuse_pitch(dict(verdicts), cfg)
        playing = int(pitch.state is MinuteState.ACTIVE_PLAY) if pitch.decided else None
    return total, playing
