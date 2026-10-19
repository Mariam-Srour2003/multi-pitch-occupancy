"""How much this computer can do, and whether the saved playgrounds ask for more.

:func:`measure_speed` times the whole job for one picture - opening a camera, decoding a whole
frame, counting - in processor time, and saves it to ``configs/computer_speed.json``.
:func:`capacity` turns that into pictures per minute and compares it with what the enabled
playgrounds ask for (each camera, once per check).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pitch_occupancy.config import CONFIGS_DIR
from pitch_occupancy.playgrounds import check, store

__all__ = ["SPEED_PATH", "HEADROOM", "load_speed", "capacity", "check_capacity",
           "measure_speed"]

#: Read at call time, so patching ``capacity.SPEED_PATH`` redirects every reader.
SPEED_PATH: Path = CONFIGS_DIR / "computer_speed.json"
#: Fraction of the measured speed the playgrounds may use. The rest is for everything else
#: the computer does - the web pages, the recorder, Windows - so the checks keep their time.
HEADROOM = 0.7
#: Used until the speed has been measured: a cautious guess for one picture on a laptop CPU.
DEFAULT_SECONDS_PER_PICTURE = 1.0


def load_speed(path: Path | None = None) -> dict | None:
    try:
        raw = json.loads((path or SPEED_PATH).read_text(encoding="utf-8"))
        return raw if float(raw.get("seconds_per_picture", 0)) > 0 else None
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def capacity(items: list[dict] | None = None, speed: dict | None = None) -> dict:
    """Pictures per minute the computer can analyse, and what the playgrounds ask for."""
    items = store.load() if items is None else items
    speed = speed if speed is not None else load_speed()
    per_picture = float(speed["seconds_per_picture"]) if speed else DEFAULT_SECONDS_PER_PICTURE
    can = 60.0 / per_picture * HEADROOM
    need = sum(len(p["cameras"]) * 60.0 / p["every_seconds"] for p in items if p["enabled"])
    examples = {every: int(can * every // 60) for every in (5, 10, 30, 60, 300)}
    return {
        "seconds_per_picture": round(per_picture, 3), "measured": speed is not None,
        "measured_at": speed.get("measured_at") if speed else None,
        "can_per_min": round(can, 1), "need_per_min": round(need, 1),
        "percent": round(100 * need / can) if can else 0,
        "cameras_at": examples,
    }


def check_capacity(proposed: list[dict]) -> None:
    """Raise ``ValueError`` (a sentence for the page) when ``proposed`` asks for too much."""
    cap = capacity(proposed)
    if cap["need_per_min"] > cap["can_per_min"]:
        raise ValueError(
            f"This computer can analyse about {cap['can_per_min']:.0f} pictures a minute, and "
            f"these settings ask for {cap['need_per_min']:.0f}. Check less often, use fewer "
            f"cameras, or switch a playground off.")


def measure_speed(path: Path | None = None, *, samples: int = 3) -> dict:
    """Time the whole job for one picture - open, decode, count - and save the result.

    Processor time rather than clock time, because waiting for a camera costs the computer
    nothing while decoding and counting cost it everything. Uses up to ``samples`` saved
    cameras that answer; if none does, counting alone on a stored picture, plus a margin for
    the decoding that could not be timed.
    """
    import numpy as np

    from pitch_occupancy import cameras

    pipeline = check._pipeline()
    blank = np.zeros((1080, 1920, 3), np.uint8)
    with check.detect_lock:
        pipeline.observe_minute("speed-test", [blank], None)  # warm the model up
    costs: list[float] = []
    for cam in cameras.list_cameras()[: samples * 3]:
        if len(costs) >= samples:
            break
        url = cameras.get_url(cam["venue"], cam["camera_id"])
        cpu = time.process_time()
        frame, _, _ = check._grab(url, check.GRAB_TIMEOUT_S)
        if frame is None:
            continue
        check._count(pipeline, cam["venue"], cam["camera_id"], frame)
        costs.append(time.process_time() - cpu)
    if costs:
        per_picture, how = sorted(costs)[len(costs) // 2], f"{len(costs)} camera(s)"
    else:
        cpu = time.process_time()
        with check.detect_lock:
            pipeline.observe_minute("speed-test", [blank], None)
        per_picture, how = (time.process_time() - cpu) + 0.5, "counting only, no camera answered"
    result = {"seconds_per_picture": round(max(per_picture, 0.02), 3), "how": how,
              "measured_at": store.now().isoformat(timespec="seconds")}
    target = path or SPEED_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
