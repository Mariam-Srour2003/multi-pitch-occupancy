"""The saved playgrounds: ``configs/playgrounds.json`` (gitignored), read, validated, written.

Every write goes through :func:`save_playground`, :func:`set_enabled` or
:func:`delete_playground`; each refuses settings the computer could not keep up with
(`capacity.check_capacity`) and wakes the runner so a change takes effect at once.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime
from datetime import time as dtime
from pathlib import Path

from pitch_occupancy.config import CONFIGS_DIR
from pitch_occupancy.jsonstore import read_json, write_json_atomic
from pitch_occupancy.playgrounds import capacity

__all__ = ["CONFIG_PATH", "EVERY_CHOICES", "COUNTING", "COUNTING_SHORT", "load",
           "save_playground", "set_enabled", "delete_playground", "in_hours"]

#: Read at call time, so patching ``store.CONFIG_PATH`` redirects every reader.
CONFIG_PATH: Path = CONFIGS_DIR / "playgrounds.json"

#: How often a playground may be checked, in seconds.
EVERY_CHOICES: tuple[int, ...] = (5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1800, 3600)
#: How a playground's cameras are combined. See the package docstring.
COUNTING: dict[str, str] = {
    "add": "Cameras see different parts - add their counts",
    "largest": "Cameras see the same area - use the biggest count",
    "average": "Cameras see the same area - use the average count",
    "vote": "Each camera decides - playing when at least half say playing",
}
#: Short names of the counting modes, for tables.
COUNTING_SHORT: dict[str, str] = {
    "add": "counts added", "largest": "biggest count", "average": "average count",
    "vote": "cameras vote",
}

_lock = threading.RLock()
#: Set after every change to the saved playgrounds; the runner waits on it between rounds.
changed = threading.Event()


def now() -> datetime:
    """The local time, aware. One function so a test can move the clock."""
    return datetime.now().astimezone()


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "playground"


def _read(path: Path | None = None) -> list[dict]:
    try:
        raw = read_json(path or CONFIG_PATH)
    except (OSError, ValueError):
        return []
    items = raw.get("playgrounds") if isinstance(raw, dict) else None
    return [p for p in items if isinstance(p, dict)] if isinstance(items, list) else []


def _write(items: list[dict], path: Path | None = None) -> None:
    write_json_atomic(path or CONFIG_PATH, {"playgrounds": items})


def load(path: Path | None = None) -> list[dict]:
    """Every saved playground, in the order they were made."""
    return [_normalise(p) for p in _read(path)]


def _normalise(p: dict) -> dict:
    hours = p.get("hours") if isinstance(p.get("hours"), dict) else None
    return {
        "id": str(p.get("id") or slug(str(p.get("name", "")))),
        "name": str(p.get("name") or p.get("id") or "Playground"),
        "cameras": [str(c) for c in p.get("cameras") or []],
        "every_seconds": int(p.get("every_seconds") or 30),
        "counting": p.get("counting") if p.get("counting") in COUNTING else "add",
        "hours": hours,
        "enabled": bool(p.get("enabled", True)),
        "save_every": max(0, int(p.get("save_every", 5) or 0)),
    }


def _hhmm(value: str, what: str) -> dtime:
    try:
        return dtime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must look like 08:00") from exc


def _validate(p: dict, others: list[dict], saved_cameras: set[str]) -> dict:
    name = str(p.get("name") or "").strip()
    if not name:
        raise ValueError("Give the playground a name.")
    if len(name) > 60:
        raise ValueError("The name is too long (60 characters at most).")
    cams = [str(c) for c in p.get("cameras") or []]
    if not cams:
        raise ValueError("Choose at least one camera.")
    if len(set(cams)) != len(cams):
        raise ValueError("A camera is listed twice.")
    unknown = [c for c in cams if c not in saved_cameras]
    if unknown:
        raise ValueError(f"Unknown camera(s): {', '.join(unknown)}. Add them on the Cameras "
                         f"page first.")
    taken = {c: o["name"] for o in others for c in o["cameras"]}
    clash = [f"{c} (already in {taken[c]})" for c in cams if c in taken]
    if clash:
        raise ValueError("A camera can belong to one playground only: " + ", ".join(clash))
    every = p.get("every_seconds")
    if isinstance(every, bool) or every not in EVERY_CHOICES:
        raise ValueError("Choose how often from the list.")
    counting = p.get("counting", "add")
    if counting not in COUNTING:
        raise ValueError("Choose how the cameras are combined.")
    hours = p.get("hours")
    if hours:
        start, end = _hhmm(hours.get("from"), "From"), _hhmm(hours.get("to"), "To")
        if start == end:
            raise ValueError("From and To are the same time; choose All day instead.")
        hours = {"from": start.strftime("%H:%M"), "to": end.strftime("%H:%M")}
    else:
        hours = None
    save_every = p.get("save_every", 5)
    if (isinstance(save_every, bool) or not isinstance(save_every, int)
            or not 0 <= save_every <= 1000):
        raise ValueError("Save pictures must be a whole number: 0 (off), 1 (every check), "
                         "5 (one check in five)...")
    return {"name": name, "cameras": cams, "every_seconds": int(every), "counting": counting,
            "hours": hours, "enabled": bool(p.get("enabled", True)), "save_every": save_every}


def _saved_cameras() -> set[str]:
    from pitch_occupancy import cameras

    return {f"{c['venue']}/{c['camera_id']}" for c in cameras.list_cameras()}


def save_playground(values: dict, *, playground_id: str | None = None,
                    path: Path | None = None) -> dict:
    """Add a playground, or replace the one with ``playground_id``. Raises ``ValueError``
    with a sentence for the page - including when the computer could not keep up."""
    with _lock:
        items = load(path)
        others = [p for p in items if p["id"] != playground_id]
        if playground_id is not None and len(others) == len(items):
            raise KeyError(playground_id)
        clean = _validate(values, others, _saved_cameras())
        if playground_id is None:
            base = slug(clean["name"])
            pid, n = base, 2
            while any(p["id"] == pid for p in items):
                pid, n = f"{base}-{n}", n + 1
        else:
            pid = playground_id
        new = {"id": pid, **clean}
        proposed = [new if p["id"] == pid else p for p in items]
        if playground_id is None:
            proposed.append(new)
        capacity.check_capacity(proposed)
        _write(proposed, path)
    changed.set()
    return new


def set_enabled(playground_id: str, enabled: bool, path: Path | None = None) -> dict:
    with _lock:
        items = load(path)
        found = next((p for p in items if p["id"] == playground_id), None)
        if found is None:
            raise KeyError(playground_id)
        found["enabled"] = bool(enabled)
        if enabled:
            capacity.check_capacity(items)
        _write(items, path)
    changed.set()
    return found


def delete_playground(playground_id: str, path: Path | None = None) -> None:
    with _lock:
        items = load(path)
        rest = [p for p in items if p["id"] != playground_id]
        if len(rest) == len(items):
            raise KeyError(playground_id)
        _write(rest, path)
    changed.set()


def in_hours(p: dict, now: datetime) -> bool:
    hours = p.get("hours")
    if not hours:
        return True
    start, end = dtime.fromisoformat(hours["from"]), dtime.fromisoformat(hours["to"])
    t = now.time().replace(tzinfo=None)
    return start <= t < end if start < end else (t >= start or t < end)  # past midnight
