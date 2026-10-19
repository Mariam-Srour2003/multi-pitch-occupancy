"""Saved pictures: drawn copies of checked frames, per playground and day, under a size cap.

The runner saves one check in ``save_every`` (`runner`). The pictures never take more than
:data:`SNAPSHOT_CAP_BYTES`: when a new one would pass it, the oldest are deleted first, so a
long run keeps the most recent pictures instead of filling the disk.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from pitch_occupancy.config import settings
from pitch_occupancy.playgrounds import store

__all__ = ["SNAPSHOT_DIR", "SNAPSHOT_CAP_BYTES", "save_snapshot", "snapshot_usage", "drawn"]

#: Both read at call time, so patching them here redirects every reader.
SNAPSHOT_DIR: Path = settings.snapshot_dir
SNAPSHOT_CAP_BYTES: int = settings.snapshot_cap_mb * 1024 * 1024

_snap_lock = threading.Lock()
#: Running total of the saved pictures' size; ``None`` until first counted from disk.
_snap_bytes: int | None = None


def _snapshot_files() -> list[Path]:
    return [f for f in SNAPSHOT_DIR.rglob("*.jpg") if f.is_file()] if SNAPSHOT_DIR.exists() else []


def snapshot_usage() -> dict:
    """How much the saved pictures take, and the cap."""
    files = _snapshot_files()
    used = sum(f.stat().st_size for f in files)
    return {"files": len(files), "bytes": used, "cap_bytes": SNAPSHOT_CAP_BYTES,
            "folder": str(SNAPSHOT_DIR)}


def drawn(frame, counted):
    """The frame with the count drawn on; the plain frame when drawing fails."""
    try:
        from pitch_occupancy.vision import overlay

        return overlay.draw_verdict(frame, counted["verdict"], instances=counted["instances"],
                                    trace=False)
    except Exception:  # noqa: BLE001 - a picture without the drawing beats no picture
        return frame


def save_snapshot(pid: str, camera: str, when: datetime, frame, counted) -> str | None:
    """Draw the count on the picture and keep it; returns the path relative to SNAPSHOT_DIR."""
    global _snap_bytes
    import cv2

    picture = drawn(frame, counted)
    h, w = picture.shape[:2]
    if w > 1280:
        picture = cv2.resize(picture, (1280, max(1, int(h * 1280 / w))),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return None
    data = buf.tobytes()
    rel = (Path(pid) / when.strftime("%Y-%m-%d")
           / f"{when.strftime('%H%M%S')}_{store.slug(camera)}.jpg")
    with _snap_lock:
        if _snap_bytes is None:
            _snap_bytes = sum(f.stat().st_size for f in _snapshot_files())
        if _snap_bytes + len(data) > SNAPSHOT_CAP_BYTES:
            for f in sorted(_snapshot_files(), key=lambda f: f.stat().st_mtime):
                if _snap_bytes + len(data) <= SNAPSHOT_CAP_BYTES * 0.95:
                    break
                try:
                    size = f.stat().st_size
                    f.unlink()
                    _snap_bytes -= size
                except OSError:
                    pass
        target, n = SNAPSHOT_DIR / rel, 2
        while target.exists():  # two checks in one second: keep both
            target = target.with_name(f"{rel.stem}_{n}.jpg")
            n += 1
        rel = target.relative_to(SNAPSHOT_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        _snap_bytes += len(data)
    return rel.as_posix()
