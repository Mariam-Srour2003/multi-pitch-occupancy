"""Reading and atomically writing the small JSON files under ``configs/`` and ``data/``.

Each caller keeps its own validation and its own policy for a broken file (degrade to a
default, or refuse to overwrite); this module only does the file handling they share.

A write goes to a temporary file beside the target, is flushed to disk and moved over the
target with :func:`os.replace`, so a reader - another process included - sees the old file or
the new one and never half of either. Windows refuses the move while another process has the
target open, so it is retried briefly.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["JSONFileError", "read_json", "write_json_atomic"]

_RETRY_SLEEP_S = 0.05


class JSONFileError(ValueError):
    """A file that exists but does not hold valid JSON.

    ``reason`` is the parser's own message and ``lineno`` the line it failed on (``None`` when
    the file is not valid UTF-8), so a caller can word its own message.
    """

    def __init__(self, path: Path, reason: str, lineno: int | None = None) -> None:
        where = f" (line {lineno})" if lineno is not None else ""
        super().__init__(f"{path.name} is not valid JSON{where}: {reason}")
        self.path = path
        self.reason = reason
        self.lineno = lineno


def read_json(path: Path, default: Any = None) -> Any:
    """The parsed contents of ``path``; ``default`` if it does not exist.

    A file that cannot be parsed raises :class:`JSONFileError` (a ``ValueError``) naming it.
    Other ``OSError``\\ s propagate.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    except UnicodeDecodeError as exc:
        raise JSONFileError(Path(path), str(exc)) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise JSONFileError(Path(path), str(exc), exc.lineno) from exc


def _backup(target: Path, history_dir: Path, keep: int) -> Path:
    """Copy ``target`` into ``history_dir`` under a timestamped name; keep the newest ``keep``."""
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    backup = history_dir / f"{target.stem}.{stamp}{target.suffix}"
    shutil.copy2(target, backup)
    for old in sorted(history_dir.glob(f"{target.stem}.*{target.suffix}"))[:-keep]:
        old.unlink(missing_ok=True)
    return backup


def write_json_atomic(path: Path, data: Any, *, indent: int | None = 2,
                      trailing_newline: bool = True,
                      default: Callable[[Any], Any] | None = None,
                      history_dir: Path | None = None, keep: int = 20,
                      attempts: int = 20) -> Path | None:
    """Write ``data`` to ``path`` atomically. Returns the backup of the old file, if one was made.

    With ``history_dir``, an existing file is first copied there as
    ``<stem>.<YYYYmmddTHHMMSSffffff><suffix>`` and all but the newest ``keep`` copies are
    removed. ``attempts`` bounds the retries of the final move on ``PermissionError``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=indent, default=default)
    if trailing_newline:
        text += "\n"
    backup = None
    if history_dir is not None and target.exists():
        backup = _backup(target, Path(history_dir), keep)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(max(1, attempts)):
            try:
                os.replace(tmp, target)
                break
            except PermissionError:
                if attempt >= attempts - 1:
                    raise
                time.sleep(_RETRY_SLEEP_S)
    finally:
        tmp.unlink(missing_ok=True)
    return backup
