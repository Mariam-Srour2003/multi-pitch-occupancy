"""The bounds every upload route shares: body size, pixel count, CPU, and clean-up.

The review routes take footage and stills from whoever can reach the server, and each of
them was bounded in one dimension and open in the others. A 25 MB cap on compressed bytes
let a 17 KiB PNG decode to 12000x12000 and take a gigabyte of RAM; a JSON body was buffered
whole before any of its caps applied; a clip analysis ran its hundreds of model passes on the
event loop, so one upload froze the dashboard and every other request for minutes; and an
upload was deleted only when a generator was finalised, which after a client disconnect meant
whenever the garbage collector next ran. These are the shared answers, kept in one place so
the routes cannot drift apart on them.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tempfile
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

__all__ = [
    "MAX_PIXELS", "HEAVY_CONCURRENCY", "UPLOAD_PREFIX", "WORKSPACE_PREFIX",
    "read_body", "read_json", "parse_model", "check_image_pixels", "check_video_pixels",
    "run_heavy", "upload_file", "upload_workspace", "sweep_stale", "CleanupStreamingResponse",
]

LOG = logging.getLogger(__name__)

_MB = 1024 * 1024

#: Pixels in one decoded image or video frame. 50 MP is above any phone or camera this
#: project sees (a 4K frame is 8.3 MP, a 200 MP phone sensor bins to 12.5 MP by default) and
#: is about 150 MB as a BGR array - large but survivable, where OpenCV's own ceiling of 2^30
#: pixels is 3 GiB per image.
MAX_PIXELS: int = 50_000_000

# OpenCV's decoder has its own ceiling, read once on the first decode. Lowered here as well,
# for the formats `check_image_pixels` cannot measure; a value set by the operator wins.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 26))

#: How many model-heavy jobs may run at once across all the review routes. They queue rather
#: than fail: on this machine's CPU two at a time already share the cores, and more would
#: only make each slower while holding more frames in memory.
HEAVY_CONCURRENCY: int = 2

#: Name prefixes for what the review routes put in the temporary directory, so a sweep can
#: recognise its own leftovers and nothing else's.
UPLOAD_PREFIX = "pitch-upload-"
WORKSPACE_PREFIX = "pitch-shots-"

#: Older than this, a leftover from the prefixes above is from a process that was killed: no
#: review request runs for six hours.
STALE_AFTER_S: float = 6 * 3600


def _too_large(what: str, limit: int) -> HTTPException:
    return HTTPException(413, f"{what} exceeds {max(1, limit // _MB)} MB")


def _content_length(request: Request) -> int | None:
    try:
        value = int(request.headers.get("content-length", ""))
    except ValueError:
        return None
    return value if value >= 0 else None


async def read_body(request: Request, max_bytes: int, *, what: str = "request body") -> bytes:
    """The body, refused with 413 as soon as it is known to exceed ``max_bytes``.

    Checked against the declared length first and then while streaming, because a body
    sent without one (chunked) declares nothing. Letting FastAPI parse a JSON body into a
    model buffers all of it before a single cap applies - a 99 MiB batch was refused only
    after the process had grown by 200 MB.
    """
    declared = _content_length(request)
    if declared is not None and declared > max_bytes:
        raise _too_large(what, max_bytes)
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > max_bytes:
            raise _too_large(what, max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


async def read_json(request: Request, max_bytes: int) -> Any:
    """The body as JSON, capped like `read_body`; malformed JSON is a 422, never a 500.

    Parsed in a worker thread: a few hundred megabytes of base64 take long enough to decode
    that doing it on the event loop stalls every other request.
    """
    body = await read_body(request, max_bytes)
    if not body.strip():
        raise HTTPException(422, "the request body is empty; it should be a JSON object")
    try:
        return await anyio.to_thread.run_sync(json.loads, body)
    except (ValueError, RecursionError) as exc:
        raise HTTPException(422, f"the request body is not valid JSON: {exc}") from exc


def parse_model[M: BaseModel](model: type[M], payload: Any) -> M:
    """Validate ``payload`` as ``model``, failing the way a FastAPI body parameter does."""
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors(include_url=False)) from exc


def check_image_pixels(raw: bytes, what: str = "image") -> None:
    """Refuse (413) an encoded image whose header declares more than :data:`MAX_PIXELS`.

    Read from the header by PIL, which decodes nothing, so the refusal costs microseconds
    rather than the gigabyte a full decode of the same file would. Bytes PIL cannot identify
    are let through: whether they are an image at all is OpenCV's decision, as it always was,
    and its own ceiling is lowered above for that case.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, f"{what} has too many pixels to decode safely") from exc
    except Exception:  # noqa: BLE001 - unidentifiable is OpenCV's call, not a failure here
        return
    if width * height > MAX_PIXELS:
        raise HTTPException(
            413, f"{what} is {width}x{height} pixels; the limit is "
                 f"{MAX_PIXELS // 1_000_000} megapixels")


def check_video_pixels(path: Path | str, what: str = "video") -> None:
    """Refuse (413) a video whose frames would each be larger than :data:`MAX_PIXELS`.

    From the container's declared size, before a frame is decoded. A file that will not open
    is let through: the caller already says so in its own words.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if width * height > MAX_PIXELS:
        raise HTTPException(
            413, f"{what} frames are {width}x{height} pixels; the limit is "
                 f"{MAX_PIXELS // 1_000_000} megapixels")


_LIMITERS: weakref.WeakKeyDictionary[Any, anyio.CapacityLimiter] = weakref.WeakKeyDictionary()
_LIMITERS_LOCK = threading.Lock()


def _limiter() -> anyio.CapacityLimiter:
    """One limiter per event loop. A limiter is bound to the loop it was made on, and the
    test client runs a fresh loop per client, so a single module-level one would break the
    second time it was used."""
    loop = asyncio.get_running_loop()
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(loop)
        if limiter is None:
            limiter = _LIMITERS[loop] = anyio.CapacityLimiter(HEAVY_CONCURRENCY)
    return limiter


async def run_heavy[T](fn: Callable[..., T], *args: Any) -> T:
    """Run ``fn(*args)`` in a worker thread, at most :data:`HEAVY_CONCURRENCY` at a time.

    For anything that decodes video, runs a model or computes a median of frames. Inside an
    ``async def`` route that work used to run on the event loop itself: with a stub model at
    0.4 s a frame, `GET /api/v1/roi` went from 0.03 s to 4.1 s while one clip was analysed.
    The body is read on the loop first, since that is I/O; only the computation moves.
    """
    return await anyio.to_thread.run_sync(fn, *args, limiter=_limiter())


_SWEPT = False
_SWEPT_LOCK = threading.Lock()


def sweep_stale(folder: Path | None = None, *, max_age_s: float = STALE_AFTER_S) -> int:
    """Remove this project's upload leftovers older than ``max_age_s``. Returns how many.

    The routes delete their uploads in ``finally`` blocks, which a killed process never
    reaches - and footage of people left in the temporary directory stays there. Only names
    with this module's prefixes are touched, and only old ones, so a concurrent request's
    upload is never removed from under it.
    """
    import shutil

    folder = Path(folder or tempfile.gettempdir())
    cutoff = time.time() - max_age_s
    removed = 0
    for prefix in (UPLOAD_PREFIX, WORKSPACE_PREFIX):
        for entry in folder.glob(f"{prefix}*"):
            try:
                if entry.stat().st_mtime > cutoff:
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def _sweep_once() -> None:
    """`sweep_stale`, once per process, on the first upload - startup has no hook here."""
    global _SWEPT
    with _SWEPT_LOCK:
        if _SWEPT:
            return
        _SWEPT = True
    try:
        sweep_stale()
    except Exception:  # noqa: BLE001 - housekeeping must never fail an upload
        LOG.exception("sweeping stale uploads failed")


def upload_file() -> tuple[int, Path]:
    """A new temporary file for an upload, as ``(fd, path)``, named so a sweep knows it."""
    _sweep_once()
    fd, name = tempfile.mkstemp(prefix=UPLOAD_PREFIX, suffix=".upload")
    return fd, Path(name)


def upload_workspace() -> Path:
    """A new temporary directory for a batch of uploaded images, named so a sweep knows it."""
    _sweep_once()
    return Path(tempfile.mkdtemp(prefix=WORKSPACE_PREFIX))


class CleanupStreamingResponse(StreamingResponse):
    """A streamed response over a sync generator that releases its files however it ends.

    Two things the plain `StreamingResponse` does not do.

    **Clean-up that does not wait for the garbage collector.** The routes deleted their
    uploads in the generator's ``finally``. When the client disconnects mid-stream the
    response stops iterating and leaves the generator suspended, and its ``finally`` runs
    only when a GC cycle collects it - measured, 3 to 6 of 30 disconnected uploads were
    still on disk 8 s later. Starlette's ``background`` task is skipped on a disconnect too.
    So the response closes the generator itself, in its own ``finally``, and then runs
    ``cleanup``.

    **Each step runs through `run_heavy`**, so streams share the same concurrency bound as
    every other model-heavy route instead of each taking a thread of its own.
    """

    def __init__(self, content: Iterator[str], *, cleanup: Callable[[], None],
                 media_type: str = "application/x-ndjson") -> None:
        self._source = content
        self._cleanup = cleanup
        self._released = False
        super().__init__(self._pump(), media_type=media_type)

    async def _pump(self):
        done = object()
        while True:
            chunk = await run_heavy(next, self._source, done)
            if chunk is done:
                return
            yield chunk

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.release()

    def release(self) -> None:
        """Close the generator and run ``cleanup``, once. Safe to call more than once."""
        if self._released:
            return
        self._released = True
        try:
            close = getattr(self._source, "close", None)
            if close is not None:
                close()
        except Exception:  # noqa: BLE001 - the files must still go
            LOG.exception("closing a streamed response's generator failed")
        finally:
            try:
                self._cleanup()
            except Exception:  # noqa: BLE001
                LOG.exception("cleaning up after a streamed response failed")
