"""The camera registry, and the one function that tells you why a camera will not connect.

Cameras live in ``configs/cameras.json`` as venue id -> camera id -> URL, the file
`worker.load_camera_urls` already reads for the live run. Until now the only way to fill it
was a text editor, which is fine for a developer and hopeless for the person installing a
camera on a pitch at 07:30. This module is what the `/cameras` page writes through.

Three things it is careful about, all for the same reason - **an RTSP URL usually carries the
password of a camera that films identifiable people**:

* **Nothing leaves this module unmasked unless a caller asks for the raw URL by name.**
  :func:`list_cameras` returns :func:`mask`-ed URLs (``rtsp://admin:***@10.0.0.5:554/...``),
  and :class:`ProbeResult` carries only the masked form. The raw URL is read with
  :func:`get_url`, which exists for the code that opens the stream and nothing else.
* **An edit may leave the password out.** Saving ``rtsp://admin@10.0.0.5/...`` or the masked
  ``rtsp://admin:***@...`` over an existing camera keeps the stored password, so the page never
  has to show it to let someone change the path.
* **Writes are atomic and the previous file is kept**, the same way `api/schedule_editor`
  writes the schedule: a temporary file beside the target moved into place with
  :func:`os.replace`, and a copy of the old file under ``configs/cameras_history/`` (gitignored,
  pruned to the last 20). Keys starting with ``_`` are comments and survive every write.

:func:`probe` answers "why will it not connect" in stages, because the fixes are different:
a TCP connect to host:port first (nothing there / something there but the port is closed),
then opening the stream (reachable, but the path or the credentials are wrong), then one frame
(opened, but nothing decodable arrived - often H.265 or a stalled stream). Both the socket
connect and the capture opener are parameters, so every stage is tested without a camera.

:func:`open_capture` is shared with `frame_source.RTSPSource`, which is how ``usb:0`` (a
local webcam, for a bench test) works in the live worker as well as on the page.

Two answers for the morning check sit here as well. :func:`probe_all` tries every saved camera
one after another with a short timeout and returns no pictures, so a page of six cameras
answers in seconds. :func:`recording_health` says when the live run last recorded a picture
from each camera, read from the database's ``frame_samples`` - a camera that answers the test
but has recorded nothing since Tuesday is the failure the test alone cannot show.
"""

from __future__ import annotations

import base64
import os
import re
import socket
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from pitch_occupancy.config import CONFIGS_DIR
from pitch_occupancy.jsonstore import JSONFileError, read_json, write_json_atomic

__all__ = [
    "CONFIG_PATH", "ProbeResult", "STAGES", "list_cameras", "get_url", "save_camera",
    "delete_camera", "mask", "validate_id", "validate_url", "merge_password", "open_capture",
    "probe", "snapshot", "jpeg_base64", "history_dir", "probe_all", "recording_health",
]

#: The file `worker.load_camera_urls` reads. A module attribute so a test can point it at a
#: temporary path; every function also takes ``path=``.
CONFIG_PATH: Path = CONFIGS_DIR / "cameras.json"

#: Previous versions kept under ``cameras_history/``. They hold credentials, which is why the
#: folder is gitignored, and why it is bounded.
KEEP_VERSIONS = 20

#: Letters, digits, ``_ . -``; must not start with ``_`` (those keys are comments).
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_USB_RE = re.compile(r"^usb:(\d{1,2})$")
_SCHEMES = ("rtsp", "rtsps", "http", "https")
_DEFAULT_PORTS = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443}
#: Query parameters some cameras take credentials in (``?user=admin&password=...``).
_SECRET_PARAMS = {"password", "pass", "passwd", "pwd", "psw", "token", "key", "auth"}
_MASK = "***"

_NEW_FILE_COMMENT = [
    "Camera URLs for the live path: venue id -> camera id -> URL. Written by the /cameras page.",
    "Gitignored on purpose: an RTSP URL usually carries the password of a camera that films",
    "identifiable people. Keys starting with '_' are comments and are kept on every save.",
]


# --- ids and URLs -------------------------------------------------------------------------

def validate_id(value: str, what: str = "id") -> str:
    """Return ``value`` stripped, or raise with a sentence a person can act on."""
    clean = (value or "").strip()
    if not _ID_RE.match(clean):
        raise ValueError(
            f"{what} {clean!r} is not allowed: use letters, digits, '_', '.' or '-' "
            f"(up to 64), not starting with '_' or a symbol"
        )
    return clean


def _split(url: str) -> tuple[str, str | None, str | None, str] | None:
    """``(scheme, user, password, rest)`` for ``scheme://[user[:password]@]rest``.

    Split by hand at the **last** ``@`` rather than by `urlsplit`, because a password that
    was not %-encoded may contain ``@`` or ``/``: `urlsplit` would then put half the password
    in the host or the path - and :func:`mask` would print it. Erring this way can only hide
    too much (an ``@`` in a path, which camera URLs do not have), never too little.
    """
    scheme, sep, rest = url.strip().partition("://")
    if not sep or not scheme:
        return None
    if "@" not in rest:
        return scheme, None, None, rest
    userinfo, _, rest = rest.rpartition("@")
    user, colon, password = userinfo.partition(":")
    return scheme, user, (password if colon else None), rest


def _join(scheme: str, user: str | None, password: str | None, rest: str) -> str:
    if user is None:
        return f"{scheme}://{rest}"
    return f"{scheme}://{user}{':' + password if password is not None else ''}@{rest}"


def _query_items(rest: str) -> tuple[str, list[tuple[str, str | None]]]:
    """``rest`` split into everything before ``?`` and raw ``(key, value)`` pairs after it.

    Raw, not decoded: re-encoding a query would change how a stored password is written.
    """
    base, q, query = rest.partition("?")
    if not q:
        return rest, []
    items = []
    for part in query.split("&"):
        key, eq, value = part.partition("=")
        items.append((key, value if eq else None))
    return base, items


def _rebuild(base: str, items: list[tuple[str, str | None]]) -> str:
    if not items:
        return base
    return base + "?" + "&".join(k if v is None else f"{k}={v}" for k, v in items)


def validate_url(url: str) -> str:
    """Return the URL stripped if its scheme and host are usable, else raise ``ValueError``."""
    clean = (url or "").strip()
    if not clean:
        raise ValueError("the camera address is empty")
    if any(c.isspace() for c in clean):
        raise ValueError("the camera address contains a space; copy it again without one")
    if clean.lower().startswith("usb"):
        if not _USB_RE.match(clean.lower()):
            raise ValueError("a local webcam is written usb:N, for example usb:0")
        return clean.lower()
    split = _split(clean)
    scheme = split[0].lower() if split else ""
    if scheme not in _SCHEMES:
        raise ValueError(
            "the address must start with rtsp://, rtsps://, http://, https:// or be usb:N")
    _, user, password, rest = split
    if user is not None and any(c in (user + (password or "")) for c in "/?#"):
        raise ValueError(
            "the user name or password contains / ? or #, which must be written %-encoded "
            "(/ as %2F, ? as %3F, # as %23) or the camera will misread the address")
    if (password or "").count("@") or (user or "").count("@"):
        raise ValueError("the password contains @, which must be written %40")
    host, port = _host_port(clean)
    if not host:
        raise ValueError("the address has no camera IP or host name")
    if port is None or not 0 < port < 65536:
        raise ValueError("the port in the address is not a valid number")
    return clean


def mask(url: str) -> str:
    """The URL with any password (and secret-looking query values) replaced by ``***``.

    Never raises, and never returns an address it could not parse - that is shown as ``***``
    rather than risked.
    """
    if not url:
        return ""
    text = url.strip()
    if _USB_RE.match(text.lower()):
        return text.lower()
    split = _split(text)
    if split is None:
        return _MASK
    scheme, user, password, rest = split
    base, items = _query_items(rest)
    items = [(k, _MASK if k.lower() in _SECRET_PARAMS and v else v) for k, v in items]
    return _join(scheme, user, _MASK if password else None, _rebuild(base, items))


def merge_password(new_url: str, stored_url: str | None) -> str:
    """Fill a blank or masked password in ``new_url`` from ``stored_url``.

    Lets the page edit a camera without ever showing its password: the form holds the masked
    URL, the person changes the path, and the stored password is kept - provided the user name
    **and the scheme, host and port** are unchanged. Applies to secret query parameters too.
    Raises when a password is left as ``***`` with nothing stored to keep: saving the literal
    ``***`` would be worse than refusing.

    The address check is what keeps the mask meaningful: without it, anyone who could see the
    masked address could point it at their own server, leave ``***`` in place and have this
    fill in the real password for them to read off the wire.
    """
    text = new_url.strip()
    if _USB_RE.match(text.lower()):
        return text.lower()
    split = _split(text)
    if split is None:
        return text
    scheme, user, password, rest = split
    old_user = old_password = None
    old_items: dict[str, str | None] = {}
    if (stored_url and (old := _split(stored_url)) is not None
            and _same_endpoint(text, stored_url)):
        _, old_user, old_password, old_rest = old
        old_items = dict(_query_items(old_rest)[1])

    if user is not None and password in (None, "", _MASK):
        if old_password and old_user == user:
            password = old_password
        elif password == _MASK:
            raise ValueError(
                "the password is hidden (***) and there is no saved password for this user "
                "name and address to keep; type the password")
        else:
            password = password or None
    base, items = _query_items(rest)
    merged = []
    for k, v in items:
        if k.lower() in _SECRET_PARAMS and v in ("", _MASK):
            if old_items.get(k):
                v = old_items[k]
            elif v == _MASK:
                raise ValueError(f"the {k} value is hidden (***) and nothing is saved to keep; "
                                 f"type it")
        merged.append((k, v))
    return _join(scheme, user, password, _rebuild(base, merged))


def _same_endpoint(a: str, b: str) -> bool:
    """Whether two camera addresses reach the same scheme, host and port."""
    sa, sb = _split(a), _split(b)
    if sa is None or sb is None or sa[0].lower() != sb[0].lower():
        return False
    host_a, port_a = _host_port(a)
    host_b, port_b = _host_port(b)
    return bool(host_a) and (host_a or "").lower() == (host_b or "").lower() and port_a == port_b


def _host_port(url: str) -> tuple[str | None, int | None]:
    split = _split(url)
    if split is None:
        return None, None
    scheme, _, _, rest = split
    try:
        parts = urlsplit(f"//{rest}")
        host = parts.hostname
    except ValueError:
        return None, None
    try:
        port = parts.port
    except ValueError:  # not a number, or out of range: reported as an invalid port
        port = -1
    return (unquote(host) if host else None), (port or _DEFAULT_PORTS.get(scheme.lower()))


# --- the registry -------------------------------------------------------------------------

def _path(path: Path | None) -> Path:
    return path or CONFIG_PATH


def history_dir(path: Path | None = None) -> Path:
    return _path(path).parent / "cameras_history"


def _read(path: Path | None = None) -> dict:
    target = _path(path)
    try:
        raw = read_json(target, {})
    except JSONFileError as exc:
        # Refuse rather than start afresh: overwriting a file someone hand-edited badly
        # would delete every camera it held.
        where = f" (line {exc.lineno})" if exc.lineno is not None else ""
        raise ValueError(f"{target.name} is not valid JSON{where}; fix or "
                         f"restore it from {history_dir(path).name}/ before saving") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{target.name} must hold an object of venues")
    return raw


def _write(data: dict, path: Path | None = None) -> Path | None:
    """Back up the current file, then replace it atomically. Returns the backup, if any."""
    return write_json_atomic(_path(path), data, history_dir=history_dir(path),
                             keep=KEEP_VERSIONS)


def _kind(url: str) -> str:
    if _USB_RE.match(url.lower()):
        return "usb"
    return urlsplit(url).scheme.lower() or "?"


def list_cameras(path: Path | None = None) -> list[dict]:
    """Every configured camera as ``{venue, camera_id, url, kind}`` - **the URL masked**."""
    out = []
    for venue, cams in sorted(_read(path).items()):
        if venue.startswith("_") or not isinstance(cams, dict):
            continue
        for cam, url in sorted(cams.items()):
            if cam.startswith("_") or not isinstance(url, str):
                continue
            out.append({"venue": venue, "camera_id": cam, "url": mask(url),
                        "kind": _kind(url)})
    return out


def get_url(venue: str, camera_id: str, path: Path | None = None) -> str:
    """The **raw** URL of one camera, for opening it. Never return this from an API."""
    url = _read(path).get(venue, {}).get(camera_id)
    if not isinstance(url, str) or venue.startswith("_") or camera_id.startswith("_"):
        raise KeyError(f"no camera {camera_id!r} at venue {venue!r}")
    return url


def venue_urls(venue: str, path: Path | None = None) -> dict[str, str]:
    """Raw URLs of one venue's cameras, comments dropped."""
    cams = _read(path).get(venue)
    if venue.startswith("_") or not isinstance(cams, dict):
        return {}
    return {c: u for c, u in cams.items() if not c.startswith("_") and isinstance(u, str)}


def save_camera(venue: str, camera_id: str, url: str, path: Path | None = None) -> dict:
    """Add or replace one camera. A blank or ``***`` password keeps the stored one.

    Returns the saved entry with the URL masked.
    """
    venue = validate_id(venue, "venue id")
    camera_id = validate_id(camera_id, "camera id")
    data = _read(path)
    current = data.get(venue)
    if current is not None and not isinstance(current, dict):
        raise ValueError(f"venue {venue!r} in the camera file is not a list of cameras")
    stored = (current or {}).get(camera_id)
    clean = validate_url(merge_password(validate_url(url), stored))
    if not data:
        data = {"_comment": list(_NEW_FILE_COMMENT)}
    data.setdefault(venue, {})[camera_id] = clean
    _write(data, path)
    return {"venue": venue, "camera_id": camera_id, "url": mask(clean), "kind": _kind(clean)}


def delete_camera(venue: str, camera_id: str, path: Path | None = None) -> None:
    """Remove one camera; a venue left with no cameras (and no comments) goes too."""
    data = _read(path)
    cams = data.get(venue)
    if (venue.startswith("_") or camera_id.startswith("_") or not isinstance(cams, dict)
            or camera_id not in cams):
        raise KeyError(f"no camera {camera_id!r} at venue {venue!r}")
    del cams[camera_id]
    if not cams:
        del data[venue]
    _write(data, path)


# --- opening and testing ------------------------------------------------------------------

def open_capture(url: str, timeout_ms: int = 5000):
    """An OpenCV capture for ``url``: ``usb:N`` is local webcam N, anything else goes
    through FFMPEG with open and read timeouts (so a dead camera costs seconds, not minutes).
    """
    match = _USB_RE.match(url.strip().lower())
    if match:
        return cv2.VideoCapture(int(match.group(1)))
    # RTSP over TCP unless the operator chose otherwise. FFMPEG's default is UDP, and on the
    # first real camera (2026-10-16, a 57 ms Wi-Fi link) UDP lost enough packets that frames
    # arrived half grey and the stream managed 5 fps, against a clean 26 fps over TCP. A
    # frame with its bottom half missing is not a frame the detector can count people in.
    # OpenCV reads this variable at every open, so it applies to the capture below.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    # Passed to the constructor, not set afterwards: the constructor *is* the open, so an
    # open timeout set on the returned capture applies to nothing (`set` returns False) and
    # a dead camera cost FFMPEG's ~30 s default per attempt instead of ``timeout_ms``.
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
    ])


def _tcp_connect(host: str, port: int, timeout_s: float) -> None:
    with socket.create_connection((host, port), timeout=timeout_s):
        pass


#: The stages :func:`probe` can stop at, in order, with the fix for each. The page's
#: troubleshooting table and `docs/cameras.md` are written from these.
STAGES: dict[str, tuple[str, str]] = {
    "address": (
        "The address is not usable",
        "Check the address is written like rtsp://user:password@192.168.1.64:554/... "
        "and that the host name (if not an IP) can be looked up from this computer."),
    "unreachable": (
        "Nothing answered at that address",
        "The camera is off, the IP is wrong, it is on another network, or a firewall drops "
        "the traffic. Ping the IP from this computer, check the cable/PoE light, and check "
        "the camera's IP in its app or the NVR."),
    "port_closed": (
        "The camera answered but the video port is closed",
        "Turn on RTSP in the camera's settings (Network > Advanced > RTSP or Integration "
        "Protocol), and check the port - 554 is the usual one."),
    "stream": (
        "The camera is reachable but the video would not open",
        "Usually the path or the user name/password is wrong. Check the URL pattern for "
        "your brand, check the password in the camera's own web page, and make sure that "
        "user is allowed to view live video. Special characters in a password must be "
        "written %-encoded (@ as %40)."),
    "frame": (
        "The video opened but no picture arrived",
        "Try the sub-stream, switch the camera's encoding from H.265 to H.264, lower the "
        "resolution, or check the network is not overloaded. A camera that is still "
        "booting can also do this - try again in a minute."),
    "ok": ("Connected - a picture was received", ""),
}


@dataclass
class ProbeResult:
    """What one connection test found. ``url`` is masked; ``frame`` is not serialised."""

    ok: bool
    stage: str
    message: str
    hint: str
    elapsed_ms: int
    url: str
    host: str | None = None
    port: int | None = None
    width: int | None = None
    height: int | None = None
    #: Where each step got to, in plain words, for the page's detail line.
    steps: list[str] = field(default_factory=list)
    frame: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop("frame", None)
        return out


def probe(
    url: str,
    *,
    timeout_ms: int = 5000,
    open_capture: Callable[[str, int], object] | None = None,
    connect: Callable[[str, int, float], None] | None = None,
) -> ProbeResult:
    """Try one camera in stages and say where it stopped and what to do about it.

    1. **address** - the URL parses, has a supported scheme and a host;
    2. **network** - a TCP connect to host:port (default 554 for RTSP, 80/443 for HTTP);
       skipped for ``usb:N``. A refusal is ``port_closed``; a timeout, an unreachable route
       or a name that does not resolve is ``unreachable`` (or ``address`` for the name);
    3. **stream** - the capture opens (wrong path or credentials fail here);
    4. **frame** - one frame is read.

    Exceptions from the opener are reported by type only: OpenCV's messages can include the
    URL, and the URL can include the password.
    """
    started = time.perf_counter()
    opener = open_capture or globals()["open_capture"]
    tcp = connect or _tcp_connect
    masked = mask(url)
    steps: list[str] = []

    def done(stage: str, *, ok: bool = False, detail: str = "", **extra) -> ProbeResult:
        message, hint = STAGES[stage]
        if detail:
            message = f"{message} ({detail})"
        return ProbeResult(ok=ok, stage=stage, message=message, hint=hint,
                           elapsed_ms=int((time.perf_counter() - started) * 1000),
                           url=masked, steps=steps, **extra)

    try:
        clean = validate_url(url)
    except ValueError as exc:
        return done("address", detail=str(exc))

    host = port = None
    if not _USB_RE.match(clean):
        host, port = _host_port(clean)
        if not host or not port or port < 0:
            return done("address", detail="no host or port")
        try:
            tcp(host, port, max(0.5, timeout_ms / 1000))
        except ConnectionRefusedError:
            steps.append(f"{host} answered, port {port} refused")
            return done("port_closed", host=host, port=port)
        except socket.gaierror:
            steps.append(f"the name {host} could not be looked up")
            return done("address", detail=f"{host} could not be looked up", host=host,
                        port=port)
        except TimeoutError:
            steps.append(f"no reply from {host}:{port} within {timeout_ms} ms")
            return done("unreachable", detail="timed out", host=host, port=port)
        except OSError as exc:
            steps.append(f"{host}:{port} could not be reached ({type(exc).__name__})")
            return done("unreachable", host=host, port=port)
        steps.append(f"{host}:{port} is reachable")
    else:
        steps.append(f"local webcam {clean}")

    try:
        capture = opener(clean, timeout_ms)
    except Exception as exc:  # noqa: BLE001 - reported by type, never by message
        steps.append(f"opening failed ({type(exc).__name__})")
        return done("stream", host=host, port=port)
    try:
        opened = bool(getattr(capture, "isOpened", lambda: True)())
        if not opened:
            steps.append("the video stream did not open")
            return done("stream", host=host, port=port)
        steps.append("the video stream opened")
        try:
            ok, image = capture.read()
            # A half-drawn first frame (stream stalled mid-picture) is not the picture to
            # show: read on for a whole one, as the recorder does (`frame_source`).
            from pitch_occupancy.frame_source import _read_whole, is_incomplete

            if ok and image is not None and is_incomplete(image):
                get = getattr(capture, "get", None)
                fps = float(get(cv2.CAP_PROP_FPS) or 25.0) if get is not None else 25.0
                ok, image = _read_whole(capture, fps)
        except Exception as exc:  # noqa: BLE001
            steps.append(f"reading failed ({type(exc).__name__})")
            return done("frame", host=host, port=port)
        if not ok or image is None or getattr(image, "size", 0) == 0:
            steps.append("no picture arrived")
            return done("frame", host=host, port=port)
    finally:
        release = getattr(capture, "release", None)
        if release is not None:
            release()

    height, width = image.shape[:2]
    steps.append(f"picture received, {width}x{height}")
    return done("ok", ok=True, host=host, port=port, width=int(width), height=int(height),
                frame=image)


def snapshot(url: str, *, timeout_ms: int = 5000,
             open_capture: Callable[[str, int], object] | None = None) -> np.ndarray | None:
    """One frame from ``url``, or ``None`` - never a placeholder image."""
    return probe(url, timeout_ms=timeout_ms, open_capture=open_capture).frame


def jpeg_base64(image_bgr: np.ndarray | None, *, max_width: int = 640,
                quality: int = 80) -> str | None:
    """A frame as base64 JPEG, downscaled to ``max_width``, for an ``<img src=data:...>``."""
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return None
    height, width = image_bgr.shape[:2]
    if width > max_width:
        scale = max_width / width
        image_bgr = cv2.resize(image_bgr, (max_width, max(1, int(height * scale))),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


# --- the daily check ----------------------------------------------------------------------

def probe_all(path: Path | None = None, *, timeout_ms: int = 3000,
              open_capture: Callable[[str, int], object] | None = None,
              connect: Callable[[str, int, float], None] | None = None) -> list[dict]:
    """:func:`probe` every saved camera, one after another, **URLs masked, no pictures**.

    One after another rather than in parallel: two cameras on one NVR, or a slow PoE switch,
    answer worse when hit at once, and a daily check is not in a hurry. With the short timeout
    a dead camera costs about ``2 x timeout_ms`` (connect, then open), so six cameras with one
    dead take well under half a minute.
    """
    out = []
    for venue, cams in sorted(_read(path).items()):
        if venue.startswith("_") or not isinstance(cams, dict):
            continue
        for cam, url in sorted(cams.items()):
            if cam.startswith("_") or not isinstance(url, str):
                continue
            r = probe(url, timeout_ms=timeout_ms, open_capture=open_capture, connect=connect)
            out.append({"venue": venue, "camera_id": cam, "url": r.url, "ok": r.ok,
                        "stage": r.stage, "message": r.message, "hint": r.hint,
                        "elapsed_ms": r.elapsed_ms, "width": r.width, "height": r.height})
    return out


def _local(stamp: str | None) -> datetime | None:
    """An ISO timestamp as a naive local time. The worker writes UTC with an offset; a value
    without one (older or simulated rows) is taken as local already."""
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(stamp).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone().replace(tzinfo=None) if parsed.tzinfo else parsed


def _span(slot_date: str, start: str, end: str) -> tuple[datetime, datetime] | None:
    try:
        day = date.fromisoformat(slot_date)
        s0 = datetime.combine(day, datetime.strptime(start, "%H:%M").time())
        s1 = datetime.combine(day, datetime.strptime(end, "%H:%M").time())
    except (TypeError, ValueError):
        return None
    if s1 <= s0:
        s1 += timedelta(days=1)
    return s0, s1


def recording_health(db_path: Path, cams: Iterable[tuple[str, str]], *,
                     now: datetime | None = None) -> dict[tuple[str, str], dict]:
    """When each ``(venue, camera_id)`` last had a picture recorded, and the last 24 hours.

    Read from ``frame_samples`` (one row per camera per recorded minute). Per camera:

    * ``last_recorded`` - the latest ``captured_at`` as local time, or ``None`` ("never");
    * ``minutes_since`` - how long ago that was;
    * ``captured_24h`` - recorded minutes in the last 24 hours;
    * ``expected_24h`` - minutes of the camera's pitch's slots in ``rental_slots`` that fall
      inside the last 24 hours, or ``None`` when the database does not say which pitch the
      camera belongs to. Only slots the live run reached are in that table (a slot is written
      when it is recorded), so a day the recorder was not running at all shows nothing
      expected rather than everything missed - the "last picture" age is what shows that.

    **Matched on the camera id exactly, never guessed.** The database knows cameras by id
    alone; when it also records a venue for that id and the venue differs from the registry's,
    the rows belong to another camera of the same name and this one is "never recorded", with
    a ``note`` saying so. A database that does not exist yet means never recorded for every
    camera, not an error. Opened read-only, so looking never creates or locks the file.
    """
    wanted = list(dict.fromkeys(cams))
    empty = {"last_recorded": None, "minutes_since": None, "captured_24h": 0,
             "expected_24h": None, "note": None}
    out = {key: dict(empty) for key in wanted}
    db = Path(db_path)
    if not wanted or not db.is_file():
        return out
    now = now or datetime.now()
    since = now - timedelta(hours=24)
    # Text bound for the index-free scan: two days back covers any timezone offset.
    floor = (since - timedelta(days=2)).date().isoformat()
    try:
        from pitch_occupancy.db.schema import open_readonly

        conn = open_readonly(db)
    except (sqlite3.Error, FileNotFoundError):
        return out
    try:
        for venue, cam in wanted:
            row = out[(venue, cam)]
            try:
                link = conn.execute(
                    "SELECT c.field_id, f.venue_id FROM cameras c "
                    "LEFT JOIN fields f ON f.field_id = c.field_id WHERE c.camera_id = ?",
                    (cam,)).fetchone()
            except sqlite3.Error:
                link = None
            field_id, db_venue = (link[0], link[1]) if link else (None, None)
            if db_venue and db_venue != venue:
                row["note"] = (f"the database has a camera called {cam} at {db_venue}, not at "
                               f"{venue}; its pictures are not counted as this camera's")
                continue
            try:
                last = conn.execute("SELECT MAX(captured_at) FROM frame_samples "
                                    "WHERE camera_id = ?", (cam,)).fetchone()[0]
                recent = [r[0] for r in conn.execute(
                    "SELECT captured_at FROM frame_samples WHERE camera_id = ? "
                    "AND captured_at >= ?", (cam, floor))]
            except sqlite3.Error:
                return out  # no such table yet: nothing recorded
            when = _local(last)
            if when is not None:
                row["last_recorded"] = when.isoformat(timespec="seconds")
                row["minutes_since"] = max(0, int((now - when).total_seconds() // 60))
            row["captured_24h"] = sum(1 for t in recent
                                      if (lt := _local(t)) is not None and since <= lt <= now)
            if field_id:
                expected = 0.0
                try:
                    slots = conn.execute(
                        "SELECT slot_date, start_time, end_time FROM rental_slots "
                        "WHERE field_id = ? AND slot_date >= ?", (field_id, floor)).fetchall()
                except sqlite3.Error:
                    slots = []
                for slot_date, start, end in slots:
                    span = _span(slot_date, start, end)
                    if span is None:
                        continue
                    overlap = (min(span[1], now) - max(span[0], since)).total_seconds()
                    expected += max(0.0, overlap) / 60
                row["expected_24h"] = int(round(expected))
    finally:
        conn.close()
    return out
