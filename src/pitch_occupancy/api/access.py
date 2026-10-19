"""A password in front of every page and endpoint, when one is set.

The pages can save camera connections, start a recording schedule and show live pictures of
people on a pitch. That is fine on the Mini-PC's own screen (the server binds to 127.0.0.1 by
default) and not fine on a shared network. So when ``PITCH_ACCESS_PASSWORD`` is set, every
request needs it, through the browser's own sign-in prompt (HTTP Basic). ``/health`` stays open
so a monitor can check the service is up without holding the password.

**Basic authentication sends the password with every request, encoded, not encrypted.** On a
local network behind the facility's router that is the usual trade; anywhere else, put the
service behind HTTPS (a reverse proxy) or a VPN. `docs/runbook.md` and the README say so.

Unset means open, and `pitch serve` refuses to listen beyond this machine while it is unset
unless told ``--no-password`` explicitly - an open camera console on a network should be a
decision somebody typed, not a default.

**`RequestGuard` stops other web pages from driving this one.** Three checks, on every
request:

- *Host.* With no password the service answers only to this machine's own names, so a web
  page that re-points its own domain at 127.0.0.1 (DNS rebinding) cannot read it.
  ``PITCH_ALLOWED_HOSTS`` lists further names when the service is reached by another one.
- *Cross-site writes.* A browser attaches a remembered sign-in to a form another site
  submits. A request that changes something is refused when the browser says it came from a
  different site (``Sec-Fetch-Site``) or its ``Origin`` names another host.
- *Repeated wrong passwords.* After ``MAX_FAILURES`` *different* wrong sign-ins from one
  address within ``FAILURE_WINDOW_S`` seconds, that address is refused for the rest of the
  window. Different, because a forgotten browser tab polling with an old password repeats
  one wrong guess every few seconds; that is not someone guessing, and must not lock the
  address out. Behind a reverse proxy the address is the client's only if the proxy sends
  ``X-Forwarded-For`` (Caddy does); otherwise everyone shares the proxy's.

Every response also carries headers that stop the pages being framed by another site
(clickjacking) and stop browsers guessing content types.
"""

from __future__ import annotations

import base64
import binascii
import secrets
import threading
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from pitch_occupancy.config import settings

__all__ = ["AccessMiddleware", "RequestGuard", "OPEN_PATHS", "LOCAL_HOSTS", "credentials_ok"]

#: Reachable without the password: a liveness probe carries no data.
OPEN_PATHS: frozenset[str] = frozenset({"/health"})

#: Bind addresses that only this machine can reach.
LOCAL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})


def credentials_ok(header: str | None, user: str, password: str) -> bool:
    """Whether an ``Authorization`` header carries ``user``/``password``. Constant-time."""
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    given_user, sep, given_password = decoded.partition(":")
    if not sep:
        return False
    # Both compared, always, so the response time does not say which half was wrong.
    user_ok = secrets.compare_digest(given_user.encode(), user.encode())
    password_ok = secrets.compare_digest(given_password.encode(), password.encode())
    return user_ok and password_ok


#: Wrong sign-ins from one address before it is refused, and the window they count in.
MAX_FAILURES = 10
FAILURE_WINDOW_S = 300.0
#: address -> {hash of the wrong credentials: when last tried}
_failures: dict[str, dict[str, float]] = {}
#: Addresses tracked at most; beyond it the oldest are forgotten first, so a flood of new
#: addresses cannot wipe the record of the one actually guessing.
MAX_TRACKED = 10_000
_failures_lock = threading.Lock()


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _locked_out(address: str, now: float) -> bool:
    with _failures_lock:
        tried = _failures.get(address)
        if not tried:
            return False
        for key in [k for k, t in tried.items() if now - t >= FAILURE_WINDOW_S]:
            del tried[key]
        if not tried:
            _failures.pop(address, None)
            return False
        return len(tried) >= MAX_FAILURES


def _record_failure(address: str, header: str, now: float) -> None:
    import hashlib

    key = hashlib.sha256(header.encode("utf-8", "replace")).hexdigest()[:16]
    with _failures_lock:
        _failures.setdefault(address, {})[key] = now
        if len(_failures) > MAX_TRACKED:
            oldest = sorted(_failures, key=lambda a: max(_failures[a].values(), default=0.0))
            for stale in oldest[: len(_failures) - MAX_TRACKED]:
                del _failures[stale]


class AccessMiddleware(BaseHTTPMiddleware):
    """Ask for the password on every request except `OPEN_PATHS`, when one is configured.

    The setting is read per request rather than at start-up, so a test (or an operator
    restarting with a new environment) is not fighting a value captured at import.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        password = settings.access_password.get_secret_value()
        if not password or request.url.path in OPEN_PATHS:
            return await call_next(request)
        address, now = _client(request), time.monotonic()
        if _locked_out(address, now):
            return PlainTextResponse("Too many wrong sign-ins; try again in a few minutes.",
                                     status_code=429)
        header = request.headers.get("authorization")
        if credentials_ok(header, settings.access_user, password):
            return await call_next(request)
        if header:
            _record_failure(address, header, now)
        return PlainTextResponse(
            "Sign-in required.", status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Pitch Occupancy", charset="UTF-8"'},
        )


#: Methods that change nothing; everything else is a write and is checked for its origin.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}


def _host_name(value: str) -> str:
    """``example.com:8000`` -> ``example.com``; ``[::1]:8000`` -> ``::1``."""
    value = value.strip().lower()
    if value.startswith("["):
        return value[1:].split("]", 1)[0]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def allowed_hosts() -> frozenset[str] | None:
    """Host names this service answers to, or None for any.

    Configured ones always; with no password and none configured, only this machine's own
    names (and ``testserver``, the test client's).
    """
    configured = {h.strip().lower() for h in settings.allowed_hosts.split(",") if h.strip()}
    if configured:
        return frozenset(configured | LOCAL_HOSTS)
    if settings.access_password.get_secret_value():
        return None
    return frozenset(LOCAL_HOSTS | {"testserver"})


class RequestGuard(BaseHTTPMiddleware):
    """Host check, cross-site write check and security headers - see the module docstring."""

    async def dispatch(self, request: Request, call_next) -> Response:
        hosts = allowed_hosts()
        host = _host_name(request.headers.get("host", ""))
        if hosts is not None and host not in hosts:
            return PlainTextResponse(
                "This address is not one the service answers to. Add it to "
                "PITCH_ALLOWED_HOSTS if it should be.", status_code=400)
        if request.method not in SAFE_METHODS and _cross_site(request, host):
            return PlainTextResponse("Refused: the request came from another website.",
                                     status_code=403)
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


def _cross_site(request: Request, host: str) -> bool:
    """Whether a write came from another site - or another port of this machine, which is
    another web server and just as untrusted."""
    fetch_site = request.headers.get("sec-fetch-site", "").lower()
    if fetch_site in ("cross-site", "same-site"):
        return True
    origin = request.headers.get("origin")
    if origin is None:
        return False
    if origin == "null":
        return True
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(origin)
        origin_host = (parts.hostname or "").lower()
        origin_port = parts.port or {"http": 80, "https": 443}.get(parts.scheme)
    except ValueError:
        return True
    raw = request.headers.get("host", "").strip().lower()
    port = None
    tail = raw.rsplit("]", 1)[-1] if raw.startswith("[") else raw
    if ":" in tail:
        try:
            port = int(tail.rsplit(":", 1)[1])
        except ValueError:
            return True
    # A proxy in front may rewrite the port; only refuse when both sides name one.
    if port is None:
        port = {"http": 80, "https": 443}.get(parts.scheme)
    return origin_host != host or (origin_port is not None and port is not None
                                   and origin_port != port)
