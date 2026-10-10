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
"""

from __future__ import annotations

import base64
import binascii
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from pitch_occupancy.config import settings

__all__ = ["AccessMiddleware", "OPEN_PATHS", "LOCAL_HOSTS", "credentials_ok"]

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


class AccessMiddleware(BaseHTTPMiddleware):
    """Ask for the password on every request except `OPEN_PATHS`, when one is configured.

    The setting is read per request rather than at start-up, so a test (or an operator
    restarting with a new environment) is not fighting a value captured at import.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        password = settings.access_password.get_secret_value()
        if not password or request.url.path in OPEN_PATHS:
            return await call_next(request)
        if credentials_ok(request.headers.get("authorization"), settings.access_user,
                          password):
            return await call_next(request)
        return PlainTextResponse(
            "Sign-in required.", status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Pitch Occupancy", charset="UTF-8"'},
        )
