"""The "Getting started" checklist: is this machine set up to record and judge slots?

A new installation fails quietly. With no cameras the worker has nothing to open, with no
schedule it never wakes, with no detector weights every minute is "not sure" - and the
dashboard just looks empty, which reads as "nothing happened" rather than "nothing is set up".
:func:`checklist` turns that into eight plain steps, each worked out from the real
configuration on this machine, so the dashboard can say what is still missing and where to fix
it.

Three things are deliberate.

**A check never raises.** Every step is computed inside its own guard: a hand-edited
``cameras.json`` that is no longer valid JSON shows as a *warning* step carrying the error
message, never as a 500 on the dashboard. The checklist is most needed exactly when something
is broken.

**Nothing heavy, nothing on the network.** The detector step looks for the weights file on
disk; it does not import the model or run it. The cameras step reads the registry; it does
not connect to a camera. The page can load this on every visit.

**Optional is not the same as to do.** Booking records and alerts make the product more
useful but are not needed to record. The sign-in password is required only once the server is
opened to the network (``settings.api_host`` not a local address). Only required steps decide
whether the card collapses itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pitch_occupancy.config import CONFIGS_DIR, settings

__all__ = ["Step", "checklist", "summary", "ALERTS_PATH", "STATES"]

#: The alerts configuration. Another part of the product writes it; here only its presence is
#: checked.
ALERTS_PATH: Path = CONFIGS_DIR / "alerts.json"

#: Every state a step can be in. ``unknown`` is for a check that cannot be made on this
#: installation (the recording program's status file support is missing).
STATES = ("done", "todo", "optional", "warning", "unknown")

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True, slots=True)
class Step:
    """One line of the checklist.

    ``link`` is a page of this site to fix it on, or None; ``hint`` is a command or a place in
    the documentation, for steps that are fixed outside the browser.
    """

    key: str
    title: str
    state: str
    detail: str
    required: bool
    link: str | None = None
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _cameras() -> list[dict]:
    from pitch_occupancy import cameras

    return cameras.list_cameras()


def _listing(items: list[str], limit: int = 3) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


# --- the steps ------------------------------------------------------------------------------


def _step_cameras() -> Step:
    rows = _cameras()
    venues = {r["venue"] for r in rows}
    if not rows:
        return Step("cameras", "Cameras added", "todo",
                    "No cameras yet. Add each camera's address on the Cameras page.",
                    True, "/cameras")
    return Step("cameras", "Cameras added", "done",
                f"{_plural(len(rows), 'camera')} for {_plural(len(venues), 'venue')}.",
                True, "/cameras")


def _step_boundaries() -> Step:
    from pitch_occupancy.vision import roi

    title = "Every camera has a pitch boundary"
    rows = _cameras()
    if not rows:
        return Step("boundaries", title, "todo", "Add cameras first, then draw a boundary "
                    "for each.", True, "/roi")
    missing = [r["camera_id"] if len({x["venue"] for x in rows}) == 1
               else f"{r['camera_id']} ({r['venue']})"
               for r in rows if roi.resolve_with_key(r["camera_id"], venue=r["venue"])[1]
               is None]
    if missing:
        return Step("boundaries", title, "todo",
                    f"No boundary drawn yet for {_listing(missing)} - the whole picture is "
                    f"counted, including people beside the pitch.", True, "/roi")
    return Step("boundaries", title, "done",
                f"All {_plural(len(rows), 'camera')} have a boundary.", True, "/roi")


def _step_schedule() -> Step:
    from pitch_occupancy.scheduler import load_schedule

    title = "Recording schedule set"
    try:
        schedule = load_schedule()
    except FileNotFoundError:
        return Step("schedule", title, "todo", "No recording schedule yet: set the times "
                    "each pitch is to be recorded.", True, "/cameras")
    rows = _cameras()
    by_venue: dict[str, set[str]] = {}
    for r in rows:
        by_venue.setdefault(r["venue"], set()).add(r["camera_id"])
    if not schedule.slots:
        return Step("schedule", title, "todo", "The schedule has no slots yet.", True,
                    "/cameras")
    if not rows:
        return Step("schedule", title, "todo", f"{_plural(len(schedule.slots), 'slot')} "
                    f"planned, but no cameras added yet to record them.", True, "/cameras")
    unknown = sorted({f"{cam} ({slot.venue_id})"
                      for slot in schedule.slots for cam in slot.cameras
                      if cam not in by_venue.get(slot.venue_id, set())})
    if unknown:
        return Step("schedule", title, "warning",
                    f"The schedule uses cameras that are not in the camera list: "
                    f"{_listing(unknown)}. Those slots cannot be recorded.", True, "/cameras")
    scheduled = {slot.venue_id for slot in schedule.slots}
    unscheduled = sorted(set(by_venue) - scheduled)
    if unscheduled:
        return Step("schedule", title, "todo",
                    f"{_listing(unscheduled)} {'has' if len(unscheduled) == 1 else 'have'} "
                    f"cameras but no recording times.", True, "/cameras")
    return Step("schedule", title, "done",
                f"{_plural(len(schedule.slots), 'slot')} for "
                f"{_plural(len(scheduled), 'venue')}.", True, "/cameras")


def _step_detector() -> Step:
    from pitch_occupancy.vision.detector import Detector, weights_path

    title = "Detector model available"
    # The file on disk, not `Detector.available`: that property builds the model, which
    # imports torch - seconds of work on every dashboard visit, to learn what `exists` says.
    path = weights_path(Detector.load("yolov8n").spec)
    if path.exists():
        return Step("detector", title, "done", f"{path.name} is on this computer.", True)
    return Step("detector", title, "todo",
                f"{path.name} has not been downloaded yet, so no one can be counted.", True,
                hint="uv run pitch fetch-weights")


def _step_bookings() -> Step:
    from pitch_occupancy.booking_sources import load_source

    title = "Booking records connected"
    source = load_source()
    if source is None:
        return Step("bookings", title, "optional", "Not connected. Occupancy works without "
                    "it; connect it to compare play with bookings.", False, "/bookings")
    kind = getattr(source, "kind", "") or "a booking source"
    return Step("bookings", title, "done", f"Connected ({kind}).", False, "/bookings")


def _step_password() -> Step:
    title = "Sign-in password set"
    hint = "README, section \"Running it\" (Sign-in)"
    network = settings.api_host not in _LOCAL_HOSTS
    if settings.access_password.get_secret_value():
        return Step("password", title, "done", f"Pages ask for a password (user "
                    f"{settings.access_user!r}).", network, hint=hint)
    if network:
        return Step("password", title, "todo",
                    f"The server is set to listen on {settings.api_host} with no password: "
                    f"anyone on the network could open it. Set PITCH_ACCESS_PASSWORD.",
                    True, hint=hint)
    return Step("password", title, "optional", "Not needed while the server is opened on this "
                "computer only. Set one before opening it to the network.", False, hint=hint)


def _age(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 120:
        return f"{s} seconds"
    if s < 7200:
        return f"{s // 60} minutes"
    if s < 172800:
        return f"{s // 3600} hours"
    return f"{s // 86400} days"


def _step_worker() -> Step:
    title = "Recording program running"
    try:
        from pitch_occupancy.status import read_status
    except ImportError:
        return Step("worker", title, "unknown", "This version cannot tell whether the "
                    "recording program is running.", True, "/cameras")
    status = read_status()
    if status is None:
        return Step("worker", title, "todo", "The recording program has never been started "
                    "on this computer.", True, "/cameras")
    age = status.get("age_s")
    heard = f" (last heard from {_age(age)} ago)" if isinstance(age, int | float) else ""
    if status.get("alive"):
        return Step("worker", title, "done", f"Running{heard}.", True, "/cameras")
    if status.get("error"):
        return Step("worker", title, "warning", f"Its status file cannot be read "
                    f"({status['error']}).", True, "/cameras")
    return Step("worker", title, "warning", f"Not running{heard}. Slots are not being "
                "recorded until it is started again.", True, "/cameras")


def _step_alerts() -> Step:
    title = "Alerts configured"
    if ALERTS_PATH.exists():
        return Step("alerts", title, "done", f"{ALERTS_PATH.name} is set up.", False)
    return Step("alerts", title, "optional", "No alerts yet. Without them, problems are seen "
                "only when someone opens the dashboard.", False)


#: In the order the page shows them, with what each is called if its check itself fails.
_STEPS: tuple[tuple[str, str, bool, str | None, Callable[[], Step]], ...] = (
    ("cameras", "Cameras added", True, "/cameras", _step_cameras),
    ("boundaries", "Every camera has a pitch boundary", True, "/roi", _step_boundaries),
    ("schedule", "Recording schedule set", True, "/cameras", _step_schedule),
    ("detector", "Detector model available", True, None, _step_detector),
    ("bookings", "Booking records connected", False, "/bookings", _step_bookings),
    ("password", "Sign-in password set", False, None, _step_password),
    ("worker", "Recording program running", True, "/cameras", _step_worker),
    ("alerts", "Alerts configured", False, None, _step_alerts),
)


def checklist() -> list[Step]:
    """Every step, worked out now from this machine's configuration. Never raises."""
    out = []
    for key, title, required, link, check in _STEPS:
        try:
            out.append(check())
        except Exception as exc:  # noqa: BLE001 - a broken config is a step, never a 500
            message = str(exc) or exc.__class__.__name__
            out.append(Step(key, title, "warning", f"Could not check this: {message}",
                            required, link))
    return out


def summary(steps: list[Step] | None = None) -> dict[str, Any]:
    """The checklist as JSON, with whether every required step is done.

    ``unknown`` does not hold the card open: a check this installation cannot make is not
    something the client can fix from the list.
    """
    steps = checklist() if steps is None else steps
    required = [s for s in steps if s.required]
    return {
        "steps": [s.to_dict() for s in steps],
        "required_done": all(s.state in ("done", "unknown") for s in required),
        "done": sum(s.state == "done" for s in steps),
        "total": len(steps),
    }
