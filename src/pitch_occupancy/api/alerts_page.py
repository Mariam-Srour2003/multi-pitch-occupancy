"""The alerts page: what the system tells you about, where it sends it, and what it sent.

Served at ``/alerts``, with its JSON under ``/api/v1/alerts/...``. The engine is
`pitch_occupancy.alerts`; this module only shows it and lets a person try it.

Like the bookings page, **there is deliberately no route that saves the settings.** They hold
webhook URLs (each one works like a password for its channel) and an email password, and a web
form that accepts them would have to store copies of them somewhere. Editing
``configs/alerts.json`` on the recording computer is the safer path, and the page shows exactly
what to write, with one copyable example per service. Everything the page shows about the
configured channels is masked by `alerts.describe_config`.

Read routes: ``GET /alerts`` (the page), ``GET /api/v1/alerts/config`` (masked settings) and
``GET /api/v1/alerts/log`` (recent deliveries, newest first). Mutating routes:
``POST /api/v1/alerts/test`` (sends one test message to one channel, writes a log line) and
``POST /api/v1/alerts/check`` (computes what would be sent now; with ``?send=true`` it also
sends, which updates the de-duplication state and the log).
"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pitch_occupancy import alerts
from pitch_occupancy.api.layout import CALL_JS, page
from pitch_occupancy.api.templating import template

__all__ = ["router", "alerts_html"]

router = APIRouter(tags=["alerts"])


class AlertTestBody(BaseModel):
    channel: str = Field(min_length=1, max_length=100)


def _config_or_problem() -> alerts.AlertConfig | None:
    try:
        return alerts.load_config()
    except ValueError as exc:
        raise HTTPException(422, f"configs/{alerts.CONFIG_PATH.name} has a problem: {exc}") \
            from None


# --- JSON ---------------------------------------------------------------------------------


@router.get("/api/v1/alerts/config")
def alerts_config() -> dict:
    """The alert settings with every secret masked, or ``{"configured": false}``."""
    return alerts.describe_config()


@router.post("/api/v1/alerts/test")
def alerts_test(body: AlertTestBody) -> dict:
    """Send a test message to one channel now (ignores its event filter, quiet hours and the
    cooldown). The result says whether the channel accepted it."""
    cfg = _config_or_problem()
    if cfg is None:
        raise HTTPException(422, "Alerts are not set up yet: create configs/"
                                 f"{alerts.CONFIG_PATH.name} as shown on the Alerts page.")
    if body.channel not in cfg.channels:
        raise HTTPException(404, f"There is no channel called {body.channel!r}. "
                                 f"Configured: {', '.join(cfg.channels) or 'none'}.")
    return alerts.send_test(body.channel, config=cfg)


@router.post("/api/v1/alerts/check")
def alerts_check(send: Annotated[bool, Query(description="also send what is found")] = False
                 ) -> dict:
    """What the checker finds right now, and what it would do with each event per channel.

    With ``send=true`` the events are also dispatched, exactly as the background checker would
    (repeats skipped, quiet hours respected).
    """
    cfg = _config_or_problem()
    if send and (cfg is None or not cfg.enabled_channels):
        raise HTTPException(422, "Nothing to send to: no channel is set up and enabled.")
    now = datetime.now()
    notes: list[str] = []
    events = alerts.check(now, config=cfg, notes=notes)
    out = {"time": now.isoformat(timespec="seconds"), "notes": notes,
           "configured": cfg is not None,
           "quiet_hours": bool(cfg and alerts.in_quiet_hours(cfg, now)),
           "events": alerts.preview(events, config=cfg, now=now), "sent": None}
    if send:
        out["sent"] = alerts.dispatch(events, config=cfg, now=now)
    return out


@router.get("/api/v1/alerts/log")
def alerts_log(limit: Annotated[int, Query(ge=1, le=alerts.LOG_KEEP)] = alerts.LOG_KEEP
               ) -> dict:
    """The recent alert log, newest first: what was sent, where, and whether it arrived."""
    return {"entries": alerts.read_log(limit)}


# --- the page -----------------------------------------------------------------------------

_ALL = list(alerts.EVENT_TYPES)

#: One copyable example per service. Placeholders only - never a real address.
_EXAMPLES: dict[str, dict] = {
    "Slack": {"channels": {"slack": {
        "type": "webhook", "url": "https://hooks.slack.com/services/T000/B000/XXXXXXXX",
        "events": _ALL}}},
    "Microsoft Teams": {"channels": {"teams": {
        "type": "webhook",
        "url": "https://prod-00.westeurope.logic.azure.com/workflows/.../invoke?...",
        "events": _ALL}}},
    "Discord": {"channels": {"discord": {
        "type": "webhook",
        "url": "https://discord.com/api/webhooks/000000000000/XXXXXXXX/slack",
        "events": _ALL}}},
    "WhatsApp, Telegram or SMS (through Zapier, Make or n8n)": {"channels": {"phone": {
        "type": "webhook", "url": "https://hooks.zapier.com/hooks/catch/000000/xxxxxx/",
        "events": ["anomaly", "worker_down", "worker_error"], "min_severity": "warning"}}},
    "Gmail": {"channels": {"email": {
        "type": "email", "host": "smtp.gmail.com", "port": 587, "security": "starttls",
        "username": "you@gmail.com", "password": "your 16-letter app password",
        "to": ["manager@example.com"], "events": _ALL}}},
    "Outlook / Microsoft 365": {"channels": {"email": {
        "type": "email", "host": "smtp.office365.com", "port": 587, "security": "starttls",
        "username": "you@yourclub.com", "password": "your app password",
        "to": ["manager@example.com"], "events": _ALL}}},
    "Everything else (thresholds, quiet hours, daily summary)": {
        "channels": {"...": "one or more of the channels above"},
        "camera_silent_minutes": 10, "cooldown_hours": 6, "check_interval_minutes": 5,
        "anomaly_lookback_hours": 48, "quiet_hours": {"start": "23:00", "end": "07:00"},
        "daily_summary": "07:30"},
}

_HOWTO: dict[str, str] = {
    "Slack": "In Slack: Apps > Incoming Webhooks > Add to Slack, pick the channel, copy the "
             "Webhook URL.",
    "Microsoft Teams": "In Teams: the channel's ... menu > Workflows > \"Post to a channel when "
                       "a webhook request is received\". Copy the address it gives you. If the "
                       "message arrives empty, edit the workflow to post the <code>text</code> "
                       "field.",
    "Discord": "In Discord: Server settings > Integrations > Webhooks > New Webhook > Copy "
               "Webhook URL, then add <code>/slack</code> to the end.",
    "WhatsApp, Telegram or SMS (through Zapier, Make or n8n)": "Create a flow that starts with "
        "\"Catch webhook\" (Zapier), \"Custom webhook\" (Make) or a Webhook node (n8n), and "
        "sends the <code>text</code> field to WhatsApp, Telegram or SMS. Paste its address.",
    "Gmail": "Turn on 2-step verification, then Google account > Security > App passwords. Use "
             "that password here, never your normal one.",
    "Outlook / Microsoft 365": "Your administrator may need to allow \"Authenticated SMTP\" "
                               "for the mailbox. Use an app password if 2-step sign-in is on.",
    "Everything else (thresholds, quiet hours, daily summary)": "All optional - leave a key out "
        "to keep its default. Every key is explained in "
        "<code>configs/alerts.example.json</code>.",
}


def _examples_html() -> str:
    parts = []
    for i, (label, example) in enumerate(_EXAMPLES.items()):
        parts.append(
            f'<details{" open" if i == 0 else ""}><summary>{escape(label)}</summary>'
            f'<p class="muted" style="font-size:13px">{_HOWTO[label]}</p>'
            f'<pre id="ex{i}">{escape(json.dumps(example, indent=2))}</pre>'
            f'<button class="ghost" data-copy="ex{i}">Copy</button></details>')
    return "".join(parts)


def _events_html() -> str:
    rows = "".join(f"<tr><td class=mono>{escape(k)}</td><td>{escape(v)}</td></tr>"
                   for k, v in alerts.EVENT_TYPES.items())
    return (f'<div class="scroll"><table><thead><tr><th>Event</th><th>When it is sent</th>'
            f"</tr></thead><tbody>{rows}</tbody></table></div>")


_BODY = template("alerts_page_body.html")

_CSS = template("alerts_page_css.css")

_SCRIPT = template("alerts_page_script.js")


def alerts_html() -> str:
    """The whole page. Built per request: it is small, and `layout` is shared and may change."""
    body = _BODY.replace("EVENTS", _events_html()).replace("EXAMPLES", _examples_html())
    return page("Pitch Occupancy - Alerts", "Alerts", body, active="alerts", css=_CSS,
                script=CALL_JS + _SCRIPT)


@router.get("/alerts", response_class=HTMLResponse, include_in_schema=False)
def alerts_page() -> HTMLResponse:
    return HTMLResponse(alerts_html())
