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
from pitch_occupancy.api.layout import page

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


_BODY = """
<p class="note"><b>What alerts are for.</b> You should not have to keep this dashboard open to
find out something is wrong. When something needs attention, the system sends one short
message to Slack, Microsoft Teams, Discord, email, or - through an automation service -
WhatsApp, Telegram or SMS. It checks every few minutes while the server is running.</p>

<h2>What you are told about</h2>
EVENTS
<p class="sub2">The same problem is not repeated on a channel for a few hours (6 by default),
so a camera that stays dark sends one message, not one every five minutes. An anomaly and a
daily summary are only ever sent once. During quiet hours nothing is sent; whatever is still a
problem when they end is sent then.</p>

<h2>Where alerts go</h2>
<p class="sub2">The settings live in a file on this computer, <code>configs/alerts.json</code>,
which is never shared or uploaded: a webhook address works like a password for its channel,
and an email login has a password. Addresses are shown here as their first part only.</p>
<div class="card" id="cfgCard"><span class="muted">Loading...</span></div>

<details class="card" style="margin-top:12px" id="howto">
<summary><b>How to set up alerts</b></summary>
<p>Create <code>configs/alerts.json</code> with one or more of these channels (several can sit
side by side under <code>channels</code>), restart the server, and press <b>Send a test
alert</b>. <code>events</code> chooses which alerts go to that channel;
<code>min_severity</code> can be <code>info</code>, <code>warning</code> or
<code>serious</code>.</p>
<div class="grid2" style="margin-top:10px">EXAMPLES</div>
</details>

<h2>Check now</h2>
<p class="sub2">Runs the same checks the background checker runs, and shows what it would send
right now and to which channel. Nothing is sent unless you tick the box.</p>
<div class="controls" style="margin-top:0">
  <label style="display:flex;gap:6px;align-items:center;font-size:13px">
    <input type="checkbox" id="sendToo" style="width:auto"> Also send them</label>
  <button class="act" id="checkBtn">Check now</button>
</div>
<div id="checkOut" style="margin-top:14px"></div>

<h2>Recent alerts</h2>
<p class="sub2">The last 200 messages sent (and any that failed), newest first.</p>
<div class="controls" style="margin-top:0">
  <button class="ghost" id="logBtn">Refresh</button></div>
<div id="logOut" style="margin-top:14px"><span class="muted">Loading...</span></div>
"""

_CSS = """
details summary{cursor:pointer}
details pre{margin:8px 0 6px;white-space:pre-wrap;word-break:break-word}
.sev{display:inline-block;font:600 11px 'JetBrains Mono',monospace;padding:2px 8px;
  border-radius:5px}
.sev.serious{background:var(--flag-soft);color:var(--flag)}
.sev.warning{background:var(--maint-soft);color:var(--maint)}
.sev.info{background:var(--surface-2);color:var(--ink-2)}
.chan{border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:0 0 10px}
.chan h3{margin:0 0 6px;font-size:14px}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:0 0 8px;
  font-size:13.5px}
dl.kv dt{color:var(--ink-3)}dl.kv dd{margin:0;word-break:break-word}
ul.ch{margin:0;padding-left:18px;font-size:12.5px}
"""

_SCRIPT = r"""
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',
  '"':'&quot;',"'":'&#39;'}[c]));
async function call(url,opts){
  const r=await fetch(url,opts);let d=null;
  try{d=await r.json()}catch(e){}
  if(!r.ok)throw new Error((d&&d.detail)?(typeof d.detail==='string'?d.detail:
    JSON.stringify(d.detail)):('HTTP '+r.status));
  return d;
}
const errBox=m=>'<p class="note err">'+esc(m)+'</p>';
const sev=s=>'<span class="sev '+esc(s)+'">'+esc(s)+'</span>';
const chans=o=>{const e=Object.entries(o||{});return e.length?'<ul class="ch">'+
  e.map(([k,v])=>'<li><b>'+esc(k)+'</b>: '+esc(v)+'</li>').join('')+'</ul>':
  '<span class="muted">-</span>'};
const table=(cols,rows)=>'<div class="scroll"><table><thead><tr>'+
  cols.map(c=>'<th>'+esc(c[1])+'</th>').join('')+'</tr></thead><tbody>'+
  rows.map(r=>'<tr>'+cols.map(c=>'<td'+(c[2]?' class="'+c[2]+'"':'')+'>'+
    (c[3]?c[3](r):esc(r[c[0]]))+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';

document.querySelectorAll('[data-copy]').forEach(b=>b.addEventListener('click',async()=>{
  const t=$(b.dataset.copy).textContent;
  try{await navigator.clipboard.writeText(t);b.textContent='Copied'}
  catch(e){b.textContent='Select and copy'}
  setTimeout(()=>b.textContent='Copy',1500);
}));

async function loadConfig(){
  const card=$('cfgCard');
  try{
    const d=await call('/api/v1/alerts/config');
    if(!d.configured){
      card.innerHTML='<b>Alerts are not set up yet.</b> <span class="muted">Create <code>'+
        esc(d.file)+'</code> as shown below - pick a service, copy its example, and put '+
        'your own address in it.</span>';
      $('howto').open=true;return;
    }
    if(!d.valid){
      card.innerHTML='<b class="bad">The alert settings have a problem, so no alerts are '+
        'being sent.</b><p>'+esc(d.problem)+'</p><p class="muted">Fix <code>'+esc(d.file)+
        '</code> and restart the server.</p>';return;
    }
    let h='';
    if(!d.channels.length)h+='<p><b>No channels yet.</b> Add one under <code>channels</code>.</p>';
    for(const c of d.channels){
      const kv=[['Type',c.type==='webhook'?'Webhook':'Email'],
        ['Status',c.enabled?'On':'Off (enabled: false)']];
      if(c.type==='webhook')kv.push(['Address',c.url]);
      else{kv.push(['Mail server',c.host+':'+c.port+' ('+c.security+')']);
        if(c.username)kv.push(['Login',c.username+(c.password_set?' (password set)':
          ' (no password)')]);
        kv.push(['From',c.sender]);kv.push(['To',(c.to||[]).join(', ')]);}
      kv.push(['Alerts',c.events.join(', ')]);
      if(c.min_severity!=='info')kv.push(['Only',c.min_severity+' and above']);
      h+='<div class="chan"><h3>'+esc(c.name)+'</h3><dl class="kv">'+kv.map(([k,v])=>
        '<dt>'+esc(k)+'</dt><dd class="mono">'+esc(v)+'</dd>').join('')+'</dl>'+
        '<button class="ghost" data-test="'+esc(c.name)+'">Send a test alert</button> '+
        '<span class="muted" id="t_'+esc(c.name)+'"></span></div>';
    }
    const s=d.settings,q=s.quiet_hours;
    h+='<p class="muted" style="margin:6px 0 0;font-size:13px">Camera silent after '+
      esc(s.camera_silent_minutes)+' min - repeats held for '+esc(s.cooldown_hours)+
      ' h - checked every '+esc(s.check_interval_minutes)+' min - quiet hours: '+
      (q?esc(q.start+'-'+q.end):'none')+' - daily summary: '+
      esc(s.daily_summary||'off')+' - background checker: '+
      (d.checker_running?'<b class="ok">running</b>':'<b class="bad">not running</b>'+
      ' (it starts with the server when a channel is enabled)')+'.</p>';
    card.innerHTML=h;
    card.querySelectorAll('[data-test]').forEach(b=>b.addEventListener('click',async()=>{
      const name=b.dataset.test,out=$('t_'+name);out.textContent='Sending...';
      try{
        const r=await call('/api/v1/alerts/test',{method:'POST',
          headers:{'Content-Type':'application/json'},body:JSON.stringify({channel:name})});
        out.innerHTML=r.ok?'<b class="ok">Sent.</b> '+esc(r.result)+' - check it arrived.':
          '<b class="bad">Not delivered:</b> '+esc(r.result);
      }catch(e){out.innerHTML='<b class="bad">'+esc(e.message)+'</b>'}
      loadLog();
    }));
  }catch(e){card.innerHTML=errBox(e.message)}
}

$('checkBtn').addEventListener('click',async()=>{
  const out=$('checkOut'),send=$('sendToo').checked;
  out.innerHTML='<p class="muted">Checking...</p>';
  try{
    const d=await call('/api/v1/alerts/check?send='+send,{method:'POST'});
    let h=d.notes.map(n=>'<p class="note">'+esc(n)+'</p>').join('');
    if(d.quiet_hours)h+='<p class="note">It is quiet hours now: nothing is sent until they '+
      'end.</p>';
    if(!d.events.length)h+='<p class="note"><b class="ok">Nothing needs attention right '+
      'now.</b></p>';
    else h+=table([['severity','Severity','',r=>sev(r.severity)],['type','Event','mono'],
      ['text','Message'],['channels',send?'Before sending':'What would happen','',
      r=>d.configured?chans(r.channels):'<span class="muted">no channels set up</span>']],
      d.events);
    if(send)h+='<p class="note">'+(d.sent&&d.sent.length?'Sent '+d.sent.length+
      ' alert(s) - see Recent alerts below.':'Nothing new was sent.')+'</p>';
    out.innerHTML=h;
    if(send)loadLog();
  }catch(e){out.innerHTML=errBox(e.message)}
});

async function loadLog(){
  const out=$('logOut');
  try{
    const d=await call('/api/v1/alerts/log');
    out.innerHTML=d.entries.length?table([['time','Time','mono'],
      ['severity','Severity','',r=>sev(r.severity)],['event','Event','mono'],
      ['text','Message'],['channels','Delivery','',r=>r.event==='checker_error'?
        '<span class="bad">check failed</span>':chans(r.channels)]],d.entries):
      '<p class="muted">Nothing has been sent yet.</p>';
  }catch(e){out.innerHTML=errBox(e.message)}
}
$('logBtn').addEventListener('click',loadLog);
loadConfig();loadLog();
"""


def alerts_html() -> str:
    """The whole page. Built per request: it is small, and `layout` is shared and may change."""
    body = _BODY.replace("EVENTS", _events_html()).replace("EXAMPLES", _examples_html())
    return page("Pitch Occupancy - Alerts", "Alerts", body, active="alerts", css=_CSS,
                script=_SCRIPT)


@router.get("/alerts", response_class=HTMLResponse, include_in_schema=False)
def alerts_page() -> HTMLResponse:
    return HTMLResponse(alerts_html())
