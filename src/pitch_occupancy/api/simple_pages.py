"""The simple front end: four pages, few words, nothing to configure on screen.

The original pages grew around the research - boundaries, decision rows, slot verdicts - and a
person who only wants to add a camera and see whether anyone is there had to read all of it.
These pages are the short path through the same machinery:

* ``/``                         Home - every camera as a card: how many people at the last
                                check. Click one to watch it.
* ``/live/<place>/<camera>``    One camera: the picture, the number of people, refreshing itself.
* ``/cameras/add``              Cameras - add (name, address, password), test, remove.
* ``/records``                  The database: for each camera and each hour and a half, were
                                there people and how many - and the on/off switch for automatic
                                checking (`monitor.py`).
* ``/more``                     The original pages, all still there.

Everything here calls the APIs the original pages already use (`cameras_page`) or the
monitor's; nothing about how a camera is read or scored differs between the two. The old
addresses of the first version (``/live``, ``/cameras/test``) redirect here.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date
from html import escape
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from pitch_occupancy import monitor
from pitch_occupancy.api.layout import page
from pitch_occupancy.config import settings

__all__ = ["router"]

router = APIRouter()

#: Records are grouped into periods of this many minutes - the hour and a half the site asked
#: for. The API still accepts other lengths; the page does not offer them.
RECORD_PERIOD = 90


# --- shared look and helpers ---------------------------------------------------------------

CSS = """
h1{font-size:26px;margin:4px 0 16px;letter-spacing:-.02em}
.top{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;
  margin-bottom:6px}
.top h1{margin:4px 0}
.cards{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(200px,1fr))}
a.cam{display:block;padding:16px 18px;border:1px solid var(--line);border-radius:12px;
  background:var(--surface);text-decoration:none;color:var(--ink);transition:border-color .15s}
a.cam:hover{border-color:var(--accent)}
a.cam b{display:block;font-size:16px}
a.cam .n{font:700 30px Archivo,sans-serif;margin-top:8px;line-height:1.1}
a.cam .w{color:var(--ink-3);font-size:13px;margin-top:2px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;
  background:var(--ink-3);vertical-align:middle}
.dot.many{background:var(--play)}.dot.few{background:var(--maint)}.dot.off{background:var(--flag)}
.empty{padding:28px;text-align:center;color:var(--ink-2)}
.form{display:grid;gap:12px 16px;grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.form label{display:block;font-weight:600;font-size:13.5px;margin-bottom:5px}
.form input,.form select,select.plain,input.plain{font:14px Archivo,sans-serif;
  padding:9px 11px;border:1px solid var(--line);border-radius:7px;background:var(--ground);
  color:var(--ink);width:100%}
select.plain,input.plain{width:auto}
details.opts{margin-top:12px}
details.opts summary{cursor:pointer;color:var(--ink-2);font-size:13.5px}
.msg{margin-top:10px;font-size:14px}.msg.good{color:var(--play)}.msg.bad{color:var(--flag)}
table.simple{width:100%;border-collapse:collapse;font-size:14.5px}
table.simple th{text-align:left;font-size:12.5px;color:var(--ink-3);font-weight:600;
  padding:8px 10px;border-bottom:1px solid var(--line)}
table.simple td{padding:10px;border-bottom:1px solid var(--line);vertical-align:middle}
table.simple img{width:160px;border-radius:6px;display:block}
.scroll{overflow-x:auto}
.yes{color:var(--play);font-weight:700}.no{color:var(--ink-3)}
.bad{color:var(--flag)}.good{color:var(--play)}
button.small,a.small{font:600 12.5px Archivo,sans-serif;padding:6px 12px;border-radius:6px;
  cursor:pointer;border:1px solid var(--line);background:var(--surface);color:var(--ink-2);
  text-decoration:none;display:inline-block}
button.small.red{border-color:var(--flag);color:var(--flag);background:transparent}
.toggle{font:700 15px Archivo,sans-serif;padding:10px 22px;border-radius:999px;cursor:pointer;
  border:1px solid var(--line);background:var(--surface-2);color:var(--ink-2)}
.toggle.on{background:var(--play);border-color:var(--play);color:#fff}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.live{display:grid;gap:20px;grid-template-columns:minmax(0,2fr) minmax(200px,1fr);
  align-items:start}
.live img{width:100%;border-radius:10px;display:block;background:var(--surface-2);
  min-height:200px}
.big{font:700 72px Archivo,sans-serif;line-height:1;letter-spacing:-.03em}
.word{font-size:20px;font-weight:600;margin-top:6px}
.hint{color:var(--ink-3);font-size:13px}
@media (max-width:760px){.live{grid-template-columns:1fr}}
.morelist a{display:block;padding:14px 0;border-bottom:1px solid var(--line);
  text-decoration:none;color:var(--ink)}
.morelist a b{color:var(--accent)}
.morelist a span{display:block;color:var(--ink-2);font-size:13.5px}
"""

#: JavaScript every page shares: escaping, a JSON fetch that surfaces the server's reason,
#: and the plain words for a number of people.
JS_COMMON = r"""
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path, opts = {}) {
  const o = {...opts};
  if (o.json !== undefined) {
    o.body = JSON.stringify(o.json);
    o.headers = {'content-type': 'application/json'};
    delete o.json;
  }
  const r = await fetch(path, o);
  let d = null;
  try { d = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) {
    const det = d && d.detail;
    throw new Error(typeof det === 'string' ? det : 'Something went wrong (' + r.status + ')');
  }
  return d;
}
const hhmm = iso => iso ? String(iso).slice(11, 16) : '';
const people = n => n == null ? '?' : n === 1 ? '1 person' : n + ' people';
// In people, not football: many cameras watch a cafe or a gate, where "Playing" means nothing.
const crowd = n => n == null ? '' : n === 0 ? 'Nobody' : n < 5 ? 'A few people' : 'Many people';
const dot = n => n == null ? 'dot' : n === 0 ? 'dot' : n < 5 ? 'dot few' : 'dot many';
const liveHref = c =>
  '/live/' + encodeURIComponent(c.venue) + '/' + encodeURIComponent(c.camera_id);
const camPath = c =>
  '/api/v1/cameras/' + encodeURIComponent(c.venue) + '/' + encodeURIComponent(c.camera_id);
"""


def _html(title: str, body: str, script: str, active: str) -> HTMLResponse:
    return HTMLResponse(page(title, "Camera monitor", body, active=active, css=CSS,
                             script=JS_COMMON + script))


def _conn():
    from pitch_occupancy.db.schema import connect, initialise

    conn = connect(settings.db_path)
    initialise(conn)
    return conn


# --- Home: every camera, one card each ------------------------------------------------------

HOME_BODY = """
<div class="top"><h1>Cameras</h1><span class="hint" id="state"></span></div>
<div class="cards" id="cards"><div class="empty">Loading…</div></div>
"""

HOME_JS = r"""
async function load() {
  const [mon, latest] = await Promise.all([api('/api/v1/monitor'), api('/api/v1/records/latest')]);
  $('state').innerHTML = mon.settings.enabled
    ? `Checked every ${mon.settings.every_minutes} min` + (mon.last_finished_at
      ? `, last at ${hhmm(mon.last_finished_at)}` : '')
    : 'Automatic checking is off - <a href="/records">turn it on</a>';
  const cams = latest.cameras;
  if (!cams.length) {
    $('cards').innerHTML = '<div class="empty">No cameras yet.<br><br>'
      + '<a class="toggle on" style="text-decoration:none" href="/cameras/add">'
      + 'Add a camera</a></div>';
    return;
  }
  $('cards').innerHTML = cams.map(c => {
    let n = '<span class="hint">Not checked yet</span>', w = 'Click to watch';
    if (c.checked_at && !c.readable) {
      n = '<span class="bad" style="font-size:18px">No picture</span>';
      w = 'at ' + hhmm(c.checked_at);
    } else if (c.checked_at) {
      n = `<span class="${dot(c.people)}"></span>${esc(people(c.people))}`;
      w = crowd(c.people) + ' · ' + hhmm(c.checked_at);
    }
    return `<a class="cam" href="${liveHref(c)}"><b>${esc(c.camera_id)}</b>
      <div class="n">${n}</div><div class="w">${esc(w)}</div></a>`;
  }).join('');
}
load().catch(e => { $('cards').innerHTML = `<div class="empty">${esc(e.message)}</div>`; });
setInterval(() => load().catch(() => {}), 30000);
"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    return _html("Cameras", HOME_BODY, HOME_JS, "home")


# --- One camera, live ---------------------------------------------------------------------------

LIVE_BODY = """
<p><a href="/">← All cameras</a></p>
<h1 id="title"></h1>
<div class="live">
  <div><img id="pic" alt="latest picture from the camera">
    <p class="hint" id="when">Getting a picture…</p></div>
  <div class="card">
    <div class="big" id="num">…</div>
    <div class="word" id="word"></div>
    <p class="msg bad" id="problem"></p>
    <div class="row" style="margin-top:18px"><button class="act" id="now">Check now</button></div>
    <p class="hint" style="margin-top:12px">Checks again by itself every 30 seconds.</p>
  </div>
</div>
"""

LIVE_JS = r"""
const CAM = JSON.parse($('cam').textContent);
$('title').textContent = CAM.camera;
document.title = CAM.camera;
let busy = false;
async function check() {
  if (busy) return;
  busy = true; $('now').disabled = true;
  try {
    const d = await api('/api/v1/cameras/predict',
      {method: 'POST', json: {venue: CAM.venue, cameras: [CAM.camera]}});
    const r = d.cameras[0];
    const t = (d.taken_at || '').slice(11, 19);
    if (r.snapshot) $('pic').src = 'data:image/jpeg;base64,' + r.snapshot;
    if (r.readable) {
      $('num').textContent = r.people == null ? '?' : r.people;
      $('word').textContent = r.people == null ? 'Could not count' : crowd(r.people);
      $('when').textContent = 'Picture at ' + t;
      $('problem').textContent = '';
    } else {
      $('num').textContent = '–';
      $('word').textContent = 'No picture';
      $('when').textContent = 'Tried at ' + t;
      $('problem').textContent = r.problem || 'The camera did not answer.';
    }
  } catch (e) {
    $('problem').textContent = e.message;
  }
  busy = false; $('now').disabled = false;
}
$('now').onclick = check;
check();
setInterval(check, 30000);
"""


@router.get("/live/{venue}/{camera}", response_class=HTMLResponse, include_in_schema=False)
def live_camera(venue: str, camera: str) -> HTMLResponse:
    from pitch_occupancy import cameras

    try:
        known = {(c["venue"], c["camera_id"]) for c in cameras.list_cameras()}
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc
    if (venue, camera) not in known:
        raise HTTPException(404, f"no camera {camera!r} at {venue!r}")
    # The camera's identity goes to the script as JSON inside a non-executable block, with
    # "<" escaped so a name can never close the tag; the page itself only ever sets text.
    data = json.dumps({"venue": venue, "camera": camera}).replace("<", "\\u003c")
    body = f'<script type="application/json" id="cam">{data}</script>\n' + LIVE_BODY
    return _html(camera, body, LIVE_JS, "home")


# --- Cameras: add, test, remove -----------------------------------------------------------------

CAMERAS_BODY = """
<div class="top"><h1>Cameras</h1>
  <div class="row"><button class="small" id="testall">Test all</button>
    <button class="act" id="addbtn">+ Add camera</button></div></div>

<div class="card" id="addcard" hidden style="margin-bottom:18px">
  <div class="form">
    <div><label for="name">Name</label><input id="name" placeholder="e.g. gate"></div>
    <div><label for="ip">Camera IP</label><input id="ip" placeholder="192.168.10.157"></div>
    <div><label for="pass">Password</label>
      <input id="pass" type="password" autocomplete="new-password"></div>
  </div>
  <details class="opts"><summary>More options</summary>
    <div class="form" style="margin-top:10px">
      <div><label for="user">Username</label>
        <input id="user" value="admin" autocomplete="off"></div>
      <div><label for="port">Port</label><input id="port" value="554"></div>
      <div><label for="stream">Stream</label><select id="stream">
        <option value="/1/1">Best quality</option>
        <option value="/1/2">Lighter (slow connection)</option></select></div>
      <div><label for="place">Group</label><input id="place" list="places">
        <datalist id="places"></datalist></div>
      <div style="grid-column:1/-1"><label for="full">Or paste a full camera link</label>
        <input id="full" placeholder="rtsp://user:password@192.168.10.157:554/1/1"></div>
    </div>
  </details>
  <div class="row" style="margin-top:14px">
    <button class="act" id="savebtn">Save</button>
    <button class="small" id="cancelbtn">Cancel</button></div>
  <div class="msg" id="msg"></div>
</div>

<div class="card scroll" id="list"><div class="empty">Loading…</div></div>
"""

CAMERAS_JS = r"""
let cams = [];
const say = (t, good) => {
  $('msg').className = 'msg ' + (good === undefined ? '' : good ? 'good' : 'bad');
  $('msg').textContent = t;
};
$('addbtn').onclick = () => { $('addcard').hidden = false; $('name').focus(); };
$('cancelbtn').onclick = () => { $('addcard').hidden = true; say(''); };

function buildUrl() {
  const full = $('full').value.trim();
  if (full) return full;
  const ip = $('ip').value.trim();
  if (!ip) throw new Error('Type the camera IP.');
  const user = $('user').value.trim(), pass = $('pass').value;
  const login = user ? encodeURIComponent(user) + ':' + encodeURIComponent(pass) + '@' : '';
  return `rtsp://${login}${ip}:${$('port').value.trim() || '554'}${$('stream').value}`;
}

$('savebtn').onclick = async () => {
  const camera_id = $('name').value.trim().replace(/\s+/g, '_');
  const venue = $('place').value.trim().replace(/\s+/g, '_') || (cams[0] ? cams[0].venue : 'site');
  if (!camera_id) { say('Give the camera a name.', false); return; }
  let url;
  try { url = buildUrl(); } catch (e) { say(e.message, false); return; }
  $('savebtn').disabled = true;
  say('Saving and testing…');
  try {
    await api('/api/v1/cameras', {method: 'PUT', json: {venue, camera_id, url}});
    const r = await api('/api/v1/cameras/test', {method: 'POST', json: {venue, camera_id}});
    say(r.ok ? `Saved. "${camera_id}" is working.`
             : `Saved, but it is not working yet: ${r.message}.`, r.ok);
    for (const id of ['name', 'ip', 'pass', 'full']) $(id).value = '';
    await loadList();
  } catch (e) { say(e.message, false); }
  $('savebtn').disabled = false;
};

async function test(i) {
  const c = cams[i], cell = $('st-' + i);
  cell.innerHTML = '<span class="hint">Testing…</span>';
  try {
    const r = await api('/api/v1/cameras/test',
      {method: 'POST', json: {venue: c.venue, camera_id: c.camera_id}});
    cell.innerHTML = r.ok
      ? `<span class="good">Working</span>` + (r.snapshot
        ? `<img alt="" src="data:image/jpeg;base64,${r.snapshot}">` : '')
      : `<span class="bad">Not working</span><div class="hint">${esc(r.message)}</div>`;
    return r.ok;
  } catch (e) { cell.innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
  return false;
}
$('testall').onclick = async () => {
  $('testall').disabled = true;
  for (let i = 0; i < cams.length; i++) await test(i);
  $('testall').disabled = false;
};
async function remove(i) {
  const c = cams[i];
  if (!confirm(`Remove "${c.camera_id}"?`)) return;
  try { await api(camPath(c), {method: 'DELETE'}); await loadList(); }
  catch (e) { alert(e.message); }
}

async function loadList() {
  cams = (await api('/api/v1/cameras/registry')).cameras || [];
  $('places').innerHTML = [...new Set(cams.map(c => c.venue))]
    .map(v => `<option value="${esc(v)}">`).join('');
  if (!cams.length) {
    $('list').innerHTML = '<div class="empty">No cameras yet. Click <b>+ Add camera</b>.</div>';
    $('addcard').hidden = false;
    return;
  }
  $('list').innerHTML = `<table class="simple"><thead><tr><th>Name</th><th>IP</th>
    <th>Status</th><th></th></tr></thead><tbody>${cams.map((c, i) => {
    const ip = (/@([^:\/]+)/.exec(c.url) || /\/\/([^:\/]+)/.exec(c.url) || [])[1] || '';
    return `<tr><td><b>${esc(c.camera_id)}</b></td><td>${esc(ip)}</td>
      <td id="st-${i}"><span class="hint">Not tested</span></td>
      <td><div class="row"><button class="small" data-test="${i}">Test</button>
        <a class="small" href="${liveHref(c)}">Watch</a>
        <button class="small red" data-del="${i}">Remove</button></div></td></tr>`;
  }).join('')}</tbody></table>`;
  $('list').querySelectorAll('[data-test]').forEach(b => b.onclick = () => test(+b.dataset.test));
  $('list').querySelectorAll('[data-del]').forEach(b => b.onclick = () => remove(+b.dataset.del));
}
loadList().catch(e => { $('list').innerHTML = `<div class="empty">${esc(e.message)}</div>`; });
"""


@router.get("/cameras/add", response_class=HTMLResponse, include_in_schema=False)
def cameras_simple() -> HTMLResponse:
    return _html("Cameras", CAMERAS_BODY, CAMERAS_JS, "cameras-simple")


@router.get("/cameras/test", include_in_schema=False)
@router.get("/live", include_in_schema=False)
def old_addresses() -> RedirectResponse:
    """The first version's separate Test and Live pages, folded into Cameras and Home."""
    return RedirectResponse("/", status_code=307)


# --- Records ------------------------------------------------------------------------------------

RECORDS_BODY = """
<div class="top"><h1>Records</h1>
  <div class="row"><span class="hint" id="monstate"></span>
    <button class="toggle" id="toggle">…</button></div></div>
<p class="hint" style="margin-top:0">When automatic checking is on, every camera is checked
  every few minutes and the number of people is kept here.</p>
<div class="row" style="margin:16px 0 12px">
  <input type="date" class="plain" id="day">
  <select class="plain" id="camera"><option value="">All cameras</option></select>
  <a class="small" id="csv" href="#" style="text-decoration:none">Download</a>
</div>
<div class="card scroll" id="table"><div class="empty">Loading…</div></div>
<details class="opts"><summary>Settings</summary>
  <div class="row" style="margin-top:10px"><label for="every">Check every</label>
    <select class="plain" id="every"></select></div>
</details>
"""

RECORDS_JS = r"""
let mon = null, lastFinished = null;
const PERIOD = __PERIOD__;
const mins = m => m < 60 ? m + ' minutes' : m === 60 ? 'hour' : m === 90 ? '1.5 hours'
  : m / 60 + ' hours';

function showMonitor(d) {
  mon = d;
  const s = d.settings;
  $('toggle').className = 'toggle' + (s.enabled ? ' on' : '');
  $('toggle').textContent = s.enabled ? 'On' : 'Off';
  $('every').value = s.every_minutes;
  $('monstate').textContent = d.checking_now ? 'Checking now…'
    : s.enabled ? `Every ${mins(s.every_minutes)}` + (d.last_finished_at
      ? ` · last ${hhmm(d.last_finished_at)}` : '') : '';
  if (d.last_finished_at && d.last_finished_at !== lastFinished) {
    if (lastFinished !== null) loadRecords().catch(() => {});
    lastFinished = d.last_finished_at;
  }
}
async function change(values) {
  try { showMonitor(await api('/api/v1/monitor', {method: 'PUT', json: values})); }
  catch (e) { $('monstate').textContent = e.message; }
}
$('toggle').onclick = () => change({enabled: !mon.settings.enabled});
$('every').onchange = () => change({every_minutes: +$('every').value});

async function loadRecords() {
  const p = new URLSearchParams({day: $('day').value, period: PERIOD});
  if ($('camera').value) p.set('camera', $('camera').value);
  $('csv').href = '/api/v1/records.csv?' + p;
  const d = await api('/api/v1/records?' + p);
  const keep = $('camera').value;
  $('camera').innerHTML = '<option value="">All cameras</option>' + d.cameras.map(c =>
    `<option value="${esc(c)}">${esc(c.split('/').pop())}</option>`).join('');
  $('camera').value = keep;
  if (!d.rows.length) {
    $('table').innerHTML = '<div class="empty">Nothing recorded on this day.</div>';
    return;
  }
  $('table').innerHTML = `<table class="simple"><thead><tr><th>Time</th><th>Camera</th>
    <th>People?</th><th>Most at once</th></tr></thead><tbody>${d.rows.map(r => `<tr>
      <td><b>${esc(r.start)}–${esc(r.end)}</b></td><td>${esc(r.camera)}</td>
      <td>${r.people_seen == null ? '<span class="bad">No picture</span>'
        : r.people_seen ? '<span class="yes">Yes</span>' : '<span class="no">No</span>'}</td>
      <td>${r.max_people ?? ''}</td></tr>`).join('')}</tbody></table>`;
}

(async () => {
  const opts = await api('/api/v1/monitor/options');
  $('every').innerHTML = opts.every_minutes
    .map(m => `<option value="${m}">${mins(m)}</option>`).join('');
  const local = new Date(Date.now() - new Date().getTimezoneOffset() * 60000);
  $('day').value = local.toISOString().slice(0, 10);
  const reload = () => loadRecords().catch(e => { $('table').textContent = e.message; });
  $('day').onchange = reload; $('camera').onchange = reload;
  showMonitor(await api('/api/v1/monitor'));
  await loadRecords();
  setInterval(() => api('/api/v1/monitor').then(showMonitor).catch(() => {}), 5000);
})().catch(e => { $('table').textContent = e.message; });
""".replace("__PERIOD__", str(RECORD_PERIOD))


@router.get("/records", response_class=HTMLResponse, include_in_schema=False)
def records_page() -> HTMLResponse:
    return _html("Records", RECORDS_BODY, RECORDS_JS, "records")


# --- More ---------------------------------------------------------------------------------------

MORE = (
    ("/client", "Pitch verdicts", "Used / not used / review for each booked slot."),
    ("/reports", "Reports", "Usage over days and weeks, busy hours."),
    ("/testing", "Testing with videos", "Upload a video or pictures and see the result."),
    ("/cameras", "Camera setup (advanced)", "Recording status and the slot schedule."),
    ("/bookings", "Bookings", "Check booking records against the cameras."),
    ("/roi", "Pitch boundaries", "Draw the pitch on each camera, so only players are counted."),
    ("/alerts", "Alerts", "Messages when a camera or the recorder stops."),
)


@router.get("/more", response_class=HTMLResponse, include_in_schema=False)
def more_page() -> HTMLResponse:
    links = "".join(f'<a href="{href}"><b>{escape(title)}</b><span>{escape(text)}</span></a>'
                    for href, title, text in MORE)
    body = f'<h1>More tools</h1><div class="card morelist">{links}</div>'
    return _html("More tools", body, "", "more")


# --- JSON: automatic checking ---------------------------------------------------------------

class MonitorIn(BaseModel):
    enabled: bool | None = None
    every_minutes: int | None = None
    save_pictures: bool | None = None
    cameras: list[str] | None = Field(default=None, max_length=500)


def _monitor_status() -> dict:
    out = monitor.status()
    try:
        out["cameras_selected"] = len(monitor.selected_cameras(out["settings"]))
    except ValueError:
        out["cameras_selected"] = 0
    return out


@router.get("/api/v1/monitor/options")
def monitor_options() -> dict:
    return {"every_minutes": list(monitor.EVERY_CHOICES),
            "period_minutes": list(monitor.PERIOD_CHOICES), "default_period": 90}


@router.get("/api/v1/monitor")
def monitor_status() -> dict:
    return _monitor_status()


@router.put("/api/v1/monitor")
def monitor_update(body: MonitorIn) -> dict:
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        monitor.save_settings(changes)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    monitor.start_background()
    return _monitor_status()


@router.post("/api/v1/monitor/run-now")
def monitor_run_now() -> dict:
    return {"started": monitor.run_now()}


# --- JSON: records ------------------------------------------------------------------------

def _day(value: str | None) -> date:
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(422, "day must look like 2026-10-17") from exc


def _camera_filter(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    venue, sep, camera = value.partition("/")
    if not sep or not venue or not camera:
        raise HTTPException(422, 'camera must look like "place/camera"')
    return venue, camera


def _records(day: str | None, period: int, camera: str | None) -> tuple[date, list[dict]]:
    d = _day(day)
    venue, cam = _camera_filter(camera)
    conn = _conn()
    try:
        rows = monitor.records(conn, d, period_minutes=period, venue=venue, camera=cam)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        conn.close()
    return d, rows


@router.get("/api/v1/records")
def records_json(day: str | None = None, period: int = Query(90),
                 camera: str | None = None) -> dict:
    d, rows = _records(day, period, camera)
    conn = _conn()
    try:
        in_db = [f"{r[0]}/{r[1]}" for r in conn.execute(
            "SELECT DISTINCT venue_id, camera_id FROM camera_checks ORDER BY 1, 2")]
        totals = conn.execute(
            """SELECT COUNT(*) AS checks, SUM(readable = 0) AS unreadable, MAX(people) AS mx
               FROM camera_checks WHERE substr(checked_at, 1, 10) = ?""",
            (d.isoformat(),)).fetchone()
    finally:
        conn.close()
    try:
        from pitch_occupancy import cameras

        saved = [f"{c['venue']}/{c['camera_id']}" for c in cameras.list_cameras()]
    except ValueError:
        saved = []
    return {
        "day": d.isoformat(), "period_minutes": period, "rows": rows,
        "cameras": sorted(set(saved) | set(in_db)),
        "totals": {"checks": totals["checks"] or 0,
                   "unreadable_checks": totals["unreadable"] or 0,
                   "max_people": totals["mx"],
                   "periods_with_people": sum(1 for r in rows if r["people_seen"])},
    }


@router.get("/api/v1/records.csv")
def records_csv(day: str | None = None, period: int = Query(90),
                camera: str | None = None) -> Response:
    d, rows = _records(day, period, camera)
    buf = io.StringIO()
    fields = ["day", "start", "end", "venue", "camera", "people_seen", "max_people",
              "avg_people", "checks", "checks_with_people", "readable", "usual_label"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        seen = {True: "yes", False: "no", None: ""}[r["people_seen"]]
        writer.writerow({**r, "people_seen": seen})
    return Response(buf.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="camera-records-{d.isoformat()}.csv"'})


@router.get("/api/v1/records/latest")
def records_latest() -> dict:
    """Each saved camera's most recent check (or none yet), saved cameras first."""
    from pitch_occupancy import cameras

    conn = _conn()
    try:
        latest = {(r["venue_id"], r["camera_id"]): dict(r) for r in conn.execute(
            """SELECT c.* FROM camera_checks c JOIN (
                   SELECT MAX(check_id) AS id FROM camera_checks GROUP BY venue_id, camera_id
               ) m ON c.check_id = m.id""")}
    finally:
        conn.close()
    try:
        saved = [(c["venue"], c["camera_id"]) for c in cameras.list_cameras()]
    except ValueError:
        saved = []
    out = []
    for venue, cam in saved:
        r = latest.get((venue, cam), {})
        out.append({"venue": venue, "camera_id": cam, "checked_at": r.get("checked_at"),
                    "readable": bool(r.get("readable")), "people": r.get("people"),
                    "label": r.get("label"), "problem": r.get("problem")})
    return {"cameras": out}


@router.get("/api/v1/records/picture/{check_id}", include_in_schema=False)
def record_picture(check_id: int) -> FileResponse:
    conn = _conn()
    try:
        row = conn.execute("SELECT picture_path FROM camera_checks WHERE check_id = ?",
                           (check_id,)).fetchone()
    finally:
        conn.close()
    if row is None or not row["picture_path"]:
        raise HTTPException(404, "no picture was kept for this check")
    root = (Path(settings.evidence_dir) / "checks").resolve()
    path = Path(row["picture_path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "the picture is no longer on disk")
    return FileResponse(path, media_type="image/jpeg")
