"""The simple front end: one page per job, in plain words.

The original pages grew around the research - boundaries, decision rows, slot verdicts - and a
person who only wants to add a camera and see whether anyone is on the pitch had to read all
of it. These pages are the short path through the same machinery:

* ``/``                 Home - status at a glance and four big buttons.
* ``/cameras/add``      Add, change or remove a camera, by address and password - no RTSP
                        link to write by hand.
* ``/cameras/test``     Test every camera, one by one, with a picture from each.
* ``/live``             Every camera, each with its own page (``/live/<place>/<camera>``) that
                        shows the picture and how many people it sees, refreshing itself.
* ``/records``          The database: for each camera and each period (an hour and a half by
                        default), were there people and how many - plus the switch for
                        automatic checking (`monitor.py`).
* ``/more``             The original pages, which are all still there.

Everything here calls the APIs the original pages already use (`cameras_page`) or the
monitor's; nothing about how a camera is read or scored differs between the two.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date
from html import escape
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from pitch_occupancy import monitor
from pitch_occupancy.api.layout import page
from pitch_occupancy.config import settings

__all__ = ["router"]

router = APIRouter()


# --- shared look and helpers ---------------------------------------------------------------

CSS = """
.hero h1{font-size:28px;margin:4px 0 6px;letter-spacing:-.02em}
.hero p{color:var(--ink-2);margin:0 0 18px;font-size:15px}
.big{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  margin:20px 0 8px}
a.bigbtn{display:block;padding:22px 20px;border:1px solid var(--line);border-radius:12px;
  background:var(--surface);text-decoration:none;color:var(--ink);transition:border-color .15s}
a.bigbtn:hover{border-color:var(--accent)}
a.bigbtn b{display:block;font-size:18px;margin-bottom:6px}
a.bigbtn span{color:var(--ink-2);font-size:14px}
.status{display:inline-block;padding:3px 11px;border-radius:999px;font-weight:600;
  font-size:13px;white-space:nowrap}
.status.good{background:var(--play-soft);color:var(--play)}
.status.bad{background:var(--flag-soft);color:var(--flag)}
.status.wait{background:var(--surface-2);color:var(--ink-2)}
.status.some{background:var(--maint-soft);color:var(--maint)}
.camgrid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));
  margin-top:14px}
.camcard{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:14px 16px}
.camcard h3{margin:0 0 2px;font-size:16px}
.camcard .place{color:var(--ink-3);font-size:12.5px}
.camcard img{width:100%;border-radius:6px;display:block;margin-top:10px;
  background:var(--surface-2)}
.camcard p{margin:8px 0 0;font-size:13.5px;color:var(--ink-2)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.spread{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.form{display:grid;gap:14px 18px;grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}
.form label,.inline label{display:block;font-weight:600;font-size:13.5px;margin-bottom:5px}
.form input,.form select,.inline select,.inline input{font:14px Archivo,sans-serif;
  padding:9px 11px;border:1px solid var(--line);border-radius:7px;background:var(--ground);
  color:var(--ink);width:100%}
.inline{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end}
.inline > div{min-width:170px}
.hint{color:var(--ink-3);font-size:12.5px;margin-top:4px}
.msg{margin-top:12px;font-size:14px}
.msg.good{color:var(--play)}.msg.bad{color:var(--flag)}
button.danger{font:600 12px Archivo,sans-serif;padding:6px 11px;border-radius:6px;
  cursor:pointer;border:1px solid var(--flag);background:transparent;color:var(--flag)}
a.btnlink{font:600 12px Archivo,sans-serif;padding:6px 11px;border-radius:6px;
  border:1px solid var(--line);background:var(--surface);color:var(--accent);
  text-decoration:none;display:inline-block}
.bignum{font:700 64px Archivo,sans-serif;line-height:1;letter-spacing:-.03em}
.livebox{display:grid;gap:20px;grid-template-columns:minmax(0,2fr) minmax(220px,1fr);
  align-items:start}
.livebox img{width:100%;border-radius:8px;display:block;background:var(--surface-2);
  min-height:200px}
@media (max-width:760px){.livebox{grid-template-columns:1fr}}
table.simple{width:100%;border-collapse:collapse;font-size:14px}
table.simple th{text-align:left;font-size:12.5px;color:var(--ink-3);font-weight:600;
  padding:8px 10px;border-bottom:1px solid var(--line)}
table.simple td{padding:9px 10px;border-bottom:1px solid var(--line)}
.scroll{overflow-x:auto}
.yes{color:var(--play);font-weight:700}.no{color:var(--ink-3)}
.toggle{font:700 14px Archivo,sans-serif;padding:10px 20px;border-radius:999px;cursor:pointer;
  border:1px solid var(--line);background:var(--surface-2);color:var(--ink-2)}
.toggle.on{background:var(--play);border-color:var(--play);color:#fff}
.morelist a{display:block;padding:14px 0;border-bottom:1px solid var(--line);
  text-decoration:none;color:var(--ink)}
.morelist a b{color:var(--accent)}
.morelist a span{display:block;color:var(--ink-2);font-size:13.5px}
"""

#: JavaScript every simple page shares: escaping, a JSON fetch that surfaces the server's
#: reason, and the camera list.
JS_COMMON = r"""
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
    const why = typeof det === 'string' ? det : (det ? JSON.stringify(det) : 'Error ' + r.status);
    throw new Error(why);
  }
  return d;
}
async function cameraList() {
  const d = await api('/api/v1/cameras/registry');
  return d.cameras || [];
}
const hhmm = iso => iso ? String(iso).slice(11, 16) : '—';
const peopleText = n => n == null ? '—' : (n === 1 ? '1 person' : n + ' people');
// In people, not football: many cameras watch a cafe or a gate, where "Playing" means nothing.
const crowd = n => n == null ? 'Not counted' : n === 0 ? 'Nobody'
  : n < 5 ? 'A few people' : 'Many people';
const crowdClass = n => n == null ? 'wait' : n === 0 ? 'wait' : n < 5 ? 'some' : 'good';
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


# --- Home -----------------------------------------------------------------------------------

HOME_BODY = """
<section class="hero">
  <h1>Pitch cameras</h1>
  <p>Add your cameras, check they work, watch them live, and look back at when people were
  there.</p>
</section>
<div class="tiles">
  <div class="tile"><div class="k">Cameras saved</div><div class="v" id="t-cams">—</div></div>
  <div class="tile"><div class="k">Automatic checking</div>
    <div class="v sm" id="t-auto">—</div></div>
  <div class="tile"><div class="k">Last check</div><div class="v sm" id="t-last">—</div></div>
  <div class="tile"><div class="k">Cameras seeing people</div>
    <div class="v" id="t-people">—</div></div>
</div>
<div class="big">
  <a class="bigbtn" href="/cameras/add"><b>1. Add cameras</b>
    <span>Type the camera's address and password.</span></a>
  <a class="bigbtn" href="/cameras/test"><b>2. Test cameras</b>
    <span>See which cameras answer, with a picture from each.</span></a>
  <a class="bigbtn" href="/live"><b>3. Watch live</b>
    <span>Open any camera and see how many people it sees right now.</span></a>
  <a class="bigbtn" href="/records"><b>4. Records</b>
    <span>When there were people, and how many - every hour and a half.</span></a>
</div>
<h2>Latest check of each camera</h2>
<p class="sub2">From automatic checking. Switch it on in <a href="/records">Records</a>.</p>
<div class="card scroll" id="latest">Loading…</div>
"""

HOME_JS = r"""
async function load() {
  const [mon, latest] = await Promise.all([api('/api/v1/monitor'), api('/api/v1/records/latest')]);
  const s = mon.settings;
  document.getElementById('t-cams').textContent = latest.cameras.length;
  document.getElementById('t-auto').innerHTML = s.enabled
    ? `<span class="status good">On</span> every ${s.every_minutes} min`
    : '<span class="status wait">Off</span>';
  document.getElementById('t-last').textContent =
    mon.last_finished_at ? hhmm(mon.last_finished_at) : 'never';
  const seeing = latest.cameras.filter(c => (c.people || 0) > 0).length;
  document.getElementById('t-people').textContent =
    latest.cameras.some(c => c.checked_at) ? seeing : '—';
  const box = document.getElementById('latest');
  if (!latest.cameras.length) {
    box.innerHTML = 'No cameras yet. <a href="/cameras/add">Add your first camera</a>.';
    return;
  }
  box.innerHTML = `<table class="simple"><thead><tr><th>Camera</th><th>Place</th>
    <th>Checked at</th><th>People</th><th>How busy</th><th></th></tr></thead><tbody>${
    latest.cameras.map(c => `<tr><td><b>${esc(c.camera_id)}</b></td><td>${esc(c.venue)}</td>
      <td>${c.checked_at ? esc(c.checked_at.slice(0, 10)) + ' ' + hhmm(c.checked_at)
        : '<span class="no">not checked yet</span>'}</td>
      <td>${!c.checked_at ? '' : c.readable ? esc(peopleText(c.people))
        : '<span class="status bad">no picture</span>'}</td>
      <td>${c.checked_at && c.readable ? esc(crowd(c.people)) : ''}</td>
      <td><a class="btnlink" href="${liveHref(c)}">Watch live</a></td></tr>`).join('')
  }</tbody></table>`;
}
load().catch(e => { document.getElementById('latest').textContent = e.message; });
setInterval(() => load().catch(() => {}), 30000);
"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    return _html("Pitch cameras", HOME_BODY, HOME_JS, "home")


# --- Add cameras ------------------------------------------------------------------------------

ADD_BODY = """
<section class="hero"><h1>Add a camera</h1>
<p>Fill in the camera's details as they appear on your recorder (NVR). Test it first if you
like, then save.</p></section>
<div class="card">
  <div class="form">
    <div><label for="name">Camera name</label><input id="name" placeholder="e.g. pitch1_left">
      <div class="hint">Letters, numbers, - or _ . No spaces.</div></div>
    <div><label for="place">Place</label>
      <input id="place" list="places" placeholder="e.g. nvr_site">
      <datalist id="places"></datalist>
      <div class="hint">Groups cameras of one site or pitch.</div></div>
    <div><label for="ip">Camera address (IP)</label>
      <input id="ip" placeholder="192.168.10.157"></div>
    <div><label for="port">Port</label><input id="port" value="554">
      <div class="hint">Almost always 554.</div></div>
    <div><label for="user">Username</label><input id="user" value="admin" autocomplete="off"></div>
    <div><label for="pass">Password</label>
      <input id="pass" type="password" autocomplete="new-password">
      <div class="hint" id="passhint">Typed as it is - special characters like @ are
        handled.</div></div>
    <div><label for="stream">Picture quality</label>
      <select id="stream">
        <option value="/1/1">Best quality (main stream)</option>
        <option value="/1/2">Lighter (sub stream) - for slow connections</option>
        <option value="custom">Other path…</option>
      </select></div>
    <div id="custombox" hidden><label for="path">Stream path</label>
      <input id="path" placeholder="/Streaming/Channels/101"></div>
  </div>
  <details style="margin-top:14px"><summary>I have a full camera link instead</summary>
    <div class="form" style="margin-top:10px"><div style="grid-column:1/-1">
      <label for="full">Camera link</label>
      <input id="full" placeholder="rtsp://user:password@192.168.10.157:554/1/1">
      <div class="hint">When filled in, this is used instead of the boxes above.</div></div></div>
  </details>
  <div class="row" style="margin-top:16px">
    <button class="ghost" id="testbtn">Test first</button>
    <button class="act" id="savebtn">Save camera</button>
    <button class="ghost" id="clearbtn">Clear</button>
  </div>
  <div class="msg" id="msg"></div>
  <img id="testpic" alt=""
    style="display:none;max-width:480px;width:100%;border-radius:8px;margin-top:12px">
</div>

<h2>Your cameras</h2>
<p class="sub2">Click <b>Change</b> to edit a camera; leave the password empty to keep it.</p>
<div class="card scroll" id="list">Loading…</div>
"""

ADD_JS = r"""
const $ = id => document.getElementById(id);
let editing = null;   // {venue, camera_id} while changing a saved camera

$('stream').onchange = () => { $('custombox').hidden = $('stream').value !== 'custom'; };

function buildUrl() {
  const full = $('full').value.trim();
  if (full) return full;
  const ip = $('ip').value.trim();
  if (!ip) throw new Error('Type the camera address (IP), for example 192.168.10.157.');
  const port = $('port').value.trim() || '554';
  let path = $('stream').value === 'custom' ? $('path').value.trim() : $('stream').value;
  if (!path) throw new Error('Type the stream path, or choose a picture quality.');
  if (!path.startsWith('/')) path = '/' + path;
  const user = $('user').value.trim();
  const pass = $('pass').value;
  const login = user ? encodeURIComponent(user) + ':' + encodeURIComponent(pass) + '@' : '';
  return `rtsp://${login}${ip}:${port}${path}`;
}
function cleanName(s) { return s.trim().replace(/\s+/g, '_'); }
function say(text, good) {
  $('msg').className = 'msg ' + (good === undefined ? '' : good ? 'good' : 'bad');
  $('msg').textContent = text;
}

$('testbtn').onclick = async () => {
  $('testpic').style.display = 'none';
  let url;
  try { url = buildUrl(); } catch (e) { say(e.message, false); return; }
  say('Testing… this can take up to 15 seconds.');
  $('testbtn').disabled = true;
  try {
    const body = {url};
    if (editing) Object.assign(body, editing);
    const r = await api('/api/v1/cameras/test', {method: 'POST', json: body});
    if (r.ok) {
      say(`Working - a ${r.width}×${r.height} picture arrived.`, true);
      if (r.snapshot) {
        $('testpic').src = 'data:image/jpeg;base64,' + r.snapshot;
        $('testpic').style.display = 'block';
      }
    } else {
      say(`Not working: ${r.message}. ${r.hint || ''}`, false);
    }
  } catch (e) { say(e.message, false); }
  $('testbtn').disabled = false;
};

$('savebtn').onclick = async () => {
  const camera_id = cleanName($('name').value);
  const venue = cleanName($('place').value) || 'site';
  if (!camera_id) { say('Give the camera a name.', false); return; }
  let url;
  try { url = buildUrl(); } catch (e) { say(e.message, false); return; }
  $('savebtn').disabled = true;
  try {
    if (editing && (editing.venue !== venue || editing.camera_id !== camera_id)) {
      if (!$('pass').value && !$('full').value.trim()) {
        throw new Error('To rename a camera or move it to another place, type its password again.');
      }
    }
    await api('/api/v1/cameras', {method: 'PUT', json: {venue, camera_id, url}});
    if (editing && (editing.venue !== venue || editing.camera_id !== camera_id)) {
      await api(camPath(editing), {method: 'DELETE'});
    }
    say(`Saved "${camera_id}". You can test it on the Test page or watch it live.`, true);
    clearForm(false);
    await loadList();
  } catch (e) { say(e.message, false); }
  $('savebtn').disabled = false;
};

function clearForm(clearMsg = true) {
  editing = null;
  for (const id of ['name', 'ip', 'pass', 'full', 'path']) $(id).value = '';
  $('port').value = '554'; $('user').value = 'admin'; $('stream').value = '/1/1';
  $('custombox').hidden = true; $('savebtn').textContent = 'Save camera';
  $('passhint').textContent = 'Typed as it is - special characters like @ are handled.';
  $('testpic').style.display = 'none';
  if (clearMsg) say('');
}
$('clearbtn').onclick = () => clearForm();

function parse(masked) {
  const re = /^rtsp:\/\/(?:([^:@\/]*)(?::[^@\/]*)?@)?([^:\/]+)(?::(\d+))?(\/.*)?$/i;
  const m = re.exec(masked || '');
  if (!m) return null;
  return {user: decodeURIComponent(m[1] || ''), ip: m[2], port: m[3] || '554',
          path: m[4] || '/'};
}
function edit(c) {
  const p = parse(c.url);
  editing = {venue: c.venue, camera_id: c.camera_id};
  $('name').value = c.camera_id; $('place').value = c.venue;
  $('pass').value = ''; $('full').value = '';
  if (p) {
    $('ip').value = p.ip; $('port').value = p.port; $('user').value = p.user;
    if (p.path === '/1/1' || p.path === '/1/2') {
      $('stream').value = p.path; $('custombox').hidden = true;
    }
    else { $('stream').value = 'custom'; $('path').value = p.path; $('custombox').hidden = false; }
  } else { $('full').value = ''; }
  $('passhint').textContent = 'Leave empty to keep the saved password.';
  $('savebtn').textContent = 'Save changes';
  say(`Changing "${c.camera_id}".`);
  window.scrollTo({top: 0, behavior: 'smooth'});
}
async function remove(c) {
  if (!confirm(`Remove camera "${c.camera_id}"?`)) return;
  try {
    const r = await api(camPath(c), {method: 'DELETE'});
    say(r.warning ? 'Removed. Note: ' + r.warning : `Removed "${c.camera_id}".`, !r.warning);
    await loadList();
  } catch (e) { say(e.message, false); }
}

let cams = [];
async function loadList() {
  cams = await cameraList();
  $('places').innerHTML = [...new Set(cams.map(c => c.venue))]
    .map(v => `<option value="${esc(v)}">`).join('');
  if (!$('place').value && cams.length) $('place').value = cams[0].venue;
  const box = $('list');
  if (!cams.length) { box.textContent = 'No cameras saved yet.'; return; }
  box.innerHTML = `<table class="simple"><thead><tr><th>Name</th><th>Place</th>
    <th>Address</th><th></th></tr></thead><tbody>${
    cams.map((c, i) => `<tr><td><b>${esc(c.camera_id)}</b></td><td>${esc(c.venue)}</td>
      <td><code>${esc(c.url)}</code></td>
      <td class="row"><button class="ghost" data-edit="${i}">Change</button>
        <a class="btnlink" href="${liveHref(c)}">Watch live</a>
        <button class="danger" data-del="${i}">Remove</button></td></tr>`).join('')
  }</tbody></table>`;
  box.querySelectorAll('[data-edit]').forEach(b => b.onclick = () => edit(cams[+b.dataset.edit]));
  box.querySelectorAll('[data-del]').forEach(b => b.onclick = () => remove(cams[+b.dataset.del]));
}
loadList().catch(e => { $('list').textContent = e.message; });
"""


@router.get("/cameras/add", response_class=HTMLResponse, include_in_schema=False)
def add_cameras() -> HTMLResponse:
    return _html("Add cameras", ADD_BODY, ADD_JS, "add")


# --- Test cameras -----------------------------------------------------------------------------

TEST_BODY = """
<section class="hero"><h1>Test cameras</h1>
<p>Checks each camera one after another and shows a picture from every camera that answers.
Nothing is saved.</p></section>
<div class="spread">
  <div class="row"><button class="act" id="all">Test all cameras</button>
    <span id="summary" class="sub2" style="margin:0"></span></div>
  <a class="btnlink" href="/cameras/add">Add a camera</a>
</div>
<div class="camgrid" id="grid"></div>
"""

TEST_JS = r"""
const grid = document.getElementById('grid');
let cams = [];
function card(c, i) {
  return `<div class="camcard" id="cam-${i}">
    <div class="spread">
      <div><h3>${esc(c.camera_id)}</h3><div class="place">${esc(c.venue)}</div></div>
      <span class="status wait" id="st-${i}">Not tested</span></div>
    <div id="res-${i}"></div>
    <div class="row" style="margin-top:10px"><button class="ghost" data-test="${i}">Test</button>
      <a class="btnlink" href="${liveHref(c)}">Watch live</a></div></div>`;
}
async function testOne(i) {
  const c = cams[i];
  const st = document.getElementById('st-' + i), res = document.getElementById('res-' + i);
  st.className = 'status wait'; st.textContent = 'Testing…'; res.innerHTML = '';
  try {
    const r = await api('/api/v1/cameras/test',
      {method: 'POST', json: {venue: c.venue, camera_id: c.camera_id}});
    if (r.ok) {
      st.className = 'status good'; st.textContent = 'Working';
      res.innerHTML = (r.snapshot ? `<img alt="picture from ${esc(c.camera_id)}"
          src="data:image/jpeg;base64,${r.snapshot}">` : '')
        + `<p>${r.width}×${r.height} picture in ${(r.elapsed_ms / 1000).toFixed(1)} s</p>`;
      return true;
    }
    st.className = 'status bad'; st.textContent = 'Not working';
    res.innerHTML = `<p><b>${esc(r.message)}</b></p>${r.hint ? `<p>${esc(r.hint)}</p>` : ''}`;
  } catch (e) {
    st.className = 'status bad'; st.textContent = 'Error';
    res.innerHTML = `<p>${esc(e.message)}</p>`;
  }
  return false;
}
document.getElementById('all').onclick = async () => {
  const btn = document.getElementById('all'), sum = document.getElementById('summary');
  btn.disabled = true;
  let ok = 0;
  for (let i = 0; i < cams.length; i++) {
    sum.textContent = `Testing ${i + 1} of ${cams.length}…`;
    if (await testOne(i)) ok++;
  }
  sum.textContent = `${ok} of ${cams.length} cameras working.`;
  btn.disabled = false;
};
cameraList().then(list => {
  cams = list;
  if (!cams.length) {
    grid.innerHTML = '<p>No cameras saved yet. <a href="/cameras/add">Add one</a>.</p>';
    return;
  }
  grid.innerHTML = cams.map(card).join('');
  grid.querySelectorAll('[data-test]').forEach(b => b.onclick = () => testOne(+b.dataset.test));
}).catch(e => { grid.textContent = e.message; });
"""


@router.get("/cameras/test", response_class=HTMLResponse, include_in_schema=False)
def test_cameras() -> HTMLResponse:
    return _html("Test cameras", TEST_BODY, TEST_JS, "test")


# --- Live -------------------------------------------------------------------------------------

LIVE_INDEX_BODY = """
<section class="hero"><h1>Watch live</h1>
<p>Pick a camera. Its page shows the picture and how many people it sees, and keeps checking
by itself. Each camera has its own link you can bookmark or share.</p></section>
<div class="camgrid" id="grid">Loading…</div>
"""

LIVE_INDEX_JS = r"""
Promise.all([cameraList(), api('/api/v1/records/latest')]).then(([cams, latest]) => {
  const grid = document.getElementById('grid');
  if (!cams.length) {
    grid.innerHTML = '<p>No cameras saved yet. <a href="/cameras/add">Add one</a>.</p>';
    return;
  }
  const last = Object.fromEntries(latest.cameras.map(c => [c.venue + '/' + c.camera_id, c]));
  grid.innerHTML = cams.map(c => {
    const l = last[c.venue + '/' + c.camera_id];
    const seen = l && l.checked_at
      ? `<p>Last check ${hhmm(l.checked_at)}:
          ${l.readable ? esc(peopleText(l.people)) : 'no picture'}</p>` : '';
    return `<a class="camcard" style="display:block;text-decoration:none;color:inherit"
      href="${liveHref(c)}">
      <h3>${esc(c.camera_id)}</h3><div class="place">${esc(c.venue)}</div>${seen}
      <p style="color:var(--accent);font-weight:600">Open live view →</p></a>`;
  }).join('');
}).catch(e => { document.getElementById('grid').textContent = e.message; });
"""


@router.get("/live", response_class=HTMLResponse, include_in_schema=False)
def live_index() -> HTMLResponse:
    return _html("Watch live", LIVE_INDEX_BODY, LIVE_INDEX_JS, "live")


LIVE_BODY = """
<p><a href="/live">← All cameras</a></p>
<section class="hero"><h1 id="title"></h1><p id="place"></p></section>
<div class="livebox">
  <div><img id="pic" alt="latest picture from the camera">
    <p class="hint" id="when">Waiting for the first picture…</p></div>
  <div class="card">
    <div class="k" style="font-weight:600;color:var(--ink-3)">People seen</div>
    <div class="bignum" id="people">…</div>
    <p style="margin:10px 0 0"><span class="status wait" id="label">Checking…</span></p>
    <p class="hint" id="verdict"></p>
    <p class="msg bad" id="problem"></p>
    <div class="inline" style="margin-top:16px">
      <div><label for="every">Check again</label>
        <select id="every"><option value="0">Only when I click</option>
          <option value="15">Every 15 seconds</option>
          <option value="30" selected>Every 30 seconds</option>
          <option value="60">Every minute</option></select></div>
    </div>
    <div class="row" style="margin-top:12px"><button class="act" id="now">Check now</button></div>
    <p class="hint" style="margin-top:14px">Link to this camera:<br><code id="link"></code></p>
  </div>
</div>
<h2>This visit</h2>
<p class="sub2">The checks made while this page is open. Automatic checking keeps them for good
- see <a href="/records">Records</a>.</p>
<div class="card scroll"><table class="simple"><thead><tr><th>Time</th><th>People</th>
  <th>How busy</th></tr></thead><tbody id="hist">
  <tr><td colspan="3" class="no">Nothing yet</td></tr></tbody></table></div>
"""

LIVE_JS = r"""
const CAM = JSON.parse(document.getElementById('cam').textContent);
const $ = id => document.getElementById(id);
$('title').textContent = CAM.camera;
$('place').textContent = 'Place: ' + CAM.venue;
$('link').textContent = location.href;
document.title = CAM.camera + ' - live';
let busy = false, timer = null;
const history = [];
async function check() {
  if (busy) return;
  busy = true; $('now').disabled = true;
  $('label').className = 'status wait'; $('label').textContent = 'Checking…';
  try {
    const d = await api('/api/v1/cameras/predict',
      {method: 'POST', json: {venue: CAM.venue, cameras: [CAM.camera]}});
    const r = d.cameras[0];
    const t = (d.taken_at || '').slice(11, 19);
    if (r.snapshot) $('pic').src = 'data:image/jpeg;base64,' + r.snapshot;
    $('when').textContent = r.readable ? 'Picture taken at ' + t : 'No picture at ' + t;
    $('people').textContent = r.people == null ? '?' : r.people;
    $('label').className = 'status ' + (r.readable ? crowdClass(r.people) : 'bad');
    $('label').textContent = r.readable ? crowd(r.people) : 'Camera not answering';
    $('verdict').textContent = r.readable && r.label ? 'Pitch rule says: ' + r.label : '';
    $('problem').textContent = r.problem ? r.problem + (r.hint ? ' ' + r.hint : '') : '';
    history.unshift({t, people: r.people, text: r.readable ? crowd(r.people) : 'no picture'});
    history.length = Math.min(history.length, 30);
    $('hist').innerHTML = history.map(h => `<tr><td>${esc(h.t)}</td>
      <td>${esc(peopleText(h.people))}</td><td>${esc(h.text)}</td></tr>`).join('');
  } catch (e) {
    $('label').className = 'status bad'; $('label').textContent = 'Error';
    $('problem').textContent = e.message;
  }
  busy = false; $('now').disabled = false;
}
function schedule() {
  clearInterval(timer);
  const s = +$('every').value;
  if (s > 0) timer = setInterval(check, s * 1000);
}
$('now').onclick = check;
$('every').onchange = schedule;
check(); schedule();
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
    body = (f'<script type="application/json" id="cam">{data}</script>\n' + LIVE_BODY)
    return _html(f"{camera} - live", body, LIVE_JS, "live")


# --- Records ----------------------------------------------------------------------------------

RECORDS_BODY = """
<section class="hero"><h1>Records</h1>
<p>Every automatic check is kept here: when each camera was checked, whether there were people,
and how many. Grouped into periods - an hour and a half unless you choose otherwise.</p></section>

<div class="card">
  <div class="spread">
    <div><b>Automatic checking</b>
      <div class="hint" id="monstate">Loading…</div></div>
    <button class="toggle" id="toggle">…</button>
  </div>
  <div class="inline" style="margin-top:14px">
    <div><label for="every">Check every camera</label><select id="every"></select></div>
    <div><label><input type="checkbox" id="pics" style="width:auto;margin-right:6px">Keep a
      picture with each check</label>
      <div class="hint">Pictures show people - only switch on if you are allowed to keep
        them.</div></div>
    <div><button class="ghost" id="runnow">Check all cameras now</button></div>
  </div>
</div>

<h2>Look back</h2>
<div class="inline">
  <div><label for="day">Day</label><input type="date" id="day"></div>
  <div><label for="period">Group by</label><select id="period"></select></div>
  <div><label for="camera">Camera</label>
    <select id="camera"><option value="">All cameras</option></select></div>
  <div><a class="btnlink" id="csv" href="#">Download as spreadsheet (CSV)</a></div>
</div>
<div class="tiles" style="margin-top:16px">
  <div class="tile"><div class="k">Checks this day</div><div class="v" id="s-checks">—</div></div>
  <div class="tile"><div class="k">Periods with people</div><div class="v" id="s-busy">—</div></div>
  <div class="tile"><div class="k">Most people at once</div><div class="v" id="s-max">—</div></div>
  <div class="tile"><div class="k">Camera not answering</div><div class="v" id="s-bad">—</div></div>
</div>
<div class="card scroll" style="margin-top:12px" id="table">Loading…</div>
"""

RECORDS_JS = r"""
const $ = id => document.getElementById(id);
let mon = null, lastFinished = null;
const periodName = m => m === 60 ? '1 hour' : m % 60 === 0 ? m / 60 + ' hours'
  : m === 90 ? '1.5 hours' : m + ' minutes';
const everyName = m => m === 1 ? 'minute' : m < 60 ? m + ' minutes'
  : m === 60 ? 'hour' : periodName(m);

function showMonitor(d) {
  mon = d;
  const s = d.settings;
  $('toggle').className = 'toggle' + (s.enabled ? ' on' : '');
  $('toggle').textContent = s.enabled ? 'On - click to stop' : 'Off - click to start';
  $('every').value = s.every_minutes;
  $('pics').checked = s.save_pictures;
  let text = s.enabled
    ? `Checks ${d.cameras_selected} camera(s) every ${everyName(s.every_minutes)}.`
    : 'Switched off - nothing is being checked.';
  if (d.checking_now) text += ` Checking ${d.checking_now} now…`;
  else if (d.last_finished_at) {
    const sm = d.last_summary;
    text += ` Last round ${hhmm(d.last_finished_at)}` + (sm
      ? `: ${sm.readable} of ${sm.checked} answered, ${sm.with_people} with people.` : '.');
  }
  if (s.enabled && d.next_at && !d.checking_now) text += ` Next at ${hhmm(d.next_at)}.`;
  if (d.last_error) text += ' Problem: ' + d.last_error;
  $('monstate').textContent = text;
  $('runnow').disabled = !!d.checking_now;
  if (d.last_finished_at && d.last_finished_at !== lastFinished) {
    if (lastFinished !== null) loadRecords();
    lastFinished = d.last_finished_at;
  }
}
async function loadMonitor() { showMonitor(await api('/api/v1/monitor')); }
async function change(values) {
  try { showMonitor(await api('/api/v1/monitor', {method: 'PUT', json: values})); }
  catch (e) { $('monstate').textContent = e.message; }
}
$('toggle').onclick = () => change({enabled: !mon.settings.enabled});
$('every').onchange = () => change({every_minutes: +$('every').value});
$('pics').onchange = () => change({save_pictures: $('pics').checked});
$('runnow').onclick = async () => {
  $('runnow').disabled = true;
  try { await api('/api/v1/monitor/run-now', {method: 'POST'}); }
  catch (e) { $('monstate').textContent = e.message; }
  setTimeout(loadMonitor, 800);
};

function query() {
  const p = new URLSearchParams({day: $('day').value, period: $('period').value});
  if ($('camera').value) p.set('camera', $('camera').value);
  return p.toString();
}
async function loadRecords() {
  const q = query();
  $('csv').href = '/api/v1/records.csv?' + q;
  const d = await api('/api/v1/records?' + q);
  const keep = $('camera').value;
  $('camera').innerHTML = '<option value="">All cameras</option>' + d.cameras.map(c =>
    `<option value="${esc(c)}">${esc(c)}</option>`).join('');
  $('camera').value = keep;
  $('s-checks').textContent = d.totals.checks;
  $('s-busy').textContent = d.totals.periods_with_people;
  $('s-max').textContent = d.totals.max_people ?? '—';
  $('s-bad').textContent = d.totals.unreadable_checks;
  const box = $('table');
  if (!d.rows.length) {
    box.innerHTML = 'No checks on this day yet. Switch automatic checking on above, '
      + 'or press <b>Check all cameras now</b>.';
    return;
  }
  box.innerHTML = `<table class="simple"><thead><tr><th>Time</th><th>Camera</th><th>People?</th>
    <th>Most at once</th><th>Average</th><th>Checks</th><th>Busiest moment</th><th></th>
    </tr></thead><tbody>${
    d.rows.map(r => `<tr><td><b>${esc(r.start)}–${esc(r.end)}</b></td>
      <td>${esc(r.camera)}<div class="hint">${esc(r.venue)}</div></td>
      <td>${r.people_seen == null ? '<span class="no">no picture</span>'
        : r.people_seen ? '<span class="yes">Yes</span>' : '<span class="no">No</span>'}</td>
      <td>${r.max_people ?? '—'}</td><td>${r.avg_people ?? '—'}</td>
      <td>${r.checks}${r.readable < r.checks
        ? ` <span class="hint">(${r.checks - r.readable} without picture)</span>` : ''}</td>
      <td>${r.max_people == null ? '' : esc(crowd(r.max_people))}</td>
      <td>${r.busiest_has_picture ? `<a class="btnlink" target="_blank"
        href="/api/v1/records/picture/${r.busiest_check_id}">Picture</a>` : ''}</td>
      </tr>`).join('')
  }</tbody></table>`;
}

(async () => {
  const opts = await api('/api/v1/monitor/options');
  $('every').innerHTML = opts.every_minutes
    .map(m => `<option value="${m}">every ${everyName(m)}</option>`).join('');
  $('period').innerHTML = opts.period_minutes
    .map(m => `<option value="${m}">${periodName(m)}</option>`).join('');
  $('period').value = opts.default_period;
  const local = new Date(Date.now() - new Date().getTimezoneOffset() * 60000);
  $('day').value = local.toISOString().slice(0, 10);
  const reload = () => loadRecords().catch(e => { $('table').textContent = e.message; });
  for (const id of ['day', 'period', 'camera']) $(id).onchange = reload;
  await loadMonitor();
  await loadRecords();
  setInterval(() => loadMonitor().catch(() => {}), 5000);
})().catch(e => { $('table').textContent = e.message; });
"""


@router.get("/records", response_class=HTMLResponse, include_in_schema=False)
def records_page() -> HTMLResponse:
    return _html("Records", RECORDS_BODY, RECORDS_JS, "records")


# --- More -------------------------------------------------------------------------------------

MORE = (
    ("/client", "Pitch verdicts", "The booking-slot dashboard: used / not used / review for each "
                                  "rental slot, with evidence pictures."),
    ("/reports", "Reports", "Usage over days and weeks, busy hours."),
    ("/testing", "Testing with videos", "Upload a video or pictures and see what the system "
                                        "decides."),
    ("/cameras", "Cameras (advanced)", "Recording status, two-camera spot checks and the slot "
                                       "schedule for the recorder."),
    ("/bookings", "Bookings", "Connect booking records and check them against the cameras."),
    ("/roi", "Pitch boundaries", "Draw the outline of the pitch on each camera, so only "
                                 "people on the pitch are counted."),
    ("/alerts", "Alerts", "Messages when a camera stops or the recorder is down."),
)


@router.get("/more", response_class=HTMLResponse, include_in_schema=False)
def more_page() -> HTMLResponse:
    links = "".join(f'<a href="{href}"><b>{escape(title)}</b><span>{escape(text)}</span></a>'
                    for href, title, text in MORE)
    body = ('<section class="hero"><h1>More tools</h1><p>The full set of pages, for '
            'setting up recording of booked slots and for checking results in detail.</p>'
            f'</section><div class="card morelist">{links}</div>')
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
