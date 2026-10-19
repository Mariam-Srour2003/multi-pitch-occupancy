"""The simple front end: playgrounds first, few words, nothing technical on screen.

* ``/``                         Home - every playground as a card: how many people at the last
                                check, playing or not. Click one to see its cameras.
* ``/playground/<id>``          One playground: the total, and each camera's count and picture.
* ``/playgrounds``              Make and change playgrounds: which cameras, how often, which
                                hours, how the cameras are combined - with what this computer
                                can manage shown as you choose (`playgrounds.capacity`).
* ``/cameras/add``              Cameras - add (name, address, password), test, remove.
* ``/live/<place>/<camera>``    One camera on its own, refreshing itself.
* ``/results``                  For showing how it went: each 1.5-hour session's verdict, the
                                people over time, and the pictures saved (`results_page.py`).
* ``/records``                  The database: for each playground and each hour and a half,
                                were there people, the most at once, how long it was played on.
* ``/more``                     The original pages, all still there.

Everything here calls `playgrounds.py` or the APIs the original pages already use
(`cameras_page`); nothing about how a camera is read or counted differs between them.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date
from html import escape

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from pitch_occupancy import playgrounds
from pitch_occupancy.api.layout import page
from pitch_occupancy.config import settings

__all__ = ["router"]

router = APIRouter()

#: Records are grouped into periods of this many minutes - the hour and a half the site asked
#: for. The API still accepts the other lengths in `playgrounds.PERIOD_CHOICES`.
RECORD_PERIOD = 90


# --- shared look and helpers ---------------------------------------------------------------

CSS = """
h1{font-size:26px;margin:4px 0 16px;letter-spacing:-.02em}
h3{margin:0}
.top{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;
  margin-bottom:6px}
.top h1{margin:4px 0}
.cards{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}
a.cam{display:block;padding:16px 18px;border:1px solid var(--line);border-radius:12px;
  background:var(--surface);text-decoration:none;color:var(--ink);transition:border-color .15s}
a.cam:hover{border-color:var(--accent)}
a.cam b{display:block;font-size:16px}
a.cam .n{font:700 30px Archivo,sans-serif;margin-top:8px;line-height:1.1}
a.cam .w{color:var(--ink-3);font-size:13px;margin-top:4px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;
  background:var(--ink-3);vertical-align:middle}
.dot.many{background:var(--play)}.dot.few{background:var(--maint)}
.tag{display:inline-block;font-size:12px;font-weight:600;padding:2px 9px;border-radius:999px;
  background:var(--surface-2);color:var(--ink-2)}
.tag.play{background:var(--play-soft);color:var(--play)}
.tag.off{background:var(--flag-soft);color:var(--flag)}
.empty{padding:28px;text-align:center;color:var(--ink-2)}
.form{display:grid;gap:12px 16px;grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.form label,.lbl{display:block;font-weight:600;font-size:13.5px;margin-bottom:5px}
.form input,.form select,select.plain,input.plain{font:14px Archivo,sans-serif;
  padding:9px 11px;border:1px solid var(--line);border-radius:7px;background:var(--ground);
  color:var(--ink);width:100%}
select.plain,input.plain{width:auto}
.choices{display:flex;flex-wrap:wrap;gap:8px}
.choices label{display:flex;gap:6px;align-items:center;padding:7px 11px;
  border:1px solid var(--line);border-radius:8px;background:var(--ground);font-size:14px;
  cursor:pointer}
.choices label.used{opacity:.45;cursor:not-allowed}
.choices input{width:auto;margin:0}
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
.toggle{font:700 14px Archivo,sans-serif;padding:7px 16px;border-radius:999px;cursor:pointer;
  border:1px solid var(--line);background:var(--surface-2);color:var(--ink-2)}
.toggle.on{background:var(--play);border-color:var(--play);color:#fff}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.meter{height:12px;border-radius:6px;background:var(--surface-2);overflow:hidden;margin:8px 0}
.meter i{display:block;height:100%;background:var(--play);width:0;transition:width .3s}
.meter.warn i{background:var(--maint)}.meter.over i{background:var(--flag)}
.camgrid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
  margin-top:16px}
.camtile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px}
.camtile img{width:100%;border-radius:6px;display:block;margin-top:8px;
  background:var(--surface-2)}
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
const dot = n => n == null || n === 0 ? 'dot' : n < 5 ? 'dot few' : 'dot many';
const every = s => s < 60 ? s + ' seconds' : s === 60 ? 'minute' : s / 60 + ' minutes';
const liveHref = c =>
  '/live/' + encodeURIComponent(c.venue) + '/' + encodeURIComponent(c.camera_id);
const camPath = c =>
  '/api/v1/cameras/' + encodeURIComponent(c.venue) + '/' + encodeURIComponent(c.camera_id);
const pgPath = id => '/api/v1/playgrounds/' + encodeURIComponent(id);
"""


def _html(title: str, body: str, script: str, active: str) -> HTMLResponse:
    return HTMLResponse(page(title, "Camera monitor", body, active=active, css=CSS,
                             script=JS_COMMON + script))


def _conn():
    from pitch_occupancy.db.schema import connect, initialise

    conn = connect(settings.db_path)
    initialise(conn)
    return conn


def _find(playground_id: str) -> dict:
    found = next((p for p in playgrounds.load() if p["id"] == playground_id), None)
    if found is None:
        raise HTTPException(404, f"no playground {playground_id!r}")
    return found


def _json_block(element_id: str, data: dict) -> str:
    """Data for a page's script, in a non-executable block with "<" escaped so a name can
    never close the tag. The page only ever sets text from it."""
    payload = json.dumps(data).replace("<", "\\u003c")
    return f'<script type="application/json" id="{element_id}">{payload}</script>\n'


# --- Home: every playground, one card each -------------------------------------------------

HOME_BODY = """
<div class="top"><h1>Playgrounds</h1>
  <a class="small" href="/playgrounds">Change playgrounds</a></div>
<div class="cards" id="cards"><div class="empty">Loading…</div></div>
"""

HOME_JS = r"""
async function load() {
  const d = await api('/api/v1/playgrounds');
  if (!d.playgrounds.length) {
    $('cards').innerHTML = '<div class="empty">No playgrounds yet.<br><br>'
      + '<a class="toggle on" style="text-decoration:none" href="/playgrounds">'
      + 'Make your first playground</a></div>';
    return;
  }
  $('cards').innerHTML = d.playgrounds.map(p => {
    const l = p.latest;
    let n = '<span class="hint" style="font-size:16px">Not checked yet</span>';
    if (l && l.people == null) n = '<span class="bad" style="font-size:18px">No picture</span>';
    else if (l) n = `<span class="${dot(l.people)}"></span>${esc(people(l.people))}`;
    const tag = !p.enabled ? '<span class="tag off">Off</span>'
      : l && l.playing ? '<span class="tag play">Playing</span>' : '';
    const k = p.cameras.length;
    const w = `${k} camera${k === 1 ? '' : 's'} · every ${every(p.every_seconds)}`
      + (l ? ` · ${hhmm(l.checked_at)}` : '');
    return `<a class="cam" href="/playground/${encodeURIComponent(p.id)}">
      <div class="row" style="justify-content:space-between"><b>${esc(p.name)}</b>${tag}</div>
      <div class="n">${n}</div><div class="w">${esc(w)}</div></a>`;
  }).join('');
}
load().catch(e => { $('cards').innerHTML = `<div class="empty">${esc(e.message)}</div>`; });
setInterval(() => load().catch(() => {}), 5000);
"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    return _html("Playgrounds", HOME_BODY, HOME_JS, "home")


# --- One playground ---------------------------------------------------------------------------

PG_BODY = """
<p><a href="/">← All playgrounds</a></p>
<div class="top"><h1 id="title"></h1>
  <button class="act" id="look">Show pictures now</button></div>
<div class="row" style="align-items:flex-end;gap:22px">
  <div class="big" id="num">…</div>
  <div><div class="word" id="word"></div><div class="hint" id="when"></div></div>
</div>
<p class="hint" id="how"></p>
<div class="camgrid" id="cams"></div>
"""

PG_JS = r"""
const PG = JSON.parse($('pg').textContent);
document.title = PG.name;
$('title').textContent = PG.name;
let pictures = {};
function show(d) {
  const l = d.latest;
  $('how').textContent = ({
    largest: 'The cameras see the same area: the biggest count is used.',
    average: 'The cameras see the same area: their counts are averaged.',
    vote: 'Each camera decides: playing when at least half the cameras see a game.',
  }[PG.counting] || 'The cameras see different parts: their counts are added.')
    + ` Playing means ${d.play_min} or more people.`;
  if (!l) { $('num').textContent = '–'; $('word').textContent = 'Not checked yet'; }
  else if (l.people == null) { $('num').textContent = '–'; $('word').textContent = 'No picture'; }
  else {
    $('num').textContent = l.people;
    $('word').textContent = l.playing ? 'Playing' : l.people === 0 ? 'Nobody'
      : people(l.people) + ', not playing';
  }
  $('when').textContent = l
    ? `Checked at ${l.checked_at.slice(11, 19)} · every ${every(PG.every_seconds)}`
    : (d.enabled ? '' : 'This playground is switched off.');
  $('cams').innerHTML = d.cameras.map(c => {
    const pic = pictures[c.camera];
    const what = !c.checked ? '<span class="hint">Not checked yet</span>'
      : !c.readable ? `<span class="bad">No picture</span>`
      : `<b>${esc(people(c.people))}</b>`;
    return `<div class="camtile"><div class="row" style="justify-content:space-between">
      <span>${esc(c.camera.split('/').pop())}</span>${what}</div>
      ${c.checked && !c.readable && c.problem ? `<div class="hint">${esc(c.problem)}</div>` : ''}
      ${pic ? `<img alt="" src="data:image/jpeg;base64,${pic}">` : ''}</div>`;
  }).join('');
}
async function refresh() { show(await api(pgPath(PG.id) + '/latest')); }
$('look').onclick = async () => {
  $('look').disabled = true; $('look').textContent = 'Taking pictures…';
  try {
    const r = await api(pgPath(PG.id) + '/look', {method: 'POST'});
    pictures = Object.fromEntries(r.cameras.map(c => [c.camera, c.picture]));
    await refresh();
  } catch (e) { $('how').textContent = e.message; }
  $('look').disabled = false; $('look').textContent = 'Show pictures now';
};
refresh().catch(e => { $('how').textContent = e.message; });
setInterval(() => refresh().catch(() => {}), 5000);
"""


@router.get("/playground/{playground_id}", response_class=HTMLResponse,
            include_in_schema=False)
def playground_page(playground_id: str) -> HTMLResponse:
    p = _find(playground_id)
    data = {k: p[k] for k in ("id", "name", "counting", "every_seconds")}
    return _html(p["name"], _json_block("pg", data) + PG_BODY, PG_JS, "home")


# --- Playgrounds: make and change -------------------------------------------------------------

PGS_BODY = """
<div class="top"><h1>Playgrounds</h1>
  <button class="act" id="newbtn">+ New playground</button></div>

<div class="card" style="margin-bottom:18px">
  <div class="row" style="justify-content:space-between"><b>This computer</b>
    <button class="small" id="measure">Measure speed</button></div>
  <div class="meter" id="meter"><i id="meterbar"></i></div>
  <div id="captext" class="hint"></div>
</div>

<div class="card" id="editcard" hidden style="margin-bottom:18px">
  <h3 id="edittitle">New playground</h3>
  <div class="form" style="margin-top:12px">
    <div><label for="pname">Name</label><input id="pname" placeholder="e.g. Playground 1"></div>
    <div><label for="pevery">Check every</label><select id="pevery"></select></div>
    <div><label for="psaveevery">Save pictures</label><select id="psaveevery">
      <option value="0">Never</option><option value="1">Every check</option>
      <option value="2">1 check in 2</option><option value="5">1 check in 5</option>
      <option value="10">1 check in 10</option><option value="30">1 check in 30</option>
    </select></div>
  </div>
  <div style="margin-top:14px"><span class="lbl">Cameras</span>
    <div class="choices" id="pcams"></div></div>
  <div style="margin-top:14px"><span class="lbl">The cameras see…</span>
    <div class="choices">
      <label><input type="radio" name="counting" value="add" checked> different parts of the
        playground - add their counts</label>
      <label><input type="radio" name="counting" value="largest"> the same area - use the
        biggest count</label>
      <label><input type="radio" name="counting" value="average"> the same area - use the
        average count</label>
      <label><input type="radio" name="counting" value="vote"> each camera decides - playing
        when at least half say playing</label></div>
    <div class="hint" style="margin-top:6px">Adding counts each person once only if each
      camera counts its own part. Draw each camera's part under More → Pitch boundaries so
      the parts do not overlap.</div></div>
  <div style="margin-top:14px"><span class="lbl">When</span>
    <div class="row"><label class="row" style="font-weight:400;gap:6px">
      <input type="checkbox" id="allday" checked style="width:auto"> All day</label>
      <span id="hoursbox" hidden>from <input type="time" class="plain" id="hfrom"
        value="08:00"> to <input type="time" class="plain" id="hto" value="23:00"></span>
    </div></div>
  <div class="hint" id="cost" style="margin-top:12px"></div>
  <div class="row" style="margin-top:14px"><button class="act" id="psave">Save</button>
    <button class="small" id="pcancel">Cancel</button></div>
  <div class="msg" id="pmsg"></div>
</div>

<div class="card scroll" id="list"><div class="empty">Loading…</div></div>
"""

PGS_JS = r"""
let D = null, editing = null;
const say = (t, good) => {
  $('pmsg').className = 'msg ' + (good === undefined ? '' : good ? 'good' : 'bad');
  $('pmsg').textContent = t;
};
function showCapacity(cap) {
  $('meterbar').style.width = Math.min(cap.percent, 100) + '%';
  $('meter').className = 'meter' + (cap.percent > 100 ? ' over' : cap.percent > 80 ? ' warn' : '');
  const at = cap.cameras_at;
  const n = k => k === 1 ? '1 camera' : k + ' cameras';
  const how = cap.measured ? 'measured ' + esc(cap.measured_at.slice(0, 10))
    : 'a guess - press Measure speed';
  $('captext').innerHTML = `Can analyse about <b>${Math.round(cap.can_per_min)}</b> pictures
    a minute (${how}). Your playgrounds use <b>${Math.round(cap.need_per_min)}</b>
    (${cap.percent}%).<br>That is enough for about ${n(at[5])} checked every 5 seconds,
    ${n(at[30])} every 30 seconds, or ${n(at[60])} every minute.`;
}
const selectedCams = () =>
  [...document.querySelectorAll('#pcams input:checked')].map(i => i.value);
function updateCost() {
  if (!D) return;
  const n = selectedCams().length, s = +$('pevery').value;
  const mine = n * 60 / s;
  const others = D.playgrounds.filter(p => p.enabled && p.id !== editing)
    .reduce((t, p) => t + p.cameras.length * 60 / p.every_seconds, 0);
  const total = others + mine, can = D.capacity.can_per_min;
  $('cost').innerHTML = !n ? '' : `This playground: ${Math.round(mine * 10) / 10} pictures a
    minute. All playgrounds: ${Math.round(total)} of ${Math.round(can)}` + (total > can
    ? ' - <b class="bad">too much for this computer</b>' : ' - fine');
}
function openForm(p) {
  editing = p ? p.id : null;
  $('edittitle').textContent = p ? 'Change ' + p.name : 'New playground';
  $('pname').value = p ? p.name : 'Playground ' + (D.playgrounds.length + 1);
  $('pevery').value = p ? p.every_seconds : 30;
  $('psaveevery').value = p ? p.save_every : 5;
  document.querySelector(`input[name=counting][value=${p ? p.counting : 'add'}]`).checked = true;
  $('allday').checked = !(p && p.hours);
  $('hoursbox').hidden = $('allday').checked;
  if (p && p.hours) { $('hfrom').value = p.hours.from; $('hto').value = p.hours.to; }
  const mine = new Set(p ? p.cameras : []);
  $('pcams').innerHTML = D.cameras.map(c => {
    const used = c.playground && c.playground !== editing;
    return `<label class="${used ? 'used' : ''}"
      title="${used ? 'In ' + esc(c.playground_name) : ''}"><input type="checkbox"
      value="${esc(c.ref)}" ${mine.has(c.ref) ? 'checked' : ''} ${used ? 'disabled' : ''}>
      ${esc(c.camera_id)}</label>`;
  }).join('') || '<span class="hint">No cameras saved yet - add them on the Cameras page.</span>';
  document.querySelectorAll('#pcams input').forEach(i => { i.onchange = updateCost; });
  say('');
  $('editcard').hidden = false;
  updateCost();
  $('pname').focus();
}
$('newbtn').onclick = () => openForm(null);
$('pcancel').onclick = () => { $('editcard').hidden = true; };
$('allday').onchange = () => { $('hoursbox').hidden = $('allday').checked; };
$('pevery').onchange = updateCost;
$('psave').onclick = async () => {
  const body = {
    name: $('pname').value.trim(), cameras: selectedCams(), every_seconds: +$('pevery').value,
    counting: document.querySelector('input[name=counting]:checked').value,
    save_every: +$('psaveevery').value,
    hours: $('allday').checked ? null : {from: $('hfrom').value, to: $('hto').value},
  };
  $('psave').disabled = true;
  try {
    if (editing) await api(pgPath(editing), {method: 'PUT', json: body});
    else await api('/api/v1/playgrounds', {method: 'POST', json: body});
    $('editcard').hidden = true;
    await load();
  } catch (e) { say(e.message, false); }
  $('psave').disabled = false;
};
async function toggle(p) {
  try {
    await api(pgPath(p.id) + '/enabled', {method: 'POST', json: {enabled: !p.enabled}});
    await load();
  } catch (e) { alert(e.message); }
}
async function remove(p) {
  if (!confirm(`Delete "${p.name}"? Its records stay.`)) return;
  try { await api(pgPath(p.id), {method: 'DELETE'}); await load(); }
  catch (e) { alert(e.message); }
}
$('measure').onclick = async () => {
  $('measure').disabled = true; $('measure').textContent = 'Measuring… (up to a minute)';
  try { await api('/api/v1/computer-speed', {method: 'POST'}); await load(); }
  catch (e) { $('captext').textContent = e.message; }
  $('measure').disabled = false; $('measure').textContent = 'Measure speed';
};
function row(p, i) {
  return `<tr><td><b>${esc(p.name)}</b>
      <div class="hint">${esc(D.counting_short[p.counting] || p.counting)}${
        p.save_every ? ' · saves 1 in ' + p.save_every : ''}</div></td>
    <td>${p.cameras.map(c => esc(c.split('/').pop())).join(', ')}</td>
    <td>${every(p.every_seconds)}</td>
    <td>${p.hours ? esc(p.hours.from) + '–' + esc(p.hours.to) : 'All day'}</td>
    <td><div class="row"><button class="toggle ${p.enabled ? 'on' : ''}" data-t="${i}">${
      p.enabled ? 'On' : 'Off'}</button>
      <button class="small" data-e="${i}">Change</button>
      <button class="small red" data-d="${i}">Delete</button></div></td></tr>`;
}
async function load() {
  D = await api('/api/v1/playgrounds');
  showCapacity(D.capacity);
  $('pevery').innerHTML = D.every_choices
    .map(s => `<option value="${s}">${every(s)}</option>`).join('');
  if (!D.playgrounds.length) {
    $('list').innerHTML = '<div class="empty">No playgrounds yet. '
      + 'Click <b>+ New playground</b>.</div>';
    return;
  }
  $('list').innerHTML = `<table class="simple"><thead><tr><th>Playground</th><th>Cameras</th>
    <th>Every</th><th>When</th><th></th></tr></thead>
    <tbody>${D.playgrounds.map(row).join('')}</tbody></table>`;
  const at = (sel, fn) => $('list').querySelectorAll(sel).forEach(b => {
    b.onclick = () => fn(D.playgrounds[+Object.values(b.dataset)[0]]);
  });
  at('[data-t]', toggle); at('[data-e]', openForm); at('[data-d]', remove);
}
load().catch(e => { $('list').innerHTML = `<div class="empty">${esc(e.message)}</div>`; });
"""


@router.get("/playgrounds", response_class=HTMLResponse, include_in_schema=False)
def playgrounds_page() -> HTMLResponse:
    return _html("Playgrounds", PGS_BODY, PGS_JS, "playgrounds")


# --- Cameras: add, test, remove ---------------------------------------------------------------

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
  const on = (sel, fn) => $('list').querySelectorAll(sel).forEach(b => {
    b.onclick = () => fn(+Object.values(b.dataset)[0]);
  });
  on('[data-test]', test); on('[data-del]', remove);
}
loadList().catch(e => { $('list').innerHTML = `<div class="empty">${esc(e.message)}</div>`; });
"""


@router.get("/cameras/add", response_class=HTMLResponse, include_in_schema=False)
def cameras_simple() -> HTMLResponse:
    return _html("Cameras", CAMERAS_BODY, CAMERAS_JS, "cameras-simple")


# --- One camera, live -------------------------------------------------------------------------

LIVE_BODY = """
<p><a href="/cameras/add">← Cameras</a></p>
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
      $('word').textContent = r.people == null ? 'Could not count' : people(r.people);
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
    body = _json_block("cam", {"venue": venue, "camera": camera}) + LIVE_BODY
    return _html(camera, body, LIVE_JS, "cameras-simple")


@router.get("/cameras/test", include_in_schema=False)
@router.get("/live", include_in_schema=False)
def old_addresses() -> RedirectResponse:
    """The first version's separate Test and Live pages, folded into Cameras and Home."""
    return RedirectResponse("/", status_code=307)


# --- Records ------------------------------------------------------------------------------------

RECORDS_BODY = """
<div class="top"><h1>Records</h1></div>
<p class="hint" style="margin-top:0">What each playground's checks found, for every hour and a
  half of the day.</p>
<div class="row" style="margin:16px 0 12px">
  <input type="date" class="plain" id="day">
  <select class="plain" id="pg"><option value="">All playgrounds</option></select>
  <a class="small" id="csv" href="#">Download</a>
</div>
<div class="card scroll" id="table"><div class="empty">Loading…</div></div>
"""

RECORDS_JS = r"""
const PERIOD = __PERIOD__;
let names = {};
async function loadRecords() {
  const p = new URLSearchParams({day: $('day').value, period: PERIOD});
  if ($('pg').value) p.set('playground', $('pg').value);
  $('csv').href = '/api/v1/records.csv?' + p;
  const d = await api('/api/v1/records?' + p);
  if (!d.rows.length) {
    $('table').innerHTML = '<div class="empty">Nothing recorded on this day.</div>';
    return;
  }
  $('table').innerHTML = `<table class="simple"><thead><tr><th>Time</th><th>Playground</th>
    <th>Most people</th><th>Playing?</th><th>Played for about</th></tr></thead><tbody>${
    d.rows.map(r => `<tr><td><b>${esc(r.start)}–${esc(r.end)}</b></td>
      <td>${esc(names[r.playground_id] || r.playground_id)}</td>
      <td>${r.max_people == null ? '<span class="bad">No picture</span>' : r.max_people}</td>
      <td>${r.playing_checks ? '<span class="yes">Yes</span>' : '<span class="no">No</span>'}</td>
      <td>${r.playing_checks ? r.minutes_playing + ' min' : ''}</td></tr>`).join('')
  }</tbody></table>`;
}
(async () => {
  const d = await api('/api/v1/playgrounds');
  names = Object.fromEntries(d.playgrounds.map(p => [p.id, p.name]));
  $('pg').innerHTML = '<option value="">All playgrounds</option>' + d.playgrounds.map(p =>
    `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join('');
  const local = new Date(Date.now() - new Date().getTimezoneOffset() * 60000);
  $('day').value = local.toISOString().slice(0, 10);
  const reload = () => loadRecords().catch(e => { $('table').textContent = e.message; });
  $('day').onchange = reload; $('pg').onchange = reload;
  await loadRecords();
  setInterval(reload, 30000);
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
    ("/roi", "Pitch boundaries", "Draw each camera's part of the playground, so people are "
                                 "counted once and only on the pitch."),
    ("/alerts", "Alerts", "Messages when a camera or the recorder stops."),
)


@router.get("/more", response_class=HTMLResponse, include_in_schema=False)
def more_page() -> HTMLResponse:
    links = "".join(f'<a href="{href}"><b>{escape(title)}</b><span>{escape(text)}</span></a>'
                    for href, title, text in MORE)
    body = f'<h1>More tools</h1><div class="card morelist">{links}</div>'
    return _html("More tools", body, "", "more")


# --- JSON: playgrounds --------------------------------------------------------------------------

class PlaygroundIn(BaseModel):
    name: str = Field(max_length=60)
    cameras: list[str] = Field(max_length=32)
    every_seconds: int
    counting: str = "add"
    hours: dict[str, str] | None = None
    save_every: int = 5


class EnabledIn(BaseModel):
    enabled: bool


def _saved_cameras() -> list[dict]:
    from pitch_occupancy import cameras

    try:
        return cameras.list_cameras()
    except ValueError:
        return []


@router.get("/api/v1/playgrounds")
def list_playgrounds() -> dict:
    items = playgrounds.load()
    state = playgrounds.status()["playgrounds"]
    conn = _conn()
    try:
        latest = playgrounds.latest(conn)
    finally:
        conn.close()
    owner = {c: p for p in items for c in p["cameras"]}
    cams = []
    for c in _saved_cameras():
        ref = f"{c['venue']}/{c['camera_id']}"
        home = owner.get(ref, {})
        cams.append({"ref": ref, "venue": c["venue"], "camera_id": c["camera_id"],
                     "playground": home.get("id"), "playground_name": home.get("name")})
    return {
        "playgrounds": [{**p, "latest": latest.get(p["id"]), "status": state.get(p["id"], {})}
                        for p in items],
        "cameras": cams,
        "capacity": playgrounds.capacity(items),
        "every_choices": list(playgrounds.EVERY_CHOICES),
        "counting_short": playgrounds.COUNTING_SHORT,
        "play_min": playgrounds.play_min(),
    }


def _saved(call):
    try:
        return call()
    except KeyError as exc:
        raise HTTPException(404, f"no playground {exc.args[0]!r}") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/api/v1/playgrounds")
def create_playground(body: PlaygroundIn) -> dict:
    return _saved(lambda: playgrounds.save_playground(body.model_dump()))


@router.put("/api/v1/playgrounds/{playground_id}")
def update_playground(playground_id: str, body: PlaygroundIn) -> dict:
    current = _find(playground_id)
    values = {**body.model_dump(), "enabled": current["enabled"]}
    return _saved(lambda: playgrounds.save_playground(values, playground_id=playground_id))


@router.delete("/api/v1/playgrounds/{playground_id}")
def delete_playground(playground_id: str) -> dict:
    _saved(lambda: playgrounds.delete_playground(playground_id))
    return {"ok": True}


@router.post("/api/v1/playgrounds/{playground_id}/enabled")
def switch_playground(playground_id: str, body: EnabledIn) -> dict:
    return _saved(lambda: playgrounds.set_enabled(playground_id, body.enabled))


@router.post("/api/v1/playgrounds/{playground_id}/look")
def look_now(playground_id: str) -> dict:
    """Check the playground now, with each camera's picture. The result is stored like any
    other check; the pictures are returned and not kept."""
    record = playgrounds.check_playground(_find(playground_id), with_pictures=True)
    keep = ("camera", "readable", "people", "problem", "picture")
    return {**{k: v for k, v in record.items() if k != "cameras"},
            "cameras": [{k: c[k] for k in keep} for c in record["cameras"]]}


@router.get("/api/v1/playgrounds/{playground_id}/latest")
def playground_latest(playground_id: str) -> dict:
    p = _find(playground_id)
    conn = _conn()
    try:
        last = playgrounds.latest(conn).get(playground_id)
        rows: dict[str, dict] = {}
        if last is not None:
            rows = {f"{r['venue_id']}/{r['camera_id']}": dict(r) for r in conn.execute(
                "SELECT * FROM camera_checks WHERE playground_check_id = ?",
                (last["check_id"],))}
    finally:
        conn.close()
    return {
        "latest": last, "enabled": p["enabled"], "play_min": playgrounds.play_min(),
        "cameras": [{"camera": ref, "checked": ref in rows,
                     "readable": bool(rows.get(ref, {}).get("readable")),
                     "people": rows.get(ref, {}).get("people"),
                     "problem": rows.get(ref, {}).get("problem")} for ref in p["cameras"]],
    }


@router.post("/api/v1/computer-speed")
def computer_speed() -> dict:
    """Measure how fast this computer handles one picture, and save it for the capacity."""
    try:
        result = playgrounds.measure_speed()
    except Exception as exc:  # noqa: BLE001 - reported, not a 500
        raise HTTPException(503, f"The speed could not be measured "
                                 f"({type(exc).__name__}).") from exc
    return {"speed": result, "capacity": playgrounds.capacity()}


# --- JSON: records ------------------------------------------------------------------------

def _day(value: str | None) -> date:
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(422, "day must look like 2026-10-17") from exc


def _records(day: str | None, period: int, playground: str | None) -> tuple[date, list[dict]]:
    d = _day(day)
    conn = _conn()
    try:
        rows = playgrounds.records(conn, d, period_minutes=period, playground_id=playground)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        conn.close()
    return d, rows


@router.get("/api/v1/records")
def records_json(day: str | None = None, period: int = Query(RECORD_PERIOD),
                 playground: str | None = None) -> dict:
    d, rows = _records(day, period, playground)
    return {"day": d.isoformat(), "period_minutes": period, "rows": rows}


@router.get("/api/v1/records.csv")
def records_csv(day: str | None = None, period: int = Query(RECORD_PERIOD),
                playground: str | None = None) -> Response:
    d, rows = _records(day, period, playground)
    names = {p["id"]: p["name"] for p in playgrounds.load()}
    buf = io.StringIO()
    fields = ["day", "start", "end", "playground", "max_people", "avg_people", "playing",
              "minutes_playing", "checks", "checks_without_picture"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({**r, "playground": names.get(r["playground_id"], r["playground_id"]),
                         "playing": "yes" if r["playing_checks"] else "no"})
    return Response(buf.getvalue(), media_type="text/csv", headers={
        "Content-Disposition":
            f'attachment; filename="playground-records-{d.isoformat()}.csv"'})
