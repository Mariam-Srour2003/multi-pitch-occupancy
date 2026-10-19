"""Results: how the playgrounds went, session by session - the page for showing it to people.

* ``/results``  A grid of playgrounds by 1.5-hour sessions (on the grid through 09:30, see
  `playgrounds.sessions`), each cell one verdict - Playing, Played a little, Mostly empty,
  Some people no game, Empty. Click a cell for that session: the people at every check drawn
  over time against the playing line, how the verdict was reached, and the pictures saved
  during it (with each camera's count drawn on), which open full screen with arrow keys.

The pictures are the ones the checker saved (one check in ``save_every``) under
`playgrounds.snapshots.SNAPSHOT_DIR`, kept under a size cap; a session whose pictures were
deleted to make room still has its numbers.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from pitch_occupancy import playgrounds
from pitch_occupancy.api.simple_pages import _html
from pitch_occupancy.db.schema import open_db

__all__ = ["router"]

router = APIRouter()

CSS_EXTRA = """
<style>
.grid{border-collapse:separate;border-spacing:4px;font-size:13.5px}
.grid th{font-size:12px;color:var(--ink-3);font-weight:600;padding:4px 6px;text-align:center;
  white-space:nowrap}
.grid th.pg{text-align:left}
.grid td.pg{white-space:nowrap;padding-right:10px}
.grid td.c{min-width:92px;height:58px;border-radius:8px;text-align:center;cursor:pointer;
  background:var(--surface-2);color:var(--ink-2);border:2px solid transparent;line-height:1.25}
.grid td.c:hover{border-color:var(--accent)}
.grid td.c.sel{border-color:var(--ink)}
.grid td.c b{display:block;font-size:13px}
.grid td.c span{font-size:11.5px}
.v-Playing{background:var(--play)!important;color:#fff!important}
.v-Played-a-little{background:var(--play-soft)!important;color:var(--play)!important}
.v-Mostly-empty{background:var(--maint-soft,var(--surface-2))!important;
  color:var(--maint)!important}
.v-Some-people-no-game{background:var(--surface-2)!important;color:var(--ink-2)!important}
.v-Empty{background:var(--surface)!important;color:var(--ink-3)!important;
  border:1px dashed var(--line)!important}
.v-No-pictures{background:var(--flag-soft)!important;color:var(--flag)!important}
.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:12.5px;margin:10px 0}
.legend i{display:inline-block;width:14px;height:14px;border-radius:4px;vertical-align:-2px;
  margin-right:5px}
.how{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}
.how .card h3{font-size:15px;margin-bottom:6px}
.how .card p{margin:4px 0;font-size:13.5px;color:var(--ink-2)}
.stats{display:flex;gap:26px;flex-wrap:wrap;margin:8px 0 14px}
.stats div b{display:block;font:700 26px Archivo,sans-serif}
.stats div span{font-size:12.5px;color:var(--ink-3)}
.chart{width:100%;height:auto;display:block;background:var(--surface);border-radius:8px}
.shots{display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
  margin-top:10px}
.shots figure{margin:0;cursor:zoom-in}
.shots img{width:100%;border-radius:6px;display:block;background:var(--surface-2)}
.shots figcaption{font-size:12px;color:var(--ink-2);margin-top:3px}
#box{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;z-index:50;
  align-items:center;justify-content:center;flex-direction:column}
#box.on{display:flex}
#box img{max-width:96vw;max-height:84vh;border-radius:6px}
#box .cap{color:#fff;font:600 16px Archivo,sans-serif;margin-top:10px;text-align:center}
#box button{position:absolute;top:50%;background:rgba(255,255,255,.15);color:#fff;border:0;
  font-size:30px;width:52px;height:72px;border-radius:8px;cursor:pointer}
#box .prev{left:14px}#box .next{right:14px}
#box .x{top:14px;right:14px;height:48px;width:48px;font-size:22px}
</style>
"""

BODY = CSS_EXTRA + """
<div class="top"><h1>Results</h1>
  <div class="row">
    <label class="hint">From <input type="datetime-local" class="plain" id="from"></label>
    <label class="hint">to <input type="datetime-local" class="plain" id="to"></label>
    <a class="small" id="csv" href="#">Download</a></div></div>

<div class="how" id="how"></div>

<h2 style="font-size:18px;margin:22px 0 4px">Every 1.5 hours, one answer per playground</h2>
<div class="legend" id="legend"></div>
<div class="card scroll" id="gridbox"><div class="empty">Loading…</div></div>

<div class="card" id="detail" style="margin-top:18px" hidden>
  <div class="row" style="justify-content:space-between"><h3 id="dtitle"></h3>
    <button class="small" id="play">Show pictures full screen</button></div>
  <div class="stats" id="dstats"></div>
  <p class="hint" id="dwhy"></p>
  <svg class="chart" id="chart" viewBox="0 0 900 220" role="img"
    aria-label="People at each check during the session"></svg>
  <div class="shots" id="shots"></div>
</div>

<div id="box"><button class="x" title="Close (Esc)">✕</button>
  <button class="prev" title="Previous (←)">‹</button><img alt="">
  <div class="cap"></div><button class="next" title="Next (→)">›</button></div>
"""

JS = r"""
const VERDICTS = ['Playing', 'Played a little', 'Mostly empty', 'Some people, no game',
                  'Empty', 'No pictures'];
const vcls = v => 'v-' + String(v).replace(/[^A-Za-z]+/g, '-').replace(/-$/, '');
const pad = n => String(n).padStart(2, '0');
const localIso = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${
  pad(d.getHours())}:${pad(d.getMinutes())}`;
let D = null, sel = null, shots = [], at = 0;

function howCards(d) {
  const modes = d.counting;
  const pgs = d.playgrounds.map(p => `<p><b>${esc(p.name)}</b> - ${p.cameras.length} camera${
    p.cameras.length === 1 ? '' : 's'}, ${esc(d.counting_short[p.counting])}, every ${
    every(p.every_seconds)}${p.save_every ? ', picture saved 1 check in ' + p.save_every : ''}</p>`
  ).join('');
  const u = d.pictures;
  $('how').innerHTML = `
  <div class="card"><h3>1. Each check</h3>
    <p>At each check every camera of a playground takes a picture at the same moment and the
    people on the pitch are counted.</p>
    <p>A playground is <b>playing</b> when it has <b>${d.play_min} or more</b> people.</p></div>
  <div class="card"><h3>2. Cameras combined</h3>
    ${Object.entries(modes).map(([k, v]) => `<p><b>${esc(d.counting_short[k])}</b>: ${
      esc(v)}</p>`).join('')}</div>
  <div class="card"><h3>3. One answer per 1.5 hours</h3>
    <p>Sessions start at 09:30, 11:00, … 21:30, 23:00, 00:30 …</p>
    <p><b>Playing</b> when ${Math.round(d.rule.played * 100)}% or more of the checks were
    playing; <b>Played a little</b> from ${Math.round(d.rule.some * 100)}%; <b>Mostly empty</b>
    below that; <b>Empty</b> when nobody was seen.</p></div>
  <div class="card"><h3>Playgrounds</h3>${pgs}
    <p class="hint">Pictures saved: ${u.files} (${(u.bytes / 1048576).toFixed(0)} MB of
    ${(u.cap_bytes / 1048576).toFixed(0)} MB allowed - the oldest are deleted beyond that)</p>
  </div>`;
}

function grid(d) {
  $('legend').innerHTML = VERDICTS.map(v => `<span><i class="${vcls(v)}"></i>${esc(v)}</span>`)
    .join('');
  if (!d.sessions.length) {
    $('gridbox').innerHTML = '<div class="empty">No checks in this time yet.</div>';
    return;
  }
  const starts = d.columns;
  const byKey = Object.fromEntries(d.rows.map(r => [r.playground_id + '|' + r.start, r]));
  const head = starts.map(s => `<th>${s.slice(5, 10).replace('-', '/')}<br>${s.slice(11, 16)}–${
    hhmm(d.ends[s])}</th>`).join('');
  const body = d.playgrounds.map(p => `<tr><td class="pg"><b>${esc(p.name)}</b>
    <div class="hint">${p.cameras.length} cam · ${esc(d.counting_short[p.counting])}</div></td>${
    starts.map(s => {
      const r = byKey[p.id + '|' + s];
      if (!r) return '<td class="c" title="No checks"><span>–</span></td>';
      const share = r.playing_share == null ? '' : Math.round(r.playing_share * 100) + '% playing';
      const key = esc(p.id + '|' + s);
      return `<td class="c ${vcls(r.verdict)} ${sel === p.id + '|' + s ? 'sel' : ''}"
        data-k="${key}"><b>${esc(r.verdict)}</b><span>avg ${r.avg_people ?? '–'} · ${share}</span>
        </td>`;
    }).join('')}</tr>`).join('');
  $('gridbox').innerHTML = `<table class="grid"><thead><tr><th class="pg">Playground</th>${head}
    </tr></thead><tbody>${body}</tbody></table>`;
  $('gridbox').querySelectorAll('td[data-k]').forEach(td => {
    td.onclick = () => openSession(td.dataset.k);
  });
}

function chart(checks, playMin, start, end) {
  const W = 900, H = 220, L = 34, R = 10, T = 12, B = 26;
  const t0 = new Date(start).getTime(), t1 = new Date(end).getTime();
  const max = Math.max(playMin + 2, ...checks.map(c => c.people ?? 0));
  const x = t => L + (W - L - R) * (t - t0) / (t1 - t0);
  const y = n => H - B - (H - T - B) * n / max;
  let g = '';
  for (let n = 0; n <= max; n += Math.max(1, Math.ceil(max / 5))) {
    g += `<line x1="${L}" x2="${W - R}" y1="${y(n)}" y2="${y(n)}" stroke="var(--line)"/>
      <text x="${L - 6}" y="${y(n) + 4}" text-anchor="end" font-size="11"
      fill="var(--ink-3)">${n}</text>`;
  }
  for (let t = t0; t <= t1; t += 15 * 60000) {
    const d = new Date(t);
    g += `<text x="${x(t)}" y="${H - 8}" text-anchor="middle" font-size="11"
      fill="var(--ink-3)">${pad(d.getHours())}:${pad(d.getMinutes())}</text>`;
  }
  g += `<line x1="${L}" x2="${W - R}" y1="${y(playMin)}" y2="${y(playMin)}"
    stroke="var(--play)" stroke-dasharray="5 4"/>
    <text x="${W - R}" y="${y(playMin) - 5}" text-anchor="end" font-size="11"
    fill="var(--play)">playing: ${playMin}+ people</text>`;
  const pts = checks.filter(c => c.people != null)
    .map(c => [x(new Date(c.checked_at).getTime()), y(c.people), c]);
  if (pts.length > 1) g += `<polyline fill="none" stroke="var(--accent)" stroke-width="2"
    points="${pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ')}"/>`;
  g += pts.map(([px, py, c]) => `<circle cx="${px}" cy="${py}" r="3.5"
    fill="${c.playing ? 'var(--play)' : 'var(--ink-3)'}"><title>${hhmm(c.checked_at)} - ${
    people(c.people)}${c.saved ? ' (picture saved)' : ''}</title></circle>`).join('');
  $('chart').innerHTML = g;
}

async function openSession(key) {
  sel = key;
  grid(D);
  const [pid, start] = key.split('|');
  const r = D.rows.find(x => x.playground_id === pid && x.start === start);
  const d = await api('/api/v1/sessions/detail?' + new URLSearchParams({playground: pid, start}));
  $('detail').hidden = false;
  $('dtitle').textContent = `${d.playground.name} · ${start.slice(0, 10)} ${start.slice(11, 16)}–${
    hhmm(r.end)} · ${r.verdict}`;
  $('dstats').innerHTML = [
    [r.verdict, 'answer for this session'],
    [r.avg_people ?? '–', 'people on average'], [r.max_people ?? '–', 'most at once'],
    [r.playing_share == null ? '–' : Math.round(r.playing_share * 100) + '%', 'checks playing'],
    [r.minutes_playing + ' min', 'played for about'], [r.checks, 'checks'],
  ].map(([b, s]) => `<div><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join('');
  $('dwhy').textContent = `${r.playing_checks} of ${r.checks - r.checks_without_picture} checks `
    + `had ${D.play_min} or more people (${esc(D.counting_short[d.playground.counting])}).`
    + (r.checks_without_picture ? ` ${r.checks_without_picture} checks got no picture.` : '')
    + (r.assist_upgrades ? ` ${r.assist_upgrades} of them were judged playing by the second `
      + `opinion after the detector counted too few.` : '');
  chart(d.checks, D.play_min, r.start, r.end);
  shots = d.pictures;
  $('shots').innerHTML = shots.length ? shots.map((s, i) => `<figure data-i="${i}">
    <img loading="lazy" alt="" src="${esc(s.url)}"><figcaption>${hhmm(s.checked_at)} · ${
    esc(s.camera)} · <b>${esc(people(s.people))}</b> · total ${s.total ?? '–'}${
    s.playing ? ' · playing' : ''}</figcaption></figure>`).join('')
    : '<div class="hint">No pictures were saved in this session.</div>';
  $('shots').querySelectorAll('figure').forEach(f => { f.onclick = () => show(+f.dataset.i); });
  $('detail').scrollIntoView({behavior: 'smooth'});
}

function show(i) {
  if (!shots.length) return;
  at = (i + shots.length) % shots.length;
  const s = shots[at];
  $('box').querySelector('img').src = s.url;
  $('box').querySelector('.cap').textContent = `${s.checked_at.slice(0, 10)} ${
    s.checked_at.slice(11, 19)} · ${s.camera}: ${people(s.people)} · playground total ${
    s.total ?? '–'}${s.playing ? ' - PLAYING' : ''}  (${at + 1}/${shots.length})`;
  $('box').classList.add('on');
}
const closeBox = () => $('box').classList.remove('on');
$('box').querySelector('.prev').onclick = () => show(at - 1);
$('box').querySelector('.next').onclick = () => show(at + 1);
$('box').querySelector('.x').onclick = closeBox;
document.addEventListener('keydown', e => {
  if (!$('box').classList.contains('on')) return;
  if (e.key === 'ArrowLeft') show(at - 1);
  else if (e.key === 'ArrowRight') show(at + 1);
  else if (e.key === 'Escape') closeBox();
});
$('play').onclick = () => show(0);

async function load() {
  const p = new URLSearchParams({start: $('from').value, end: $('to').value});
  $('csv').href = '/api/v1/sessions.csv?' + p;
  D = await api('/api/v1/sessions?' + p);
  howCards(D);
  grid(D);
}
const now = new Date();
$('to').value = localIso(new Date(now.getTime() + 90 * 60000));
$('from').value = localIso(new Date(now.getTime() - 24 * 3600000));
const reload = () => load().catch(e => { $('gridbox').textContent = e.message; });
$('from').onchange = reload; $('to').onchange = reload;
reload();
setInterval(() => { if (!$('box').classList.contains('on')) reload(); }, 60000);
"""


@router.get("/results", response_class=HTMLResponse, include_in_schema=False)
def results_page() -> HTMLResponse:
    return _html("Results", BODY, JS, "results")


def _when(value: str | None, default: datetime) -> datetime:
    if not value:
        return default
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(422, "times must look like 2026-10-18T21:30") from exc
    return parsed if parsed.tzinfo else parsed.astimezone()


def _range(start: str | None, end: str | None) -> tuple[datetime, datetime]:
    now = datetime.now().astimezone()
    lo = _when(start, now - timedelta(days=1))
    hi = _when(end, now + timedelta(minutes=playgrounds.SESSION_MINUTES))
    # Whole sessions: widen both ends to the session they fall in.
    return playgrounds.session_bounds(lo)[0], playgrounds.session_bounds(hi)[1]


def _session_rows(start: str | None, end: str | None) -> tuple[datetime, datetime, list[dict]]:
    lo, hi = _range(start, end)
    conn = open_db()
    try:
        rows = playgrounds.sessions.sessions(conn, lo, hi)
    finally:
        conn.close()
    return lo, hi, rows


@router.get("/api/v1/sessions")
def sessions_json(start: str | None = None, end: str | None = None) -> dict:
    lo, hi, rows = _session_rows(start, end)
    columns = sorted({r["start"] for r in rows})
    return {
        "start": lo.isoformat(timespec="minutes"), "end": hi.isoformat(timespec="minutes"),
        "rows": rows, "sessions": columns, "columns": columns,
        "ends": {r["start"]: r["end"] for r in rows},
        "playgrounds": playgrounds.load(), "play_min": playgrounds.check.play_min(),
        "counting": playgrounds.COUNTING, "counting_short": playgrounds.COUNTING_SHORT,
        "rule": {"played": playgrounds.SESSION_PLAYED, "some": playgrounds.SESSION_SOME},
        "pictures": playgrounds.snapshot_usage(),
    }


@router.get("/api/v1/sessions/detail")
def session_detail(playground: str, start: str) -> dict:
    found = next((p for p in playgrounds.load() if p["id"] == playground), None)
    if found is None:
        raise HTTPException(404, f"no playground {playground!r}")
    lo, hi = playgrounds.session_bounds(_when(start, datetime.now().astimezone()))
    args = (playground, lo.isoformat(timespec="seconds"), hi.isoformat(timespec="seconds"))
    conn = open_db()
    try:
        checks = [dict(r) for r in conn.execute(
            """SELECT * FROM playground_checks WHERE playground_id = ? AND checked_at >= ?
               AND checked_at < ? ORDER BY checked_at""", args)]
        pics = [dict(r) for r in conn.execute(
            """SELECT c.camera_id, c.checked_at, c.people, c.picture_path, p.people AS total,
                      p.playing
               FROM camera_checks c JOIN playground_checks p
                 ON p.check_id = c.playground_check_id
               WHERE p.playground_id = ? AND p.checked_at >= ? AND p.checked_at < ?
                 AND c.picture_path IS NOT NULL ORDER BY c.checked_at, c.camera_id""", args)]
    finally:
        conn.close()
    saved_at = {p["checked_at"] for p in pics}
    for c in checks:
        c["saved"] = c["checked_at"] in saved_at
    folder = playgrounds.snapshots.SNAPSHOT_DIR
    pictures = [{"camera": p["camera_id"], "checked_at": p["checked_at"], "people": p["people"],
                 "total": p["total"], "playing": p["playing"],
                 "url": "/snapshots/" + p["picture_path"]}
                for p in pics if (folder / p["picture_path"]).is_file()]
    return {"playground": found, "start": lo.isoformat(timespec="minutes"),
            "end": hi.isoformat(timespec="minutes"), "checks": checks, "pictures": pictures}


@router.get("/snapshots/{path:path}", include_in_schema=False)
def snapshot(path: str) -> FileResponse:
    root = playgrounds.snapshots.SNAPSHOT_DIR.resolve()
    target = (root / Path(path)).resolve()
    if root not in target.parents or target.suffix != ".jpg" or not target.is_file():
        raise HTTPException(404, "no such picture")
    return FileResponse(target, media_type="image/jpeg")


@router.get("/api/v1/sessions.csv")
def sessions_csv(start: str | None = None, end: str | None = None) -> Response:
    lo, _, rows = _session_rows(start, end)
    buf = io.StringIO()
    fields = ["playground", "start", "end", "verdict", "avg_people", "max_people",
              "playing_share", "minutes_playing", "checks", "checks_without_picture",
              "saved_pictures", "assist_upgrades"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="sessions-{lo.date().isoformat()}.csv"'})
