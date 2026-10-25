
const VERDICTS = ['Playing', 'Played a little', 'Mostly empty', 'Some people, no game',
                  'Empty', 'Not sure', 'No pictures'];
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
