
const TAG = {'Playing': 'v-Playing', 'People, not playing': 'v-Some-people-no-game',
             'Empty': 'v-Empty', 'Not sure': 'v-Not-sure'};
const pct = x => x == null ? '–' : Math.round(x * 100) + '%';
let D = null, shown = [], at = 0;

function query() {
  const q = new URLSearchParams();
  if ($('day').value) q.set('day', $('day').value);
  if ($('pg').value) q.set('playground', $('pg').value);
  if ($('lab').value) q.set('label', $('lab').value);
  return q;
}

function card(r, i) {
  const pic = r.url ? `<img loading="lazy" alt="" src="${esc(r.url)}" data-i="${i}">`
    : '<div class="gone">Picture deleted to stay under the size limit</div>';
  const ball = r.ball == null ? '–' : r.ball ? 'yes (' + pct(r.ball_confidence) + ')' : 'no';
  const pg = r.playground_playing == null ? 'not decided'
    : (r.playground_playing ? 'playing' : 'not playing')
      + (r.playing_source === 'assist' ? ' (second opinion)' : '');
  return `<div class="pic">${pic}<div class="meta">
    <span class="tag ${TAG[r.label] || ''}">${esc(r.label || '?')}</span>
    <b>${esc(hhmm(r.checked_at))}</b> · ${esc(r.playground || '')} · ${esc(r.camera)}<br>
    People inside: <b>${r.people ?? '–'}</b> · seen in picture: ${r.people_seen ?? '–'}
    · ball: ${esc(ball)} · confidence ${pct(r.confidence)}<br>
    Playground: ${r.playground_people ?? '–'} people, ${esc(pg)}<br>
    <span class="hint">${esc(r.reason || '')}</span>${r.second_opinion
      ? `<br>Second opinion: <b>${esc(r.second_opinion)}</b> <span class="hint">${
        esc(r.second_opinion_reason || '')}</span>` : ''}
  </div></div>`;
}

async function load() {
  const d = await api('/api/v1/pictures?' + query());
  D = d;
  if (!$('day').options.length || !d.days.includes($('day').value)) {
    const days = d.days.length ? d.days : [d.day];
    $('day').innerHTML = days.map(x => `<option>${esc(x)}</option>`).join('');
    $('day').value = d.day;
  }
  if ($('pg').options.length === 1) {
    $('pg').innerHTML += d.playgrounds.map(p =>
      `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join('');
  }
  if ($('lab').options.length === 1) {
    $('lab').innerHTML += d.labels.map(l => `<option>${esc(l)}</option>`).join('');
  }
  $('csv').href = '/api/v1/pictures.csv?' + query();
  const mb = Math.round(d.usage.bytes / 1e6);
  $('usage').textContent = `${d.usage.files} pictures kept, ${mb} MB of ${
    Math.round(d.usage.cap_bytes / 1e6)} MB; the oldest are deleted first.`;
  $('counts').innerHTML = Object.entries(d.counts).map(([k, n]) =>
    `<span class="tag ${TAG[k] || ''}">${esc(k)}: ${n}</span>`).join('');
  shown = d.rows.filter(r => r.url);
  $('pics').innerHTML = d.rows.length ? d.rows.map((r) => card(r, shown.indexOf(r))).join('')
    : '<div class="empty">No pictures saved on this day yet.</div>';
  $('pics').querySelectorAll('img[data-i]').forEach(img => {
    img.onclick = () => show(+img.dataset.i);
  });
}

function show(i) {
  if (!shown.length) return;
  at = (i + shown.length) % shown.length;
  const r = shown[at];
  $('box').querySelector('img').src = r.url;
  $('box').querySelector('.cap').textContent = `${r.checked_at.slice(0, 10)} ${
    r.checked_at.slice(11, 19)} · ${r.playground} · ${r.camera}: ${r.label} · ${
    people(r.people)} inside · ${r.reason || ''}  (${at + 1}/${shown.length})`;
  $('box').classList.add('on');
}
const closeBox = () => $('box').classList.remove('on');
$('box').querySelector('.prev').onclick = () => show(at - 1);
$('box').querySelector('.next').onclick = () => show(at + 1);
$('box').querySelector('.x').onclick = closeBox;
document.addEventListener('keydown', e => {
  if (!$('box').classList.contains('on')) return;
  if (e.key === 'Escape') closeBox();
  if (e.key === 'ArrowLeft') show(at - 1);
  if (e.key === 'ArrowRight') show(at + 1);
});
['day', 'pg', 'lab'].forEach(id => { $(id).onchange = load; });
load().catch(e => { $('pics').innerHTML = `<div class="empty">${esc(e.message)}</div>`; });
setInterval(() => { if (!$('box').classList.contains('on')) load(); }, 60000);
