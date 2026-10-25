
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
