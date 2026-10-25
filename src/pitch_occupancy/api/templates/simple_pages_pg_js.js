
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
