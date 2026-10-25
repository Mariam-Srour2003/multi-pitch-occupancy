
async function load() {
  const d = await api('/api/v1/playgrounds');
  const on = d.playgrounds.filter(p => p.enabled).length;
  $('runner').hidden = !on || d.running;
  $('runner').textContent = 'The playground checks are not running. Restart the server.';
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
      : p.overdue ? '<span class="tag off" title="Not checked for more than three times its '
        + 'interval">Not checked lately</span>'
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
