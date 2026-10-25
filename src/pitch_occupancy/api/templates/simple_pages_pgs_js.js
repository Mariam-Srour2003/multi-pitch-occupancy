
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
