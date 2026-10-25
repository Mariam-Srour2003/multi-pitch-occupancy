
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
