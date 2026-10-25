
const DAYS = ['mon','tue','wed','thu','fri','sat','sun'];
let CAMS = [];
let STATUS = {};
const key = (c) => c.venue + '/' + c.camera_id;

function ago(minutes) {
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const h = Math.floor(minutes / 60);
  if (h < 24) return `${h} hour${h === 1 ? '' : 's'} ago`;
  const d = Math.floor(h / 24);
  return `${d} day${d === 1 ? '' : 's'} ago`;
}
function lastCell(rec) {
  if (!rec) return '<span class="muted">-</span>';
  const when = rec.last_recorded ? ago(rec.minutes_since) : 'never';
  const cls = rec.stale ? 'bad' : (rec.last_recorded ? '' : 'muted');
  let tip = rec.last_recorded ? rec.last_recorded.replace('T', ' ') : 'no picture recorded';
  if (rec.expected_24h) tip += ` - last 24 h: ${rec.captured_24h} of ${rec.expected_24h} ` +
    'minutes recorded';
  if (rec.note) tip += ` - ${rec.note}`;
  const extra = rec.expected_24h ? `<br><span class="muted">${rec.captured_24h} of ` +
    `${rec.expected_24h} min in 24 h</span>` : '';
  return `<span class="${cls}" title="${esc(tip)}">${esc(when)}</span>${extra}`;
}
function statusCell(c) {
  const r = STATUS[key(c)];
  if (!r) return '<span class="muted">not tested</span>';
  if (r.ok) return `<span class="ok">answered in ${r.elapsed_ms} ms</span>`;
  return `<span class="bad">${esc(r.message)}</span>` +
    (r.hint ? `<span class="hint">${esc(r.hint)}</span>` : '');
}

async function api(method, url, body) {
  const r = await fetch(url, {method, headers: {'Content-Type': 'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body)});
  let data = null;
  try { data = await r.json(); } catch (e) { data = null; }
  if (!r.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === 'string' ? d : (d ? JSON.stringify(d) : 'request failed'));
  }
  return data;
}

function venues() { return [...new Set(CAMS.map((c) => c.venue))].sort(); }

function camBoxes(target, venue) {
  const cams = CAMS.filter((c) => c.venue === venue);
  $(target).innerHTML = cams.length ? cams.map((c) =>
    `<label><input type="checkbox" value="${esc(c.camera_id)}" checked>` +
    `${esc(c.camera_id)}</label>`).join('') : '<span class="muted">no cameras</span>';
}
function checked(target) {
  return [...$(target).querySelectorAll('input:checked')].map((i) => i.value);
}

function fillVenues() {
  for (const [sel, boxes] of [['p_venue', 'p_cams'], ['s_venue', 's_cams']]) {
    const keep = $(sel).value;
    $(sel).innerHTML = venues().map((v) => `<option>${esc(v)}</option>`).join('');
    if (venues().includes(keep)) $(sel).value = keep;
    camBoxes(boxes, $(sel).value);
  }
}

async function loadCams() {
  const data = await api('GET', '/api/v1/cameras/registry');
  CAMS = data.cameras;
  $('camproblem').hidden = !data.problem;
  $('camproblem').textContent = data.problem || '';
  $('camrows').innerHTML = CAMS.length ? CAMS.map((c, i) => `<tr>
    <td>${esc(c.venue)}</td><td><b>${esc(c.camera_id)}</b></td>
    <td class="mono">${esc(c.url)}</td>
    <td>${c.boundary ? '<span class="ok">drawn</span>'
         : '<a href="/roi" class="bad">not drawn yet</a>'}</td>
    <td class="status">${statusCell(c)}</td><td class="last">${lastCell(c.recording)}</td>
    <td style="white-space:nowrap"><button class="ghost" data-test="${i}">Test</button>
    <button class="ghost" data-edit="${i}">Edit</button>
    <button class="ghost" data-del="${i}">Delete</button></td></tr>`).join('')
    : '<tr><td colspan="7" class="muted">No cameras yet - add the first one below.</td></tr>';
  fillVenues();
}

function showTest(r, where) {
  const img = r.snapshot ? `<img src="data:image/jpeg;base64,${r.snapshot}" alt="snapshot">` : '';
  const size = r.width ? ` - picture ${r.width}x${r.height}` : '';
  $(where).innerHTML = `<div class="snaps"><figure>${img}<figcaption>
    <div class="${r.ok ? 'ok' : 'bad'}"><b>${esc(r.message)}</b></div>
    <div class="mono">${esc(r.url)}</div>
    ${r.hint ? `<p class="note warn">${esc(r.hint)}</p>` : ''}
    <div class="muted">${esc((r.steps || []).join(' -> '))} (${r.elapsed_ms} ms${size})</div>
    </figcaption></figure></div>`;
}

async function runTest(body, where) {
  $(where).innerHTML = '<p class="muted">Connecting (up to ~10 seconds)...</p>';
  try { showTest(await api('POST', '/api/v1/cameras/test', body), where); }
  catch (e) { $(where).innerHTML = `<p class="note err">${esc(e.message)}</p>`; }
}

$('camrows').addEventListener('click', async (ev) => {
  const b = ev.target.closest('button'); if (!b) return;
  const c = CAMS[+(b.dataset.test ?? b.dataset.edit ?? b.dataset.del)];
  if (b.dataset.test !== undefined) {
    runTest({venue: c.venue, camera_id: c.camera_id}, 'testout');
    $('testout').scrollIntoView({behavior: 'smooth', block: 'nearest'});
  } else if (b.dataset.edit !== undefined) {
    $('f_venue').value = c.venue; $('f_cam').value = c.camera_id; $('f_url').value = c.url;
    $('formtitle').textContent = `Edit ${c.venue} / ${c.camera_id}`;
    $('f_url').focus();
  } else if (b.dataset.del !== undefined) {
    if (!confirm(`Delete camera ${c.camera_id} at ${c.venue}?`)) return;
    try {
      const r = await api('DELETE',
        `/api/v1/cameras/${encodeURIComponent(c.venue)}/${encodeURIComponent(c.camera_id)}`);
      $('f_msg').innerHTML = r.warning ? `<p class="note warn">${esc(r.warning)}</p>` : '';
      await loadCams();
    } catch (e) { $('f_msg').innerHTML = `<p class="note err">${esc(e.message)}</p>`; }
  }
});

function formBody() {
  return {venue: $('f_venue').value.trim(), camera_id: $('f_cam').value.trim(),
          url: $('f_url').value.trim()};
}
$('f_save').onclick = async () => {
  try {
    const r = await api('PUT', '/api/v1/cameras', formBody());
    $('f_msg').innerHTML = `<p class="note">Saved <b>${esc(r.camera.venue)} / ` +
      `${esc(r.camera.camera_id)}</b> as <span class="mono">${esc(r.camera.url)}</span>. ` +
      `Press Test to check it.</p>`;
    $('f_url').value = r.camera.url;
    await loadCams();
  } catch (e) { $('f_msg').innerHTML = `<p class="note err">${esc(e.message)}</p>`; }
};
$('f_test').onclick = () => runTest(formBody(), 'f_msg');
$('f_clear').onclick = () => {
  for (const id of ['f_venue', 'f_cam', 'f_url']) $(id).value = '';
  $('formtitle').textContent = 'Add a camera'; $('f_msg').innerHTML = '';
};
$('t_go').onclick = () => runTest({url: $('t_url').value.trim()}, 'testout');

$('testall').onclick = async () => {
  $('testall').disabled = true;
  $('testallsum').className = 'muted';
  $('testallsum').textContent = `Testing ${CAMS.length} camera(s), one after another...`;
  try {
    const r = await api('POST', '/api/v1/cameras/test-all');
    STATUS = {};
    for (const c of r.cameras) STATUS[key(c)] = c;
    $('testallsum').className = r.answered === r.total ? 'ok' : 'bad';
    $('testallsum').textContent = `${r.summary} (${r.tested_at.replace('T', ' ')}).`;
    await loadCams();
  } catch (e) {
    $('testallsum').className = 'bad'; $('testallsum').textContent = e.message;
  } finally { $('testall').disabled = false; }
};

// --- live check ---
let busy = false, timer = null;
const yesno = (v) => v === null || v === undefined ? '-' : (v ? 'yes' : 'no');
async function predict() {
  if (busy || !$('p_venue').value) return;
  busy = true; $('p_go').disabled = true; $('busy').classList.add('on');
  try {
    const r = await api('POST', '/api/v1/cameras/predict',
      {venue: $('p_venue').value, cameras: checked('p_cams')});
    const p = r.pitch;
    const cams = r.cameras.map((c) => `<figure>
      ${c.snapshot ? `<img src="data:image/jpeg;base64,${c.snapshot}" alt="${esc(c.camera_id)}">`
        : ''}
      <figcaption><b>${esc(c.camera_id)}</b>
      ${c.state ? ` <span class="pill ${esc(c.state)}">${esc(c.label)}</span>` : ''}<br>
      ${c.readable ? `People on its half: <b>${c.people ?? '-'}</b> &middot; ball seen: ` +
        `<b>${yesno(c.ball)}</b><br><span class="muted">${c.boundary ? 'counted inside the ' +
        'drawn boundary' : 'no boundary drawn - whole picture counted'}</span>`
        : `<span class="bad">${esc(c.problem || 'no picture')}</span>` +
          (c.hint ? `<p class="note warn">${esc(c.hint)}</p>` : '')}
      ${c.readable && c.problem ? `<div class="bad">${esc(c.problem)}</div>` : ''}
      </figcaption></figure>`).join('');
    $('predout').innerHTML = `<div class="verdictbar">
      <span class="pill ${esc(p.state)}">${esc(p.label)}</span>
      <span>People on the pitch: <b>${p.people ?? '-'}</b></span>
      <span>Ball seen: <b>${yesno(p.ball)}</b></span>
      <span class="spacer muted">${esc(r.taken_at.replace('T', ' '))}</span></div>
      ${(r.notes || []).map((n) => `<p class="note warn">${esc(n)}</p>`).join('')}
      <div class="snaps">${cams}</div>
      <details class="g" style="margin-top:12px"><summary>Why this answer</summary><div>
      <pre>${esc((p.trace || []).join('\n'))}</pre></div></details>`;
  } catch (e) {
    $('predout').innerHTML = `<p class="note err">${esc(e.message)}</p>`;
  } finally {
    busy = false; $('p_go').disabled = false; $('busy').classList.remove('on');
  }
}
$('p_go').onclick = predict;
$('p_venue').onchange = () => camBoxes('p_cams', $('p_venue').value);
$('p_auto').onchange = () => {
  clearInterval(timer); timer = null;
  if ($('p_auto').checked) { predict(); timer = setInterval(predict, 60000); }
};

// --- schedule ---
$('s_days').innerHTML = DAYS.map((d) =>
  `<label><input type="checkbox" value="${d}" checked>${d}</label>`).join('');
$('s_venue').onchange = () => { camBoxes('s_cams', $('s_venue').value); stale(); };
$('s_mode').onchange = () => {
  $('s_countbox').hidden = $('s_mode').value !== 'count';
  $('s_endbox').hidden = $('s_mode').value !== 'end'; stale();
};
function stale() { $('s_save').disabled = true; }
for (const id of ['s_field', 's_start', 's_len', 's_count', 's_end', 's_days', 's_cams'])
  $(id).addEventListener('input', stale);
$('s_days').addEventListener('change', stale); $('s_cams').addEventListener('change', stale);

function schedBody() {
  const b = {venue_id: $('s_venue').value, field_id: $('s_field').value.trim(),
    cameras: checked('s_cams'), start: $('s_start').value, slot_minutes: +$('s_len').value,
    days: checked('s_days')};
  if ($('s_mode').value === 'count') b.count = +$('s_count').value; else b.end = $('s_end').value;
  return b;
}
function showPlan(r, saved) {
  const rows = r.slots.map((s, i) => `<tr><td class="num">${i + 1}</td>` +
    `<td class="mono">${esc(s.start)} - ${esc(s.end)}</td><td class="num">${s.minutes}</td></tr>`)
    .join('');
  const cmds = r.commands;
  $('schedout').innerHTML = `
    ${r.problems.map((p) => `<p class="note err">${esc(p)}</p>`).join('')}
    ${r.warnings.map((p) => `<p class="note warn">${esc(p)}</p>`).join('')}
    <p class="sub2" style="margin-top:12px"><b>${esc(r.venue_id)} / ${esc(r.field_id)}</b>,
      cameras ${esc(r.cameras.join(', '))}, on ${esc(r.days.join(' '))}.
      ${saved ? 'Saved' : 'Saving will'} replace ${r.replaced} existing slot(s) for this pitch
      and keep ${r.kept} other slot(s).</p>
    <div class="scroll"><table style="min-width:0"><thead><tr><th>#</th><th>Slot</th>
      <th>Minutes</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${saved ? `<div class="card" style="margin-top:14px"><h3 class="ok">Schedule saved</h3>
      <p>Now start the live recorder on the Mini-PC, in the project folder:</p>
      <p class="muted">1. Check everything is in place (opens no camera):</p>
      <pre>${esc(cmds.check)}</pre>
      <p class="muted">2. Start recording (it asks you to type <b>yes</b>):</p>
      <pre>${esc(cmds.start)}</pre>
      <p class="muted">Only if you need pictures kept as evidence:</p>
      <pre>${esc(cmds.with_images)}</pre>
      <ul>${r.results.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>
      <p class="muted">Previous schedule kept as ${esc(r.previous_version)}.</p></div>` : ''}`;
}
$('s_preview').onclick = async () => {
  try {
    const r = await api('POST', '/api/v1/cameras/schedule/preview', schedBody());
    showPlan(r, false); $('s_save').disabled = !r.ok;
  } catch (e) { $('schedout').innerHTML = `<p class="note err">${esc(e.message)}</p>`; stale(); }
};
$('s_save').onclick = async () => {
  try { showPlan(await api('PUT', '/api/v1/cameras/schedule', schedBody()), true); stale(); }
  catch (e) { $('schedout').innerHTML = `<p class="note err">${esc(e.message)}</p>`; }
};

// --- recording status (the live worker's heartbeat) ---
function since(seconds) {
  if (seconds === null || seconds === undefined) return 'at an unknown time';
  return ago(Math.floor(seconds / 60));
}
const hhmm = (iso) => iso ? String(iso).slice(11, 16) : '-';
const dayhhmm = (iso) => iso ? String(iso).slice(0, 16).replace('T', ' ') : '-';
function recCameras(cams) {
  return Object.entries(cams || {}).map(([id, c]) =>
    `${esc(id)}: ${c.frames} picture(s), ${c.missed} missed, last ` +
    `${c.last_frame_at ? esc(hhmm(c.last_frame_at)) : 'none yet'}`).join('<br>');
}
async function loadRecording() {
  let r;
  try { r = await api('GET', '/api/v1/recording/status'); }
  catch (e) { $('recstatus').innerHTML = `<p class="bad">${esc(e.message)}</p>`; return; }
  const s = r.status;
  if (!s) {
    const never = 'The recording program has not been started on this computer.';
    $('recstatus').innerHTML = `<p><b>${never}</b> Start it in the project folder with:</p>
      <pre>${esc(r.start_command)}</pre>
      <p class="muted">To keep it running after a restart, see ${esc(r.service_doc)}.</p>`;
    return;
  }
  const head = s.alive
    ? `<span class="pill C2_ACTIVE_PLAY">Running</span><span class="muted">last seen ` +
      `${esc(since(s.age_s))}${s.model ? ' - model ' + esc(s.model) : ''}</span>`
    : `<span class="pill UNCERTAIN bad">${s.state === 'stopped' ? 'Stopped' : 'Not running'}` +
      `</span><span class="bad">last seen ${esc(since(s.age_s))}</span>` +
      (s.state === 'stopped' ? '' : '<span class="muted">(it stopped without saying so - ' +
       'check the computer and the log)</span>');
  const running = (s.slots_running || []).map((x) => `<tr><td class="mono">${esc(x.slot_id)}` +
    `</td><td>${esc(hhmm(x.start))} - ${esc(hhmm(x.end))}</td><td class="num">` +
    `${x.minutes_done} / ${x.minutes_expected}</td><td>${recCameras(x.cameras)}</td></tr>`)
    .join('');
  const next = (s.next_slots || []).map((x) => `<tr><td class="mono">${esc(x.slot_id)}</td>` +
    `<td>${esc(dayhhmm(x.start))}</td></tr>`).join('');
  const done = (s.last_finished || []).map((x) => `<tr><td class="mono">${esc(x.slot_id)}` +
    `</td><td>${esc(x.status)}</td><td>${esc(dayhhmm(x.finished_at))}</td></tr>`).join('');
  const errs = (s.errors || []).map((x) => `<li><span class="muted">${esc(dayhhmm(x.at))}` +
    `</span> ${esc(x.message)}</li>`).join('');
  $('recstatus').innerHTML = `<div class="head">${head}</div>
    ${s.alive ? '' : `<p class="note warn">Nothing is being recorded while it is not running.
      Start it with:</p><pre>${esc(r.start_command)}</pre>`}
    <h4>Recording now</h4>${running ? `<div class="scroll"><table><thead><tr><th>Slot</th>
      <th>Time</th><th>Minutes done</th><th>Cameras</th></tr></thead><tbody>${running}
      </tbody></table></div>` : '<p class="muted">No slot is being recorded right now.</p>'}
    <h4>Next slots</h4>${next ? `<div class="scroll"><table><tbody>${next}</tbody></table>
      </div>` : '<p class="muted">None today or tomorrow.</p>'}
    <h4>Last finished</h4>${done ? `<div class="scroll"><table><thead><tr><th>Slot</th>
      <th>Result</th><th>Finished</th></tr></thead><tbody>${done}</tbody></table></div>`
      : '<p class="muted">None since it started.</p>'}
    ${errs ? `<h4>Recent errors</h4><ul>${errs}</ul>` : ''}`;
}
loadRecording();
setInterval(loadRecording, 30000);

loadCams().catch((e) => {
  $('camrows').innerHTML = `<tr><td colspan="7" class="bad">${esc(e.message)}</td></tr>`;
});
