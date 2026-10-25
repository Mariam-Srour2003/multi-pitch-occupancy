
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
