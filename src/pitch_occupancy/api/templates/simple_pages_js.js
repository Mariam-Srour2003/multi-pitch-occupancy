
async function api(path, opts = {}) {
  const o = {...opts};
  if (o.json !== undefined) {
    o.body = JSON.stringify(o.json);
    o.headers = {'content-type': 'application/json'};
    delete o.json;
  }
  const r = await fetch(path, o);
  let d = null;
  try { d = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) {
    const det = d && d.detail;
    throw new Error(typeof det === 'string' ? det : 'Something went wrong (' + r.status + ')');
  }
  return d;
}
const hhmm = iso => iso ? String(iso).slice(11, 16) : '';
const people = n => n == null ? '?' : n === 1 ? '1 person' : n + ' people';
const dot = n => n == null || n === 0 ? 'dot' : n < 5 ? 'dot few' : 'dot many';
const every = s => s < 60 ? s + ' seconds' : s === 60 ? 'minute' : s / 60 + ' minutes';
const liveHref = c =>
  '/live/' + encodeURIComponent(c.venue) + '/' + encodeURIComponent(c.camera_id);
const camPath = c =>
  '/api/v1/cameras/' + encodeURIComponent(c.venue) + '/' + encodeURIComponent(c.camera_id);
const pgPath = id => '/api/v1/playgrounds/' + encodeURIComponent(id);
