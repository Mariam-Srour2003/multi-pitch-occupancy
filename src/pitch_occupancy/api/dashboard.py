"""The operator dashboard, served at `/` (WP6-T6).

Distinct from the project site in `results/project_site.html`: that one reports the
research, this one is the tool a facility manager opens. It reads live database state
through the same `/api/v1` endpoints an integrator would use - nothing here has privileged
access, so if the page can show it, the API can serve it.

The flow it is shaped around is the one the blueprint asks for: land on what needs
attention, open a slot's evidence, agree or correct in one click. Anything that does not
serve that flow is not here.
"""

from __future__ import annotations

from fastapi.responses import HTMLResponse

__all__ = ["DASHBOARD_HTML"]

DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pitch Occupancy - Operator Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap">
<style>
:root{
  color-scheme:light;
  --ground:#f6f8f7;--surface:#fff;--surface-2:#eef2f1;--line:#dde4e2;
  --ink:#111817;--ink-2:#4d5c5a;--ink-3:#7a8886;
  --accent:#0d6d78;--accent-soft:#d7ebed;
  --used:#2c7a52;--used-soft:#dcefe3;--notused:#5c6a6d;--notused-soft:#e7ebea;
  --review:#8a6d1f;--review-soft:#f6ecd4;--serious:#a8402f;--serious-soft:#f6dfda;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --ground:#0e1414;--surface:#161e1e;--surface-2:#1c2625;--line:#2b3736;
  --ink:#eaf1ef;--ink-2:#a3b2af;--ink-3:#7b8a88;
  --accent:#4fb3bf;--accent-soft:#12363a;
  --used:#5cb884;--used-soft:#152f24;--notused:#8b9a97;--notused-soft:#222c2b;
  --review:#cfae57;--review-soft:#2c2614;--serious:#e08a76;--serious-soft:#331d18;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#0e1414;--surface:#161e1e;--surface-2:#1c2625;--line:#2b3736;
  --ink:#eaf1ef;--ink-2:#a3b2af;--ink-3:#7b8a88;
  --accent:#4fb3bf;--accent-soft:#12363a;
  --used:#5cb884;--used-soft:#152f24;--notused:#8b9a97;--notused-soft:#222c2b;
  --review:#cfae57;--review-soft:#2c2614;--serious:#e08a76;--serious-soft:#331d18;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.55 Archivo,ui-sans-serif,system-ui,sans-serif}
header{border-bottom:1px solid var(--line);background:var(--surface)}
.hin{max-width:1080px;margin:0 auto;padding:16px 24px;display:flex;align-items:baseline;
  gap:14px;flex-wrap:wrap}
.brand{font-weight:700;letter-spacing:-.01em}.brand span{color:var(--accent)}
.hin .sub{color:var(--ink-3);font-size:13px}
.hin a{color:var(--accent);font-size:13px;text-decoration:none}
.hin a:first-of-type{margin-left:auto}
main{max-width:1080px;margin:0 auto;padding:26px 24px 70px}
h2{font-size:17px;margin:30px 0 4px;font-weight:600;letter-spacing:-.01em}
.sub2{color:var(--ink-2);font-size:13px;margin:0 0 14px}
.tiles{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 15px}
.tile .k{font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:5px}
.tile .v{font:700 26px 'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;
  letter-spacing:-.02em}
.tile.alert{border-left:4px solid var(--serious)}
.tile.alert .v{color:var(--serious)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;min-width:620px}
th{font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-3);text-align:left;padding:11px 14px;border-bottom:1px solid var(--line);
  background:var(--surface-2);white-space:nowrap}
td{padding:11px 14px;border-bottom:1px solid var(--line);font-size:13.5px;vertical-align:middle}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--surface-2)}
.mono{font-family:'JetBrains Mono',monospace;font-size:12.5px}
.num{font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;text-align:right}
.pill{display:inline-block;font:600 11px 'JetBrains Mono',monospace;letter-spacing:.04em;
  padding:3px 9px;border-radius:5px;white-space:nowrap}
.USED{background:var(--used-soft);color:var(--used)}
.NOTUSED{background:var(--notused-soft);color:var(--notused)}
.REVIEW{background:var(--review-soft);color:var(--review)}
.sev-serious{background:var(--serious-soft);color:var(--serious)}
.sev-warning{background:var(--review-soft);color:var(--review)}
.bar{height:7px;border-radius:2px;background:var(--surface-2);overflow:hidden;min-width:74px;
  display:inline-block;vertical-align:middle}
.bar i{display:block;height:100%;background:var(--used)}
button.act{font:600 12px Archivo,sans-serif;padding:6px 11px;border-radius:6px;cursor:pointer;
  border:1px solid var(--line);background:var(--surface);color:var(--ink-2)}
button.act:hover{border-color:var(--accent);color:var(--accent)}
button.act:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.empty{padding:26px;color:var(--ink-2);font-size:14px}
.empty code{background:var(--surface-2);padding:2px 6px;border-radius:4px;
  font-family:'JetBrains Mono',monospace;font-size:12.5px}
dialog{border:1px solid var(--line);border-radius:12px;background:var(--surface);color:var(--ink);
  padding:0;max-width:620px;width:calc(100% - 40px)}
dialog::backdrop{background:rgba(0,0,0,.45)}
.dh{padding:18px 22px;border-bottom:1px solid var(--line)}
.dh h3{margin:0 0 4px;font-size:16px}
.db{padding:18px 22px;max-height:52vh;overflow-y:auto}
.df{padding:14px 22px;border-top:1px solid var(--line);display:flex;gap:8px;flex-wrap:wrap;
  align-items:center}
.df input{font:13px Archivo,sans-serif;padding:7px 10px;border:1px solid var(--line);
  border-radius:6px;background:var(--ground);color:var(--ink)}
.df .spacer{margin-left:auto}
.ev{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 16px}
.ev span{font-family:'JetBrains Mono',monospace;font-size:11.5px;background:var(--accent-soft);
  color:var(--accent);padding:4px 9px;border-radius:5px}
.ev figure{margin:0;display:flex;flex-direction:column;gap:5px}
.ev figure img{width:210px;height:140px;object-fit:cover;border-radius:7px;
  border:1px solid var(--line);background:var(--accent-soft);display:block}
.ev figcaption{font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--muted)}
/* An evidence frame retention has deleted must read as gone, never as a blank picture that
   an operator might take for an empty pitch. */
.ev figure.gone img{display:none}
.ev figure.gone figcaption::after{content:' - image no longer on disk';color:var(--alert)}
.matrix-head{display:flex;align-items:center;gap:9px;margin:0 0 12px;font-size:12.5px;
  color:var(--ink-2)}
.matrix-head input{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;
  background:var(--ground);color:var(--ink)}
.mx{border-collapse:separate;border-spacing:5px 6px}
.mx th{font-size:11px;color:var(--ink-3);font-weight:500;text-align:left;padding:0 2px}
.mx td.field{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--ink-2);
  padding-right:10px;white-space:nowrap}
.chip{display:inline-flex;flex-direction:column;gap:2px;min-width:74px;padding:6px 9px;
  border-radius:7px;font-size:11.5px;line-height:1.25;cursor:pointer;border:1px solid transparent}
.chip .t{font-family:'JetBrains Mono',monospace;font-weight:600}
.chip .c{font-size:10px;opacity:0.75}
/* A slot with no verdict is hollow, never a filled neutral chip: an unobserved hour and an
   empty pitch are different claims and must not share a colour. */
.chip.none{background:transparent;border:1px dashed var(--line);color:var(--ink-3)}
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);background:var(--ink);
  color:var(--ground);padding:11px 18px;border-radius:8px;font-size:13.5px;z-index:50}
.toast[hidden]{display:none}
</style></head><body>

<header><div class="hin">
  <div class="brand">Pitch Occupancy<span>.</span></div>
  <div class="sub">Operator dashboard</div>
  <a href="/">&larr; Thesis</a><a href="/clip" style="margin-left:14px">Clip reviewer</a><a href="/docs" style="margin-left:14px">API docs &rarr;</a>
</div></header>

<main>
  <div class="tiles" id="tiles"></div>

  <h2>Needs attention</h2>
  <p class="sub2">Discrepancies between the booking record and what the cameras observed.
  Slots the system was unsure about are not listed here &mdash; uncertainty is not an
  accusation.</p>
  <div class="scroll"><table>
    <thead><tr><th>Slot</th><th>Anomaly</th><th>Severity</th><th>Explanation</th></tr></thead>
    <tbody id="anoms"></tbody>
  </table></div>

  <h2>Field matrix</h2>
  <p class="sub2">Every pitch, hour by hour, for one day. A slot with no verdict is drawn
  hollow &mdash; an hour nobody looked at is the case reconciliation exists to catch, so it
  must not look the same as an empty pitch.</p>
  <div class="matrix-head"><label for="mday">Day</label>
    <input type="date" id="mday" onchange="loadMatrix()"></div>
  <div class="scroll" id="matrix"></div>

  <h2>Slots</h2>
  <p class="sub2">Click a slot to see the evidence behind its verdict and agree or correct it.</p>
  <div class="scroll"><table>
    <thead><tr><th>Slot</th><th>Verdict</th><th class="num">Play</th><th>Activity</th>
      <th class="num">Samples</th><th class="num">Confidence</th><th></th></tr></thead>
    <tbody id="slots"></tbody>
  </table></div>
</main>

<dialog id="dlg"><form method="dialog">
  <div class="dh"><h3 id="dt">Slot</h3><div class="sub2" id="dr"></div></div>
  <div class="db" id="dbody"></div>
  <div class="df">
    <input id="op" placeholder="Your name" aria-label="Operator name">
    <button class="act" value="cancel">Close</button>
    <span class="spacer"></span>
    <button class="act" id="bUSED" type="button">Mark USED</button>
    <button class="act" id="bNOTUSED" type="button">Mark NOTUSED</button>
  </div>
</form></dialog>
<div class="toast" id="toast" hidden></div>

<script>
const api = p => fetch("/api/v1" + p).then(r => r.json());
let current = null;

function pill(s){ return `<span class="pill ${s}">${s}</span>`; }
function toast(msg){
  const t = document.getElementById("toast");
  t.textContent = msg; t.hidden = false;
  setTimeout(() => { t.hidden = true; }, 2600);
}

async function loadMatrix(){
  const input = document.getElementById("mday");
  if (!input.value) {
    // Default to the most recent day that has any evaluation, not to today: a demo or a
    // fresh install with no slots today would otherwise show an empty grid and read as a
    // broken page rather than as a quiet day.
    // The date is pulled out with a character class rather than \\d, because this JavaScript
    // lives inside a non-raw Python string and a lone backslash-d there is an invalid escape
    // that Python only warns about - it happens to work, which is the worst kind of working.
    const recent = await api("/slots?limit=1");
    const dated = recent.length
      ? recent[0].slot_id.match(/[0-9]{4}-[0-9]{2}-[0-9]{2}/) : null;
    input.value = dated ? dated[0] : new Date().toISOString().slice(0, 10);
  }
  const fields = await api(`/fields/day/${input.value}`);
  const box = document.getElementById("matrix");
  if (!fields.length) {
    box.innerHTML = `<p class="empty">No slots scheduled on this day.</p>`;
    return;
  }
  box.innerHTML = `<table class="mx"><tbody>${fields.map(f => `<tr>
      <td class="field">${f.field_id}</td>
      ${f.slots.map(s => {
        const verdict = s.is_overridden ? s.override_status : s.status;
        const conf = s.mean_confidence == null ? "" : (s.mean_confidence * 100).toFixed(0) + "%";
        if (!verdict) {
          return `<td><span class="chip none" title="scheduled, never evaluated">
            <span class="t">${s.start_time}</span><span class="c">no verdict</span></span></td>`;
        }
        return `<td><span class="chip pill ${verdict}" title="${s.slot_id}"
          onclick="openSlot('${s.slot_id}')">
          <span class="t">${s.start_time}</span>
          <span class="c">${verdict}${conf ? " · " + conf : ""}${
            s.is_overridden ? " · corrected" : ""}</span></span></td>`;
      }).join("")}</tr>`).join("")}</tbody></table>`;
}

async function load(){
  const m = await api("/meters");
  document.getElementById("tiles").innerHTML = [
    ["Slots evaluated", m.slots_evaluated, ""],
    ["Used", m.used, ""],
    ["Not used", m.notused, ""],
    ["Awaiting review", m.review, ""],
    ["Open anomalies", m.open_anomalies, m.open_anomalies > 0 ? "alert" : ""],
    ["Review rate", (m.review_rate * 100).toFixed(0) + "%", ""],
  ].map(([k, v, cls]) =>
    `<div class="tile ${cls}"><div class="k">${k}</div><div class="v">${v}</div></div>`
  ).join("");

  const an = await api("/anomalies");
  document.getElementById("anoms").innerHTML = an.length ? an.map(a => `<tr>
      <td class="mono">${a.slot_id}</td>
      <td class="mono">${a.anomaly}</td>
      <td><span class="pill sev-${a.severity}">${a.severity}</span></td>
      <td style="color:var(--ink-2)">${a.explanation}</td></tr>`).join("")
    : `<tr><td colspan="4" class="empty">Nothing to act on &mdash; every slot agrees with its
       booking record.</td></tr>`;

  await loadMatrix();

  const slots = await api("/slots?limit=100");
  document.getElementById("slots").innerHTML = slots.length ? slots.map(s => `<tr>
      <td class="mono">${s.slot_id}</td>
      <td>${pill(s.is_overridden ? s.override_status : s.status)}${
        s.is_overridden ? ' <span class="mono" style="color:var(--ink-3)">corrected</span>' : ""}</td>
      <td class="num">${s.play_ratio.toFixed(2)}</td>
      <td><span class="bar"><i style="width:${(s.play_ratio * 100).toFixed(0)}%"></i></span></td>
      <td class="num">${s.n_samples}</td>
      <td class="num">${s.mean_confidence.toFixed(2)}</td>
      <td><button class="act" data-slot="${s.slot_id}">Evidence</button></td></tr>`).join("")
    : `<tr><td colspan="7" class="empty">No slots evaluated yet. Populate the database with
       <code>uv run pitch seed</code>, then reload.</td></tr>`;

  document.querySelectorAll("button[data-slot]").forEach(b => {
    b.onclick = () => openSlot(b.dataset.slot);
  });
}

async function openSlot(id){
  current = id;
  const ev = await api(`/slots/${id}/evidence`);
  document.getElementById("dt").textContent = id;
  document.getElementById("dr").textContent = ev.reason;
  const byMinute = {};
  ev.samples.forEach(s => { (byMinute[s.minute_index] ??= []).push(s); });
  const minutes = Object.keys(byMinute).map(Number).sort((a, b) => a - b);
  document.getElementById("dbody").innerHTML = `
    <div>${pill(ev.status)}</div>
    <div class="ev">${ev.evidence_paths.length
      ? ev.evidence_paths.map((p, i) => `<figure>
          <img src="/api/v1/slots/${encodeURIComponent(id)}/evidence/${i}"
               alt="evidence frame ${i + 1} for ${id}" loading="lazy"
               onerror="this.closest('figure').classList.add('gone')">
          <figcaption>${p}</figcaption>
        </figure>`).join("")
      : `<span>no evidence images saved for this slot - the run was made without
         evidence_dir, so there is nothing to review</span>`}</div>
    <div class="sub2">${minutes.length} sampled minutes, ${ev.samples.length} camera
      observations</div>
    <div class="scroll"><table><thead><tr><th>Minute</th><th>Cameras</th></tr></thead><tbody>
      ${minutes.slice(0, 40).map(mi => `<tr><td class="mono">${mi}</td>
        <td class="mono">${byMinute[mi].map(s =>
          `${s.camera_id.replace(/^slot_\\d+_\\d+_/, "")}:${s.predicted.replace(/^C\\d_/, "")}`
        ).join("  ")}</td></tr>`).join("")}
    </tbody></table></div>
    ${minutes.length > 40 ? '<p class="sub2">Showing the first 40 minutes.</p>' : ""}`;
  document.getElementById("dlg").showModal();
}

async function override(status){
  const operator = document.getElementById("op").value.trim();
  if (!operator) { toast("Enter your name — a correction is an audit record."); return; }
  const r = await fetch(`/api/v1/slots/${current}/override`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, operator }),
  });
  if (!r.ok) { toast("Could not save that correction."); return; }
  document.getElementById("dlg").close();
  toast(`Corrected to ${status}. The model's verdict is kept alongside it.`);
  load();
}
document.getElementById("bUSED").onclick = () => override("USED");
document.getElementById("bNOTUSED").onclick = () => override("NOTUSED");

load();
</script></body></html>
"""


def dashboard_response() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)
