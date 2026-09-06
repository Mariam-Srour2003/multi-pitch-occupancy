"""The interactive preprocessing-search panel (WP3-T8).

Run the search from the page and watch it fill in. The chart is the point: which switch
helps, by how much, and whether it bought that recall honestly.

Two things it is built to make visible rather than leave in a column:

* **Direction.** Bars diverge from the baseline, so a configuration that made things worse
  reads as worse before any number is parsed.
* **How the recall was earned.** Every cross-venue test set here is entirely active play,
  so recall can be bought by simply predicting "playing" more often. A configuration that
  gains recall while also raising false alarms is marked, not ranked.
"""

from __future__ import annotations

__all__ = ["PANEL_HTML", "PANEL_STYLES"]

PANEL_STYLES = """
.runbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:var(--surface);
 border:1px solid var(--line);border-radius:11px;padding:15px 18px;margin:16px 0}
.runbar label{font:500 12px 'JetBrains Mono',monospace;color:var(--ink-3);
 letter-spacing:.06em;text-transform:uppercase}
.runbar select{font:13px Archivo,sans-serif;padding:7px 10px;border:1px solid var(--line);
 border-radius:7px;background:var(--ground);color:var(--ink)}
.runbar .go{font:600 13px Archivo,sans-serif;padding:9px 18px;border-radius:8px;border:0;
 background:var(--accent);color:#fff;cursor:pointer}
.runbar .go:hover{filter:brightness(1.08)}
.runbar .go:disabled{opacity:.5;cursor:not-allowed}
.runbar .go:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.runbar .ghost{background:none;border:1px solid var(--line);color:var(--ink-2);
 font:500 12.5px Archivo,sans-serif;padding:8px 13px;border-radius:8px;cursor:pointer}
.runbar .ghost:hover{border-color:var(--accent);color:var(--accent)}
.runbar .state{margin-left:auto;font:500 12.5px 'JetBrains Mono',monospace;color:var(--ink-3)}
.runbar .state.live{color:var(--accent)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent);
 margin-right:6px;vertical-align:middle}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.dot.live{animation:pulse 1.4s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.dot.live{animation:none}}
.chart{background:var(--surface);border:1px solid var(--line);border-radius:11px;
 padding:18px 20px;margin:16px 0;overflow-x:auto}
.chart h4{margin:0 0 3px;font-size:14.5px;font-weight:600}
.chart .cs{margin:0 0 14px;font-size:12.5px;color:var(--ink-3);max-width:70ch}
.chart svg{display:block;min-width:520px;width:100%}
.chart text{font-family:'JetBrains Mono',monospace;font-size:11px}
.lbl{fill:var(--ink-2)} .val{fill:var(--ink);font-weight:700}
.axis{stroke:var(--line);stroke-width:1} .zero{stroke:var(--ink-3);stroke-width:1}
.b-up{fill:var(--up,#2c7a52)} .b-down{fill:var(--down,#a8512f)}
.b-base{fill:var(--ink-3)} .b-flag{fill:var(--warn,#8a6d1f)}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;font-size:12px;color:var(--ink-3)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;
 vertical-align:-1px}
.note{font-size:12.5px;color:var(--ink-3);margin-top:10px;max-width:72ch}
.alert{background:var(--warn-soft);color:var(--ink-2);border-radius:9px;padding:12px 16px;
 margin:12px 0;font-size:13.5px;max-width:74ch}
.alert b{color:var(--ink)}
"""

PANEL_HTML = """
<div class="runbar">
  <label for="sModel">Backbone</label>
  <select id="sModel">
    <option value="convnextv2">ConvNeXtV2 (fastest)</option>
    <option value="dinov2">DINOv2</option>
    <option value="vit">ViT</option>
  </select>
  <label for="sScope">Frames</label>
  <select id="sScope">
    <option value="">All 1,578 - quotable</option>
    <option value="500">500 - quick check only</option>
  </select>
  <button class="go" id="sRun">Run search</button>
  <button class="ghost" id="sClear">Clear results</button>
  <span class="state" id="sState"></span>
</div>
<div id="sAlert"></div>
<div id="sChart"></div>
"""

PANEL_SCRIPT = """
const sState = document.getElementById("sState");
const sChart = document.getElementById("sChart");
const sAlert = document.getElementById("sAlert");
const sRun = document.getElementById("sRun");
let poller = null;

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function barChart(rows, baseline) {
  if (!rows.length) return "";
  const shown = rows.slice(0, 18);
  const rowH = 26, padL = 210, padR = 74, padT = 8, w = 900;
  const h = padT + shown.length * rowH + 26;
  const span = Math.max(0.06, ...shown.map(r => Math.abs(r.delta))) * 1.15;
  const plotW = w - padL - padR;
  const zero = padL + plotW / 2;
  const scale = d => (d / span) * (plotW / 2);

  const bars = shown.map((r, i) => {
    const y = padT + i * rowH;
    const len = Math.abs(scale(r.delta));
    const x = r.delta >= 0 ? zero : zero - len;
    // gained recall but also raised false alarms -> boundary shift, not better sight
    const flagged = r.delta > 0.002 && r.false_play > 0.001;
    const cls = r.label === "baseline" ? "b-base" : flagged ? "b-flag" : r.delta >= 0 ? "b-up" : "b-down";
    const name = r.label.length > 26 ? r.label.slice(0, 25) + "\\u2026" : r.label;
    return `
      <text class="lbl" x="${padL - 10}" y="${y + 15}" text-anchor="end">${esc(name)}</text>
      <rect class="${cls}" x="${x}" y="${y + 5}" width="${Math.max(len, 1.5)}" height="14" rx="2">
        <title>${esc(r.label)} — recall ${r.recall.toFixed(4)}, worst fold ${r.worst.toFixed(3)}, false-play ${r.false_play.toFixed(4)}</title>
      </rect>
      <text class="val" x="${w - padR + 8}" y="${y + 15}">${r.recall.toFixed(3)}</text>`;
  }).join("");

  return `<div class="chart">
    <h4>Change from the untouched baseline</h4>
    <p class="cs">Cross-venue play recall. Bars run right when a switch helped and left when
    it hurt; the number on the right is the absolute recall. Baseline ${baseline.toFixed(3)}.</p>
    <svg viewBox="0 0 ${w} ${h}" role="img"
      aria-label="Change in cross-venue recall for each preprocessing configuration">
      <line class="zero" x1="${zero}" y1="${padT}" x2="${zero}" y2="${h - 24}"/>
      ${bars}
      <text class="lbl" x="${zero}" y="${h - 6}" text-anchor="middle">baseline</text>
      <text class="lbl" x="${padL}" y="${h - 6}">worse</text>
      <text class="lbl" x="${w - padR}" y="${h - 6}" text-anchor="end">better</text>
    </svg>
    <div class="legend">
      <span><i class="b-up" style="background:var(--up,#2c7a52)"></i>improved</span>
      <span><i class="b-down" style="background:var(--down,#a8512f)"></i>worse</span>
      <span><i class="b-flag" style="background:var(--warn,#8a6d1f)"></i>gained recall but also
        raised false alarms</span>
      <span><i class="b-base" style="background:var(--ink-3)"></i>baseline</span>
    </div>
    <p class="note">Every cross-venue test set is entirely active play, so recall can be
    bought by predicting &ldquo;playing&rdquo; more often. Amber bars did exactly that and
    are not improvements.</p>
  </div>`;
}

function renderSearch(s) {
  sRun.disabled = s.running;
  sRun.textContent = s.running ? "Running\\u2026" : "Run search";
  sState.className = "state" + (s.running ? " live" : "");
  sState.innerHTML = s.running
    ? `<span class="dot live"></span>${s.evaluations} evaluated${
        s.rounds_done.length ? " \\u00b7 round " + Math.max(...s.rounds_done) : ""}`
    : s.evaluations
      ? `${s.evaluations} evaluations \\u00b7 ${s.n_frames.join(", ")} frames`
      : "not started";

  sAlert.innerHTML = s.warning ? `<div class="alert"><b>Careful.</b> ${esc(s.warning)}</div>` : "";

  if (!s.results.length) {
    sChart.innerHTML = `<p class="missing">No results yet. Pick a backbone and press
      <b>Run search</b> \\u2014 a full-size run takes a few hours and keeps going if you
      close this tab.</p>`;
    return;
  }
  const best = s.results[0];
  const head = `<div class="alert" style="background:var(--surface-2)">
    <b>Best so far:</b> <code>${esc(best.label)}</code> at ${best.recall.toFixed(4)} recall,
    worst fold ${best.worst.toFixed(3)}${
      best.delta ? `, ${best.delta >= 0 ? "+" : ""}${best.delta.toFixed(3)} against baseline` : ""}.</div>`;
  sChart.innerHTML = head + barChart(s.results, s.baseline ?? 0);
}

async function pollSearch() {
  try {
    const s = await fetch("/api/v1/search/preprocess").then(r => r.json());
    renderSearch(s);
    if (s.running && !poller) poller = setInterval(pollSearch, 5000);
    if (!s.running && poller) { clearInterval(poller); poller = null; }
  } catch { /* the server may be reloading; the next tick retries */ }
}

sRun.onclick = async () => {
  sRun.disabled = true;
  const limit = document.getElementById("sScope").value;
  const r = await fetch("/api/v1/search/preprocess", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: document.getElementById("sModel").value,
      rounds: 3,
      limit: limit ? Number(limit) : null,
    }),
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    sAlert.innerHTML = `<div class="alert"><b>Not started.</b> ${esc(e.detail || "Unknown error")}</div>`;
    sRun.disabled = false;
    return;
  }
  pollSearch();
};

document.getElementById("sClear").onclick = async () => {
  const r = await fetch("/api/v1/search/preprocess", { method: "DELETE" });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    sAlert.innerHTML = `<div class="alert"><b>Not cleared.</b> ${esc(e.detail || "")}</div>`;
    return;
  }
  pollSearch();
};

pollSearch();
"""
