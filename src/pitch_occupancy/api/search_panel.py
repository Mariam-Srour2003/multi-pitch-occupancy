"""The saved preprocessing searches (WP3-T8).

Six finished runs - three backbones over two frame counts - each charted on demand. The
chart is the point: which switch helps, by how much, and whether it bought that recall
honestly.

Two things it is built to make visible rather than leave in a column:

* **Direction.** Bars diverge from the baseline, so a configuration that made things worse
  reads as worse before any number is parsed.
* **How the recall was earned.** Every cross-venue test set here is entirely active play,
  so recall can be bought by simply predicting "playing" more often. A configuration that
  gains recall while also raising false alarms is marked, not ranked.
"""

from __future__ import annotations

__all__ = ["PANEL_STYLES", "PANEL_SCRIPT", "ARCHIVE_HTML"]

PANEL_STYLES = """
.runbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:var(--surface);
 border:1px solid var(--line);border-radius:11px;padding:15px 18px;margin:16px 0}
.runbar label{font:500 12px 'JetBrains Mono',monospace;color:var(--ink-3);
 letter-spacing:.06em;text-transform:uppercase}
.runbar select{font:13px Archivo,sans-serif;padding:7px 10px;border:1px solid var(--line);
 border-radius:7px;background:var(--ground);color:var(--ink)}
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
.b-stale{fill:none;stroke:var(--ink-3);stroke-width:1;stroke-dasharray:3 2}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;font-size:12px;color:var(--ink-3)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;
 vertical-align:-1px}
.note{font-size:12.5px;color:var(--ink-3);margin-top:10px;max-width:72ch}
.alert{background:var(--warn-soft);color:var(--ink-2);border-radius:9px;padding:12px 16px;
 margin:12px 0;font-size:13.5px;max-width:74ch}
.alert b{color:var(--ink)}
.runbar .when{font:500 12.5px 'JetBrains Mono',monospace;color:var(--ink-3)}
/* The grid of six. Every cell is listed whether or not it has been run, because "which of
   these is missing" is the question the selector is answering. */
.cells{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0 0}
.cells button{font:500 12px 'JetBrains Mono',monospace;padding:6px 10px;border-radius:7px;
 border:1px solid var(--line);background:var(--surface);color:var(--ink-2);cursor:pointer}
.cells button:hover{border-color:var(--accent);color:var(--accent)}
.cells button[aria-current="true"]{background:var(--ink);color:var(--ground);
 border-color:var(--ink)}
.cells button.empty{color:var(--ink-3);border-style:dashed}
.cells button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""

#: The saved-run viewer. Six runs - three backbones over two frame counts - and the live
#: state file holds one of them at a time, so without this every finished run would be
#: destroyed by the next one being started.
ARCHIVE_HTML = """
<h2>View old results</h2>
<p class="note">Three backbones over two frame counts. Each finished run is saved here, so the six can be compared without re-running any of them.</p>
<div class="runbar">
  <label for="aModel">Backbone</label>
  <select id="aModel">
    <option value="dinov2">DINOv2</option>
    <option value="convnextv2">ConvNeXtV2</option>
    <option value="vit">ViT</option>
  </select>
  <label for="aScope">Frames</label>
  <select id="aScope">
    <option value="all">All frames</option>
    <option value="500">500 frames</option>
  </select>
  <span class="when" id="aWhen"></span>
</div>
<div class="cells" id="aCells"></div>
<div id="aAlert"></div>
<div id="aChart"></div>
"""

PANEL_SCRIPT = """
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
    // An entry the repaired control never scored gets its own class and is never drawn as
    // a win. `false_play > 0.001` treated the old placeholder 0.0 as a clean sheet, so 45
    // broken rows rendered green - the opposite of the safeguard this panel describes.
    const stale = r.rescored === false;
    const cls = r.label === "baseline" ? "b-base"
      : stale ? "b-stale"
      : flagged ? "b-flag" : r.delta >= 0 ? "b-up" : "b-down";
    const name = r.label.length > 26 ? r.label.slice(0, 25) + "\\u2026" : r.label;
    return `
      <text class="lbl" x="${padL - 10}" y="${y + 15}" text-anchor="end">${esc(name)}</text>
      <rect class="${cls}" x="${x}" y="${y + 5}" width="${Math.max(len, 1.5)}" height="14" rx="2">
        <title>${esc(r.label)} — recall ${r.recall.toFixed(4)}, worst fold ${r.worst.toFixed(3)}, false-play ${stale ? "not re-scored" : r.false_play.toFixed(4)}</title>
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
      <span><i class="b-stale" style="background:transparent;border:1px dashed var(--ink-3)"></i>not
        re-scored - predates the false-play repair, so it has no usable false-play figure</span>
    </div>
    <p class="note">Every cross-venue test set is entirely active play, so recall can be
    bought by predicting &ldquo;playing&rdquo; more often. Amber bars did exactly that and
    are not improvements.</p>
  </div>`;
}

/* --- saved runs ----------------------------------------------------------------
   The six cells are rendered from the server's index rather than from this list of
   options, so a cell that has never been run says so instead of drawing an empty chart
   that looks like a finished one. Both are empty; only one of them means "run it". */
const aModel = document.getElementById("aModel");
const aScope = document.getElementById("aScope");
const aCells = document.getElementById("aCells");
const aChart = document.getElementById("aChart");
const aAlert = document.getElementById("aAlert");
const aWhen = document.getElementById("aWhen");

function renderCells(cells) {
  const m = aModel.value, sc = aScope.value;
  aCells.innerHTML = cells.map(c => `
    <button data-model="${c.model}" data-scope="${c.scope}"
      class="${c.saved ? "" : "empty"}"
      aria-current="${c.model === m && c.scope === sc}">
      ${esc(c.model_label)} · ${esc(c.scope_label)} · ${
        c.saved ? c.scored + "/" + c.evaluations + " scored" : "not run"}
    </button>`).join("");
  [...aCells.children].forEach(b => {
    b.onclick = () => {
      aModel.value = b.dataset.model;
      aScope.value = b.dataset.scope;
      loadSaved();
    };
  });
}

async function loadSaved() {
  const m = aModel.value, sc = aScope.value;
  let r;
  try {
    r = await fetch(`/api/v1/search/runs/${m}/${sc}`).then(x => x.json());
  } catch { return; }
  fetch("/api/v1/search/runs").then(x => x.json()).then(g => renderCells(g.cells))
    .catch(() => {});
  aWhen.textContent = r.saved
    ? `${r.evaluations} evaluations · ${r.n_frames.join(", ") || "?"} frames${
        r.generated ? " · " + r.generated.slice(0, 16).replace("T", " ") : ""}`
    : "";
  aAlert.innerHTML = r.warning
    ? `<div class="alert"><b>Careful.</b> ${esc(r.warning)}</div>` : "";
  if (!r.saved) {
    aChart.innerHTML = `<p class="missing"><b>${esc(r.model_label)} · ${
      esc(r.scope_label)}</b> has not been run yet. Six runs fill this grid: three
      backbones over two frame counts. Run
      <code>uv run python experiments/search_all.py</code> to fill every empty cell in
      order.</p>`;
    return;
  }
  const scored = r.results.filter(x => typeof x.recall === "number");
  if (!scored.length) {
    aChart.innerHTML = `<p class="missing">${r.evaluations} evaluation${
      r.evaluations === 1 ? "" : "s"} saved, none of them scored — every
      <code>play_recall</code> in this run is <code>NaN</code>. There is nothing to chart
      until it is re-run.</p>`;
    return;
  }
  const best = scored[0];
  aChart.innerHTML = `<div class="alert" style="background:var(--surface-2)">
    <b>Best on ${esc(r.model_label)}, ${esc(r.scope_label).toLowerCase()}:</b>
    <code>${esc(best.label)}</code> at ${best.recall.toFixed(4)} recall, worst fold
    ${best.worst.toFixed(3)}.</div>` + barChart(scored, r.baseline ?? 0);
}

aModel.onchange = loadSaved;
aScope.onchange = loadSaved;
loadSaved();
"""
