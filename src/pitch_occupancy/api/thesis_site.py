"""The thesis frontend, served at `/` (WP8).

Two audiences, two front ends. This one is for the researcher and the supervisor: every
finding, every experiment, the dataset's limits, both configuration searches, the code map.
The client-facing operator dashboard lives at `/client` and shows live slot state instead.

Everything here is read from disk **at request time** - the experiment log, the
pre-registration, the RQ matrix, the result CSVs. Rerun an experiment and reload the page;
there is no build step to forget and no copy to drift.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from pitch_occupancy.api.markdown import render

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
THESIS = ROOT / "thesis"

#: tab id -> (label, source document)
DOCUMENTS = {
    "findings": ("Findings", RESULTS / "EXPERIMENT_LOG.md"),
    "prereg": ("Pre-registration", THESIS / "preregistration.md"),
    "questions": ("Questions", THESIS / "rq_matrix.md"),
    "dataset": ("Dataset", DOCS / "data_layout.md"),
    "code": ("Code", DOCS / "CODEBASE.md"),
    "ideas": ("Ideas", DOCS / "IDEAS.md"),
    "ethics": ("Ethics", THESIS / "ethics.md"),
}


def _doc(path: Path) -> str:
    if not path.exists():
        return f"<p class='missing'>Not generated yet: <code>{path.name}</code></p>"
    return render(path.read_text(encoding="utf-8"))


def _csv(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _search_summary() -> str:
    """The live state of the preprocessing search, however far it has got."""
    path = RESULTS / "preprocess_search.json"
    if not path.exists():
        return ("<p class='missing'>No preprocessing search has completed yet. Run "
                "<code>uv run python experiments/preprocess_search.py</code>.</p>")
    state = json.loads(path.read_text(encoding="utf-8"))
    evals = state.get("evaluations", [])
    if not evals:
        return "<p class='missing'>The search has started but scored nothing yet.</p>"
    frames = {e.get("n_frames") for e in evals}
    base = next((e for e in evals if e["label"] == "baseline"), None)
    rows = sorted(evals, key=lambda e: -e["play_recall"])
    body = "".join(
        f"<tr><td class='mono'>{e['label']}</td>"
        f"<td class='num'>{e['play_recall']:.4f}</td>"
        f"<td class='num'>{(e['play_recall'] - base['play_recall']):+.3f}</td>"
        f"<td class='num'>{e['worst_fold']:.3f}</td>"
        f"<td class='num'>{e['false_play']:.4f}</td>"
        f"<td class='num'>{e['round']}</td></tr>"
        for e in rows
    ) if base else ""
    note = (f"<p>{len(evals)} evaluations, scored on "
            f"{', '.join(str(f) for f in sorted(frames) if f)} frames.</p>")
    return note + (
        "<div class='scroll'><table><thead><tr><th>Configuration</th><th class='num'>Recall</th>"
        "<th class='num'>vs baseline</th><th class='num'>Worst fold</th>"
        "<th class='num'>False-play</th><th class='num'>Round</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _prompt_summary() -> str:
    rows = _csv("prompt_search.csv")
    if not rows:
        return "<p class='missing'>No prompt search results yet.</p>"
    top = sorted(rows, key=lambda r: -float(r["balanced"]))[:10]
    body = "".join(
        f"<tr><td class='mono'>{r.get('desc_ACTIVE_PLAY','')}</td>"
        f"<td class='num'>{r['n_templates']}</td>"
        f"<td class='num'>{float(r['play_recall']):.3f}</td>"
        f"<td class='num'>{float(r['false_play']):.4f}</td>"
        f"<td class='num'>{float(r['balanced']):.3f}</td></tr>"
        for r in top
    )
    return (f"<p>{len(rows)} prompt sets scored exhaustively.</p>"
            "<div class='scroll'><table><thead><tr><th>Descriptor for active play</th>"
            "<th class='num'>Templates</th><th class='num'>Recall</th>"
            "<th class='num'>False-play</th><th class='num'>Balanced</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


@lru_cache(maxsize=1)
def _shell() -> str:
    return SHELL


def page() -> str:
    tabs = "".join(
        f'<button data-view="{k}">{label}</button>' for k, (label, _) in DOCUMENTS.items()
    )
    views = "".join(
        f'<section class="view" data-view="{k}" hidden><div class="doc">{_doc(path)}</div></section>'
        for k, (_, path) in DOCUMENTS.items()
    )
    searches = (
        '<section class="view" data-view="searches" hidden><div class="doc">'
        "<h1>Configuration searches</h1>"
        "<p>Both are scored on cross-venue transfer and guarded by a false-play control: "
        "every cross-venue test set is entirely active play, so recall can be bought by "
        "saying &ldquo;playing&rdquo; more often.</p>"
        "<h2>Zero-shot prompt search</h2>" + _prompt_summary() +
        "<h2>Preprocessing search</h2>" + _search_summary() +
        "</div></section>"
    )
    return _shell().replace("__TABS__", tabs).replace("__VIEWS__", views + searches)


SHELL = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pitch Occupancy - Thesis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap">
<style>
:root{color-scheme:light;
 --ground:#f6f8f7;--surface:#fff;--surface-2:#eef2f1;--line:#dde4e2;
 --ink:#111817;--ink-2:#4b5a58;--ink-3:#7a8886;--accent:#0d6d78;--accent-soft:#d7ebed;
 --warn:#8a6d1f;--warn-soft:#f6ecd4}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --ground:#0e1414;--surface:#161e1e;--surface-2:#1c2625;--line:#2b3736;
 --ink:#eaf1ef;--ink-2:#a3b2af;--ink-3:#7b8a88;--accent:#4fb3bf;--accent-soft:#12363a;
 --warn:#cfae57;--warn-soft:#2c2614}}
:root[data-theme="dark"]{color-scheme:dark;
 --ground:#0e1414;--surface:#161e1e;--surface-2:#1c2625;--line:#2b3736;
 --ink:#eaf1ef;--ink-2:#a3b2af;--ink-3:#7b8a88;--accent:#4fb3bf;--accent-soft:#12363a;
 --warn:#cfae57;--warn-soft:#2c2614}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
 font:15px/1.62 Archivo,ui-sans-serif,system-ui,sans-serif}
nav{position:sticky;top:0;z-index:20;background:var(--ground);border-bottom:1px solid var(--line)}
.navin{max-width:1000px;margin:0 auto;padding:0 24px;display:flex;align-items:center;
 gap:16px;flex-wrap:wrap;min-height:56px}
.brand{font-weight:700;letter-spacing:-.01em;white-space:nowrap}
.brand span{color:var(--accent)}
.tag{font:500 10px 'JetBrains Mono',monospace;letter-spacing:.1em;text-transform:uppercase;
 background:var(--accent-soft);color:var(--accent);padding:3px 8px;border-radius:5px}
#tabs{display:flex;gap:2px;flex-wrap:wrap;margin-left:auto}
#tabs button{font:500 13px Archivo,sans-serif;background:none;border:0;color:var(--ink-2);
 padding:8px 11px;border-radius:7px;cursor:pointer}
#tabs button:hover{background:var(--surface-2);color:var(--ink)}
#tabs button[aria-current="true"]{background:var(--ink);color:var(--ground)}
#tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.client{font-size:13px;color:var(--accent);text-decoration:none;white-space:nowrap}
main{max-width:1000px;margin:0 auto;padding:30px 24px 80px}
.view[hidden]{display:none}
.doc h1{font-size:clamp(25px,3.4vw,34px);line-height:1.15;margin:0 0 14px;font-weight:700;
 letter-spacing:-.022em;text-wrap:balance}
.doc h2{font-size:19px;margin:34px 0 8px;font-weight:600;letter-spacing:-.012em;
 padding-top:14px;border-top:1px solid var(--line)}
.doc h3{font-size:15.5px;margin:22px 0 6px;font-weight:600}
.doc h4{font-size:14px;margin:18px 0 4px;font-weight:600;color:var(--ink-2)}
.doc p{max-width:72ch;color:var(--ink-2);margin:0 0 12px}
.doc strong{color:var(--ink);font-weight:600}
.doc a{color:var(--accent)}
.doc ul{color:var(--ink-2);max-width:72ch;padding-left:20px;margin:0 0 14px}
.doc li{margin-bottom:6px}
.doc hr{border:0;border-top:1px solid var(--line);margin:26px 0}
.doc blockquote{margin:14px 0;padding:13px 17px;background:var(--warn-soft);
 border-radius:9px;color:var(--ink-2);max-width:74ch}
.doc blockquote strong{color:var(--ink)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:9px;
 background:var(--surface);margin:14px 0}
table{border-collapse:collapse;width:100%;min-width:480px}
th{font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.08em;text-transform:uppercase;
 color:var(--ink-3);text-align:left;padding:10px 13px;border-bottom:1px solid var(--line);
 background:var(--surface-2);white-space:nowrap}
td{padding:9px 13px;border-bottom:1px solid var(--line);font-size:13.5px;color:var(--ink-2)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--surface-2)}
td strong{color:var(--ink)}
.mono{font-family:'JetBrains Mono',monospace;font-size:12.5px}
.num{font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;text-align:right;
 white-space:nowrap}
pre{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:13px 16px;
 overflow-x:auto;font:12.5px/1.65 'JetBrains Mono',monospace;color:var(--ink);margin:12px 0}
code{font-family:'JetBrains Mono',monospace;font-size:12.5px;background:var(--surface-2);
 padding:1px 5px;border-radius:4px}
pre code{background:none;padding:0}
.missing{color:var(--ink-3);font-style:italic}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body>

<nav><div class="navin">
  <div class="brand">Pitch Occupancy<span>.</span></div>
  <span class="tag">thesis</span>
  <div id="tabs">__TABS__<button data-view="searches">Searches</button></div>
  <a class="client" href="/client">Client dashboard &rarr;</a>
</div></nav>

<main>__VIEWS__</main>

<script>
const views = [...document.querySelectorAll(".view")];
const tabs = document.getElementById("tabs");
const ids = views.map(v => v.dataset.view);

function show(name){
  views.forEach(v => { v.hidden = v.dataset.view !== name; });
  [...tabs.children].forEach(b =>
    b.setAttribute("aria-current", String(b.dataset.view === name)));
  if (location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
  window.scrollTo({ top: 0, behavior: "instant" });
}
[...tabs.children].forEach(b => { b.onclick = () => show(b.dataset.view); });
show(ids.includes(location.hash.slice(1)) ? location.hash.slice(1) : ids[0]);
window.addEventListener("hashchange", () => {
  const h = location.hash.slice(1);
  if (ids.includes(h)) show(h);
});
</script></body></html>
"""
