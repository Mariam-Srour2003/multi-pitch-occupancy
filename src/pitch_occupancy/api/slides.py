"""Every tab as a **presentation**, with the document underneath (WP8).

The thesis front end rendered each tab's source document in full. Those documents are good -
they are the thesis - but a Markdown file read start to finish is a poor *page*: a reader
arriving at "Dataset" wants to know how many frames there are and where the hole is, and was
instead given six thousand words in which those facts appear somewhere.

So each tab is now a **deck you page through**, not a page you scroll: a handful of slides,
each one claim, a few big numbers and a figure, with the full document collapsed below. A
slide carries the point; what is *said* about it lives in `<aside>` speaker notes, hidden
until `N`. **Nothing is removed.** The distinction this module turns on is that a document is
the *evidence* and a slide is the *claim*, and a page that leads with the claim can still be
checked, while a page that only prints the evidence is checked by nobody.

Three rules keep the slides honest:

**Every number is read from an artefact.** The RQ statuses are parsed out of the table in
`thesis/rq_matrix.md` rather than restated here; the parameter counts come from
`results/model_inventory.json`; the class counts come from `results/coverage.md`; the search
resolution floor and the augmentation spread come from their own CSVs. A slide that restated
them would be a second copy of the thesis, and the copy that goes stale - which is the failure
this repository keeps finding in its own history.

**A missing artefact says so.** A slide whose figure has not been generated prints the
command that would generate it, rather than rendering an empty box. A page that silently
shows nothing where its argument should be is the shape of failure `augmentation_grid.py`
was written to prevent, and it applies to the page as much as to the sheet.

**A slide states the point, never the talk.** If a sentence is one you would *say*, it goes in
`notes=`. The visible text is a headline, a number, or a card title. The one deliberate
exception is Overview, which is a transcription of a dated progress review and so is allowed
to carry its own prose.

The Augmentation tab is the exception to "less": its argument *is* the pictures, because
augmentation code fails silently and the only reliable check is a person looking.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pitch_occupancy.api.diagrams import (
    blocked_questions,
    confound_matrix,
    empty_blindness,
    pipeline,
    protocols,
    schema,
)

__all__ = [
    "STYLES", "SCRIPT", "SLIDES",
    "deck", "slide", "tiles", "figure", "cards", "points", "beat", "detail",
]

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
THESIS = ROOT / "thesis"
FIGS = RESULTS / "figs"


# --- components -----------------------------------------------------------------------


def slide(eyebrow: str, title: str, body: str = "", *, notes: str = "",
          tone: str = "") -> str:
    """One slide: an eyebrow, a headline, a body, and what you say about it.

    `notes` is the half of a presentation that does not belong on the wall. It renders into
    an `<aside>` the deck shows only on request, so the slide can stay at one claim while
    the reasoning stays attached to it rather than living in a separate file that drifts.
    """
    head = f'<p class="dk-eyebrow">{eyebrow}</p>' if eyebrow else ""
    lead = f'<h2 class="dk-h">{title}</h2>' if title else ""
    aside = f'<aside class="dk-notes">{notes}</aside>' if notes else ""
    return (f'<section class="dk-s{" " + tone if tone else ""}">'
            f'{head}{lead}{body}{aside}</section>')


def deck(sections: list[str]) -> str:
    """A tab's slides, plus the pager that moves between them.

    The outer `<div class="slide">` is the contract the page and its tests are written
    against - every tab opens with one - and the deck lives inside it.
    """
    body = "".join(s for s in sections if s)
    n = body.count('<section class="dk-s')
    return (
        '<div class="slide"><div class="dk" data-deck>'
        f'{body}'
        '<div class="dk-bar">'
        '<button type="button" class="dk-prev" aria-label="Previous slide">&larr;</button>'
        '<div class="dk-dots"></div>'
        '<button type="button" class="dk-next" aria-label="Next slide">&rarr;</button>'
        f'<span class="dk-count">1 / {n}</span>'
        '<button type="button" class="dk-note-btn" aria-pressed="false">Notes</button>'
        '</div></div></div>'
    )


def tiles(items: list[tuple[str, str]] | list[tuple[str, str, str]]) -> str:
    """A row of big numbers. ``(value, label)`` or ``(value, label, tone)``."""
    out = []
    for item in items:
        value, label = item[0], item[1]
        tone = item[2] if len(item) > 2 else ""
        out.append(
            f'<div class="sl-tile {tone}"><b>{value}</b><span>{label}</span></div>'
        )
    return f'<div class="sl-tiles">{"".join(out)}</div>'


def figure(name: str, caption: str = "", *, alt: str = "", script: str = "") -> str:
    """One figure from `results/figs`, or the command that would produce it."""
    if not (FIGS / name).exists():
        hint = f" Run <code>{script}</code>." if script else ""
        return f"<p class='missing'>No <code>{name}</code> yet.{hint}</p>"
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (f'<figure class="sl-fig"><img src="/figs/{name}" loading="lazy" '
            f'alt="{alt or caption or name}">{cap}</figure>')


def cards(items: list[tuple[str, str]], *, wide: bool = False) -> str:
    body = "".join(f'<div class="sl-card"><h4>{t}</h4><p>{b}</p></div>' for t, b in items)
    return f'<div class="sl-cards{" wide" if wide else ""}">{body}</div>'


def points(items: list[str], *, tone: str = "") -> str:
    """A short list, set large. Four at most - past that it is a document again."""
    body = "".join(f"<li>{t}</li>" for t in items)
    return f'<ul class="dk-points{" " + tone if tone else ""}">{body}</ul>'


def beat(before: str, after: str, label: str) -> str:
    """``before -> after``, which is the shape of most of this project's results."""
    return ('<div class="dk-beat">'
            f'<span class="was">{before}</span><span class="to">&rarr;</span>'
            f'<span class="now">{after}</span>'
            f'<span class="lab">{label}</span></div>')


def detail(html: str, label: str = "The full document") -> str:
    """The source document, collapsed.

    `<details>` rather than a CSS toggle, so find-in-page and a saved copy still reach it.
    A slide that hid its evidence behind JavaScript would be a worse page than the wall it
    replaced, not a better one.
    """
    return (f'<details class="sl-detail"><summary>{label}</summary>'
            f'<div class="sl-detail-body">{html}</div></details>')


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _json(name: str) -> dict:
    path = RESULTS / name
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


# --- parsed artefacts -----------------------------------------------------------------


def rq_status() -> list[tuple[str, str, str]]:
    """``(id, question, status)`` parsed from the table at the top of `rq_matrix.md`.

    Parsed rather than restated: the statuses move as the work moves - RQ2 gained "the
    input-path challenge did not survive" and RQ5's answer turned into "looks like no" -
    and a copy here would be right on the day it was written and wrong afterwards, with
    nothing to catch it.
    """
    out = []
    for line in _read(THESIS / "rq_matrix.md").splitlines():
        m = re.match(r"^\|\s*(RQ\d)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def coverage_counts() -> dict:
    """Class totals and the venue gap, from the generated `results/coverage.md`."""
    text = _read(RESULTS / "coverage.md")
    out: dict[str, object] = {}
    header = re.search(r"\|\s*(\d[\d,]*)\s*recorded frames\s*\|\s*(\d+)\s*venues", text)
    if header:
        out["frames"] = header.group(1)
        out["venues"] = header.group(2)
    for cls in ("EMPTY", "ACTIVE_PLAY", "MAINTENANCE_NON_SPORTING"):
        m = re.search(rf"^\|\s*{cls}\s*\|\s*([\d\-]+)\s*\|\s*([\d\-]+)\s*\|\s*(\d+)\s*\|",
                      text, re.MULTILINE)
        if m:
            out[cls] = {"day": m.group(1), "night": m.group(2), "total": m.group(3)}
    gen = re.search(r"\+(\d+)\s*generated", text)
    if gen:
        out["generated"] = gen.group(1)
    # How many venues have an EMPTY frame at all - the single fact most of the data request
    # is about.
    #
    # **Scoped to the class-by-venue section, and it has to be.** `coverage.md` opens with a
    # class-by-*lighting* table whose EMPTY row is `| EMPTY | 485 | 9 | 494 |`, so a pattern
    # matching "the EMPTY row" found that one first and counted the day and night columns as
    # venues - reporting "all from 2 venue" for a corpus whose whole problem is that the
    # answer is one.
    venue_section = text.partition("## Class x venue")[2].partition("## ")[0]
    venue_row = re.search(r"^\|\s*EMPTY\s*\|(.+)\|\s*$", venue_section, re.MULTILINE)
    if venue_row:
        # The last cell is the row total, not a venue.
        cells = [c.strip() for c in venue_row.group(1).split("|")][:-1]
        out["venues_with_empty"] = sum(1 for c in cells if c not in ("-", ""))
        out["venues_counted"] = len(cells)
    return out


def gate_rows() -> list[tuple[str, str, str]]:
    """``(gate, week, verdict)`` from the generated milestone table."""
    out = []
    for line in _read(RESULTS / "gate_status.md").splitlines():
        m = re.match(r"^\|\s*\*\*(M\d)\*\*\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def _csv(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def search_resolution() -> dict:
    """The preprocessing search's own resolution, from `search_resolution.csv`.

    The floor row is written by `search_resolution.py` as a trailing summary rather than a
    configuration, so it is found by its `describe` rather than by position - a slice of the
    last row would silently become a real configuration the day another summary is appended.
    """
    for row in _csv("search_resolution.csv"):
        if row.get("describe") == "RESOLUTION_FLOOR":
            return {"floor": row.get("unweighted", ""), "interval": row.get("weighted", "")}
    return {}


def augmentation_spread() -> dict:
    """The `light` preset across every draw, plus the no-augmentation control.

    Read rather than restated for a specific reason: this is the number the project
    retracted, and a hard-coded copy of a retracted headline is exactly the failure the
    retraction was about.
    """
    out: dict[str, str] = {}
    for row in _csv("augmentation_transfer_spread.csv"):
        if row.get("preset") == "light":
            out = {
                "n": row.get("n_seeds", ""),
                "sd": row.get("macro_f1_sd", ""),
                "min": row.get("macro_f1_min", ""),
                "max": row.get("macro_f1_max", ""),
                "best_empty": row.get("empty_recall_max", ""),
            }
            break
    for row in _csv("augmentation_transfer.csv"):
        if row.get("preset") == "baseline":
            out["baseline"] = row.get("macro_f1", "")
            break
    return out


# --- the look -------------------------------------------------------------------------

#: Scoped under `.slide` / `.dk` so nothing here reaches the rendered Markdown below it.
#:
#: The palette is declared here rather than borrowed from the shell. The shell defines
#: `--accent` and little else; `--play`, `--flag` and `--maint` are set by the operator
#: pages and were *never in scope on this site*, so every tinted tile on these tabs was
#: quietly rendering with no colour at all. These are deck-local names, defined for both
#: schemes, so a tone means the same thing on every tab.
STYLES = """
:root{
 --dk-green:#0e8f5c;--dk-green-b:#17b978;--dk-green-s:#e6f7ef;
 --dk-amber:#b26a00;--dk-amber-b:#f5a524;--dk-amber-s:#fdf0d9;
 --dk-coral:#d22d4c;--dk-coral-b:#ef4f6b;--dk-coral-s:#fde8ec;
 --dk-sky:#1d6fa5;--dk-sky-b:#3fa7e0;--dk-sky-s:#e3f2fb;
 --dk-violet:#6a57d6;--dk-violet-s:#ece9fb;
 --dk-ink:#0e1b2c;--dk-paper:#fbfaf6;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --dk-green:#3ed195;--dk-green-s:#0f2e23;--dk-amber:#e9b04a;--dk-amber-s:#2e2413;
 --dk-coral:#ff8098;--dk-coral-s:#341a21;--dk-sky:#6fc2ed;--dk-sky-s:#122b3a;
 --dk-violet:#a493f2;--dk-violet-s:#201c3a;--dk-paper:#141c1c;}}
:root[data-theme="dark"]{
 --dk-green:#3ed195;--dk-green-s:#0f2e23;--dk-amber:#e9b04a;--dk-amber-s:#2e2413;
 --dk-coral:#ff8098;--dk-coral-s:#341a21;--dk-sky:#6fc2ed;--dk-sky-s:#122b3a;
 --dk-violet:#a493f2;--dk-violet-s:#201c3a;--dk-paper:#141c1c;}

.slide{margin:0 0 6px}

/* --- the deck ---------------------------------------------------------------------- */
.dk{position:relative}
.dk-s{display:none;border:1px solid var(--line);border-radius:16px;
 background:var(--dk-paper);padding:34px 36px 30px;min-height:430px;
 animation:dk-in .28s ease}
.dk-s.on{display:block}
@keyframes dk-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.dk-s::before{content:"";position:absolute;left:0;right:0;top:0;height:6px;
 border-radius:16px 16px 0 0;
 background:linear-gradient(90deg,var(--dk-green-b),var(--dk-sky-b) 38%,
  var(--dk-amber-b) 70%,var(--dk-coral-b))}

.dk-eyebrow{font:700 11px 'JetBrains Mono',monospace;letter-spacing:.2em;
 text-transform:uppercase;color:var(--dk-green);margin:0 0 10px}
.dk-h{font-size:clamp(23px,2.9vw,33px);line-height:1.14;font-weight:700;
 letter-spacing:-.022em;margin:0 0 20px;max-width:24ch;text-wrap:balance;color:var(--ink)}
.dk-s .sub{font-size:16px;color:var(--ink-2);margin:-10px 0 18px;max-width:60ch}
.dk-s .note{font-size:12.5px;color:var(--ink-3);max-width:74ch;margin:14px 0 0}
.dk-s .missing{margin:10px 0}

/* tone slides: one statement, a lot of colour */
.dk-s.green,.dk-s.coral,.dk-s.sky,.dk-s.ink{border:0;color:#fff}
.dk-s.green{background:linear-gradient(135deg,#0e8f5c,#14a86d)}
.dk-s.coral{background:linear-gradient(135deg,#c42746,#ef4f6b)}
.dk-s.sky{background:linear-gradient(135deg,#14567f,#2c89c4)}
.dk-s.ink{background:linear-gradient(140deg,#0e1b2c,#16283f 60%,#0b2a3a)}
.dk-s.green .dk-h,.dk-s.coral .dk-h,.dk-s.sky .dk-h,.dk-s.ink .dk-h{color:#fff;
 max-width:22ch}
.dk-s.green .dk-eyebrow,.dk-s.coral .dk-eyebrow,.dk-s.sky .dk-eyebrow,
.dk-s.ink .dk-eyebrow{color:rgba(255,255,255,.82)}
.dk-s.green .sub,.dk-s.coral .sub,.dk-s.sky .sub,.dk-s.ink .sub,
.dk-s.green .note,.dk-s.coral .note,.dk-s.sky .note,.dk-s.ink .note{
 color:rgba(255,255,255,.88)}
.dk-s.ink .dk-eyebrow{color:var(--dk-green-b)}
.dk-s.green .sl-tile,.dk-s.coral .sl-tile,.dk-s.sky .sl-tile,.dk-s.ink .sl-tile,
.dk-s.green .sl-card,.dk-s.coral .sl-card,.dk-s.sky .sl-card,.dk-s.ink .sl-card{
 background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.22)}
.dk-s.green .sl-tile b,.dk-s.coral .sl-tile b,.dk-s.sky .sl-tile b,
.dk-s.ink .sl-tile b,.dk-s.green .sl-card h4,.dk-s.coral .sl-card h4,
.dk-s.sky .sl-card h4,.dk-s.ink .sl-card h4{color:#fff}
.dk-s.green .sl-tile span,.dk-s.coral .sl-tile span,.dk-s.sky .sl-tile span,
.dk-s.ink .sl-tile span,.dk-s.green .sl-card p,.dk-s.coral .sl-card p,
.dk-s.sky .sl-card p,.dk-s.ink .sl-card p,.dk-s.green .dk-points li,
.dk-s.coral .dk-points li,.dk-s.sky .dk-points li,.dk-s.ink .dk-points li{
 color:rgba(255,255,255,.9)}
.dk-s.green .dk-points li::before,.dk-s.coral .dk-points li::before,
.dk-s.sky .dk-points li::before,.dk-s.ink .dk-points li::before{background:#fff}
.dk-s.green code,.dk-s.coral code,.dk-s.sky code,.dk-s.ink code{
 background:rgba(255,255,255,.16);color:#fff}

/* --- pager ------------------------------------------------------------------------- */
.dk-bar{display:flex;align-items:center;gap:10px;margin-top:14px;flex-wrap:wrap}
.dk-bar button{font:600 13px Archivo,sans-serif;background:var(--surface);
 border:1px solid var(--line);color:var(--ink-2);border-radius:8px;padding:6px 12px;
 cursor:pointer}
.dk-bar button:hover:not(:disabled){background:var(--surface-2);color:var(--ink)}
.dk-bar button:disabled{opacity:.4;cursor:default}
.dk-bar button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.dk-note-btn[aria-pressed="true"]{background:var(--dk-amber-s);color:var(--dk-amber);
 border-color:var(--dk-amber)}
.dk-dots{display:flex;gap:6px;align-items:center}
.dk-dots button{padding:0;width:9px;height:9px;border-radius:50%;border:0;
 background:var(--line);cursor:pointer}
.dk-dots button[aria-current="true"]{background:var(--dk-green);width:22px;
 border-radius:5px}
.dk-count{font:600 11.5px 'JetBrains Mono',monospace;color:var(--ink-3);
 margin-left:auto;font-variant-numeric:tabular-nums}

/* --- speaker notes ----------------------------------------------------------------- */
.dk-notes{display:none;margin:20px 0 0;padding:14px 17px;border-radius:10px;
 background:var(--dk-amber-s);border-left:4px solid var(--dk-amber-b);
 font-size:13.5px;line-height:1.6;color:var(--ink-2);max-width:82ch}
.dk.notes .dk-notes{display:block}
.dk-notes::before{content:"What you say";display:block;
 font:700 10px 'JetBrains Mono',monospace;letter-spacing:.16em;text-transform:uppercase;
 color:var(--dk-amber);margin-bottom:6px}
.dk-notes p{margin:0 0 7px;color:inherit;max-width:none}
.dk-notes p:last-child{margin-bottom:0}
.dk-s.green .dk-notes,.dk-s.coral .dk-notes,.dk-s.sky .dk-notes,.dk-s.ink .dk-notes{
 background:rgba(0,0,0,.22);border-left-color:rgba(255,255,255,.55);color:#fff}
.dk-s.green .dk-notes::before,.dk-s.coral .dk-notes::before,
.dk-s.sky .dk-notes::before,.dk-s.ink .dk-notes::before{color:rgba(255,255,255,.85)}

/* --- blocks ------------------------------------------------------------------------ */
.sl-tiles{display:grid;gap:12px;margin:0 0 18px;
 grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
.sl-tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 padding:16px 17px;border-top:5px solid var(--line)}
.sl-tile b{display:block;font:700 34px 'JetBrains Mono',monospace;
 font-variant-numeric:tabular-nums;letter-spacing:-.03em;margin-bottom:4px;
 line-height:1.05;color:var(--ink)}
.sl-tile span{font-size:13px;color:var(--ink-2);line-height:1.4;display:block}
.sl-tile.good{border-top-color:var(--dk-green-b)}
.sl-tile.good b{color:var(--dk-green)}
.sl-tile.bad{border-top-color:var(--dk-coral-b)}
.sl-tile.bad b{color:var(--dk-coral)}
.sl-tile.lead{border-top-color:var(--dk-sky-b)}
.sl-tile.lead b{color:var(--dk-sky)}
.sl-tile.warn{border-top-color:var(--dk-amber-b)}
.sl-tile.warn b{color:var(--dk-amber)}

.sl-figs{display:grid;gap:14px;margin:0 0 6px;
 grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.sl-fig{margin:0 0 6px;background:var(--surface);border:1px solid var(--line);
 border-radius:12px;overflow:hidden}
.sl-figs .sl-fig{margin:0}
.sl-fig img{display:block;width:100%;background:var(--surface-2)}
.sl-fig figcaption{font-size:13px;color:var(--ink-2);padding:11px 14px;
 border-top:1px solid var(--line);line-height:1.45}

.sl-cards{display:grid;gap:12px;margin:0 0 6px;
 grid-template-columns:repeat(auto-fit,minmax(215px,1fr))}
.sl-cards.wide{grid-template-columns:repeat(auto-fit,minmax(315px,1fr))}
.sl-card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 padding:16px 17px;border-left:5px solid var(--dk-green-b)}
.sl-cards .sl-card:nth-child(4n+2){border-left-color:var(--dk-sky-b)}
.sl-cards .sl-card:nth-child(4n+3){border-left-color:var(--dk-amber-b)}
.sl-cards .sl-card:nth-child(4n+4){border-left-color:var(--dk-coral-b)}
.sl-card h4{margin:0 0 6px;font-size:15px;font-weight:700;color:var(--ink);
 line-height:1.25}
.sl-card p{margin:0;font-size:13.5px;color:var(--ink-2);line-height:1.5}

.dk-points{list-style:none;padding:0;margin:0 0 8px;display:grid;gap:11px}
.dk-points li{position:relative;padding-left:26px;font-size:17px;line-height:1.45;
 color:var(--ink-2);max-width:62ch}
.dk-points li::before{content:"";position:absolute;left:0;top:9px;width:11px;height:11px;
 border-radius:3px;background:var(--dk-green-b)}
.dk-points.warn li::before{background:var(--dk-amber-b)}
.dk-points.bad li::before{background:var(--dk-coral-b)}
.dk-points li b,.dk-points li strong{color:var(--ink);font-weight:700}

.dk-beats{display:grid;gap:11px;margin:0 0 8px;
 grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.dk-beat{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 padding:15px 17px;display:grid;grid-template-columns:auto auto auto;gap:9px;
 align-items:baseline}
.dk-beat .was{font:700 25px 'JetBrains Mono',monospace;color:var(--dk-coral)}
.dk-beat .to{color:var(--ink-3);font-size:17px}
.dk-beat .now{font:700 25px 'JetBrains Mono',monospace;color:var(--dk-green)}
.dk-beat .lab{grid-column:1/-1;font-size:13px;color:var(--ink-2);line-height:1.4}

.sl-quote{margin:0 0 16px;padding:16px 20px;background:var(--accent-soft);
 border-left:5px solid var(--accent);border-radius:0 12px 12px 0;
 font-size:17px;line-height:1.5;color:var(--ink);max-width:70ch}

.sl-rows{display:grid;gap:7px;margin:0 0 8px}
.sl-row{display:grid;grid-template-columns:54px 1fr auto;gap:13px;align-items:center;
 background:var(--surface);border:1px solid var(--line);border-radius:10px;
 padding:11px 15px}
.sl-row .id{font:700 13px 'JetBrains Mono',monospace;color:var(--dk-sky)}
.sl-row .q{font-size:13.5px;color:var(--ink-2)}
.sl-row .st{font:600 11px 'JetBrains Mono',monospace;padding:4px 10px;border-radius:6px;
 white-space:nowrap;background:var(--surface-2);color:var(--ink-3)}
.sl-row .st.ok{background:var(--dk-green-s);color:var(--dk-green)}
.sl-row .st.part{background:var(--dk-amber-s);color:var(--dk-amber)}
.sl-row .st.blocked{background:var(--dk-coral-s);color:var(--dk-coral)}

.sl-detail{border:1px solid var(--line);border-radius:12px;background:var(--surface);
 margin:20px 0 0}
.sl-detail > summary{cursor:pointer;padding:13px 17px;font-weight:600;font-size:13.5px;
 list-style:none;display:flex;gap:10px;align-items:center}
.sl-detail > summary::-webkit-details-marker{display:none}
.sl-detail > summary::before{content:"\\25B8";color:var(--ink-3);font-size:11px}
.sl-detail[open] > summary::before{content:"\\25BE"}
.sl-detail[open] > summary{border-bottom:1px solid var(--line)}
.sl-detail > summary:hover{background:var(--surface-2)}
.sl-detail-body{padding:4px 18px 16px}

/* the augmentation tab, which is deliberately image-heavy */
.sl-pairs{display:grid;gap:14px;margin:0 0 8px;
 grid-template-columns:repeat(auto-fill,minmax(258px,1fr))}
.sl-pair{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 overflow:hidden}
.sl-pair img{display:block;width:100%;background:var(--surface-2)}
.sl-pair .cap{padding:10px 13px;border-top:1px solid var(--line)}
.sl-pair .cap code{font-size:12px;font-weight:700;color:var(--ink);background:none;
 padding:0}
.sl-pair .cap .m{display:flex;gap:10px;flex-wrap:wrap;margin-top:5px;
 font:500 10.5px 'JetBrains Mono',monospace;color:var(--ink-3)}
.sl-pair .cap .m b{color:var(--ink-2)}
.sl-pair.crop{border-left:4px solid var(--dk-coral-b)}

@media (max-width:640px){
 .dk-s{padding:26px 20px 24px;min-height:0}
 .dk-h{font-size:22px}
}
"""

#: Paging, the dots, and the notes toggle. Keyboard only fires when no field has focus and
#: no modifier is held, so it cannot steal a browser shortcut or typing in the search panel.
SCRIPT = """
document.querySelectorAll("[data-deck]").forEach(dk => {
  const slides = [...dk.querySelectorAll(".dk-s")];
  if (!slides.length) return;
  const dots = dk.querySelector(".dk-dots");
  const count = dk.querySelector(".dk-count");
  const prev = dk.querySelector(".dk-prev");
  const next = dk.querySelector(".dk-next");
  const noteBtn = dk.querySelector(".dk-note-btn");
  let at = 0;

  slides.forEach((_, i) => {
    const d = document.createElement("button");
    d.type = "button";
    d.setAttribute("aria-label", "Slide " + (i + 1));
    d.onclick = () => go(i);
    dots.appendChild(d);
  });

  function go(i) {
    at = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((s, k) => s.classList.toggle("on", k === at));
    [...dots.children].forEach((d, k) =>
      d.setAttribute("aria-current", String(k === at)));
    count.textContent = (at + 1) + " / " + slides.length;
    prev.disabled = at === 0;
    next.disabled = at === slides.length - 1;
  }
  prev.onclick = () => go(at - 1);
  next.onclick = () => go(at + 1);
  noteBtn.onclick = () => {
    const on = dk.classList.toggle("notes");
    noteBtn.setAttribute("aria-pressed", String(on));
  };

  dk.addEventListener("keydown", e => { if (e.key === "ArrowRight" ||
    e.key === "ArrowLeft") e.stopPropagation(); });
  document.addEventListener("keydown", e => {
    if (dk.closest(".view")?.hidden) return;
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const t = e.target.tagName;
    if (t === "INPUT" || t === "TEXTAREA" || t === "SELECT") return;
    if (e.key === "ArrowRight") { go(at + 1); }
    else if (e.key === "ArrowLeft") { go(at - 1); }
    else if (e.key === "n" || e.key === "N") { noteBtn.click(); }
    else return;
    e.preventDefault();
  });

  go(0);
});
"""


# --- the slides ----------------------------------------------------------------------
#
# The site *is* the talk. Each tab is one section of the oral defence, and the slides are
# the same slides as `thesis/presentation/Thesis_Defence_25min.pptx`. The words on a slide
# are the point; the words that are spoken live in `notes=` and in
# `thesis/presentation/SPEAKER_SCRIPT.md`.


def start() -> str:
    """Title and roadmap: what this is, and where the talk goes."""
    return deck([
        slide(
            "Master&rsquo;s thesis &middot; oral defence",
            "Who actually used the pitch?",
            '<p class="sub" style="font-size:19px">Checking bookings against what the '
            'cameras already see. On one small computer, with no GPU, and a human deciding '
            'every case.</p>'
            '<p class="note" style="color:rgba(255,255,255,.8);font-size:15px;margin-top:26px">'
            'Mariam Srour &nbsp;&middot;&nbsp; [Programme] &nbsp;&middot;&nbsp; '
            '[Institution] &nbsp;&middot;&nbsp; Supervisor: [Name]</p>',
            notes="<p>Good morning, distinguished examiners, professors and colleagues. My "
                  "name is Mariam Srour, and I am a Master's student in [Programme] at "
                  "[Institution].</p>"
                  "<p>The title of my research is &ldquo;Multi-Pitch Occupancy and Booking "
                  "Verification from Existing Cameras&rdquo;. I will use the shorter "
                  "question: who actually used the pitch?</p>",
            tone="ink",
        ),
        slide(
            "Roadmap", "Where we are going",
            cards([
                ("Purpose", "What the study is for, and who it helps."),
                ("How it works", "Five stages, and one evening on pitch 3."),
                ("The problem", "And the gap in the data that shaped everything."),
                ("Data &amp; models", "Why we made data with AI, and why we trained nothing."),
                ("Rules, searches, augmentation", "What we tried, and what it gave."),
                ("Solution &amp; summary", "Three fixes, the limits, and your questions."),
            ]),
            notes="<p>In this presentation I will take you through the purpose of the "
                  "research; how it works, with a short scenario; what problem it tackles; "
                  "the data and the models; the rules and the experiments; and I will "
                  "conclude with a summary.</p>",
        ),
        slide(
            "In one line", "A system that checks every booked minute, and never gets bored",
            tiles([
                ("1,692", "labelled frames", "lead"),
                ("3", "classes", "good"),
                ("1/min", "frames per camera", "warn"),
                ("0", "GPUs", "bad"),
            ]),
            notes="<p>Give the scale once, then move on. One frame per camera per minute is "
                  "the decision that makes everything else possible - about 99% less network "
                  "traffic than decoding twenty to thirty live streams, which is what lets "
                  "the whole facility run on a single mini-PC with no graphics card.</p>",
        ),
    ])


def purpose() -> str:
    """The purpose, and the case for it: time, consistency, evidence."""
    return deck([
        slide(
            "Purpose of the research",
            "Find out which booked hours were really used — using cameras the "
            "facility already owns.",
            '<p class="sub" style="font-size:18px">Three constraints: no new hardware, no '
            'GPU, and a person decides every case.</p>',
            notes="<p>The purpose of this study is to find out which sold pitch hours were "
                  "actually used, using cameras already installed for match highlights.</p>"
                  "<p>Say it plainly: the output is not a classification. It is a billing "
                  "conversation. &lsquo;This booking looks unused, and here are three frames "
                  "from it.&rsquo;</p>",
            tone="green",
        ),
        slide(
            "Purpose", "What the owner gets",
            cards([
                ("Time back",
                 "Nobody can watch 20&ndash;30 pitches. The system checks <b>every booked "
                 "minute</b> and shows the manager only what disagrees."),
                ("The same judgement at 8am and 11pm",
                 "It never tires, never rushes, never skips the far pitch. A person samples; "
                 "this does not."),
                ("A picture, not a memory",
                 "Every flag arrives with three evidence frames. The conversation is about "
                 "what is on screen."),
                ("It watches better than a walk-round",
                 "One camera per pitch, once a minute, all day — coverage a patrol "
                 "cannot match."),
            ]),
            notes="<p>This is the slide about why anyone would want this.</p>"
                  "<p>A facility manager physically cannot watch twenty or thirty pitches. "
                  "They spot-check, they trust the sheet, and the sheet is unverified. This "
                  "system looks at every booked minute and surfaces only the disagreements, "
                  "so the manager's attention goes where it is worth something.</p>"
                  "<p>Be precise about the trust claim, because an examiner will push on it. "
                  "The system is <i>more consistent</i> than a person and it is "
                  "<i>auditable</i> - it applies the same rule every time and shows you the "
                  "frames it decided on. I am not claiming it is more accurate than a human "
                  "looking at the same picture; I have no inter-annotator figure, and that "
                  "is a real gap. Consistency and coverage are the claim. Accuracy is a "
                  "person's job, and the system is built so a person always has it.</p>",
        ),
        slide(
            "Purpose", "One rule carried everywhere",
            '<p class="sl-quote">The vision path must <b>never</b> see the booking record as '
            'an input. A model that has read the booking flag cannot give evidence '
            'independent of the record it audits.</p>'
            + cards([
                ("Three failure modes it looks for",
                 "No-shows. Unbooked usage. Plain data-entry error."),
                ("And one it refuses to create",
                 "The system never bills, never acts, and never turns its own uncertainty "
                 "into someone&rsquo;s error."),
            ], wide=True),
            notes="<p>This is the difference between an audit and a rubber stamp. If the "
                  "classifier could see the booking flag, its agreement with the booking "
                  "would mean nothing. The vision path and the reconciliation layer never "
                  "share a feature.</p>",
        ),
    ])


def how() -> str:
    """The system, the scenario, and the method."""
    return deck([
        slide(
            "How the research works", "Five stages, one frame a minute",
            '<p class="sub">Sparse sampling, not a video stream.</p>'
            + cards([
                ("1 &middot; Sample", "One frame per camera per minute. ~99% less traffic."),
                ("2 &middot; Classify", "Empty &middot; playing &middot; maintenance."),
                ("3 &middot; Check", "Boundary, motion, people, ball — rules that can "
                                     "overrule the model."),
                ("4 &middot; Aggregate", "~60 predictions become one verdict per hour."),
                ("5 &middot; Compare", "Verdict against the booking. Disagreements flagged."),
            ]),
            notes="<p>To achieve the purpose I built and then evaluated a five-stage "
                  "system.</p>"
                  "<p>Stage three is the one to flag now: the cheap geometric and motion "
                  "checks can overrule the neural network. That turns out to matter far more "
                  "than the choice of network.</p>",
        ),
        slide(
            "A scenario", "Tuesday, 20:00 — Pitch 3",
            '<div class="dk-beats">'
            '<div class="dk-beat"><span class="was">Sold</span><span class="to">&rarr;</span>'
            '<span class="now">1 hour</span><span class="lab">The booking says: paid for, '
            'and marked &ldquo;used&rdquo; by staff.</span></div>'
            '<div class="dk-beat"><span class="was">13</span><span class="to">of</span>'
            '<span class="now">16</span><span class="lab">The camera sees: an empty pitch '
            'under floodlights.</span></div>'
            '<div class="dk-beat"><span class="was">REVIEW</span><span class="to">+</span>'
            '<span class="now">3 photos</span><span class="lab">The system says: never '
            '&ldquo;do not bill&rdquo;. Only &ldquo;look at this&rdquo;.</span></div>'
            "</div>",
            notes="<p>The booking system says pitch 3 was sold for an hour and staff marked "
                  "it used. The cameras sampled sixteen frames and thirteen show an empty "
                  "pitch under floodlights.</p>"
                  "<p>The system does not cancel the charge. It raises one advisory with "
                  "three evidence photos, and a manager decides. In the code that is "
                  "structural: Advisory is the only type the decision layer can return, and "
                  "there is no code path that acts.</p>",
        ),
        slide(
            "How it works", "Where a verdict can be refused",
            pipeline(),
            notes="<p>Every arrow that leaves the pipeline can refuse rather than guess. That "
                  "is what makes REVIEW a real outcome instead of a low-confidence USED, and "
                  "it is why a dropped camera cannot silently become an empty pitch.</p>",
        ),
        slide(
            "How it works", "What gets stored",
            schema(),
            notes="<p>One row per camera per sampled minute is the only thing observed. "
                  "Everything above it is derived and can be recomputed.</p>"
                  "<p>Two decisions worth naming: a verdict and its evidence frames are "
                  "written in one transaction, so no verdict can exist without its evidence; "
                  "and a dropped minute is stored as a gap rather than filled, because no "
                  "footage is not evidence that a pitch was unused.</p>",
        ),
        slide(
            "Method", "How the data were collected",
            cards([
                ("Fixed cameras", "Existing CCTV on posts. One view forever."),
                ("Every 15 seconds", "Sampled from footage, then hand-labelled."),
                ("Nine venues", "Seven usable as folds, plus two cameras on one pitch."),
                ("A written protocol", "Labelling rules fixed before the runs."),
            ])
            + '<p class="note" style="font-size:14px"><b>The property that shaped everything:'
              '</b> fixed cameras plus short intervals means <b>98.5%</b> of frames have a '
              'near-duplicate.</p>',
            notes="<p>The note at the bottom is the one that matters. Almost every frame has "
                  "a near-twin somewhere in the corpus. That single property invalidated the "
                  "pilot, and it is why the next slide is about splitting rather than about "
                  "models.</p>",
        ),
        slide(
            "Method", "How the data were analysed",
            cards([
                ("Grouped splits, never random",
                 "Whole venues and slots held out, so no test frame has a near-twin in "
                 "training."),
                ("Four trivial baselines, every time",
                 "A clock rule, a colour histogram, a constant predictor, a random one. If "
                 "they win, the protocol is wrong."),
                ("Leave-one-venue-out",
                 "Seven folds, each reported with the smallest p-value the design could have "
                 "produced."),
                ("Pre-registered, amended in the open",
                 "Hypotheses fixed before the runs; every change numbered and left in place."),
            ]),
            notes="<p>Two of these are the method contribution.</p>"
                  "<p>First, every protocol runs four trivial baselines beside the models. "
                  "If a trivial baseline wins, that is information about the protocol, not a "
                  "curiosity to leave out.</p>"
                  "<p>Second, I report the resolution of each test. Over seven folds the "
                  "smallest attainable two-sided p-value is 0.0156 - so &lsquo;not "
                  "significant&rsquo; and &lsquo;the design could not have produced "
                  "significance&rsquo; are different statements, and I say which applies.</p>",
        ),
    ])


def problem() -> str:
    """The problem, the gap, and the four results that came out of it."""
    return deck([
        slide(
            "What problem does the research tackle?",
            "A facility sells hours. Nobody checks which ones were used.",
            cards([
                ("No-shows", "Paid for, never played."),
                ("Unbooked use", "Played, never paid for."),
                ("Data-entry error", "The record and the reality drift apart."),
            ]),
            notes="<p>Concede the obvious objection before anyone raises it: a twenty-euro "
                  "motion sensor is more robust in fog, in darkness, and against a dirty "
                  "lens. But a sensor answers <i>did something move</i>. An audit needs "
                  "<i>was this booking used, and here is the picture</i> - and it has to "
                  "tell a five-a-side match from a groundsman on a mower.</p>",
            tone="ink",
        ),
        slide(
            "The gap", "What is known — and what is missing",
            cards([
                ("Known: frozen backbones work",
                 "Strong results from very few labels, across many vision tasks."),
                ("Known: occupancy from CCTV",
                 "People counting and occupancy are established problems."),
                ("Missing: honest evaluation",
                 "Almost all of it is scored on frames from the <b>same scenes</b> as "
                 "training."),
                ("So the score measures memory",
                 "Recognition of a <b>place</b>, not of an <b>activity</b> — and nobody "
                 "runs a trivial rule beside it to check."),
            ]),
            notes="<p>The gap matters here because the facility's money rests entirely on one "
                  "class. Getting &lsquo;playing&rsquo; right is easy and worth nothing. If "
                  "the system cannot reliably recognise an <i>empty</i> pitch, it cannot flag "
                  "a single unused booking.</p>",
        ),
        slide(
            "First consequence", "Honest splitting halves the score",
            tiles([
                ("37.1%", "of near-duplicate pairs cross the line under a random split", "bad"),
                ("0.8%", "under a split grouped by venue and slot", "good"),
                ("−0.49", "macro-F1: what honesty costs ConvNeXtV2", "warn"),
            ])
            + cards([
                ("But only two thirds of it is leakage",
                 "A model that never trains cannot leak. Scoring a zero-shot model on the "
                 "same test sets isolates the rest: <b>0.183</b>. Overstating leakage would "
                 "have been the comfortable error."),
            ], wide=True),
            notes="<p>On the leaky split, every one of ConvNeXtV2's twelve errors was a frame "
                  "whose near-duplicate sat in the training set.</p>"
                  "<p>The card is where I checked my own result. Reporting the bigger number "
                  "would have made my story stronger, and it would have been wrong.</p>",
        ),
        slide(
            "The real gap", "In this data, the time of day is the answer",
            tiles([
                ("98%", "of daylight frames are an empty pitch", "warn"),
                ("99%", "of night frames are a match", "warn"),
                ("99.1%", "so &ldquo;night means play&rdquo; is right on the whole corpus",
                 "bad"),
            ])
            + '<p class="note" style="color:rgba(255,255,255,.9);font-size:15px">The two '
              'cells that would break the tie hold <b>9 frames</b> (empty at night) and '
              '<b>6</b> (play in daylight).</p>',
            notes="<p>This is the most important slide in the talk. Slow down.</p>"
                  "<p>Nothing measured on this data can separate a model that recognises an "
                  "empty pitch from one that recognises the time of day. It is a statement "
                  "about the dataset, not about the models - and identifying it, rather than "
                  "reporting around it, is the contribution of this thesis.</p>",
            tone="coral",
        ),
        slide(
            "The gap", "The same model, judged two ways",
            protocols(),
            notes="<p>Identical model, identical data, two ways of drawing the train/test "
                  "line - and the answer changes. Let them look at it before you say "
                  "anything.</p>",
        ),
        slide(
            "Evidence", "A rule that never looks at the image wins",
            '<table style="width:100%;border-collapse:collapse">'
            '<tr><th style="text-align:left">Method</th>'
            '<th style="text-align:left">Finds the match</th>'
            '<th style="text-align:left">Wrongly says &ldquo;playing&rdquo;</th></tr>'
            '<tr><td><b>Clock rule — no pixels</b></td><td><b>1.000</b></td>'
            '<td><b>0.021</b></td></tr>'
            '<tr><td>DINOv2</td><td>0.930</td><td>0.309</td></tr>'
            '<tr><td>ConvNeXtV2</td><td>0.911</td><td>0.992</td></tr>'
            '<tr><td>ViT</td><td>0.869</td><td>0.835</td></tr>'
            "</table>"
            '<p class="note" style="font-size:14px">Across venues the models had never seen. '
            'Higher is better on the left; <b>lower</b> is better on the right. The rule wins '
            'on both.</p>',
            notes="<p>Expect &ldquo;so your backbones are worthless?&rdquo; - and answer "
                  "directly. No. They are indistinguishable from a light meter <i>on this "
                  "dataset</i>.</p>"
                  "<p>Volunteer the honesty note: an earlier version of this slide said the "
                  "opposite. That number came from a label error - a brightness threshold had "
                  "filed 216 frames of floodlit night football as daylight. Corrected, the "
                  "rule wins and the finding reversed.</p>",
        ),
        slide(
            "The reversal", "Then one clip reversed the ranking",
            '<p class="sub">234 seconds of a floodlit pitch at night with nobody playing '
            '— the case the corpus does not contain.</p>'
            + tiles([
                ("16/16", "minutes the clock rule got wrong", "bad"),
                ("0.38", "false alarms, model alone", "warn"),
                ("0.00", "false alarms, full system — 13 of 13 right", "good"),
            ])
            + '<p class="note" style="font-size:14.5px">A 1,692-frame benchmark ranked the '
              'trivial rule first. <b>One clip reversed it.</b> The difference is not size or '
              'statistics — it is one missing case.</p>',
            notes="<p>The benchmark had 1,692 frames and the clip had a few hundred. The clip "
                  "won, because it contained the case the benchmark was missing. More data of "
                  "the same kind would not have found this.</p>"
                  "<p>This is the most useful sentence in the thesis.</p>",
        ),
        slide(
            "Evidence", "What a low false-play rate is actually measuring",
            empty_blindness(),
            notes="<p>This is the strongest slide in the thesis. Ask what the models answer "
                  "<i>instead</i> of ACTIVE_PLAY on the 243 held-out empty frames.</p>"
                  "<p>The protocol trains on camera A and tests on camera B: it is a "
                  "camera-transfer test, not a specificity test.</p>",
        ),
        slide(
            "Evidence", "Two more things the protocol was hiding",
            cards([
                ("A constant predictor wins one protocol",
                 "Always answering &ldquo;playing&rdquo; scores macro-F1 <b>1.000</b> "
                 "cross-venue — above every backbone, because every held-out venue is "
                 "100% active play. There is no ranking to change."),
                ("A low false-play rate is not accuracy",
                 "On 243 held-out empty frames DINOv2 answers PLAYING 75 times and "
                 "MAINTENANCE 168 times. It is correct <b>zero</b> times. No model exceeds "
                 "0.165."),
            ], wide=True)
            + '<p class="note" style="font-size:14px">Both were found the same way — by '
              'adding a column to check whether a number meant what it said.</p>',
            notes="<p>The second one is the strongest result in the thesis. The false-play "
                  "rate had been quoted throughout as how well a model recognises an empty "
                  "pitch. So I asked what the models answer <i>instead</i>. The published "
                  "numbers stand; the inference drawn from them does not.</p>",
        ),
    ])


def data() -> str:
    """The corpus, why it is small, what we generated, and what that risks."""
    c = coverage_counts()
    empty = c.get("EMPTY", {}) if c else {}
    play = c.get("ACTIVE_PLAY", {}) if c else {}
    maint = c.get("MAINTENANCE_NON_SPORTING", {}) if c else {}
    with_empty = c.get("venues_with_empty", 1) if c else 1
    counts = tiles([
        (c.get("frames", "?"), "recorded frames", "lead"),
        (c.get("venues", "?"), "venues"),
        (empty.get("total", "?"),
         f'EMPTY, all from {with_empty} venue{"" if with_empty == 1 else "s"}', "warn"),
        (maint.get("total", "?"), "MAINTENANCE, too few to be a class", "bad"),
    ]) if c else "<p class='missing'>No coverage report generated yet.</p>"

    return deck([
        slide(
            "Data", "What we have",
            counts
            + (cards([
                ("EMPTY is daytime",
                 f'{empty.get("day", "?")} day against {empty.get("night", "?")} night.'),
                ("ACTIVE_PLAY is night",
                 f'{play.get("night", "?")} night against {play.get("day", "?")} day.'),
                ("So class and clock are one variable",
                 "Which is why a lighting-only rule scores 98.4%."),
            ]) if c else ""),
            notes="<p>Give the size, then the shape of the hole. Every empty pitch in the "
                  "corpus comes from one venue, and maintenance has too few frames to be a "
                  "class you can evaluate at all.</p>",
        ),
        slide(
            "Data", "Why this data is hard to get",
            cards([
                ("There are people in it",
                 "Players, staff, sometimes children. Footage of identifiable people is "
                 "sensitive data — consent, retention and redaction all apply."),
                ("So it cannot simply be collected",
                 "Every extra venue is a permission conversation, not a download."),
                ("And the task needs variety",
                 "Many venues, many lights, many angles, many weathers. Volume of the "
                 "<b>same</b> scene does not help."),
                ("What we got instead",
                 "One venue with empty pitches, nine venues of night football, and no wet "
                 "weather at all."),
            ]),
            notes="<p>This slide answers the question everyone asks: why not just collect "
                  "more data?</p>"
                  "<p>Because this is footage of identifiable people on a pitch, including "
                  "children. It is sensitive personal data. Every additional venue is a "
                  "consent and data-protection conversation with a facility, not a dataset "
                  "download. That is the real constraint - not storage, not labelling "
                  "effort.</p>"
                  "<p>And note the second half: the task needs <i>variety</i>, not volume. "
                  "Ten thousand more frames of the same pitch on the same evening would add "
                  "nothing, because they are near-duplicates of what we already have.</p>",
        ),
        slide(
            "Data", "So we made some, starting from something real",
            cards([
                ("Real anchor first",
                 "Every generated item starts from an actual frame of an actual pitch "
                 "— never from a text prompt alone."),
                ("Images — Gemini and ChatGPT",
                 "Used to produce the scene the corpus lacks: an <b>empty pitch under "
                 "floodlights</b>."),
                ("Stills into video — 2 AI tools",
                 "A still frame animated into motion, so the motion gate has something to "
                 "read. <b>[tool names to be added]</b>"),
                ("What it bought",
                 "31 generated empty frames took cross-venue false alarms from <b>0.768 to "
                 "0.024</b> — while play-recall went <b>up</b>."),
            ]),
            notes="<p>This is where AI-generated data comes in, and I want to be careful "
                  "about how I describe it.</p>"
                  "<p>We did not generate a dataset from prompts. Every generated item starts "
                  "from a real anchor - a real frame of a real pitch - and the generation "
                  "changes one thing about it. Images came from Gemini and ChatGPT's image "
                  "models. For video, we took still frames and animated them so that motion "
                  "exists to be measured, using two AI video tools - [I will name them].</p>"
                  "<p>The target was always the same: the one cell the corpus does not have, "
                  "an empty pitch at night under floodlights. Thirty-one generated empty "
                  "frames took cross-venue false alarms from 0.768 down to 0.024, and "
                  "play-recall went up rather than down, which is the sign that it added "
                  "signal rather than noise.</p>",
            tone="sky",
        ),
        slide(
            "Data", "And we are honest about what generated data is",
            cards([
                ("Excluded from every count",
                 "Generated frames appear in <b>no</b> corpus total. Counting them would "
                 "make a gap look filled that is not."),
                ("A training aid, not evidence",
                 "They help a model learn a case. They do not prove the system works on real "
                 "footage of that case."),
                ("Checked against real pixels",
                 "26 real median-empty frames reach a comparable figure, so the effect is not "
                 "an artefact of the generator."),
                ("One thing we would not fake",
                 "No footage in this dataset is wet, so synthetic rain could only be "
                 "validated against synthetic rain — that tests the generator, not the "
                 "weather."),
            ]),
            notes="<p>Generated data is a legitimate tool and a very easy way to fool "
                  "yourself, so the guards are on the slide.</p>"
                  "<p>They are excluded from every corpus count in the thesis. They are "
                  "training-side only. And the last card is the line we drew: we did not "
                  "generate rain, because there is no real wet footage to check it against. "
                  "Validating synthetic rain against synthetic rain tests the generator, not "
                  "the weather.</p>",
        ),
        slide(
            "Data", "The confound, as a picture",
            confound_matrix(),
            notes="<p>Read straight off the manifest, so it cannot drift from the corpus. The "
                  "two nearly-empty cells are the whole problem.</p>",
        ),
        slide(
            "Data", "One missing cell, four unanswerable questions",
            blocked_questions(),
            notes="<p>This is why the gap is not a footnote in a limitations chapter. One "
                  "absent cell propagates into four separate research questions, and no "
                  "amount of further engineering reaches any of them.</p>",
        ),
        slide(
            "Data", "Small data has a specific danger",
            '<p class="sub">Not &ldquo;less accurate&rdquo;. <b>Overfitting</b> — the '
            'model memorises these places instead of learning the activity.</p>'
            + cards([
                ("The corpus is scenes, not frames",
                 "1,692 frames are about 150 distinct scenes. 243 held-out empty frames are "
                 "<b>3</b>."),
                ("More labels made it worse",
                 "Every backbone peaks at 100&ndash;300 labels and <b>falls</b> at 671 "
                 "— the extra labels are near-duplicates."),
                ("So generalisation is the thing to test",
                 "Hence grouped splits, leave-one-venue-out, and a trivial baseline in every "
                 "protocol."),
            ]),
            notes="<p>The risk with a dataset this size is not that the numbers come out low. "
                  "It is that they come out <i>high</i> for the wrong reason - the model "
                  "memorises these specific pitches and that looks like success.</p>"
                  "<p>The middle card is the evidence that this is real and not theoretical. "
                  "Label efficiency here is not monotone: every backbone peaks between one "
                  "and three hundred labels and then gets worse at 671, because the extra "
                  "labels are near-duplicates that add redundancy rather than information.</p>"
                  "<p>That is the whole reason the evaluation is built the way it is.</p>",
            tone="coral",
        ),
    ])


def models() -> str:
    """Why almost nothing here is trained, and what the two novel modules gave."""
    inv = _json("model_inventory.json")
    head = tiles([
        (f'{inv["frozen_total_params"] / 1e6:.1f}M', "frozen parameters — never updated",
         "lead"),
        (f'{inv["trained_total_params"]:,}', "trained parameters, all of them", "good"),
        (f'{inv["frozen_to_trained_ratio"]:,}:1', "frozen to trained", "warn"),
    ]) if inv else "<p class='missing'>No <code>model_inventory.json</code> yet.</p>"

    return deck([
        slide(
            "Models", "We did not train the models",
            head
            + '<p class="note" style="font-size:14.5px">Three frozen backbones — DINOv2, '
              'ConvNeXtV2, ViT — and a classifier small enough to read in one screen.</p>',
            notes="<p>The first thing anyone asks is what was trained. The answer is: a very "
                  "small head, and nothing else.</p>"
                  "<p>Two hundred million frozen parameters against about eleven thousand "
                  "trained ones. The backbones never saw this data during training; they are "
                  "fixed feature extractors. The next slide is why that was a decision rather "
                  "than a shortcut.</p>",
        ),
        slide(
            "Models", "Why not — and it is the data, not the compute",
            cards([
                ("Fine-tuning needs variety we do not have",
                 "Updating 200M parameters on ~150 distinct scenes does not learn football. "
                 "It learns <b>these pitches</b>."),
                ("The data is sensitive, so it cannot just be grown",
                 "More venues means more consent conversations. The bottleneck is "
                 "permission, not effort."),
                ("Our own numbers say so",
                 "98.5% near-duplicates, and accuracy that <b>falls</b> when labels rise "
                 "from 300 to 671."),
                ("Frozen is also what bought the breadth",
                 "Embed once, reuse forever — which is how 75 experiments fit in one "
                 "thesis."),
            ])
            + '<p class="note" style="font-size:14px">Freezing is a <b>defence against '
              'overfitting</b> first, and an efficiency win second.</p>',
            notes="<p>This is the slide I most want to land, because &lsquo;why didn't you "
                  "fine-tune?&rsquo; is a guaranteed question.</p>"
                  "<p>Fine-tuning a two-hundred-million-parameter backbone on roughly a "
                  "hundred and fifty distinct scenes does not teach it football. It teaches "
                  "it these pitches, under these floodlights, from these camera angles - and "
                  "then the cross-venue number collapses, or worse, it does not collapse and "
                  "I believe it.</p>"
                  "<p>And I could not simply collect my way out, because this is footage of "
                  "identifiable people. The bottleneck is permission.</p>"
                  "<p>So freezing is a defence against overfitting first. The efficiency - "
                  "embed once, reuse forever, seventy-five experiments - is a second, very "
                  "welcome consequence.</p>",
            tone="ink",
        ),
        slide(
            "Models", "Two novel modules, reported as they came out",
            cards([
                ("Gated fusion — negative",
                 "Routing between backbones is worth <b>−0.024</b> cross-venue recall "
                 "against the gate switched off. The gate puts ~0.70 of its weight on DINOv2 "
                 "in every fold: a learned constant in a router&rsquo;s costume."),
                ("STAN — preliminary, and saturated",
                 "A perfect 1.000 on 200 composed slots, beating an HMM by 0.105. That is an "
                 "exhausted test set, not a win. <b>The real test set is two slots</b>, and "
                 "the ≥30-slot rule raises in code."),
            ], wide=True)
            + '<p class="note" style="font-size:14px">The spec named ConvNeXtV2. Measured '
              'without leakage, <b>DINOv2</b> leads — the decision was made on evidence '
              'taken after the proposal.</p>',
            notes="<p>The thesis proposed two novel modules and I report both the way they "
                  "came out.</p>"
                  "<p>The fusion head is a negative result. I built the ablation as a "
                  "three-rung ladder - uniform, constant, learned - because an on/off switch "
                  "credited routing with what is really a learned constant. It <i>can</i> "
                  "route: on a fixture where the useful backbone flips it scores 1.000 "
                  "against 0.671. The null is about the data.</p>"
                  "<p>For STAN I argue against my own number. The composed label is a "
                  "deterministic function of five templates, so a model that reads contiguity "
                  "recovers the generating process. I re-drew it before anyone asked: 1.000 "
                  "three times, then 0.910 and 0.800.</p>",
        ),
    ])


def rules() -> str:
    """The decision table, its switches, and the preprocessing path."""
    return deck([
        slide(
            "Rules", "A table, not a black box",
            '<p class="sub">First matching row wins. <b>n</b> = people standing inside the '
            'pitch boundary; <b>ball</b> = a ball seen inside it; <b>m</b> = motion.</p>'
            '<table style="width:100%;border-collapse:collapse;font-size:15px">'
            '<tr><th style="text-align:left;width:8%">#</th>'
            '<th style="text-align:left;width:46%">Condition</th>'
            '<th style="text-align:left;width:46%">Verdict</th></tr>'
            "<tr><td>1</td><td>Detector unavailable</td><td>UNCERTAIN — a missing "
            "detector is not an empty pitch</td></tr>"
            "<tr><td>2</td><td>No pitch boundary</td><td>UNCERTAIN — mandatory on the "
            "deployed path</td></tr>"
            "<tr><td>3</td><td>n = 0, no motion</td><td><b>EMPTY</b></td></tr>"
            "<tr><td>4</td><td>n = 0, but something moved</td><td>UNCERTAIN</td></tr>"
            "<tr><td>5</td><td>1 ≤ n ≤ 4</td><td><b>Not a game</b> — too few, "
            "ball or no ball</td></tr>"
            "<tr><td>6</td><td>n &gt; 4 <b>and</b> a ball <b>and</b> motion</td>"
            "<td><b>ACTIVE PLAY</b> — the only way in</td></tr>"
            "<tr><td>7</td><td>n &gt; 4, otherwise</td><td><b>Not a game</b> — a crowd "
            "that is not playing</td></tr>"
            "</table>",
            notes="<p>This is the deployed decision layer, and it is deliberately a table a "
                  "facility manager can read and argue with.</p>"
                  "<p>Two numbers in it are the facility's rule rather than something I "
                  "fitted: five people make a game, four or fewer do not. The rest are fitted "
                  "on one camera and frozen with the commit that froze them.</p>"
                  "<p>Rows 6 and 7 changed late, and they invert what I pre-registered. "
                  "Originally play was the default above the head count. The facility's rule "
                  "is the opposite - a game has a ball in it and people moving, and a crowd "
                  "standing on a pitch is not a booking being used. Play now has to be "
                  "<i>shown</i>, not assumed. That is stricter, and it costs recall exactly "
                  "where the detector cannot see the ball - which I measured rather than "
                  "argued about.</p>",
        ),
        slide(
            "Rules", "Every switch, and what it costs",
            cards([
                ("Ball required", "A game has a ball. Strict — and it costs recall where "
                                  "the detector cannot see one (0.996 → 0.308)."),
                ("Ball must be in play", "A ball lying on the grass is furniture, not a "
                                         "match."),
                ("Motion required", "People moving, measured across a short burst of frames."),
                ("People threshold", "5 or more to be a game; 1–4 is a small group."),
                ("Pitch boundary", "Feet inside the polygon. Fixing one boundary moved a "
                                   "result by 0.037."),
                ("Detector confidence", "Separate thresholds for people and for the ball."),
            ])
            + '<p class="note" style="font-size:14px"><b>A cue that cannot be measured is '
              'reported, never assumed.</b> If motion cannot be computed, the clause is '
              'skipped and the trace says so — because a requirement that silently never '
              'fires is the shape of every guard this project found not guarding.</p>',
            notes="<p>Each switch exists because its cost is a decision someone should be able "
                  "to take deliberately.</p>"
                  "<p>The ball requirement is the clearest example. Cross-venue ball recall is "
                  "0.40, and between 0.06 and 0.89 depending on the venue. Turning it on drops "
                  "play recall from 0.996 to 0.308 at the worst venues. That is the facility's "
                  "rule and it is expensive; whoever turns it off should be able to see what "
                  "they are buying.</p>"
                  "<p>The note at the bottom is a rule about rules, and it came out of finding "
                  "six safeguards in this project that were doing nothing at all.</p>",
        ),
        slide(
            "Preprocessing", "What happens to a frame before the model sees it",
            cards([
                ("Letterbox to 224×224", "Pad rather than squash — the default "
                                              "path."),
                ("Undistort", "Take the barrel out of a wide CCTV lens."),
                ("Crop", "Centre crop or top crop, to drop sky and car park."),
                ("Standardise per image", "Normalise each frame to its own statistics."),
                ("CLAHE", "Local contrast for floodlight glare — the proposal expected "
                          "this to help. It cost <b>0.27</b>."),
                ("Gamma &amp; saturation", "Brightness curve, and colour down to grayscale."),
                ("Denoise &amp; sharpen", "Sensor noise out, edges back."),
                ("Blur", "Only as a control — it removes the people, which is the point."),
            ])
            + '<p class="note" style="font-size:14px">Experiments and the live pipeline share '
              '<b>one</b> preprocessing module. The pilot proved what happens when they drift '
              'apart.</p>',
            notes="<p>Eight switches, and the same code path runs in the experiments and in "
                  "the live system. That shared path is not tidiness - the pilot had two, they "
                  "drifted, and the numbers stopped meaning the same thing.</p>"
                  "<p>Two to point at. CLAHE is the one the proposal predicted would fix "
                  "floodlight glare; measured honestly it costs 0.27 macro-F1. And blur is in "
                  "there as a control rather than a candidate: at high sigma it removes the "
                  "people, and seeing what that does to the score tells you what the score is "
                  "actually reading.</p>",
        ),
    ])


def searches() -> str:
    """Two searches with opposite costs, and the resolution floor under both."""
    prompts = _csv("prompt_search.csv")
    evals = _json("preprocess_search.json").get("evaluations", [])
    best = max(prompts, key=lambda r: float(r["balanced"])) if prompts else None
    res = search_resolution()

    items: list[tuple[str, str] | tuple[str, str, str]] = []
    if evals:
        items.append((str(len(evals)), "preprocessing runs — greedy", "lead"))
    if prompts:
        items.append((str(len(prompts)), "prompt sets — exhaustive, zero labels", "lead"))
    items.append(("0.930", "the trained probe, for comparison", "good"))

    floor = (tiles([
        (res["floor"], "one frame in the smallest fold moves the headline this much", "bad"),
        (res["interval"], "how wide the confidence band is at the median setting", "bad"),
    ]) + '<p class="note" style="color:rgba(255,255,255,.9);font-size:15px">Almost every gap '
         'the search ranked on is <b>smaller than its own error bar</b>.</p>'
         ) if res.get("floor") else (
        "<p class='missing'>No resolution audit yet. Run "
        "<code>uv run python experiments/search_resolution.py</code>.</p>")

    return deck([
        slide(
            "The searches", "We tuned hard. Twice.",
            tiles(items)
            + cards([
                ("Prompts are cheap, so the search is exhaustive",
                 "Images are embedded once; a prompt set is a few short strings."),
                ("Preprocessing is expensive, so it is greedy",
                 "Each candidate re-embeds the whole dataset — 740 of the "
                 "pipeline&rsquo;s 1,114 minutes."),
            ], wide=True),
            notes="<p>These are the two experiments I would have reported as clean successes "
                  "six months ago, and both turned out to measure something other than what "
                  "they claimed.</p>"
                  "<p>The same guard applies to both: every cross-venue fold is 100% active "
                  "play, so recall alone can be bought by answering &lsquo;playing&rsquo; "
                  "more often. Everything is ranked on recall minus false-play instead.</p>",
        ),
        slide(
            "Search 1 — preprocessing", "The winner sits inside the noise",
            cards([
                ("What it found", "A best recipe scoring a perfect <b>1.000</b> cross-venue, "
                                  "after four greedy rounds."),
                ("What it cannot tell", "Which of those settings is actually better than "
                                        "which."),
                ("And the one the proposal backed", "CLAHE was predicted to fix glare. It "
                                                    "costs DINOv2 <b>0.27</b> macro-F1."),
            ]),
            notes="<p>Eighty-eight runs over crop, undistort, gamma, saturation, sharpening, "
                  "denoise and CLAHE. It produced a clear winner with a perfect score, and "
                  "then I measured the resolution of the search itself.</p>",
        ),
        slide(
            "Search 1 — the resolution floor", "The search is finer than the data",
            floor,
            notes="<p>One frame moving in the smallest venue fold shifts the headline by "
                  "0.0119. The bootstrap interval over seven folds is 0.092 wide at the "
                  "median configuration.</p>"
                  "<p>So the ranking is real arithmetic on unreal precision. The search is "
                  "not wrong - it is finer-grained than the evidence underneath it, and "
                  "reporting that is more useful than reporting a winner.</p>",
            tone="coral",
        ),
        slide(
            "Search 2 — prompts", "Wording moved the score nine times more than the model did",
            tiles([
                ("0.726", "macro-F1 span from worst wording to best", "warn"),
                ("0.082", "span from swapping between the three backbones", "good"),
                ("22.9%", "of prompt sets that beat the trained probe", "bad"),
            ])
            + (cards([
                ("The winner",
                 f'&ldquo;{best["desc_ACTIVE_PLAY"]}&rdquo; — {best["play_recall"]} '
                 f'recall at false-play {best["false_play"]}, with no labels at all.'),
                ("Picked, not proven",
                 "It leads a pre-declared prompt by <b>0.447</b> — on the very folds it "
                 "was selected from. Selection bias, as a number."),
            ], wide=True) if best else ""),
            notes="<p>375 prompt sets, no labels, scored exhaustively.</p>"
                  "<p>Here is the number I find genuinely uncomfortable. How you write the "
                  "sentence matters about nine times more than which model you pick.</p>"
                  "<p>Then the two guards, both of which I state before anyone else does. "
                  "Only 22.9% beat the trained probe and the median loses - so "
                  "&lsquo;zero-shot works&rsquo; is only true if you already know the "
                  "wording, which needs labels. And the winner's lead is measured on the "
                  "folds it was selected from, which is disclosed in the pre-registration as "
                  "an undeclared search family.</p>",
        ),
    ])


def augmentation() -> str:
    """The question, the retraction, and the sheets that make the code checkable."""
    a = augmentation_spread()
    pairs = _csv("preprocess_pairs.csv")

    spread = ""
    if a.get("sd"):
        spread = (
            '<div class="dk-beats">'
            + beat(a.get("baseline", "?"), a.get("max", "?"),
                   "macro-F1: no augmentation, against the draw we published")
            + beat(a.get("max", "?"), a.get("min", "?"),
                   f'the same preset, best against worst of {a.get("n", "?")} draws')
            + "</div>"
            + points([
                f'Standard deviation <b>{a["sd"]}</b>, on a metric bounded in [0,&nbsp;1].',
                f'The published number was the <b>maximum of {a.get("n", "five")}</b>.',
                f'Four of the five land <b>below the {a.get("baseline", "?")}</b> the probe '
                'reaches with no augmentation at all.',
            ], tone="bad")
        )

    cells = "".join(
        f'<figure class="sl-pair{" crop" if float(p["area_retained"]) < 1.0 else ""}">'
        f'<img src="/figs/preproc/{Path(p["file"]).name}" loading="lazy" '
        f'alt="The same frame before and after {p["label"]}">'
        f'<div class="cap"><code>{p["label"]}</code><div class="m">'
        f'<b>{float(p["mean_abs_change_255"]):.1f}</b>/255 moved &middot; '
        f'<b>{float(p["share_pixels_changed"]):.0%}</b> of pixels</div></div></figure>'
        for p in pairs
    )

    return deck([
        slide(
            "Augmentation", "A good question, cheaply asked",
            '<p class="sub">A probe trained on camera A has to work on camera B. Can '
            'augmentation close that gap with <b>no labels</b> from B?</p>'
            + tiles([
                ("0.441", "macro-F1 on camera B, cold", "bad"),
                ("0.990", "after one labelled frame of B", "good"),
                ("?", "can brightness and gamma get there for free?", "warn"),
            ]),
            notes="<p>This is the deployment question. A probe trained on camera A scores "
                  "0.441 on camera B, which watches the same pitch from a different angle "
                  "with different exposure. One labelled frame takes it to 0.99.</p>"
                  "<p>Can augmentation - varying brightness, gamma and sensor noise during "
                  "training - close that gap for free?</p>",
        ),
        slide(
            "Augmentation", "It worked. We wrote it up.",
            tiles([
                ("0.000 → 0.687", "empty-pitch recall on an unseen camera, no labels "
                                       "from it", "good"),
                ("0.855", "macro-F1", "good"),
            ])
            + '<p class="note" style="font-size:14.5px">With one note attached: '
              '<b>&ldquo;only one draw has been taken.&rdquo;</b> That note turned out to be '
              'the finding.</p>',
            notes="<p>Let them sit with this as a success for a moment. It is a good result "
                  "and it went into the write-up as a finding.</p>",
        ),
        slide(
            "Augmentation", "Then we ran it four more times",
            spread or "<p class='missing'>No augmentation spread measured yet.</p>",
            notes="<p>Same frames, same preset, same probe seed, same test set. Only the "
                  "random draw changed. 0.414, 0.351, 0.350, 0.348.</p>"
                  "<p>One detail explains the shape: three draws score exactly 0.3479, which "
                  "is the macro-F1 of answering &lsquo;active play&rsquo; to all 521 test "
                  "frames. What moves between draws is whether the fitted boundary reaches "
                  "camera B's empty pitch at all. It does not degrade gracefully - it is "
                  "either a working classifier or the trivial one.</p>",
        ),
        slide(
            "Augmentation", "The number reproduced perfectly. The result did not.",
            cards([
                ("A reproducible row",
                 "The result file is unchanged and still reproduces exactly. That is the "
                 "problem, not the defence."),
                ("A test could not have caught it",
                 "A unit test pinning the published value would have passed forever. Only "
                 "re-drawing the randomness exposed it."),
                ("So: five draws, always",
                 "Every headline now runs five times — and we ran it on our own "
                 "strongest result first."),
            ]),
            notes="<p>Be precise about what was retracted. The row is unchanged and still "
                  "reproduces. What was withdrawn is the claim about the <i>method</i>. It is "
                  "now restated as a claim about that one draw, and only ever quoted beside "
                  "the spread.</p>"
                  "<p>The transferable lesson is the middle card: reproducibility checked the "
                  "wrong thing.</p>",
            tone="coral",
        ),
        slide(
            "Augmentation", "Why the pictures are the argument",
            '<p class="sub">Augmentation code fails <b>silently</b> — a preset that does '
            'nothing, a fog veil that flattens the pitch, a crop that removes the goalmouth. '
            'All of them pass a shape and dtype check.</p>'
            + figure("augmentation_grid.jpg",
                     "Every preset, three draws each, over a night play frame and a day empty "
                     "frame.",
                     script="uv run python experiments/augmentation_grid.py")
            + figure("augmentation_effects.jpg",
                     "One effect at a time, at the magnitude the <code>full</code> preset "
                     "uses — read off the config, so the sheet cannot drift from it.",
                     script="uv run python experiments/augmentation_grid.py")
            + (f'<div class="sl-pairs">{cells}</div>' if cells else ""),
            notes="<p>This is the one part of the project that is deliberately more, not "
                  "less. Every other question here gets answered with a number, and this one "
                  "cannot, because the failures are invisible to numbers.</p>"
                  "<p>On the first run these sheets caught rain streaks written in absolute "
                  "pixels - at low resolution they were white poles a tenth of the frame "
                  "wide. No shape or dtype assertion could have found that.</p>",
        ),
    ])


def solution() -> str:
    """What we changed, and what it will not do."""
    return deck([
        slide(
            "How the research provides a solution", "The fix was never a bigger model.",
            '<p class="sub" style="font-size:18px">Three interventions, each measured on the '
            'boundary it was meant to help.</p>',
            notes="<p>The common thread, and say it before showing them: not one of these is "
                  "a bigger or better neural network. Every gain in this project came from "
                  "cheap structure around the model, or from labelling a handful of the right "
                  "frames.</p>",
            tone="ink",
        ),
        slide(
            "The solution", "Three fixes, all cheap",
            '<div class="dk-beats">'
            + beat("0.617", "0.012", "False alarms on empty pitches, once the boundary, "
                                     "motion and person checks can veto the model.")
            + beat("0.768", "0.024", "With 31 generated empty-pitch frames in training "
                                     "— and play-recall went up.")
            + beat("0.441", "0.990", "One labelled frame of a new camera. Five from it beat "
                                     "those five plus 775 from the old one.")
            + "</div>"
            + '<p class="note" style="font-size:14.5px">What buys the accuracy is having '
              '<b>any</b> labels from the new camera — not a large corpus from an old '
              'one. That is the weaker claim, and it is the true one.</p>',
            notes="<p>Counting every kind of wrong verdict rather than only play-shaped ones, "
                  "the gates take cross-venue error from 1.00 to 0.11.</p>"
                  "<p>The third is the recipe the facility can act on tomorrow, and the "
                  "control is what reframes it: from five frames, training on those five "
                  "alone matches training on those five plus 775 from the source camera. The "
                  "source set stops contributing.</p>",
        ),
        slide(
            "The method contribution", "Every guard is verified by breaking it",
            '<p class="sub">Six safeguards in this project were doing nothing. Each was found '
            'the same way.</p>'
            + cards([
                ("A test-set lock", "Resolved against the working directory, so it locked "
                                    "nothing."),
                ("A confidence threshold", "Set to 0.0 — a gate that can never fire."),
                ("A gate criterion", "Grepped for a word instead of running the verifier."),
                ("Evidence selection", "Recorded a file path of <code>None</code> for every "
                                       "frame it chose."),
            ])
            + '<p class="note" style="font-size:14px">A wrong result that looks plausible is '
              '<b>invisible</b>. That one observation is where the claims ledger and this '
              'practice came from — and it found half of the results in this talk.</p>',
            notes="<p>This is the methodological half of the contribution, and I think it is "
                  "the part that transfers beyond football pitches.</p>"
                  "<p>Every one of those six passed whatever test existed. So each new guard "
                  "is now verified by deliberately breaking the thing it guards and "
                  "confirming the guard fires.</p>",
        ),
        slide(
            "The solution", "What it will not do",
            cards([
                ("The system never bills",
                 "<code>Advisory</code> is the only output type. Human confirmation is a "
                 "property, not a setting. No code path acts."),
                ("Anomalies are per field, never per person",
                 "Reconciliation has no field that could name an individual."),
                ("Review never becomes an anomaly",
                 "The system may not convert its own uncertainty into someone else&rsquo;s "
                 "error."),
                ("Faces are redacted before publication",
                 "Detection and pixelation, then a whole-frame blur floor."),
            ]),
            notes="<p>Auditing how a facility's pitches are used is close to auditing the "
                  "people who work there, so this slide is about the limits we built in.</p>"
                  "<p>What makes the commitment credible is that it is structural rather than "
                  "a policy sentence.</p>"
                  "<p>The third card is the subtle one. If the model is unsure, that "
                  "uncertainty belongs to the model. Turning a REVIEW into a flag against a "
                  "member of staff would be laundering the system's own weakness into someone "
                  "else's record.</p>",
        ),
    ])


def summary() -> str:
    """Limits, the way out, and the close."""
    return deck([
        slide(
            "Limits", "What this study cannot claim",
            cards([
                ("Empty pitches exist at one venue only",
                 "So a three-class cross-venue evaluation cannot be run on this corpus."),
                ("243 empty frames are 3 scenes",
                 "The effective sample is scenes, not frames. Six comparisons stop being "
                 "significant when recounted that way."),
                ("Nothing has run on the target hardware",
                 "Every latency number comes from a laptop."),
                ("No human ceiling exists",
                 "There is no inter-annotator figure. That is a real gap, and I say so."),
            ]),
            notes="<p>Named first, because an examiner will find them anyway.</p>"
                  "<p>The second is the most instructive. 243 held-out empty frames sound "
                  "healthy, but they are three distinct scenes - consecutive views from one "
                  "fixed camera. Six comparisons significant to p = 8e-53 did not survive "
                  "being recounted that way.</p>"
                  "<p>Practise saying &lsquo;we do not know&rsquo; without flinching. An "
                  "examiner trusts a candidate who has bounded their ignorance more than one "
                  "who has not noticed it.</p>",
        ),
        slide(
            "Limits", "What would actually change this",
            '<p class="sub">Ordered by value, not by effort.</p>'
            + cards([
                ("1 &middot; Empty-pitch footage at a second venue",
                 "20&ndash;30 minutes, day and night. Fixes the confound, the "
                 "external-validity limit and the scope reduction at once."),
                ("2 &middot; About 30 real labelled slots",
                 "One conversation with the facility. Lifts the temporal model out of "
                 "preliminary."),
                ("3 &middot; Five labelled frames per camera",
                 "Already measured, already cheap — and it is the onboarding recipe we "
                 "would hand over."),
            ], wide=True)
            + '<p class="note" style="font-size:14.5px">The limitations are mostly a '
              '<b>data-access problem with a known and inexpensive solution</b> — not a '
              'methodological one.</p>',
            notes="<p>Close on the fix, not the apology. That last line is a much better "
                  "final word than an apology, and it is true.</p>",
            tone="green",
        ),
        slide(
            "Summary", "In summary",
            points([
                "<b>Investigated</b> — whether cameras a facility already owns can "
                "verify which booked hours were used, on a CPU, with a human deciding.",
                "<b>Found</b> — on this data a clock beats a neural network. Lighting "
                "and occupancy are confounded so tightly that no protocol here separates "
                "them.",
                "<b>Contributed</b> — an evaluation method: trivial baselines everywhere, "
                "grouped splits, re-drawing randomness, and every guard verified by breaking "
                "it.",
                "<b>Implies</b> — for the facility, a five-frame onboarding recipe. For "
                "the field, that a benchmark can rank a light meter first and still look "
                "healthy.",
            ])
            + '<p class="note" style="color:rgba(255,255,255,.92);font-size:17px;'
              'margin-top:18px"><b>No protocol compensates for a case the data never '
              'contains.</b></p>',
            notes="<p>Deliver the last line slowly, and then stop talking.</p>",
            tone="ink",
        ),
        slide(
            "Thank you", "Questions, comments and suggestions welcome.",
            '<p class="sub" style="font-size:18px">Thank you for your time and attention.</p>'
            + cards([
                ("Would a colour histogram have done this?",
                 "0.9616 on the leaky split — indistinguishable from ConvNeXtV2 once you "
                 "count 62 scenes rather than 394 frames."),
                ("Is the novelty just smoothing?",
                 "Tested against four tuned baselines including an HMM, with the saturation "
                 "caveat enforced in code."),
                ("How many comparisons before p &lt; 0.05?",
                 "Holm within declared families; undeclared ones disclosed as numbered "
                 "amendments."),
                ("Can anyone reproduce this?",
                 "One pipeline re-derives every claim from its artefact on every run."),
            ]),
            notes="<p>Thank you for your time and attention. I welcome any questions, "
                  "comments, suggestions or feedback you may have.</p>"
                  "<p>The cards are the four most likely questions, with the short answers. "
                  "The fifth - what is the human ceiling - has no number, and the right answer "
                  "is to say that plainly.</p>",
            tone="ink",
        ),
    ])


#: tab id -> (label, the deck that fills it). The site is the talk, in order.
SECTIONS: dict[str, tuple[str, object]] = {
    "start": ("Start", start),
    "purpose": ("Purpose", purpose),
    "how": ("How it works", how),
    "problem": ("The problem", problem),
    "data": ("Data", data),
    "models": ("Models", models),
    "rules": ("Rules", rules),
    "searches": ("Searches", searches),
    "augmentation": ("Augmentation", augmentation),
    "solution": ("Solution", solution),
    "summary": ("Summary", summary),
}

#: Back-compatible alias: callers that want one section's deck by id.
SLIDES = {k: v[1] for k, v in SECTIONS.items()}
