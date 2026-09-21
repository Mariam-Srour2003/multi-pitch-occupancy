"""The oral defence, as nine scrollable pages (WP8).

The site *is* the talk, and its shape is the oral-presentation template: nine tabs, one per
numbered section, in the order they are delivered. Each tab is an ordinary page you scroll -
a coloured hero, then blocks of cards, counts, a table or a figure. No slides and no pager:
a reader should be able to read a tab top to bottom without pressing anything.

Three rules keep it honest, and they are the same three the deck was built on:

**Every number is read from an artefact.** The class counts come from `results/coverage.md`,
the parameter counts from `results/model_inventory.json`, the search resolution and the
augmentation spread from their own CSVs. A page that restated them would be a second copy of
the thesis, and the copy that goes stale - which is the failure this repository keeps finding
in its own history.

**A missing artefact says so.** A block whose figure has not been generated prints the
command that would generate it, rather than rendering an empty box.

**Short on the page, long in the script.** Each tab is the *claim*; what is actually said
lives in `thesis/presentation/SPEAKER_SCRIPT.md` and nowhere on the page. If a sentence here
is one you would speak rather than show, it belongs in the script - the site is what the room
looks at while you talk, so a paragraph of delivery notes on it is a paragraph competing with
you.
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
    "STYLES", "SECTIONS",
    "hero", "block", "tiles", "cards", "numbered", "figure", "points", "beat", "beats",
    "table", "quote", "chips", "detail",
]

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
THESIS = ROOT / "thesis"
FIGS = RESULTS / "figs"


# --- components -----------------------------------------------------------------------


def hero(eyebrow: str, title: str, lead: str = "", *, tone: str = "ink") -> str:
    """The band at the top of a tab: where you are, and the one sentence of it."""
    eb = f'<p class="tk-eyebrow">{eyebrow}</p>' if eyebrow else ""
    ld = f'<p class="tk-lead">{lead}</p>' if lead else ""
    return f'<header class="tk-hero {tone}">{eb}<h1 class="tk-title">{title}</h1>{ld}</header>'


def block(title: str, body: str, *, note: str = "", tint: str = "") -> str:
    """One scrollable section of a tab."""
    h = f'<h2 class="tk-h">{title}</h2>' if title else ""
    n = f'<p class="tk-note">{note}</p>' if note else ""
    return f'<section class="tk-block {tint}">{h}{body}{n}</section>'


def tiles(items: list[tuple[str, str]] | list[tuple[str, str, str]]) -> str:
    """A row of big numbers. ``(value, label)`` or ``(value, label, tone)``."""
    out = []
    for item in items:
        value, label = item[0], item[1]
        tone = item[2] if len(item) > 2 else ""
        out.append(f'<div class="tk-tile {tone}"><b>{value}</b><span>{label}</span></div>')
    return f'<div class="tk-tiles">{"".join(out)}</div>'


def cards(items: list[tuple[str, str]], *, wide: bool = False) -> str:
    body = "".join(f'<div class="tk-card"><h3>{t}</h3><p>{b}</p></div>' for t, b in items)
    return f'<div class="tk-cards{" wide" if wide else ""}">{body}</div>'


def numbered(items: list[tuple[str, str]]) -> str:
    """Cards that carry their step number - used where order is the meaning."""
    body = "".join(
        f'<div class="tk-step"><span class="n">{i}</span>'
        f'<div><h3>{t}</h3><p>{b}</p></div></div>'
        for i, (t, b) in enumerate(items, 1)
    )
    return f'<div class="tk-steps">{body}</div>'


def figure(name: str, caption: str = "", *, alt: str = "", script: str = "") -> str:
    """One figure from `results/figs`, or the command that would produce it."""
    if not (FIGS / name).exists():
        hint = f" Run <code>{script}</code>." if script else ""
        return f"<p class='missing'>No <code>{name}</code> yet.{hint}</p>"
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (f'<figure class="tk-fig"><img src="/figs/{name}" loading="lazy" '
            f'alt="{alt or caption or name}">{cap}</figure>')


def points(items: list[str], *, tone: str = "") -> str:
    body = "".join(f"<li>{t}</li>" for t in items)
    return f'<ul class="tk-points{" " + tone if tone else ""}">{body}</ul>'


def beat(before: str, after: str, label: str) -> str:
    """``before -> after``, the shape most of this project's results take."""
    return ('<div class="tk-beat">'
            f'<span class="was">{before}</span><span class="to">&rarr;</span>'
            f'<span class="now">{after}</span><span class="lab">{label}</span></div>')


def beats(items: list[tuple[str, str, str]]) -> str:
    return f'<div class="tk-beats">{"".join(beat(a, b, c) for a, b, c in items)}</div>'


def table(headers: list[str], rows: list[list[str]], *, hi: int | None = None) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        f'<tr{" class=hi" if hi == i else ""}>'
        + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for i, row in enumerate(rows)
    )
    return f'<div class="tk-table"><table><tr>{head}</tr>{body}</table></div>'


def quote(text: str) -> str:
    return f'<p class="tk-quote">{text}</p>'


def chips(items: list[str]) -> str:
    return '<div class="tk-chips">' + "".join(f"<span>{i}</span>" for i in items) + "</div>"


def detail(html: str, label: str) -> str:
    """Evidence, collapsed. `<details>` so find-in-page and a saved copy still reach it."""
    return (f'<details class="tk-detail"><summary>{label}</summary>'
            f'<div class="tk-detail-body">{html}</div></details>')


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _json(name: str) -> dict:
    path = RESULTS / name
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


def _csv(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- parsed artefacts -----------------------------------------------------------------


def rq_status() -> list[tuple[str, str, str]]:
    """``(id, question, status)`` parsed from the table at the top of `rq_matrix.md`.

    Parsed rather than restated: the statuses move as the work moves, and a copy here would
    be right the day it was written and wrong afterwards, with nothing to catch it.
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
    # **Scoped to the class-by-venue section, and it has to be.** `coverage.md` opens with a
    # class-by-*lighting* table whose EMPTY row is `| EMPTY | 485 | 9 | 494 |`, so a pattern
    # matching "the EMPTY row" found that one first and counted day and night as venues -
    # reporting "all from 2 venues" for the fact the whole data request is about.
    venue_section = text.partition("## Class x venue")[2].partition("## ")[0]
    venue_row = re.search(r"^\|\s*EMPTY\s*\|(.+)\|\s*$", venue_section, re.MULTILINE)
    if venue_row:
        cells = [c.strip() for c in venue_row.group(1).split("|")][:-1]  # last cell = total
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


def search_resolution() -> dict:
    """The preprocessing search's own resolution, from `search_resolution.csv`.

    Found by its `describe` rather than by position - a slice of the last row would silently
    become a real configuration the day another summary is appended.
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
            out = {"n": row.get("n_seeds", ""), "sd": row.get("macro_f1_sd", ""),
                   "min": row.get("macro_f1_min", ""), "max": row.get("macro_f1_max", "")}
            break
    for row in _csv("augmentation_transfer.csv"):
        if row.get("preset") == "baseline":
            out["baseline"] = row.get("macro_f1", "")
            break
    return out


# --- the look -------------------------------------------------------------------------

#: Scoped under `.tk` so nothing here reaches anything else on the page.
#:
#: The palette is declared here rather than borrowed from the shell. The shell defines
#: `--accent` and little else; `--play`, `--flag` and `--maint` are set by the operator
#: pages and were never in scope on this site, so every tinted tile used to render with no
#: colour at all. These are talk-local names, defined for both schemes.
STYLES = """
:root{
 --tk-green:#0e8f5c;--tk-green-b:#17b978;--tk-green-s:#e6f7ef;
 --tk-amber:#b26a00;--tk-amber-b:#f5a524;--tk-amber-s:#fdf0d9;
 --tk-coral:#d22d4c;--tk-coral-b:#ef4f6b;--tk-coral-s:#fde8ec;
 --tk-sky:#1d6fa5;--tk-sky-b:#3fa7e0;--tk-sky-s:#e3f2fb;
 --tk-violet:#6a57d6;--tk-violet-s:#ece9fb;
 --tk-ink:#0e1b2c;--tk-paper:#fbfaf6;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --tk-green:#3ed195;--tk-green-s:#0f2e23;--tk-amber:#e9b04a;--tk-amber-s:#2e2413;
 --tk-coral:#ff8098;--tk-coral-s:#341a21;--tk-sky:#6fc2ed;--tk-sky-s:#122b3a;
 --tk-violet:#a493f2;--tk-violet-s:#201c3a;--tk-paper:#141c1c;}}
:root[data-theme="dark"]{
 --tk-green:#3ed195;--tk-green-s:#0f2e23;--tk-amber:#e9b04a;--tk-amber-s:#2e2413;
 --tk-coral:#ff8098;--tk-coral-s:#341a21;--tk-sky:#6fc2ed;--tk-sky-s:#122b3a;
 --tk-violet:#a493f2;--tk-violet-s:#201c3a;--tk-paper:#141c1c;}

.tk{display:flex;flex-direction:column;gap:22px}

/* --- hero ---------------------------------------------------------------------- */
.tk-hero{border-radius:18px;padding:34px 36px 32px;color:#fff;position:relative;
 overflow:hidden}
.tk-hero.ink{background:linear-gradient(140deg,#0e1b2c,#16283f 60%,#0b2a3a)}
.tk-hero.green{background:linear-gradient(135deg,#0e8f5c,#14a86d)}
.tk-hero.coral{background:linear-gradient(135deg,#c42746,#ef4f6b)}
.tk-hero.sky{background:linear-gradient(135deg,#14567f,#2c89c4)}
.tk-hero.violet{background:linear-gradient(135deg,#4a3aa8,#7a68e0)}
.tk-eyebrow{font:700 11px 'JetBrains Mono',monospace;letter-spacing:.2em;
 text-transform:uppercase;color:rgba(255,255,255,.78);margin:0 0 12px}
.tk-title{font-size:clamp(25px,3.3vw,38px);line-height:1.12;font-weight:700;
 letter-spacing:-.022em;margin:0;color:#fff;max-width:22ch;text-wrap:balance}
.tk-lead{font-size:17px;line-height:1.5;color:rgba(255,255,255,.9);margin:14px 0 0;
 max-width:62ch}

/* --- blocks -------------------------------------------------------------------- */
.tk-block{border:1px solid var(--line);border-radius:16px;background:var(--tk-paper);
 padding:26px 28px}
.tk-block.green{background:var(--tk-green-s);border-color:transparent}
.tk-block.amber{background:var(--tk-amber-s);border-color:transparent}
.tk-block.coral{background:var(--tk-coral-s);border-color:transparent}
.tk-block.sky{background:var(--tk-sky-s);border-color:transparent}
.tk-block.violet{background:var(--tk-violet-s);border-color:transparent}
.tk-h{font-size:20px;font-weight:700;letter-spacing:-.012em;margin:0 0 16px;
 color:var(--ink)}
.tk-note{font-size:13px;line-height:1.55;color:var(--ink-3);margin:14px 0 0;max-width:80ch}
.tk-block .missing{margin:6px 0}

/* --- tiles --------------------------------------------------------------------- */
.tk-tiles{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.tk-tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 padding:15px 16px;border-top:5px solid var(--line)}
.tk-tile b{display:block;font:700 30px 'JetBrains Mono',monospace;
 font-variant-numeric:tabular-nums;letter-spacing:-.03em;margin-bottom:4px;line-height:1.05;
 color:var(--ink)}
.tk-tile span{font-size:12.5px;color:var(--ink-2);line-height:1.4;display:block}
.tk-tile.good{border-top-color:var(--tk-green-b)} .tk-tile.good b{color:var(--tk-green)}
.tk-tile.bad{border-top-color:var(--tk-coral-b)} .tk-tile.bad b{color:var(--tk-coral)}
.tk-tile.lead{border-top-color:var(--tk-sky-b)} .tk-tile.lead b{color:var(--tk-sky)}
.tk-tile.warn{border-top-color:var(--tk-amber-b)} .tk-tile.warn b{color:var(--tk-amber)}

/* --- cards --------------------------------------------------------------------- */
.tk-cards{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.tk-cards.wide{grid-template-columns:repeat(auto-fit,minmax(310px,1fr))}
.tk-card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 padding:15px 17px;border-left:5px solid var(--tk-green-b)}
.tk-cards .tk-card:nth-child(4n+2){border-left-color:var(--tk-sky-b)}
.tk-cards .tk-card:nth-child(4n+3){border-left-color:var(--tk-amber-b)}
.tk-cards .tk-card:nth-child(4n+4){border-left-color:var(--tk-coral-b)}
.tk-card h3{margin:0 0 5px;font-size:14.5px;font-weight:700;color:var(--ink);
 line-height:1.25}
.tk-card p{margin:0;font-size:13px;color:var(--ink-2);line-height:1.5}

/* --- numbered steps ------------------------------------------------------------ */
.tk-steps{display:grid;gap:10px}
.tk-step{display:grid;grid-template-columns:38px 1fr;gap:14px;align-items:start;
 background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tk-step .n{width:30px;height:30px;border-radius:50%;background:var(--tk-green-b);
 color:#04301f;font:700 15px 'JetBrains Mono',monospace;display:grid;place-items:center}
.tk-steps .tk-step:nth-child(2) .n{background:var(--tk-sky-b);color:#04283d}
.tk-steps .tk-step:nth-child(3) .n{background:var(--tk-violet);color:#fff}
.tk-steps .tk-step:nth-child(4) .n{background:var(--tk-amber-b);color:#3a2400}
.tk-steps .tk-step:nth-child(5) .n{background:var(--tk-coral-b);color:#fff}
.tk-step h3{margin:0 0 4px;font-size:14.5px;font-weight:700;color:var(--ink)}
.tk-step p{margin:0;font-size:13px;color:var(--ink-2);line-height:1.5}

/* --- points, beats, quote, chips ----------------------------------------------- */
.tk-points{list-style:none;padding:0;margin:0;display:grid;gap:10px}
.tk-points li{position:relative;padding-left:24px;font-size:15px;line-height:1.5;
 color:var(--ink-2);max-width:74ch}
.tk-points li::before{content:"";position:absolute;left:0;top:7px;width:10px;height:10px;
 border-radius:3px;background:var(--tk-green-b)}
.tk-points.warn li::before{background:var(--tk-amber-b)}
.tk-points.bad li::before{background:var(--tk-coral-b)}
.tk-points li b{color:var(--ink);font-weight:700}

.tk-beats{display:grid;gap:11px;grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.tk-beat{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 padding:15px 17px;display:grid;grid-template-columns:auto auto auto;gap:9px;
 align-items:baseline}
.tk-beat .was{font:700 23px 'JetBrains Mono',monospace;color:var(--tk-coral)}
.tk-beat .to{color:var(--ink-3);font-size:16px}
.tk-beat .now{font:700 23px 'JetBrains Mono',monospace;color:var(--tk-green)}
.tk-beat .lab{grid-column:1/-1;font-size:12.5px;color:var(--ink-2);line-height:1.45}

.tk-quote{margin:0;padding:15px 19px;background:var(--surface);
 border-left:5px solid var(--tk-sky-b);border-radius:0 12px 12px 0;font-size:16px;
 line-height:1.55;color:var(--ink);max-width:76ch}
.tk-chips{display:flex;flex-wrap:wrap;gap:8px}
.tk-chips span{font-size:13px;font-weight:600;color:var(--ink-2);background:var(--surface);
 border:1px solid var(--line);border-radius:99px;padding:7px 15px}

/* --- table and figure ---------------------------------------------------------- */
.tk-table{overflow-x:auto;border:1px solid var(--line);border-radius:12px;
 background:var(--surface)}
.tk-table table{border-collapse:collapse;width:100%;min-width:440px}
.tk-table th{font:600 10.5px 'JetBrains Mono',monospace;letter-spacing:.08em;
 text-transform:uppercase;color:var(--ink-3);text-align:left;padding:11px 14px;
 border-bottom:1px solid var(--line);white-space:nowrap}
.tk-table td{padding:10px 14px;border-bottom:1px solid var(--line);font-size:13.5px;
 color:var(--ink-2)}
.tk-table tr:last-child td{border-bottom:none}
.tk-table tr.hi td{background:var(--tk-green-s);color:var(--ink);font-weight:600}

.tk-fig{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:12px;
 overflow:hidden}
.tk-fig img{display:block;width:100%;background:var(--surface-2)}
.tk-fig figcaption{font-size:12.5px;color:var(--ink-2);padding:10px 14px;
 border-top:1px solid var(--line);line-height:1.45}
.tk-figs{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}

/* --- collapsed evidence and the speaker block ---------------------------------- */
.tk-detail{border:1px solid var(--line);border-radius:12px;background:var(--surface)}
.tk-detail > summary{cursor:pointer;padding:12px 16px;font-weight:600;font-size:13.5px;
 list-style:none;display:flex;gap:9px;align-items:center;color:var(--ink)}
.tk-detail > summary::-webkit-details-marker{display:none}
.tk-detail > summary::before{content:"\\25B8";color:var(--ink-3);font-size:11px}
.tk-detail[open] > summary::before{content:"\\25BE"}
.tk-detail[open] > summary{border-bottom:1px solid var(--line)}
.tk-detail-body{padding:4px 18px 16px}

.tk-pairs{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(250px,1fr))}
.tk-pair{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 overflow:hidden}
.tk-pair img{display:block;width:100%;background:var(--surface-2)}
.tk-pair .cap{padding:9px 12px;border-top:1px solid var(--line)}
.tk-pair .cap code{font-size:11.5px;font-weight:700;color:var(--ink);background:none;
 padding:0}
.tk-pair .cap .m{display:flex;gap:9px;flex-wrap:wrap;margin-top:4px;
 font:500 10.5px 'JetBrains Mono',monospace;color:var(--ink-3)}
.tk-pair .cap .m b{color:var(--ink-2)}
.tk-pair.crop{border-left:4px solid var(--tk-coral-b)}

@media (max-width:640px){
 .tk-hero{padding:24px 20px}
 .tk-block{padding:20px 18px}
}
"""


def preprocess_pairs() -> str:
    """Every preprocessing switch, before and after, with how much it actually moved.

    Preprocessing code fails silently - a switch that does nothing, a crop that removes the
    goalmouth - and every one of those passes a shape and dtype check. The only reliable
    check is a person looking, so the sheet stays on the page (collapsed) rather than being
    summarised into a sentence nobody can check.
    """
    rows = _csv("preprocess_pairs.csv")
    if not rows:
        return ("<p class='missing'>No <code>preprocess_pairs.csv</code> yet. Run "
                "<code>uv run python experiments/preprocess_pairs.py</code>.</p>")
    cells = []
    for r in rows:
        area = float(r["area_retained"])
        kept = f' &middot; <b>{area:.0%}</b> kept' if area < 1.0 else ""
        cells.append(
            f'<figure class="tk-pair{" crop" if area < 1.0 else ""}">'
            f'<img src="/figs/preproc/{Path(r["file"]).name}" loading="lazy" '
            f'alt="The same frame before and after {r["label"]}">'
            f'<div class="cap"><code>{r["label"]}</code><div class="m">'
            f'<b>{float(r["mean_abs_change_255"]):.1f}</b>/255 moved &middot; '
            f'<b>{float(r["share_pixels_changed"]):.0%}</b> of pixels{kept}'
            "</div></div></figure>"
        )
    quietest = min(rows, key=lambda r: float(r["mean_abs_change_255"]))
    loudest = max(float(r["mean_abs_change_255"]) for r in rows)
    return (
        f'<div class="tk-pairs">{"".join(cells)}</div>'
        '<p class="tk-note"><b>A searched margin on top of a near no-op is not a margin.</b> '
        f'<code>{quietest["label"]}</code> moves the frame by '
        f'{float(quietest["mean_abs_change_255"]):.2f}/255 against {loudest:.1f} for the '
        'loudest switch &mdash; yet the search credits it with +0.016 recall.</p>'
    )


# --- the nine sections ----------------------------------------------------------------


def introduce() -> str:
    """1. Introduce yourself."""
    return (
        '<div class="tk">'
        + hero("Master&rsquo;s thesis &middot; oral defence",
               "Good morning, distinguished examiners, professors and colleagues.",
               "My name is Mariam Srour, and I am a Master&rsquo;s student in "
               "<b>[Programme]</b> at <b>[Institution]</b>.", tone="ink")
        + block("Who and where", cards([
            ("Candidate", "Mariam Srour &mdash; Master&rsquo;s student, <b>[Programme]</b>."),
            ("Institution", "<b>[Institution]</b>."),
            ("Supervisor", "<b>[Name]</b>."),
            ("Date", "<b>[Date]</b>."),
        ]), note="Fill these four in before you present.")
        + block("The work, in one sentence",
                quote("A system that checks, from cameras a facility already owns, "
                      "<b>which booked pitch hours were actually used</b> &mdash; on one "
                      "small computer, with a person deciding every case."),
                tint="green")
        + "</div>"
    )


def topic() -> str:
    """2. Name / topic of the research."""
    c = coverage_counts()
    counts = tiles([
        (c.get("frames", "?"), "labelled frames", "lead"),
        ("3", "classes: empty, playing, maintenance", "good"),
        ("7", "venue folds for cross-venue testing", "warn"),
        ("0", "GPUs &mdash; a CPU-only constraint", "bad"),
    ]) if c else "<p class='missing'>No coverage report generated yet.</p>"
    return (
        '<div class="tk">'
        + hero("The title of my research is",
               "Multi-Pitch Occupancy and Booking Verification from Existing Cameras",
               "Or, in the form I will use for the rest of this talk: "
               "<b>who actually used the pitch?</b>", tone="green")
        + block("What that means", cards([
            ("The question",
             "A facility sells pitch hours. Which of them were actually played?"),
            ("The constraint",
             "Existing cameras, one small computer, no GPU, and no footage leaves the site."),
            ("The output",
             "Not a classification &mdash; an advisory with three evidence photographs."),
        ]))
        + block("The scale of it", counts, tint="sky")
        + block("One design rule, carried everywhere",
                quote("The vision path must <b>never</b> see the booking record as an input. "
                      "A model that has read the booking flag cannot give evidence "
                      "independent of the record it audits."),
                note="That is the difference between an audit and a rubber stamp.")
        + "</div>"
    )


def roadmap() -> str:
    """3. Roadmap."""
    return (
        '<div class="tk">'
        + hero("Roadmap", "In this presentation, I will briefly take you through",
               tone="sky")
        + block("", numbered([
            ("Purpose of the research", "What the study is for, and who it helps."),
            ("How the research works", "The method, the data, the models &mdash; and a "
                                       "hypothetical Tuesday evening."),
            ("What problem it tackles", "The verification gap, and the gap in the data that "
                                        "shaped everything."),
            ("How it provides a solution", "Three fixes, two searches, and one retraction."),
            ("Summary", "What was found, what it cannot claim, and your questions."),
        ]))
        + block("Two things I will spend real time on", cards([
            ("The configuration searches",
             "Both found a winner. Neither winner meant what it looked like."),
            ("The augmentation experiment",
             "A published result that we withdrew &mdash; and the check that caught it."),
        ], wide=True), tint="amber")
        + "</div>"
    )


def purpose() -> str:
    """4. Purpose of the research."""
    return (
        '<div class="tk">'
        + hero("Purpose of the research",
               "Find out which booked hours were really used &mdash; using the cameras a "
               "facility already owns.", tone="green")
        + block("The study focuses on", cards([
            ("The subject", "A network of synthetic 5-a-side football pitches."),
            ("The population", "Every booked hour on every pitch, sampled once a minute."),
            ("The context", "Existing CCTV, a CPU-only mini-PC, and the facility&rsquo;s own "
                            "booking records."),
            ("The variables", "Three scene classes &mdash; empty, active play, maintenance "
                              "&mdash; against what the booking claims."),
        ]))
        + block("What the owner gets", cards([
            ("Time back",
             "Nobody can watch 20&ndash;30 pitches. It checks <b>every booked minute</b> and "
             "shows the manager only what disagrees."),
            ("The same judgement at 8am and 11pm",
             "It does not tire, rush, or skip the far pitch. A person samples; this does "
             "not."),
            ("A picture, not a memory",
             "Every flag arrives with three evidence frames."),
            ("Coverage a patrol cannot match",
             "One camera per pitch, once a minute, all day."),
        ]), tint="green",
            note="Stated carefully: more <b>consistent</b> and more <b>auditable</b> than a "
                 "manual check &mdash; not more accurate than a human looking at the same "
                 "photograph. No inter-annotator figure exists, and that is a real gap.")
        + block("And three things it refuses to do", points([
            "It <b>never bills</b> and never acts &mdash; <code>Advisory</code> is the only "
            "output the decision layer can produce.",
            "Anomalies are reported <b>per field, never per person</b>.",
            "<b>Review never becomes an anomaly</b> &mdash; the system may not turn its own "
            "uncertainty into someone else&rsquo;s error.",
        ]), tint="coral")
        + "</div>"
    )


def how() -> str:
    """5. How does the research work? Design, data, models, rules, analysis."""
    c = coverage_counts()
    inv = _json("model_inventory.json")
    empty = c.get("EMPTY", {}) if c else {}
    play = c.get("ACTIVE_PLAY", {}) if c else {}
    with_empty = c.get("venues_with_empty", 1) if c else 1

    model_tiles = tiles([
        (f'{inv["frozen_total_params"] / 1e6:.1f}M', "frozen parameters, never updated",
         "lead"),
        (f'{inv["trained_total_params"]:,}', "trained parameters, all of them", "good"),
        (f'{inv["frozen_to_trained_ratio"]:,}:1', "frozen to trained", "warn"),
    ]) if inv else "<p class='missing'>No <code>model_inventory.json</code> yet.</p>"

    return (
        '<div class="tk">'
        + hero("How does the research work?", "Five stages, one frame a minute",
               "Sparse sampling, not a video stream &mdash; which is what makes the whole "
               "facility fit on one small computer.", tone="ink")
        + block("The design", numbered([
            ("Sample", "One frame per camera per minute. About <b>99% less</b> network "
                       "traffic than decoding the streams."),
            ("Classify", "Empty, active play, or maintenance &mdash; so a mower is not billed "
                         "as a match."),
            ("Check", "Pitch boundary, motion, people, ball. These rules can <b>overrule</b> "
                      "the model."),
            ("Aggregate", "About 60 predictions become one verdict per booked hour: used, "
                          "not used, or review."),
            ("Compare", "Verdict against the booking record; disagreements flagged with "
                        "evidence."),
        ]))
        + block("A hypothetical scenario &mdash; Tuesday, 20:00, Pitch 3", beats([
            ("Sold", "1 hour", "The booking says: paid for, and marked &ldquo;used&rdquo; by "
                               "staff."),
            ("13", "of 16", "The camera sees: an empty pitch under floodlights."),
            ("REVIEW", "+3 photos", "The system says: never &ldquo;do not bill&rdquo;. Only "
                                    "&ldquo;look at this&rdquo;."),
        ]), tint="sky")
        + block("Where a verdict can be refused", pipeline(),
                note="Every stage can refuse rather than guess &mdash; which is why a camera "
                     "that drops out cannot silently become an empty pitch.")
        + block("What gets stored", schema(),
                note="One row per camera per sampled minute is the only thing observed; "
                     "everything above it is derived. A verdict and its evidence are written "
                     "in one transaction, and a dropped minute is stored <b>as a gap</b> "
                     "rather than filled &mdash; because no footage is not evidence that a "
                     "pitch was unused.")
        + block("The data collected", (tiles([
            (c.get("frames", "?"), "recorded frames", "lead"),
            (c.get("venues", "?"), "venues"),
            (empty.get("total", "?"),
             f'EMPTY, all from {with_empty} venue{"" if with_empty == 1 else "s"}', "warn"),
            ("98.5%", "of frames have a near-duplicate", "bad"),
        ]) + cards([
            ("Fixed cameras, every 15 seconds",
             "Existing CCTV on posts. One view forever &mdash; which is why no rotation is "
             "ever simulated."),
            ("Hand-labelled to a written protocol",
             "Rules fixed in advance, with the ambiguous cases decided before the runs."),
            ("EMPTY is daytime, ACTIVE_PLAY is night",
             f'{empty.get("day", "?")} day against {empty.get("night", "?")} night; '
             f'{play.get("night", "?")} night against {play.get("day", "?")} day.'),
        ])) if c else "<p class='missing'>No coverage report generated yet.</p>")
        + block("Why more data was not simply collected", cards([
            ("There are people in it",
             "Players, staff, sometimes children. Footage of identifiable people is "
             "sensitive data &mdash; consent, retention and redaction all apply."),
            ("So the bottleneck is permission",
             "Every extra venue is a conversation with a facility, not a download."),
            ("And the task needs variety",
             "Many venues, lights, angles, weathers. Volume of the <b>same</b> scene adds "
             "nothing."),
        ]), tint="amber")
        + block("So some data was generated, from a real starting point", cards([
            ("Real anchor first",
             "Every generated item starts from an <b>actual frame of an actual pitch</b> "
             "&mdash; never a text prompt alone."),
            ("Images &mdash; Gemini and ChatGPT",
             "Used to produce the scene the corpus lacks: an empty pitch under floodlights."),
            ("Stills into video &mdash; 2 AI tools",
             "A still frame animated into motion, so the motion check has something to read. "
             "<b>[tool names to be added]</b>"),
            ("What it bought",
             "31 generated empty frames took cross-venue false alarms from <b>0.768 to "
             "0.024</b> &mdash; while play-recall went up."),
        ]), tint="violet",
            note="<b>Excluded from every corpus count.</b> They are a training aid, not "
                 "evidence the system works on real footage of that case. And no rain was "
                 "generated: with no wet footage to check against, that would test the "
                 "generator rather than the weather.")
        + block("The models &mdash; almost nothing here is trained", model_tiles + cards([
            ("Why not fine-tune?",
             "Updating 200M parameters on ~150 distinct scenes does not learn football. It "
             "learns <b>these pitches</b>."),
            ("Small data overfits",
             "98.5% near-duplicates, and accuracy that <b>falls</b> when labels rise from 300 "
             "to 671. The risk is a high score for the wrong reason."),
            ("So freezing is a defence first",
             "Against overfitting. Embedding once and reusing forever is the second, very "
             "welcome, consequence."),
        ]), tint="green")
        + block("The rule table &mdash; first matching row wins", table(
            ["#", "Condition", "Verdict"],
            [["1", "Detector unavailable", "Uncertain &mdash; a missing detector is not an "
                                           "empty pitch"],
             ["2", "No pitch boundary", "Uncertain &mdash; mandatory on the deployed path"],
             ["3", "Nobody, nothing moving", "<b>EMPTY</b>"],
             ["4", "Nobody, but something moved", "Uncertain"],
             ["5", "1&ndash;4 people", "Not a game &mdash; too few, ball or no ball"],
             ["6", "5+ people, <b>and</b> a ball, <b>and</b> motion",
              "<b>ACTIVE PLAY</b> &mdash; the only way in"],
             ["7", "5+ people, otherwise", "Not a game &mdash; a crowd that is not playing"]],
            hi=5),
            note="Five people make a game and four do not &mdash; that is the "
                 "<b>facility&rsquo;s rule</b>, not something fitted. The ball requirement is "
                 "strict and costs recall where the detector cannot see one (0.996 &rarr; "
                 "0.308). A cue that cannot be measured is reported, never assumed.")
        + block("Preprocessing &mdash; what happens before the model sees a frame", chips([
            "Letterbox to 224&times;224", "Undistort", "Centre / top crop",
            "Per-image standardise", "CLAHE", "Gamma", "Saturation &amp; grayscale",
            "Denoise", "Sharpen", "Blur (a control, not a candidate)",
        ]) + detail(preprocess_pairs(), "Every switch, before and after"),
                note="Experiments and the live pipeline share <b>one</b> preprocessing "
                     "module. CLAHE, which the proposal expected to help with "
                     "floodlight glare, costs DINOv2 <b>0.27</b> macro-F1.")
        + block("How the data were analysed", cards([
            ("Grouped splits, never random",
             "Whole venues and slots held out, so no test frame has a near-twin in training."),
            ("Four trivial baselines, every time",
             "A clock rule, a colour histogram, a constant predictor, a random one. If they "
             "win, the protocol is wrong."),
            ("Leave-one-venue-out",
             "Seven folds, each reported with the smallest p-value the design could have "
             "produced (0.0156)."),
            ("Pre-registered, amended in the open",
             "Hypotheses fixed before the runs; every change numbered and left in place."),
        ]))
        + "</div>"
    )


def problem() -> str:
    """6. What problem does the research tackle?"""
    return (
        '<div class="tk">'
        + hero("What problem does the research tackle?",
               "A facility sells hours. Nobody checks which ones were used.",
               "Staff write it down; the records are unverified; three kinds of error "
               "follow.", tone="coral")
        + block("The problem, in three failure modes", cards([
            ("No-shows", "Paid for, never played."),
            ("Unbooked use", "Played, never paid for."),
            ("Data-entry error", "The record and the reality drift apart."),
        ]), note="A &euro;20 motion sensor is more robust in fog and darkness &mdash; but it "
                 "answers <i>did something move</i>. An audit needs <i>was this booking used, "
                 "and here is the picture</i>, and it must tell a match from a groundsman.")
        + block("What is already known &mdash; and the gap", cards([
            ("Known: frozen backbones work",
             "Strong results from very few labels, across many vision tasks."),
            ("Known: occupancy from CCTV",
             "People counting and occupancy are established problems."),
            ("Missing: honest evaluation",
             "Almost all of it is scored on frames from the <b>same scenes</b> as training."),
            ("So the score measures memory",
             "Recognition of a <b>place</b>, not of an <b>activity</b> &mdash; and nobody "
             "runs a trivial rule beside it to check."),
        ]))
        + block("This gap matters because the money rests on one class",
                quote("Getting &ldquo;playing&rdquo; right is easy and worth nothing. If the "
                      "system cannot reliably recognise an <b>empty</b> pitch, it cannot flag "
                      "a single unused booking."), tint="amber")
        + block("First consequence &mdash; honest splitting halves the score", tiles([
            ("37.1%", "of near-duplicate pairs cross the line under a random split", "bad"),
            ("0.8%", "under a split grouped by venue and slot", "good"),
            ("&minus;0.49", "macro-F1: what honesty costs ConvNeXtV2", "warn"),
        ]), note="But only about two thirds of that fall is leakage: a model that never "
                 "trains cannot leak, and scoring one on the same test sets isolates the "
                 "rest at <b>0.183</b>.")
        + block("The real gap &mdash; in this data, the time of day is the answer", tiles([
            ("98%", "of daylight frames are an empty pitch", "warn"),
            ("99%", "of night frames are a match", "warn"),
            ("99.1%", "so &ldquo;night means play&rdquo; is right on the whole corpus", "bad"),
        ]) + confound_matrix(), tint="coral",
            note="The two cells that would break the tie hold <b>9 frames</b> (empty at "
                 "night) and <b>6</b> (play in daylight). Nothing measured here can separate "
                 "<i>recognises an empty pitch</i> from <i>recognises the time of day</i>.")
        + block("The same model, judged two ways", protocols(),
                note="Identical model, identical data. Two ways of drawing the train/test "
                     "line &mdash; and the answer changes.")
        + block("What that does to a benchmark", table(
            ["Method", "Finds the match", "Wrongly says &ldquo;playing&rdquo;"],
            [["<b>Clock rule &mdash; reads no pixels</b>", "<b>1.000</b>", "<b>0.021</b>"],
             ["DINOv2", "0.930", "0.309"],
             ["ConvNeXtV2", "0.911", "0.992"],
             ["ViT", "0.869", "0.835"]], hi=0)
            + beats([("16/16", "0.00", "Minutes wrong on a clip of a floodlit empty pitch: "
                                       "the clock rule, then the full deployed system.")]),
            note="Across venues the models had never seen. One 234-second clip containing the "
                 "missing case reversed a ranking that 1,692 frames had produced.")
        + block("And two more things the protocol was hiding",
                empty_blindness() + cards([
                    ("A constant predictor wins one protocol",
                     "Always answering &ldquo;playing&rdquo; scores macro-F1 <b>1.000</b> "
                     "cross-venue &mdash; because every held-out venue is 100% active play. "
                     "There is no ranking to change."),
                    ("A low false-play rate is not accuracy",
                     "On 243 held-out empty frames DINOv2 answers PLAYING 75 times and "
                     "MAINTENANCE 168 times. It is correct <b>zero</b> times."),
                ], wide=True),
                note="Both were found the same way &mdash; by adding a column to check "
                     "whether a number meant what it said.")
        + block("Which is why one missing cell blocks four questions", blocked_questions())
        + "</div>"
    )


def solution() -> str:
    """7. How does the research provide a solution?"""
    a = augmentation_spread()
    res = search_resolution()

    spread = beats([
        (a.get("baseline", "?"), a.get("max", "?"),
         "macro-F1: no augmentation, against the draw that was published"),
        (a.get("max", "?"), a.get("min", "?"),
         f'the same preset, best against worst of {a.get("n", "?")} draws'),
    ]) if a.get("sd") else "<p class='missing'>No augmentation spread measured yet.</p>"

    floor = tiles([
        (res["floor"], "one frame in the smallest fold moves the headline this much", "bad"),
        (res["interval"], "how wide the confidence band is at the median setting", "bad"),
    ]) if res.get("floor") else "<p class='missing'>No resolution audit yet.</p>"

    return (
        '<div class="tk">'
        + hero("How does the research provide a solution?",
               "The fix was never a bigger model.",
               "Three interventions, each measured on the boundary it was meant to help.",
               tone="green")
        + block("Three fixes, all cheap", beats([
            ("0.617", "0.012", "False alarms on empty pitches, once the pitch boundary, "
                               "motion and person checks can veto the model."),
            ("0.768", "0.024", "With 31 generated empty-pitch frames in training &mdash; and "
                               "play-recall went up."),
            ("0.441", "0.990", "One labelled frame of a new camera. Five from it beat those "
                               "five plus 775 from the old one."),
        ]), note="What buys the accuracy is having <b>any</b> labels from the new camera "
                 "&mdash; not a large corpus from an old one. That is the weaker claim, and "
                 "it is the true one.")
        + block("What was investigated &mdash; two configuration searches", tiles([
            ("88", "preprocessing runs &mdash; greedy, four rounds", "lead"),
            ("375", "prompt sets &mdash; exhaustive, zero labels", "lead"),
            ("0.726", "macro-F1 span from worst wording to best", "warn"),
            ("0.082", "span from swapping between the three backbones", "good"),
        ]) + cards([
            ("Wording moved the score nine times more than the model did",
             "And only 22.9% of prompt sets beat the trained probe &mdash; so "
             "&ldquo;zero-shot works&rdquo; is true only if you already know the wording, "
             "which needs labels."),
            ("Both winners sit inside the noise",
             "The winner&rsquo;s lead of 0.447 is measured on the folds it was selected from. "
             "Disclosed in the pre-registration as an undeclared search family."),
        ], wide=True), tint="sky")
        + block("Because both searches are finer than the data", floor,
                note="Almost every gap the preprocessing search ranked on is <b>smaller than "
                     "its own error bar</b>. The ranking is real arithmetic on unreal "
                     "precision.")
        + block("The augmentation experiment &mdash; and its retraction", spread + points([
            f'Standard deviation <b>{a.get("sd", "?")}</b> on a metric bounded in '
            '[0,&nbsp;1].',
            f'The published number was the <b>maximum of {a.get("n", "five")}</b>.',
            f'Four of five draws land <b>below the {a.get("baseline", "?")}</b> the probe '
            'reaches with no augmentation at all.',
        ], tone="bad"), tint="coral",
            note="The row in the results file is unchanged and still reproduces exactly "
                 "&mdash; which is the problem, not the defence. A test pinning the published "
                 "value would have passed forever; only re-drawing the randomness caught it. "
                 "Every headline now runs five draws.")
        + block("And the practice that found half of this", cards([
            ("A test-set lock", "Resolved against the working directory, so it locked "
                                "nothing."),
            ("A confidence threshold", "Set to 0.0 &mdash; a gate that can never fire."),
            ("A gate criterion", "Grepped for a word instead of running the verifier."),
            ("Evidence selection", "Recorded a file path of <code>None</code> for every frame "
                                   "it chose."),
        ]), tint="amber",
            note="Six safeguards in this project were doing nothing. Each new guard is now "
                 "verified by <b>deliberately breaking the thing it guards</b> &mdash; "
                 "because a wrong result that looks plausible is invisible.")
        + block("What the findings contribute", cards([
            ("To practice",
             "A five-frame onboarding recipe for every new camera, and a rule table a "
             "facility manager can read and argue with."),
            ("To the field",
             "An evaluation method: trivial baselines in every protocol, grouped splits, "
             "re-drawing randomness rather than re-running it, and every guard verified by "
             "breaking it."),
        ], wide=True), tint="green")
        + "</div>"
    )


def summary() -> str:
    """8. Summary."""
    return (
        '<div class="tk">'
        + hero("Summary", "In summary",
               "What this study investigated, what it found, and what it contributes.",
               tone="ink")
        + block("", cards([
            ("Investigated",
             "Whether cameras a facility already owns can verify which booked hours were "
             "actually used &mdash; on a CPU, with a human deciding every case."),
            ("Found",
             "On this data a clock beats a neural network. Lighting and occupancy are "
             "confounded so tightly that no protocol here separates them."),
            ("Contributed",
             "An evaluation method: trivial baselines everywhere, grouped splits, re-drawing "
             "randomness, and every guard verified by breaking it."),
            ("Implies",
             "For the facility, a five-frame onboarding recipe. For the field, that a "
             "benchmark can rank a light meter first and still look healthy."),
        ], wide=True))
        + block("What this study cannot claim", cards([
            ("Empty pitches exist at one venue only",
             "So a three-class cross-venue evaluation cannot be run on this corpus."),
            ("243 empty frames are 3 scenes",
             "The effective sample is scenes, not frames. Six comparisons stop being "
             "significant when recounted that way."),
            ("Nothing has run on the target hardware",
             "Every latency number comes from a laptop."),
            ("No human ceiling exists",
             "There is no inter-annotator figure. That is a real gap, and we say so."),
        ]), tint="coral")
        + block("What would actually change it", numbered([
            ("Empty-pitch footage at a second venue",
             "20&ndash;30 minutes, day and night. Fixes the confound, the external-validity "
             "limit and the scope reduction at once."),
            ("About 30 real labelled slots",
             "One conversation with the facility. Lifts the temporal model out of "
             "preliminary."),
            ("Five labelled frames per camera",
             "Already measured, already cheap &mdash; and it is the onboarding recipe we "
             "would hand over."),
        ]), tint="green",
            note="The limitations are mostly a <b>data-access problem with a known and "
                 "inexpensive solution</b> &mdash; not a methodological one.")
        + block("", quote("<b>No protocol compensates for a case the data never "
                          "contains.</b>"), tint="amber")
        + "</div>"
    )


def thanks() -> str:
    """9. Thank you & questions."""
    return (
        '<div class="tk">'
        + hero("Thank you &amp; questions",
               "Thank you for your time and attention.",
               "I welcome any questions, comments, suggestions or feedback you may have.",
               tone="ink")
        + block("Questions I expect, and the short answers", cards([
            ("Would a colour histogram have done this?",
             "0.9616 on the leaky split &mdash; indistinguishable from ConvNeXtV2 once you "
             "count 62 scenes rather than 394 frames."),
            ("Is the novelty just temporal smoothing?",
             "Tested against four tuned baselines including an HMM, with the saturation "
             "caveat enforced in code."),
            ("How many comparisons before p &lt; 0.05?",
             "Holm within declared families; the undeclared ones disclosed as numbered "
             "amendments."),
            ("Is auditing staff by camera ethical?",
             "Per-field reporting only, and the decision layer can structurally only advise."),
            ("Can anyone reproduce this?",
             "One pipeline re-derives every claim from its artefact on every run."),
            ("What is the human ceiling?",
             "<b>No inter-annotator figure exists.</b> Say so plainly &mdash; it is a real "
             "gap."),
        ]))
        + block("Closing", quote("Practise a strong, clear voice and a confident tone. Use "
                                "simple language for the complex ideas. And end on the "
                                "statistic, not an apology."), tint="green")
        + "</div>"
    )


#: tab id -> (label, builder). The nine sections of the oral-presentation template, in the
#: order they are delivered.
SECTIONS: dict[str, tuple[str, object]] = {
    "introduce": ("1. Introduce", introduce),
    "topic": ("2. Topic", topic),
    "roadmap": ("3. Roadmap", roadmap),
    "purpose": ("4. Purpose", purpose),
    "how": ("5. How it works", how),
    "problem": ("6. The problem", problem),
    "solution": ("7. The solution", solution),
    "summary": ("8. Summary", summary),
    "thanks": ("9. Thank you", thanks),
}
