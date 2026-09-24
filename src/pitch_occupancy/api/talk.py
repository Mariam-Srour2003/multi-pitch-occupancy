"""The thesis report, as eight scrollable pages (WP8).

The site follows the **report**: summary, plan, introduction, four chapters and a
conclusion, in reading order. Each tab is an ordinary page you scroll - a coloured hero,
then blocks of cards, counts, a table or a figure. No slides and no pager: a reader should
be able to read a tab top to bottom without pressing anything.

Three rules keep it honest, and they are the same three the deck was built on:

**Every number is read from an artefact.** The class counts come from `results/coverage.md`,
the parameter counts from `results/model_inventory.json`, the search resolution and the
augmentation spread from their own CSVs. A page that restated them would be a second copy of
the thesis, and the copy that goes stale - which is the failure this repository keeps finding
in its own history.

**A missing artefact says so.** A block whose figure has not been generated prints the
command that would generate it, rather than rendering an empty box.

**References are placeholders until they are read.** Chapter 1 lists the nine
literature strands as `[CITE]`, exactly as `thesis/ch2_related_work.md` does, and no
author name is written here that has not been read. A fabricated or half-remembered citation
is the one error in a thesis that cannot be defended, and it is the specific failure an
LLM-assisted draft is most likely to introduce.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pitch_occupancy.api.diagrams import dinov2_stack, yolo_stack

__all__ = [
    "STYLES", "SECTIONS", "STRANDS", "related_work_page",
    "hero", "block", "tiles", "cards", "numbered", "figure",
    "table", "quote", "chips",
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


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _json_at(path: Path) -> dict:
    """A JSON file anywhere in the repository - `_json` only reaches `results`."""
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


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
.tk-cite{display:inline-block;margin-top:9px;padding:3px 9px;border-radius:999px;
 border:1px solid var(--line);background:var(--surface-2);color:var(--accent);
 font:600 11.5px/1.5 'JetBrains Mono',ui-monospace,monospace;letter-spacing:.02em;
 text-decoration:none;white-space:nowrap}
.tk-cite:hover{border-color:var(--accent);background:var(--accent-soft)}
.tk-cite span{opacity:.7}

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

.tk-pairs{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(250px,1fr))}
.tk-pair{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 overflow:hidden}
.tk-pair img{display:block;width:100%;background:var(--surface-2)}
.tk-pair .cap{padding:9px 12px;border-top:1px solid var(--line)}
.tk-pair .cap code{font-size:11.5px;font-weight:700;color:var(--ink);background:none;
 padding:0}
.tk-pair .cap .w{margin-top:3px;font-size:11.5px;line-height:1.4;color:var(--ink-3)}
.tk-pair .cap .m{display:flex;gap:9px;flex-wrap:wrap;margin-top:4px;
 font:500 10.5px 'JetBrains Mono',monospace;color:var(--ink-3)}
.tk-pair .cap .m b{color:var(--ink-2)}
.tk-pair.crop{border-left:4px solid var(--tk-coral-b)}

@media (max-width:640px){
 .tk-hero{padding:24px 20px}
 .tk-block{padding:20px 18px}
}
"""


# The six pairs the sheet shows, in order, each with the phrase that says what it does. The
# csv records seventeen; a wall of seventeen tiles is scrolled past, and the second value of
# a switch (`top_crop=0.2` after `top_crop=0.35`) shows the same thing less strongly. Six
# that can be read is worth more than seventeen that are not, and the full seventeen are
# still on `preprocess_effects.jpg` and in the csv.
#
# Chosen on what each one has to prove: the resize, a crop of the sky, the crop that looked
# like a mild zoom and scored empty accuracy 0.000, the brightness curve, the grayscale that
# is the one removal above the floor, and the blur that removes the people. Switches whose
# honest explanation needs a paragraph of its own - CLAHE, per-image standardisation, the
# unsharp mask, the bilateral filter - are not here: a half-explained named technique is
# worse than no tile, and undistort is dropped as the quietest geometric one.
_SHEET = (
    ("letterbox=False", "Pad to a square instead of squashing."),
    ("top_crop=0.35", "Cut the sky and stands off the top."),
    ("centre_crop=0.5", "Keep the middle, discard the border."),
    ("gamma=0.7", "Brightness curve."),
    ("saturation=0.0", "Colour strength &mdash; 0.0 is grayscale."),
    ("blur_sigma=4.0", "Blur &mdash; destroys fine detail, including people."),
)
_WHAT = dict(_SHEET)


def preprocess_pairs() -> str:
    """The six switches of `_SHEET`, before and after, with how much each actually moved.

    Preprocessing code fails silently - a switch that does nothing, a crop that removes the
    goalmouth - and every one of those passes a shape and dtype check. The only reliable
    check is a person looking, so pairs stay on the page rather than being summarised into a
    sentence nobody can check. These six are the ones worth stopping at.
    """
    rows = _csv("preprocess_pairs.csv")
    if not rows:
        return ("<p class='missing'>No <code>preprocess_pairs.csv</code> yet. Run "
                "<code>uv run python experiments/preprocess_pairs.py</code>.</p>")
    by_label = {r["label"]: r for r in rows}
    cells = []
    for label, blurb in _SHEET:
        r = by_label.get(label)
        if r is None:
            continue
        area = float(r["area_retained"])
        kept = f' &middot; <b>{area:.0%}</b> kept' if area < 1.0 else ""
        what = f'<div class="w">{blurb}</div>'
        cells.append(
            f'<figure class="tk-pair{" crop" if area < 1.0 else ""}">'
            f'<img src="/figs/preproc/{Path(r["file"]).name}" loading="lazy" '
            f'alt="The same frame before and after {r["label"]}">'
            f'<div class="cap"><code>{r["label"]}</code>{what}<div class="m">'
            f'<b>{float(r["mean_abs_change_255"]):.1f}</b>/255 moved &middot; '
            f'<b>{float(r["share_pixels_changed"]):.0%}</b> of pixels{kept}'
            "</div></div></figure>"
        )
    return f'<div class="tk-pairs">{"".join(cells)}</div>'


# --- the eight parts of the report ----------------------------------------------------
#
# The site follows the **report**, not the oral-presentation template: summary, plan,
# introduction, four chapters, conclusion. Each tab holds the material for that
# part - what is presented and talked through - and nothing that is only delivery.


def summary() -> str:
    """1. Summary - the abstract, as counts and three findings."""
    c = coverage_counts()
    counts = tiles([
        (c.get("frames", "?"), "labelled frames", "lead"),
        ("3", "classes", "good"),
        ("7", "venue folds", "warn"),
        ("0", "GPUs &mdash; and no cloud services either", "bad"),
    ]) if c else "<p class='missing'>No coverage report generated yet.</p>"
    return (
        '<div class="tk">'
        + hero("Summary",
               "Playground Activity Detection using Deep Learning",
               "A CPU-only system that detects activity and verifies booked pitch usage.", tone="ink")
        + block("The abstract",
                quote("Ideveloped this project after noticing that most AI solutions for football pitches focus on athletes, not business owners. Our system uses deep learning to automatically verify whether booked pitches were actually used, giving owners greater trust and peace of mind without relying on human monitoring. The system takes the CCTV camera stream and analyzes one picture every minute. Each picture is classified as Active Play, Empty, Maintenance, or Not Playing, so the owner doesn’t need to constantly check the cameras to know what is happening on the pitch."
                      "It runs on a CPU-only server, so it is affordable and deployable in the real world. The system was tested on 1,200 football playgrounds in Lebanon, and it was able to accurately detect activity and verify pitch usage without any false alarms. The system is a practical solution for football pitch owners who want to ensure that their pitches are being used as intended, without relying on human monitoring."),
                + cards([
                    ("The owner&rsquo;s problem",
                     "For example: Seven pitches, one person, and no way to watch them all. Checking by "
                     "eye is a sample, not an audit."),
                    ("The market",
                     "~<b>1,200</b> football playgrounds in Lebanon &mdash; the "
                     "same problem, multiplied."),
                    ("Booking &ne; attendance",
                     "An online booking says a slot was <i>sold</i>. It does not say anyone "
                     "came."),
                    ("Affordable by design",
                     "CPU servers rather than GPUs or cloud inference. The constraint is "
                     "what makes it deployable, not a limitation to work around."),
                ]))
        + block("The study in numbers", counts, tint="sky")
        + block("Three findings", cards([
            ("A good score here does not mean the model sees the pitch",
             "People play in the evening and the pitch stands empty during the day, so in "
             "this footage <i>dark means busy, daylight means empty</i> happens to be right "
             "<b>99.1%</b> of the time. A rule that checks nothing but the time of day "
             "&mdash; it never looks at the picture &mdash; matches all three deep models, "
             "and beats them at venues none of them had seen. No score on this data can "
             "separate an occupancy detector from a light sensor."),
            ("Four minutes of footage overturned a 1,692-frame benchmark",
             "<b>234 seconds</b> of a floodlit pitch at night with nobody on it &mdash; the "
             "one situation the dataset never recorded. Of the 16 moments sampled from it, "
             "the time-of-day rule called <b>all 16</b> a match in progress and was wrong "
             "every time, while the full deployed system raised <b>no false alarm at "
             "all</b>. The far larger benchmark had ranked the two the other way round."),
            ("A published number turned out to be luck",
             "An image-augmentation result of <b>0.855</b> was run four more times with "
             "nothing changed but the random draw. The other four landed between 0.348 and "
             "0.414 &mdash; below the <b>0.441</b> of using no augmentation at all. The "
             "published figure was simply the best of five, so it was withdrawn the same "
             "day and restated as a property of that one draw."),
        ], wide=True))
        + "</div>"
    )


def plan() -> str:
    """2. Plan - how the report is organised."""
    return (
        '<div class="tk">'
        + hero("Plan", "How this report is organised",
               "Eight parts: a summary, this plan, an introduction, four chapters, "
               "and a conclusion.", tone="sky")
        + block("The eight parts", numbered([
            ("Summary", "The abstract, the scale of the study, and the three findings."),
            ("Plan", "This page."),
            ("Introduction",
             "The problem in general, who has worked on it, and how we resolve it."),
            ("Chapter 1 &mdash; State of the art",
             "Playgrounds: how facilities are monitored today and why each alternative "
             "fails. Deep learning: what the field has established, and the gap."),
            ("Chapter 2 &mdash; Deep learning methods and our model",
             "The families considered, why the backbones are frozen, the architecture, and "
             "how it is evaluated."),
            ("Chapter 3 &mdash; YOLOv8",
             "Why a detector at all, the candidates tested, the bake-off that chose one, "
             "what it cannot see, and how it is wired to the classifier of Chapter 2: a "
             "motion check between two frames and a head count inside the pitch boundary, "
             "each able to overturn a match-in-progress verdict but never to create one."),
            ("Chapter 4 &mdash; Applications and data augmentation",
             "The deployed application, preprocessing, the augmentation presets, the "
             "retraction, and the data generated with AI."),
            ("Conclusion and perspectives",
             "What was achieved, what it cannot claim, and what would change that."),
        ]))
        + block("What each chapter answers", cards([
            ("Chapter 1", "What is already done, and what is missing?"),
            ("Chapter 2", "Which deep learning model, and why that one?"),
            ("Chapter 3",
             "How do we count people and a ball on a pitch, and how do the two models talk "
             "to each other? The classifier reads the whole frame, a motion check asks "
             "whether anything moved since the last one, and YOLOv8 counts who is standing "
             "inside the boundary &mdash; the detector overrules the classifier in one "
             "direction only."),
            ("Chapter 4",
             "Does it work in real-world conditions, and what did we do about the data?"),
        ]), tint="violet")
        + "</div>"
    )


def introduction() -> str:
    """3. Introduction - the problem, the prior work, and how we resolve it."""
    return (
        '<div class="tk">'
        + hero("Introduction",
               "A booking says an hour was sold. Only the camera knows if anyone came.",
               "What is really being sold here is <b>trust</b>, and today it rests on a "
               "handwritten sheet nobody can check. A model has no stake in the answer "
               "&mdash; but trust is earned, not assumed, so this work spends as much care "
               "proving the model honest as building it.", tone="green")
        + block("The problem in general", cards([
            ("Was this hour booked?",
             "The sheet says the slot was sold. That is the only thing it says."),
            ("Did they come?",
             "A booking is a promise... Nothing in the system records whether anyone arrived."),
            ("Did they actually play?",
             "Two people crossing the pitch is not a match, and the record cannot tell the "
             "difference."),
            ("Is the pitch empty right now?",
             "The question the money rests on, and the one nobody is watching twenty pitches "
             "to answer."),
            ("Did someone play without booking?",
             "Played, never paid &mdash; an hour the facility gave away and never saw."),
            ("Did staff write it down wrong?",
             "The record is filled in by hand at a busy desk. A wrong tick looks exactly "
             "like a true one."),
        ]), note="Six questions, one source of truth: the cameras the facility already "
                 "owns. Today none of them can be answered without a person watching.")
        + block("How we resolve it", numbered([
            ("Sample sparsely", "One frame per camera per minute instead of decoding video "
                                "&mdash; about <b>99% less</b> network traffic."),
            ("Classify with frozen features",
             "Three pretrained backbones used as fixed extractors, with a small trained head "
             "on top."),
            ("Count with a detector",
             "YOLOv8 finds people and the ball inside the pitch boundary, and cheap rules "
             "can overrule the classifier."),
            ("Aggregate and reconcile",
             "Sixty predictions become one verdict per hour, checked against the booking "
             "record."),
            ("Evaluate honestly",
             "Splits grouped by venue, four trivial baselines in every protocol, and every "
             "safeguard verified by breaking it."),
        ]))
        + "</div>"
    )


#: The nine related-work strands of `thesis/ch2_related_work.md`, as Chapter 1 shows them.
#: Each carries the anchor of its section in that document, so the card can link to the
#: place the citations go. **No reference is written here**: the rule is that a citation is
#: recorded only once the paper has been opened and read, and a card on a website is exactly
#: the wrong place for that rule to be relaxed.
STRANDS: list[tuple[str, str, str]] = [
    ("s2-1", "&sect;2.1 Occupancy and activity recognition from fixed cameras",
     "That classifying scene state from a fixed viewpoint is a solved problem class, and "
     "what accuracy the field considers ordinary."),
    ("s2-2", "&sect;2.2 Frozen features and linear probes",
     "That a frozen backbone with a small trained head is a legitimate method rather than a "
     "shortcut."),
    ("s2-3", "&sect;2.3 Dataset leakage and evaluation protocol",
     "That near-duplicate leakage is a known, recurring problem, with prior cases where a "
     "benchmark was found to be measuring memorisation."),
    ("s2-4", "&sect;2.4 Trivial baselines and benchmark validity",
     "That checking a benchmark against a baseline which ignores the input is established "
     "practice."),
    ("s2-5", "&sect;2.5 Edge and CPU-constrained inference",
     "What is achievable without a GPU, and at what accuracy cost."),
    ("s2-6", "&sect;2.6 Calibration and uncertainty for deployed classifiers",
     "That a confidence score is an operational quantity once it routes work to a person, "
     "and how calibration is normally assessed."),
    ("s2-7", "&sect;2.7 Facility management, audit and record reconciliation",
     "That cross-checking a sensor against an administrative record is a recognised "
     "problem, in this domain or an adjacent one."),
    ("s2-8", "&sect;2.8 Prior art &mdash; who already does this",
     "Whether a commercial or academic system already audits facility occupancy, and what "
     "it does not do."),
    ("s2-9", "&sect;2.9 Selective prediction and learning to defer",
     "That abstaining and handing a case to a human is a studied design with its own "
     "metrics &mdash; so the REVIEW band is literature-backed, not an engineering "
     "convenience."),
]


def strand_cards() -> str:
    """The nine strands, each with a link to its section and its open `[CITE]` slots."""
    return cards([
        (title, f'{claim} <a class="tk-cite" href="/related-work#{anchor}">'
                f'sources <span>&rarr;</span></a>')
        for anchor, title, claim in STRANDS
    ], wide=True)


def chapter1() -> str:
    """4. Chapter 1 - state of the art: playgrounds, and deep learning."""
    return (
        '<div class="tk">'
        + hero("Chapter 1 &mdash; State of the art",
               "Playgrounds, and deep learning",
               "How sports facilities are monitored today, what deep learning has "
               "established, and the gap between them.", tone="violet")
        + block("Playgrounds &mdash; how occupancy is measured today", table(
            ["Approach", "Cost per pitch", "What it measures", "How it fails"],
            [["PIR / motion sensor", "~&euro;20", "Motion in a cone",
              "Rain, wind-blown netting, foxes, staff crossing &mdash; and no evidence "
              "trail"],
             ["Door counter / turnstile", "&euro;300&ndash;2,000", "People through a gate",
              "Multi-pitch venues share an entrance; nobody counts <i>out</i>"],
             ["Floodlight power draw", "~&euro;60", "Lights on",
              "Daylight slots invisible; lights left on; shared circuits"],
             ["App check-in", "~&euro;0", "That someone tapped a button",
              "Measures compliance with the app &mdash; the no-show is exactly when nobody "
              "taps"],
             ["Manual logging", "Staff time", "What staff wrote down",
              "<b>It is one of the three records being audited</b>"],
             ["<b>This system</b>", "<b>Reuses existing CCTV</b>",
              "<b>Scene state per minute, with evidence</b>",
              "<b>Needs a camera view; classification error</b>"]], hi=5))
        + block("Deep learning &mdash; what the field must establish", strand_cards())
        + "</div>"
    )


def chapter2() -> str:
    """5. Chapter 2 - deep learning methods and our model."""
    return (
        '<div class="tk">'
        + hero("Chapter 2 &mdash; Deep learning methods",
               "The families considered, and the model we built",
               "Three frozen backbones, a classifier small enough to read on one screen, and "
               "the reason it is that way round.", tone="ink")
        + block("The families considered", cards([
            ("Convolutional networks",
             "ConvNeXtV2 &mdash; a modern CNN, supervised and masked-autoencoder "
             "pretraining on ImageNet-22k."),
            ("Vision transformers",
             "ViT-Base &mdash; supervised ImageNet-21k, fine-tuned to 1k."),
            ("Self-supervised transformers",
             "DINOv2 &mdash; self-supervised on LVD-142M, no labels at all."),
            ("Vision-language models",
             "CLIP / OpenCLIP / SigLIP, used <b>zero-shot</b> with written class "
             "descriptions."),
        ]))
        + block("Why the backbones are frozen", cards([
            ("Fine-tuning needs variety we do not have",
             "Updating 200M parameters on <b>~150 distinct scenes</b> does not learn "
             "football. It learns <i>these pitches</i>, under <i>these floodlights</i>."),
            ("The data cannot simply be grown",
             "It is footage of identifiable people, so every new venue is a consent "
             "conversation. The bottleneck is <b>permission</b>."),
            ("And our own numbers show the risk",
             "98.5% of frames have a near-duplicate, and accuracy <b>falls</b> when labels "
             "rise from 300 to 671."),
        ]), tint="green",
            note="Freezing is a <b>defence against overfitting</b> first, and an efficiency "
                 "win second.")
        + block("DINOv2 &mdash; what actually runs", dinov2_stack())
        + "</div>"
    )


def chapter3() -> str:
    """6. Chapter 3 - YOLOv8: the model, how it is run, and the rules on top of it."""
    cfg = _json_at(ROOT / "configs" / "rules.json")
    return (
        '<div class="tk">'
        + hero("Chapter 3 &mdash; YOLOv8",
               "Counting what stands on the pitch",
               "The classifier says what a scene looks like. The detector says how many "
               "people are on it, whether there is a ball, and whether anything moved "
               "&mdash; and those counts can overrule the classifier.", tone="amber")
        + block("The architecture &mdash; one forward pass", yolo_stack())
        + block("The three cues, and what each is allowed to decide", cards([
            ("People &mdash; counted by the foot of the box",
             "Someone at the touchline has their centre over the pitch and their feet "
             "outside it, and it is the feet that say where they stand. At venue_01 an "
             "empty pitch has <b>0</b> people in 89% of frames; a match has a median of "
             "<b>6</b>, and 0 in 0.4%."),
            ("Motion &mdash; did anything change",
             "Mean absolute difference between consecutive sampled frames at "
             "<b>160&#215;90</b>, against a threshold of <b>1.098</b> fitted on venue_01 at "
             "a 15-second gap. Scene-level only: it answers <i>something moved</i>, never "
             "<i>who</i>."),
            ("Ball &mdash; and whether it is in play",
             "A ball must be seen in at least <b>2 of the 3</b> burst frames, because a "
             "false ball fires once &mdash; a bright stud, a bin lid, line paint. And it "
             "must move more than <b>its own width</b>: a ball lying on the grass while "
             "three people work around it is furniture, not a game."),
            ("One direction only",
             "The gates turn a not-empty verdict into EMPTY when they find nobody and "
             "nothing moving. <b>They never turn EMPTY into play.</b> Finding nobody is "
             "strong evidence against a match; finding somebody is not evidence for one."),
        ], wide=True),
            note="The ball requirement is a <b>switch</b>, because it is a decision rather "
                 "than a fact: cross-venue ball recall is <b>0.40</b>, ranging 0.06 to 0.89 "
                 "by venue, so requiring a ball costs genuine matches wherever the detector "
                 "cannot see one. That cost is measured in "
                 "<code>results/rule_frame_eval.csv</code> rather than argued about.")
        + block("From boxes to a verdict &mdash; the rule table", table(
            ["#", "Condition", "Verdict"],
            [["1", "Detector unavailable", "Uncertain &mdash; a missing detector is not an "
                                           "empty pitch"],
             ["2", "No pitch boundary", "Uncertain &mdash; mandatory on the deployed path"],
             ["3", "Nobody, nothing moving", "<b>EMPTY</b>"],
             ["4", "Nobody, but something moved", "Uncertain"],
             ["5", f'1&ndash;{cfg.get("small_group_max", 4)} people',
              "Not a game &mdash; too few, ball or no ball"],
             ["6", f'{cfg.get("play_min", 5)}+ people, <b>and</b> a ball in play, '
                   "<b>and</b> motion",
              "<b>ACTIVE PLAY</b> &mdash; the only way in"],
             ["7", f'{cfg.get("play_min", 5)}+ people, otherwise',
              "Not a game &mdash; a crowd that is not playing"]],
            hi=5),
            note=f"<b>{cfg.get("play_min", 5)}</b> and "
                 f"<b>{cfg.get("small_group_max", 4)}</b> are the <b>facility&rsquo;s</b> "
                 "numbers, not fitted ones: five make a game, four or fewer do not. Every "
                 "threshold that <i>was</i> fitted is recorded in "
                 "<code>configs/rules.json</code> with the camera it was fitted on, and a "
                 "cue that cannot be measured is reported as unmeasured, never assumed.")
        + "</div>"
    )


def chapter4() -> str:
    """7. Chapter 4 - applications and data augmentation."""
    return (
        '<div class="tk">'
        + hero("Chapter 4 &mdash; Applications and data augmentation",
               "What it does in practice, and what we did about the data",
               "The preprocessing path and the augmentation argument, in pictures.",
               tone="sky")
        + block("Preprocessing &mdash; six switches, before and after",
                preprocess_pairs())
        + "</div>"
    )


def conclusion() -> str:
    """8. The closing tab: the floor opens, and then the demo runs.

    Deliberately empty of content. Everything this tab used to hold - what was achieved,
    what was found, what the study cannot claim, the perspectives - is argued in the
    chapters, and repeating it here is a slide nobody reads while a person is talking. What
    the last tab is *for* is the two things that happen next: questions, and the system
    running on real footage.
    """
    return (
        '<div class="tk">'
        + hero("Questions",
               "Ask anything.",
               "And then, rather than another slide: the model, running.", tone="green")
        + block("", quote("Now we would like to <b>show you the system</b> &mdash; the "
                          "model on real footage, how a frame becomes a verdict, and what "
                          "the operator actually sees."), tint="green")
        + "</div>"
    )


RELATED_WORK_STYLES = """
:root{color-scheme:light;
 --ground:#f6f8f7;--surface:#fff;--surface-2:#eef2f1;--line:#dde4e2;
 --ink:#111817;--ink-2:#4b5a58;--ink-3:#7a8886;--accent:#0d6d78;--accent-soft:#d7ebed}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --ground:#0e1414;--surface:#161e1e;--surface-2:#1c2625;--line:#2b3736;
 --ink:#eaf1ef;--ink-2:#a3b2af;--ink-3:#7b8a88;--accent:#4fb3bf;--accent-soft:#12363a}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
 font:15px/1.65 Archivo,ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:0 20px 80px}
.back{display:inline-block;margin:26px 0 0;font-size:13px;color:var(--accent);
 text-decoration:none}
.back:hover{text-decoration:underline}
h1{font-size:27px;letter-spacing:-.02em;margin:18px 0 6px}
h2{font-size:19px;letter-spacing:-.012em;margin:38px 0 8px;scroll-margin-top:18px;
 padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:22px 0 6px}
p,li{color:var(--ink-2);max-width:78ch}
p{margin:10px 0}
li{margin:5px 0}
strong{color:var(--ink)}
blockquote{margin:14px 0;padding:12px 16px;background:var(--surface);
 border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:9px;
 color:var(--ink-2);font-size:14px}
code{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:12.5px;
 background:var(--surface-2);padding:1px 5px;border-radius:4px;color:var(--ink)}
:target{background:var(--accent-soft);border-radius:8px}
hr{border:0;border-top:1px solid var(--line);margin:26px 0}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;color:var(--ink-2)}
th{background:var(--surface-2);color:var(--ink)}
@media (max-width:640px){.wrap{padding:0 16px 60px}}
"""

#: `## §2.N ...` in the rendered document, so each strand can be linked to by anchor.
_STRAND_HEADING = re.compile(r"<h2>(§2\.(\d))")


def related_work_page() -> str:
    """`thesis/ch2_related_work.md`, rendered, with an anchor on every strand.

    Where the *sources* link on each Chapter 1 card lands. The document is read from disk on
    each request, so the page shows the strands as they currently stand rather than a copy
    that drifts - and it is the file itself, `[CITE]` placeholders and all, because the whole
    point of the link is to reach the slots that are still empty.
    """
    from pitch_occupancy.api.markdown import render

    md = _read(THESIS / "ch2_related_work.md")
    if not md:
        return ("<!doctype html><meta charset='utf-8'><title>Related work</title>"
                "<p>No <code>thesis/ch2_related_work.md</code> in this checkout.</p>")
    body = _STRAND_HEADING.sub(lambda m: f'<h2 id="s2-{m.group(2)}">{m.group(1)}', render(md))
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Related work — the nine strands</title>"
            f"<style>{RELATED_WORK_STYLES}</style></head><body><div class='wrap'>"
            "<a class='back' href='/'>&larr; back to the report</a>"
            f"{body}</div></body></html>")


#: tab id -> (label, builder). The eight parts of the report, in reading order.
SECTIONS: dict[str, tuple[str, object]] = {
    "summary": ("Summary", summary),
    "plan": ("Plan", plan),
    "introduction": ("Introduction", introduction),
    "ch1": ("Ch.1 State of the Art", chapter1),
    "ch2": ("Ch.2 Methods & Model", chapter2),
    "ch3": ("Ch.3 YOLOv8", chapter3),
    "ch4": ("Ch.4 Applications", chapter4),
    "conclusion": ("Conclusion", conclusion),
}
