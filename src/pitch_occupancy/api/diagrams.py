"""Diagrams for the thesis frontend.

Each one earns its place by showing a mechanism the prose can only assert: that class and
scene are the same variable in this dataset, that one missing cell propagates into four
unanswerable questions, and where in the pipeline a verdict can be refused rather than
guessed.

Hand-authored inline SVG. Strokes and text use `currentColor` so both themes work from one
drawing; a literal hue is spent only where it carries meaning.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"

__all__ = ["confound_matrix", "blocked_questions", "pipeline", "protocols", "schema",
           "DIAGRAM_STYLES"]

DIAGRAM_STYLES = """
figure{margin:20px 0}
figure svg{display:block;max-width:100%;height:auto;color:var(--ink-2)}
figcaption{font-size:12.5px;color:var(--ink-3);margin-top:9px;max-width:72ch}
.dg-t{fill:var(--ink);font-family:Archivo,sans-serif;font-size:12.5px;font-weight:600}
.dg-l{fill:var(--ink-2);font-family:'JetBrains Mono',monospace;font-size:11px}
.dg-s{fill:var(--ink-3);font-family:'JetBrains Mono',monospace;font-size:10px}
.dg-box{fill:var(--surface);stroke:currentColor;stroke-width:1;opacity:.95}
.dg-line{stroke:currentColor;stroke-width:1.4;fill:none;opacity:.55}
.dg-hot{stroke:var(--accent);stroke-width:2;fill:none}
"""


def _counts() -> dict[tuple[str, str], int]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return Counter((r["class3"], r["lighting"]) for r in rows)


def confound_matrix() -> str:
    """Class against lighting, with cell area proportional to count.

    The claim: the labelled data puts EMPTY in daylight and ACTIVE_PLAY at night, so a rule
    that reads only the clock separates them. Drawn as a grid because the *shape* of the
    occupancy is the finding - two filled corners on a diagonal, everything else near-empty.
    """
    c = _counts()
    if not c:
        return ""
    classes = [
        ("C1_EMPTY", "EMPTY"),
        ("C2_ACTIVE_PLAY", "ACTIVE PLAY"),
        ("C3_MAINTENANCE_NON_SPORTING", "MAINTENANCE"),
    ]
    lights = [("day", "daylight"), ("night", "floodlit")]
    biggest = max(c.values())

    cw, ch, x0, y0 = 150, 74, 168, 54
    cells = ""
    for r, (ck, cl) in enumerate(classes):
        for k, (lk, _) in enumerate(lights):
            n = c.get((ck, lk), 0)
            x, y = x0 + k * cw, y0 + r * ch
            frac = (n / biggest) ** 0.5 if n else 0
            w, h = max(4, cw * 0.86 * frac), max(4, ch * 0.74 * frac)
            hot = n / biggest > 0.3
            fill = "var(--accent)" if hot else "currentColor"
            op = "0.9" if hot else "0.22"
            cells += (
                f'<rect x="{x + (cw - w) / 2:.0f}" y="{y + (ch - h) / 2:.0f}" '
                f'width="{w:.0f}" height="{h:.0f}" rx="3" fill="{fill}" opacity="{op}"/>'
                f'<text class="dg-l" x="{x + cw / 2:.0f}" y="{y + ch / 2 + 4:.0f}" '
                f'text-anchor="middle" fill="{"#fff" if hot else "currentColor"}">{n}</text>'
            )
        cells += (
            f'<text class="dg-t" x="{x0 - 14}" y="{y0 + r * ch + ch / 2 + 4}" '
            f'text-anchor="end">{cl}</text>'
        )
    heads = "".join(
        f'<text class="dg-t" x="{x0 + k * cw + cw / 2}" y="{y0 - 14}" '
        f'text-anchor="middle">{ll}</text>'
        for k, (_, ll) in enumerate(lights)
    )
    h = y0 + len(classes) * ch + 56
    diag = (
        f'<path class="dg-hot" d="M {x0 + cw * 0.5} {y0 + ch} '
        f'L {x0 + cw * 1.5} {y0 + ch}" stroke-dasharray="5 4"/>'
    )
    return f"""<figure>
<svg viewBox="0 0 {x0 + 2 * cw + 20} {h}" role="img"
  aria-label="Frame counts by class and lighting: empty pitches are almost all daylight and
  active play almost all floodlit, so lighting alone separates the two classes.">
  {heads}{cells}{diag}
  <text class="dg-s" x="{x0}" y="{h - 22}">A rule reading only the clock separates these
  two cells &mdash; and scores 98.4%.</text>
</svg>
<figcaption>Area is proportional to frame count. The two large cells sit on a diagonal:
lighting predicts the class without any image being examined. This is why accuracy measured
on this data cannot separate classifying occupancy from recognising the time of day.</figcaption>
</figure>"""


def blocked_questions() -> str:
    """One missing cell, four unanswerable questions.

    The claim: the gap is not "we would like more data" but a single specific absence that
    propagates. Drawn as one source fanning into four consequences, because the fan is the
    point - a list would let the reader think they are four separate problems.
    """
    items = [
        ("Three-class evaluation", "at a venue never seen"),
        ("Calibration study", "no class mix to calibrate on"),
        ("Risk / coverage point", "nothing to trade against"),
        ("Clean leakage number", "the honest split goes single-class"),
    ]
    w, h = 780, 250
    bx, by, bw, bh = 24, 92, 210, 66
    rows = ""
    for i, (title, why) in enumerate(items):
        y = 22 + i * 54
        rows += (
            f'<rect class="dg-box" x="352" y="{y}" width="404" height="42" rx="7"/>'
            f'<text class="dg-t" x="368" y="{y + 19}">{title}</text>'
            f'<text class="dg-s" x="368" y="{y + 34}">{why}</text>'
            f'<path class="dg-line" d="M {bx + bw} {by + bh / 2} '
            f'C 300 {by + bh / 2}, 310 {y + 21}, 352 {y + 21}" marker-end="url(#ar)"/>'
        )
    return f"""<figure>
<svg viewBox="0 0 {w} {h}" role="img"
  aria-label="A single missing data cell - no empty pitch outside one venue - blocks four
  separate research questions.">
  <defs><marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
    orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor" opacity="0.55"/></marker></defs>
  <rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="var(--accent)" opacity="0.12"
    stroke="var(--accent)" stroke-width="1.5"/>
  <text class="dg-t" x="{bx + 16}" y="{by + 26}" fill="var(--accent)">No empty pitch</text>
  <text class="dg-l" x="{bx + 16}" y="{by + 44}" fill="var(--accent)">outside one venue</text>
</svg>
<figcaption>Four questions fail for the same reason, not four reasons. Full-length
recordings from any second venue would close all of them at once, because a complete slot
contains the empty periods before and after play.</figcaption>
</figure>""".replace("</svg>", rows + "</svg>")


def pipeline() -> str:
    """Where a verdict can be refused rather than guessed.

    The claim: three points in the chain decline to invent an observation, and that is what
    makes the audit defensible. Drawn as the flow with those three refusals marked, because
    the refusals are invisible in a plain box-and-arrow diagram of the same pipeline.
    """
    stages = ["frame", "preprocess", "classify", "fuse\\ncameras", "aggregate\\nslot", "reconcile"]
    # six boxes of bw separated by gap, from x0, reach x0 + 5*(bw+gap) + bw = 902
    w, h, bw, bh, gap = 916, 210, 122, 54, 30
    x0, y0 = 20, 46
    boxes = ""
    for i, s in enumerate(stages):
        x = x0 + i * (bw + gap)
        lines = s.split("\\n")
        label = "".join(
            f'<text class="dg-t" x="{x + bw / 2}" y="{y0 + bh / 2 + 5 - (len(lines) - 1) * 7 + j * 14}"'
            f' text-anchor="middle">{ln}</text>' for j, ln in enumerate(lines)
        )
        boxes += f'<rect class="dg-box" x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="8"/>{label}'
        if i < len(stages) - 1:
            boxes += (f'<path class="dg-line" d="M {x + bw} {y0 + bh / 2} '
                      f'L {x + bw + gap - 6} {y0 + bh / 2}" marker-end="url(#ar2)"/>')

    refusals = [
        (2, "no frame &rarr; gap,", "never an invented EMPTY"),
        (3, "no camera &rarr; raise,", "never a default verdict"),
        (5, "unsure &rarr; REVIEW,", "never an anomaly"),
    ]
    marks = ""
    for i, l1, l2 in refusals:
        x = x0 + i * (bw + gap) + bw / 2
        marks += (
            f'<path class="dg-hot" d="M {x} {y0 + bh} L {x} {y0 + bh + 22}"/>'
            f'<text class="dg-l" x="{x}" y="{y0 + bh + 40}" text-anchor="middle" '
            f'fill="var(--accent)">{l1}</text>'
            f'<text class="dg-s" x="{x}" y="{y0 + bh + 55}" text-anchor="middle">{l2}</text>'
        )
    return f"""<figure>
<svg viewBox="0 0 {w} {h}" role="img"
  aria-label="The pipeline from frame to reconciliation, marking the three points where the
  system refuses to invent an observation rather than guessing.">
  <defs><marker id="ar2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
    orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor" opacity="0.55"/></marker></defs>
  <text class="dg-s" x="{x0}" y="26">one frame per camera per minute</text>
  {boxes}{marks}
</svg>
<figcaption>The marked points are where the system declines to produce an answer. A camera
outage becomes a gap that weakens the verdict rather than an empty pitch; a slot with no
readable camera raises rather than defaulting; an uncertain verdict routes to a person
instead of becoming an accusation.</figcaption>
</figure>"""


def protocols() -> str:
    """Why the three protocols disagree.

    The claim: they disagree because they cut the data differently, not because one is
    noisier. Drawn as the same frame pool partitioned three ways, so the reader can see the
    leak in the first one - train and test drawing from the same scene.
    """
    w, h = 820, 262  # the legend sits at y0 + 3*rh + 16 = 246
    rows = [
        ("random split", "frames shuffled", [("train", 0, 9, False), ("test", 9, 3, True)],
         "near-identical frames land on both sides"),
        ("grouped split", "whole slots move together",
         [("train", 0, 5, False), ("test", 5, 7, True)],
         "honest, but the test side goes 99% one class"),
        ("cross-venue", "whole venues held out",
         [("train", 0, 8, False), ("test", 8, 4, True)],
         "the only protocol that answers transfer"),
    ]
    x0, cw, y0, rh, bh = 152, 42, 44, 62, 26
    body = ""
    for i, (name, sub, parts, note) in enumerate(rows):
        y = y0 + i * rh
        body += (f'<text class="dg-t" x="{x0 - 14}" y="{y + 17}" text-anchor="end">{name}</text>'
                 f'<text class="dg-s" x="{x0 - 14}" y="{y + 31}" text-anchor="end">{sub}</text>')
        for label, start, count, is_test in parts:
            for k in range(count):
                x = x0 + (start + k) * cw
                fill = "var(--accent)" if is_test else "currentColor"
                op = "0.85" if is_test else "0.18"
                body += (f'<rect x="{x}" y="{y}" width="{cw - 5}" height="{bh}" rx="3" '
                         f'fill="{fill}" opacity="{op}"/>')
        body += f'<text class="dg-s" x="{x0 + 12 * cw + 10}" y="{y + 17}">{note}</text>'
    return f"""<figure>
<svg viewBox="0 0 {w} {h}" role="img"
  aria-label="Three evaluation protocols cutting the same frames differently: shuffled,
  grouped by slot, and held out by venue.">
  <text class="dg-s" x="{x0}" y="30">the same frames, cut three ways</text>
  {body}
  <text class="dg-s" x="{x0}" y="{y0 + 3 * rh + 16}">
    <tspan fill="var(--accent)">&#9632;</tspan> test &nbsp;
    <tspan>&#9632;</tspan> train
  </text>
</svg>
<figcaption>The protocols disagree because they partition differently, not because one is
noisier. Shuffling puts frames sampled seconds apart on both sides; holding out whole
venues is the only cut that asks whether the model transfers.</figcaption>
</figure>"""


DB_PATH = ROOT / "data" / "db" / "pitch_monitor.db"


def _table_counts() -> dict[str, int]:
    """Live row counts, so the diagram shows the database that exists."""
    if not DB_PATH.exists():
        return {}
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        names = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {n: conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}
    except sqlite3.Error:
        return {}
    finally:
        try:
            conn.close()
        except (NameError, sqlite3.Error):
            pass


def schema() -> str:
    """The three layers a verdict has to connect, and the rows joining them.

    The claim: an auditable verdict needs the estate it came from, the observations behind
    it, and the record it disagreed with - and the schema exists to keep those linked.
    Drawn in layers because the layering is the argument; an entity-relationship sketch of
    the same tables would show the joins and hide the reason for them.
    """
    counts = _table_counts()

    def box(x, y, w, name, sub, *, accent=False):
        n = counts.get(name)
        fill = 'fill="var(--accent)" opacity="0.1"' if accent else 'class="dg-box"'
        stroke = ' stroke="var(--accent)" stroke-width="1.4"' if accent else ""
        badge = (
            f'<text class="dg-s" x="{x + w - 12}" y="{y + 20}" text-anchor="end">{n} rows</text>'
            if n is not None else ""
        )
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="46" rx="7" {fill}{stroke}/>'
            f'<text class="dg-t" x="{x + 13}" y="{y + 21}">{name}</text>'
            f'<text class="dg-s" x="{x + 13}" y="{y + 36}">{sub}</text>{badge}'
        )

    lanes = [
        (34, "the estate", [("venues", "facility", 150), ("fields", "a pitch", 150),
                            ("cameras", "two per field", 178)]),
        (140, "what was observed", [("rental_slots", "the schedule", 178),
                                    ("frame_samples", "one row per camera-minute", 300)]),
        (246, "what was decided", [("slot_evaluations", "verdict + override", 246),
                                   ("bookings", "what records claim", 232)]),
        (352, "what disagreed", [("reconciliations", "typed anomalies", 246)]),
    ]
    body, w = "", 860
    for y, lane, boxes in lanes:
        body += f'<text class="dg-s" x="20" y="{y + 27}">{lane}</text>'
        x = 176
        for name, sub, bw in boxes:
            body += box(x, y, bw, name, sub, accent=name in ("frame_samples", "reconciliations"))
            x += bw + 22
    links = "".join(
        f'<path class="dg-line" d="M {x} {y1} L {x} {y2}" marker-end="url(#ar3)"/>'
        for x, y1, y2 in ((250, 80, 140), (400, 186, 246), (300, 292, 352))
    )
    h = 424
    return f"""<figure>
<svg viewBox="0 0 {w} {h}" role="img"
  aria-label="Database schema in four layers: the physical estate, the observations, the
  decisions and bookings, and the reconciliation outcomes.">
  <defs><marker id="ar3" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
    orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor" opacity="0.55"/></marker></defs>
  {body}{links}
  <text class="dg-s" x="20" y="{h - 12}">row counts are live from the seeded database</text>
</svg>
<figcaption>The schema exists to keep a verdict connected to its evidence. The highlighted
tables are the two that make it auditable: every sampled minute behind a decision, and every
disagreement with the booking record.</figcaption>
</figure>"""
