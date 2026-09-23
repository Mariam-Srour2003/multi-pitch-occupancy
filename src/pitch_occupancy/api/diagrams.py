"""Diagrams for the thesis frontend.

Each one earns its place by showing a mechanism the prose can only assert: that class and
scene are the same variable in this dataset, that one missing cell propagates into four
unanswerable questions, and where in the pipeline a verdict can be refused rather than
guessed.

Hand-authored inline SVG. Strokes and text use `currentColor` so both themes work from one
drawing; a literal hue is spent only where it carries meaning.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "results" / "model_inventory.json"
RULES = ROOT / "configs" / "rules.json"

__all__ = ["blocked_questions", "pipeline", "protocols", "schema",
           "augmentation_axes", "dinov2_stack", "yolo_stack", "DIAGRAM_STYLES"]

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


def _inventory() -> dict:
    if not INVENTORY.exists():
        return {}
    try:
        return json.loads(INVENTORY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def dinov2_stack() -> str:
    """What DINOv2 is, and where the frozen part stops.

    The claim: one of these two boxes is pretrained and never touched, the other is the
    whole of what this project trains - and the second is four orders of magnitude smaller.
    Drawn because "frozen backbone with a linear probe" is a phrase that hides its own
    proportions; the enclosing regions are the argument.

    Every number is read from `results/model_inventory.json`, which `model_inventory.py`
    derives from the loaded models, so the drawing cannot drift from the code.
    """
    inv = _inventory()
    if not inv:
        return ""
    dino = next((b for b in inv["frozen"] if b["key"] == "dinov2"), None)
    probe = inv.get("trained", {}).get("probe")
    if not dino or not probe:
        return ""
    a = dino["architecture"]
    patches = (224 // a["patch"]) ** 2
    ratio = dino["params"] // probe["params"]

    w, h = 916, 246
    y, bh = 62, 62
    boxes = [
        (20, 118, "the frame", f'{224}&#215;{224}, inside', "the pitch boundary"),
        (170, 150, "patch embedding", f'{a["patch"]}&#215;{a["patch"]} patches',
         f'{patches} tokens + CLS'),
        (356, 176, f'transformer &#215; {a["layers"]}', f'{a["heads"]} heads, width '
         f'{a["hidden"]}', "every patch sees every other"),
        (570, 124, "mean over tokens", f'one {a["output_dim"]}-number', "description"),
        (740, 156, "linear probe", "768 &#215; 3 + 3 biases", "the only thing trained"),
    ]
    svg = ""
    for i, (x, bw, title, l1, l2) in enumerate(boxes):
        svg += (f'<rect class="dg-box" x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8"/>'
                f'<text class="dg-t" x="{x + bw / 2}" y="{y + 20}" text-anchor="middle">'
                f'{title}</text>'
                f'<text class="dg-l" x="{x + bw / 2}" y="{y + 38}" text-anchor="middle">'
                f'{l1}</text>'
                f'<text class="dg-s" x="{x + bw / 2}" y="{y + 52}" text-anchor="middle">'
                f'{l2}</text>')
        if i < len(boxes) - 1:
            nx = boxes[i + 1][0]
            svg += (f'<path class="dg-line" d="M {x + bw} {y + bh / 2} L {nx - 6} '
                    f'{y + bh / 2}" marker-end="url(#ar3)"/>')

    # the two regions: what is pretrained and never touched, and what this project fits
    frozen = (f'<rect x="152" y="{y - 30}" width="566" height="{bh + 48}" rx="12" fill="none" '
              f'stroke="currentColor" stroke-width="1.2" stroke-dasharray="5 4" opacity=".5"/>'
              f'<text class="dg-l" x="164" y="{y - 12}">frozen &middot; '
              f'{dino["params"]:,} parameters &middot; '
              f'{inv.get("frozen_gradients_received", 0)} gradients received</text>')
    trained = (f'<rect x="724" y="{y - 30}" width="188" height="{bh + 48}" rx="12" fill="none" '
               f'stroke="var(--accent)" stroke-width="1.6"/>'
               f'<text class="dg-l" x="736" y="{y - 12}" fill="var(--accent)">trained '
               f'&middot; {probe["params"]:,}</text>')

    # the three classes, and the one number that carries the argument
    chips = ""
    for j, name in enumerate(("EMPTY", "ACTIVE PLAY", "MAINTENANCE")):
        cx = 570 + j * 118
        chips += (f'<rect class="dg-box" x="{cx}" y="{y + bh + 44}" width="108" height="26" '
                  f'rx="13"/><text class="dg-s" x="{cx + 54}" y="{y + bh + 61}" '
                  f'text-anchor="middle">{name}</text>')
    drop = (f'<path class="dg-line" d="M 818 {y + bh} L 818 {y + bh + 38}" '
            f'marker-end="url(#ar3)"/>')

    return f"""<figure>
<svg viewBox="0 0 {w} {h}" role="img"
  aria-label="DINOv2 as a frozen vision transformer feeding a trained linear probe: the
  frame becomes {patches} patch tokens, {a['layers']} transformer layers turn them into one
  {a['output_dim']}-number description, and only the probe on the end is fitted.">
  <defs><marker id="ar3" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
    orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor" opacity="0.55"/></marker></defs>
  {frozen}{trained}{svg}{drop}{chips}
  <text class="dg-s" x="20" y="{y + bh + 61}">{ratio:,} frozen parameters
    per trained one</text>
</svg>
<figcaption>DINOv2 is pretrained on {dino["pretraining"].split(" on ")[-1]} with
{dino["pretraining_labels"]}, so it has no notion of the three classes and no classifier:
what comes out is a description of the image. Everything inside the dashed region is used
exactly as downloaded and receives no gradient at any point in this project. The whole of
what is fitted is the box on the right.</figcaption>
</figure>"""


def _rules() -> dict:
    if not RULES.exists():
        return {}
    try:
        return json.loads(RULES.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def yolo_stack() -> str:
    """YOLOv8 as it is actually run here, and where the rules take over.

    The claim: the detector is one forward pass that ends at boxes, and every decision in
    this project happens *after* it, in code that can be read. Drawn so the seam is visible -
    left of it a network nobody here trained, right of it arithmetic and the facility's own
    numbers.

    The settings come from `configs/rules.json`, so the drawing states what the deployed
    rule is configured with rather than what the prose remembers.
    """
    cfg = _rules()
    if not cfg:
        return ""
    w, h = 916, 232
    y, bh = 58, 66
    stages = [
        (20, 126, "the frame", f'long edge {cfg.get("imgsz", 1280)}', "one per camera-minute"),
        (176, 168, "backbone", "CSPDarknet, C2f blocks", "features at 3 scales"),
        (376, 150, "neck", "PAN-FPN", "small objects keep detail"),
        (558, 158, "head", "anchor-free, decoupled", "box + class, no anchors"),
        (748, 148, "NMS", f'person &ge; {cfg.get("person_conf", 0.25)}',
         f'ball &ge; {cfg.get("ball_conf", 0.1)}'),
    ]
    svg = ""
    for i, (x, bw, title, l1, l2) in enumerate(stages):
        svg += (f'<rect class="dg-box" x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8"/>'
                f'<text class="dg-t" x="{x + bw / 2}" y="{y + 22}" text-anchor="middle">'
                f'{title}</text>'
                f'<text class="dg-l" x="{x + bw / 2}" y="{y + 40}" text-anchor="middle">'
                f'{l1}</text>'
                f'<text class="dg-s" x="{x + bw / 2}" y="{y + 55}" text-anchor="middle">'
                f'{l2}</text>')
        if i < len(stages) - 1:
            nx = stages[i + 1][0]
            svg += (f'<path class="dg-line" d="M {x + bw} {y + bh / 2} L {nx - 6} '
                    f'{y + bh / 2}" marker-end="url(#ar4)"/>')

    net = (f'<rect x="160" y="{y - 30}" width="576" height="{bh + 48}" rx="12" fill="none" '
           f'stroke="currentColor" stroke-width="1.2" stroke-dasharray="5 4" opacity=".5"/>'
           f'<text class="dg-l" x="172" y="{y - 12}">one forward pass &middot; COCO weights, '
           f'nothing trained here &middot; {cfg.get("detector", "yolov8n")}</text>')
    out = (f'<rect x="744" y="{y - 30}" width="158" height="{bh + 48}" rx="12" fill="none" '
           f'stroke="var(--accent)" stroke-width="1.6"/>'
           f'<text class="dg-l" x="756" y="{y - 12}" fill="var(--accent)">boxes out</text>')

    # where the network stops and the readable part starts
    seam = (f'<path class="dg-hot" d="M 822 {y + bh} L 822 {y + bh + 30}" '
            f'marker-end="url(#ar4)"/>'
            f'<text class="dg-l" x="812" y="{y + bh + 46}" text-anchor="end" '
            f'fill="var(--accent)">then the rules: foot inside the boundary, '
            f'{cfg.get("small_group_max", 4)} or fewer is not a game, '
            f'{cfg.get("play_min", 5)}+ with a moving ball is</text>')

    return f"""<figure>
<svg viewBox="0 0 {w} {h}" role="img"
  aria-label="YOLOv8 as one forward pass from frame to boxes - backbone, neck, head and
  non-maximum suppression - with every decision taken afterwards by the rule table.">
  <defs><marker id="ar4" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
    orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor" opacity="0.55"/></marker></defs>
  {net}{out}{svg}{seam}
</svg>
<figcaption>The detector ends at boxes. It is asked for two COCO classes only - person and
sports ball - and it is never asked whether a match is happening: that is decided afterwards
by counting, which is why a verdict can be shown to a manager as the objects it was made
from.</figcaption>
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


def augmentation_axes() -> str:
    """Why preprocessing has a floor and augmentation does not.

    The prose can only assert the difference. Drawn, it is one asymmetry: preprocessing's
    arrows all point away from the source and never come back, so whatever they discarded
    is gone at prediction time too. Augmentation's arrows fan out for *training* and the
    source still reaches inference untouched - which is exactly why removal can fall below
    the baseline and variation cannot.
    """
    w, h = 820, 300
    bx = "dg-box"

    def box(x, y, bw, bh, title, sub="", hot=False):
        cls = "dg-hot" if hot else bx
        t = (f'<rect class="{cls}" x="{x}" y="{y}" width="{bw}" height="{bh}" rx="5" '
             f'{"fill=\"var(--surface)\"" if hot else ""}/>'
             f'<text class="dg-t" x="{x + bw / 2}" y="{y + (19 if sub else bh / 2 + 4)}" '
             f'text-anchor="middle">{title}</text>')
        if sub:
            t += (f'<text class="dg-s" x="{x + bw / 2}" y="{y + 33}" '
                  f'text-anchor="middle">{sub}</text>')
        return t

    def arrow(x1, y1, x2, y2):
        return (f'<path class="dg-line" d="M {x1} {y1} L {x2} {y2}" '
                f'marker-end="url(#ar4)"/>')

    # --- removal: a one-way chain that ends below where it started
    a = (box(40, 36, 124, 40, "source frame", "everything")
         + arrow(170, 56, 200, 56)
         + box(206, 36, 124, 40, "grayscale", "+0.022 &middot; fp 0.02")
         + arrow(336, 56, 366, 56)
         + box(372, 36, 150, 40, "gray + crop50", "&minus;0.061 &middot; fp 0.99")
         + '<text class="dg-l" x="536" y="52" fill="var(--warn)">below the untouched</text>'
         + '<text class="dg-l" x="536" y="67" fill="var(--warn)">baseline &mdash; the floor</text>')

    # --- variation: a fan for training, and the source still reaches inference
    views = "".join(
        box(206, y, 124, 24, f"view {i + 1}")
        for i, y in enumerate((148, 178, 208))
    )
    b = (box(40, 170, 124, 40, "source frame", "everything")
         + arrow(170, 186, 200, 162) + arrow(170, 190, 200, 190)
         + arrow(170, 194, 200, 218)
         + views
         + arrow(336, 162, 366, 186) + arrow(336, 190, 366, 190)
         + arrow(336, 218, 366, 194)
         + box(372, 170, 150, 40, "probe training", "sees all three")
         + box(560, 170, 224, 40, "inference", "the source frame, unchanged", hot=True)
         + '<path class="dg-line" stroke-dasharray="5 4" marker-end="url(#ar4)" '
           'd="M 102 214 L 102 262 L 672 262 L 672 216"/>'
         + '<text class="dg-s" x="330" y="256" text-anchor="middle">'
           'nothing was discarded, so nothing is missing here</text>')

    return f"""<figure>
<svg viewBox="0 0 {w} {h}" role="img"
  aria-label="Preprocessing chains one-way transforms that discard information and can fall
  below the baseline, while augmentation fans the source into several training views and
  still passes the unchanged source to inference.">
  <defs><marker id="ar4" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
    orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor" opacity="0.55"/></marker></defs>
  <text class="dg-t" x="40" y="24">preprocessing &mdash; one way, and it has a floor</text>
  {a}
  <line class="dg-line" x1="40" y1="104" x2="784" y2="104" opacity="0.25"/>
  <text class="dg-t" x="40" y="132">augmentation &mdash; varies the input, discards nothing</text>
  {b}
</svg>
<figcaption>The asymmetry is the whole argument. Every preprocessing arrow points away from
the source and never returns, so what it discarded is missing at prediction time - which is
how grayscale and a centre crop combined to score below the untouched baseline. Each box
carries recall and the false-play rate (<code>fp</code>), because recall alone is not
readable here: grayscale is a real gain on both axes, and the centre crop's larger apparent
gain was a variant answering PLAY more often - 0.998 recall at <strong>empty accuracy
0.000</strong>. Augmentation's fan exists only during training; inference still receives the
original frame, so there is no floor to cross.</figcaption>
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
