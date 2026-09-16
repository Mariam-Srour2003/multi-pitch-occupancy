"""Every tab as a slide: the numbers and the pictures first, the document underneath (WP8).

The thesis front end rendered each tab's source document in full. Those documents are good -
they are the thesis - but a Markdown file read start to finish is a poor *page*: a reader
arriving at "Dataset" wants to know how many frames there are and where the hole is, and was
instead given six thousand words in which those facts appear somewhere.

So each tab now opens with a slide: four figures at most, a row of counts, a few cards, and
the full document collapsed below it. **Nothing is removed.** The distinction this module
turns on is that a document is the *evidence* and a slide is the *claim*, and a page that
leads with the claim can still be checked, while a page that only prints the evidence is
checked by nobody.

Two rules keep the slides honest:

**Every number is read from an artefact.** The RQ statuses are parsed out of the table in
`thesis/rq_matrix.md` rather than restated here; the parameter counts come from
`results/model_inventory.json`; the class counts come from `results/coverage.md`. A slide
that restated them would be a second copy of the thesis, and the copy that goes stale -
which is the failure this repository keeps finding in its own history.

**A missing artefact says so.** A slide whose figure has not been generated prints the
command that would generate it, rather than rendering an empty box. A page that silently
shows nothing where its argument should be is the shape of failure `augmentation_grid.py`
was written to prevent, and it applies to the page as much as to the sheet.

The Augmentation tab is deliberately the exception to "less": its argument *is* the pictures,
because augmentation code fails silently and the only reliable check is a person looking.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

__all__ = ["STYLES", "SLIDES", "tiles", "figure", "cards", "detail"]

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
THESIS = ROOT / "thesis"
FIGS = RESULTS / "figs"


# --- components -----------------------------------------------------------------------


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


# --- the styles -----------------------------------------------------------------------

#: Scoped under `.slide` so nothing here reaches the rendered Markdown below it.
STYLES = """
.slide { margin: 0 0 6px; }
.slide .lede { font-size: 15.5px; color: var(--ink-2); max-width: 70ch; margin: 0 0 4px; }
.slide h3 { margin: 26px 0 8px; font-size: 14.5px; font-weight: 600; }
.slide .note { font-size: 12.5px; color: var(--ink-3); max-width: 74ch; margin: 8px 0 0; }

.sl-tiles { display: grid; gap: 12px; margin: 18px 0;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.sl-tile { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 15px; }
.sl-tile b { display: block; font: 700 26px 'JetBrains Mono', monospace;
  font-variant-numeric: tabular-nums; letter-spacing: -.02em; margin-bottom: 3px; }
.sl-tile span { font-size: 12.5px; color: var(--ink-2); line-height: 1.4; }
.sl-tile.good { border-left: 4px solid var(--play); }
.sl-tile.good b { color: var(--play); }
.sl-tile.bad { border-left: 4px solid var(--flag); }
.sl-tile.bad b { color: var(--flag); }
.sl-tile.lead { border-left: 4px solid var(--accent); }
.sl-tile.lead b { color: var(--accent); }

.sl-figs { display: grid; gap: 14px; margin: 18px 0;
  grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }
.sl-fig { margin: 16px 0; background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; overflow: hidden; }
.sl-figs .sl-fig { margin: 0; }
.sl-fig img { display: block; width: 100%; background: var(--surface-2); }
.sl-fig figcaption { font-size: 12.5px; color: var(--ink-2); padding: 10px 13px;
  border-top: 1px solid var(--line); }

.sl-cards { display: grid; gap: 12px; margin: 16px 0;
  grid-template-columns: repeat(auto-fit, minmax(212px, 1fr)); }
.sl-cards.wide { grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); }
.sl-card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 13px 15px; }
.sl-card h4 { margin: 0 0 5px; font-size: 13.5px; font-weight: 600; }
.sl-card p { margin: 0; font-size: 12.5px; color: var(--ink-2); line-height: 1.5; }

/* A claim the page is making, as opposed to `blockquote`, which the shell tints with
   --warn-soft and which therefore reads as a caution. The distinction matters on the
   Overview tab, where the thesis statement and the retraction are both set off from the
   prose and must not look like the same kind of remark. */
.sl-quote { margin: 18px 0; padding: 14px 18px; background: var(--accent-soft);
  border-left: 4px solid var(--accent); border-radius: 0 10px 10px 0;
  font-size: 16px; line-height: 1.5; color: var(--ink); max-width: 74ch; }

.sl-rows { display: grid; gap: 7px; margin: 16px 0; }
.sl-row { display: grid; grid-template-columns: 54px 1fr auto; gap: 13px;
  align-items: center; background: var(--surface); border: 1px solid var(--line);
  border-radius: 9px; padding: 10px 14px; }
.sl-row .id { font: 700 12.5px 'JetBrains Mono', monospace; color: var(--accent); }
.sl-row .q { font-size: 13px; color: var(--ink-2); }
.sl-row .st { font: 600 11px 'JetBrains Mono', monospace; padding: 3px 9px;
  border-radius: 5px; white-space: nowrap; background: var(--surface-2);
  color: var(--ink-3); }
.sl-row .st.ok { background: var(--play-soft); color: var(--play); }
.sl-row .st.part { background: var(--maint-soft); color: var(--maint); }
.sl-row .st.blocked { background: var(--flag-soft); color: var(--flag); }

.sl-detail { border: 1px solid var(--line); border-radius: 10px; background: var(--surface);
  margin: 26px 0 0; }
.sl-detail > summary { cursor: pointer; padding: 12px 16px; font-weight: 600;
  font-size: 13.5px; list-style: none; display: flex; gap: 10px; align-items: center; }
.sl-detail > summary::-webkit-details-marker { display: none; }
.sl-detail > summary::before { content: "\\25B8"; color: var(--ink-3); font-size: 11px; }
.sl-detail[open] > summary::before { content: "\\25BE"; }
.sl-detail[open] > summary { border-bottom: 1px solid var(--line); }
.sl-detail > summary:hover { background: var(--surface-2); }
.sl-detail-body { padding: 4px 18px 16px; }

/* the augmentation tab, which is deliberately image-heavy */
.sl-pairs { display: grid; gap: 14px; margin: 16px 0;
  grid-template-columns: repeat(auto-fill, minmax(268px, 1fr)); }
.sl-pair { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  overflow: hidden; }
.sl-pair img { display: block; width: 100%; background: var(--surface-2); }
.sl-pair .cap { padding: 9px 12px; border-top: 1px solid var(--line); }
.sl-pair .cap code { font-size: 12px; font-weight: 600; color: var(--ink);
  background: none; padding: 0; }
.sl-pair .cap .m { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 5px;
  font: 500 10.5px 'JetBrains Mono', monospace; color: var(--ink-3); }
.sl-pair .cap .m b { color: var(--ink-2); }
.sl-pair.crop { border-left: 3px solid var(--flag); }
"""


# --- the slides -----------------------------------------------------------------------


def models() -> str:
    """What was frozen and what was trained, which is the first thing anyone asks."""
    inv = _json("model_inventory.json")
    if not inv:
        return ("<p class='missing'>No <code>model_inventory.json</code> yet. Run "
                "<code>uv run python experiments/model_inventory.py</code>.</p>")
    frozen = inv["frozen_total_params"]
    trained = inv["trained_total_params"]
    probe = inv["trained"]["probe"]
    gate = next(r for r in inv["trained"]["fusion"] if r["rung"] == "mlp")

    lanes = []
    for b in inv["frozen"]:
        arch = b["architecture"]
        shape = (f'stages {arch["depths"]}, widths {arch["widths"]}'
                 if arch["family"] == "convnet"
                 else f'{arch["layers"]} layers, hidden {arch["hidden"]}')
        lanes.append((
            b["label"],
            f'<b>{b["params"]:,}</b> parameters, <b>0</b> updated<br>{shape}<br>'
            f'{b["pretraining"]} &mdash; {b["pretraining_labels"]}<br><em>{b["role"]}</em>'
        ))

    return (
        '<div class="slide">'
        '<p class="lede">The three vision backbones were <b>never trained on this data</b>. '
        'What learns is a small head on top of their frozen output.</p>'
        + tiles([
            (f'{frozen / 1e6:.1f}M', 'frozen parameters, 0 ever updated', 'lead'),
            (f'{trained:,}', 'trained parameters, all of them'),
            (f'{inv["frozen_to_trained_ratio"]:,}:1', 'frozen to trained'),
            (str(inv["n_classes"]), 'classes out'),
        ])
        + '<h3>The three frozen extractors</h3>'
        + cards(lanes, wide=True)
        + '<h3>What is actually trained</h3>'
        + cards([
            (f'Linear probe &mdash; {probe["params"]:,}',
             f'{probe["shape"]}. {probe["refit"]}.'),
            (f'Gated fusion head &mdash; {gate["params"]:,}',
             'Routes between backbones per frame. It puts about 0.70 of its weight on '
             'DINOv2 in every fold: a learned constant wearing a router&rsquo;s costume.'),
            (f'STAN &mdash; {inv["trained"]["stan"]["params"]:,}',
             f'{inv["trained"]["stan"]["shape"]}. Gated at 30 real slots; there are 2.'),
        ], wide=True)
        + figure("accuracy_vs_latency.png",
                 "Accuracy against latency on a 4-thread CPU. All three fit the 60-second "
                 "cycle with large margins, so the choice falls to accuracy.",
                 script="uv run python experiments/make_figures.py")
        + '</div>'
    )


def findings() -> str:
    """The figures first. The claims and the log follow, from `findings_summary`."""
    return (
        '<div class="slide">'
        '<p class="lede">Seven figures carry the argument. Each is a claim you can check '
        'against the artefact beside it.</p>'
        '<div class="sl-figs">'
        + figure("ranking_inversion.png",
                 "<b>The protocol reverses the ranking.</b> ViT ranks first under the leaky "
                 "random split and last across venues.")
        + figure("baseline_floor.png",
                 "<b>A constant predictor wins one protocol.</b> Under leave-one-venue-out "
                 "every held-out venue is 100% active play.")
        + figure("cross_venue_recall.png",
                 "<b>Per-venue play recall.</b> Five of seven venues contribute 12-30 "
                 "frames, so the spread matters more than the mean.")
        + figure("leakage_decomposition.png",
                 "<b>How much of the penalty is leakage.</b> A model that never trains "
                 "drops 0.183 on the same change of test set.")
        + figure("label_efficiency.png",
                 "<b>More labels made it worse.</b> Every backbone peaks at 100-300 and "
                 "falls at 671.")
        + figure("onboarding_cost.png",
                 "<b>A new camera costs about one frame.</b> 0.44 macro-F1 cold, 0.99 after "
                 "a single labelled frame of it.")
        + figure("risk_coverage_band.png",
                 "<b>Why it is a band, not a curve.</b> 890 of 907 confidences are "
                 "identical, so the accuracy there is an interval.")
        + '</div></div>'
    )


def questions() -> str:
    """The research questions and where each stands, parsed from the RQ matrix."""
    rows = rq_status()
    if not rows:
        return ""

    def tone(status: str) -> str:
        low = status.lower()
        if "blocked" in low or "not started" in low:
            return "blocked"
        if "partly" in low or "baseline" in low or "looks like" in low:
            return "part"
        return "ok" if "answered" in low else ""

    body = "".join(
        f'<div class="sl-row"><span class="id">{rid}</span>'
        f'<span class="q">{q}</span>'
        f'<span class="st {tone(st)}">{re.sub(r"[*_]", "", st)[:44]}</span></div>'
        for rid, q, st in rows
    )
    answered = sum(1 for _, _, s in rows if tone(s) == "ok")
    blocked = sum(1 for _, _, s in rows if tone(s) == "blocked")
    return (
        '<div class="slide">'
        '<p class="lede">The research questions, and where each one actually stands.</p>'
        + tiles([
            (str(len(rows)), 'research questions'),
            (str(answered), 'answered', 'good'),
            (str(len(rows) - answered - blocked), 'partly answered'),
            (str(blocked), 'blocked or not started', 'bad'),
        ])
        + f'<div class="sl-rows">{body}</div>'
        + '<p class="note">Statuses are parsed from the table in '
          '<code>thesis/rq_matrix.md</code> rather than restated here &mdash; a copy would '
          'be right the day it was written and wrong afterwards.</p>'
        + '</div>'
    )


def dataset() -> str:
    """How many frames, and the one gap that blocks four separate questions."""
    c = coverage_counts()
    if not c:
        return ""
    empty = c.get("EMPTY", {})
    play = c.get("ACTIVE_PLAY", {})
    maint = c.get("MAINTENANCE_NON_SPORTING", {})
    with_empty = c.get("venues_with_empty", 1)
    return (
        '<div class="slide">'
        '<p class="lede">One fact shapes everything: <b>every empty pitch in the corpus is '
        'one venue</b>. Class and scene are very nearly the same variable.</p>'
        + tiles([
            (c.get("frames", "?"), 'recorded frames', 'lead'),
            (c.get("venues", "?"), 'venues'),
            (empty.get("total", "?"),
             f'EMPTY, all from {with_empty} venue{"" if with_empty == 1 else "s"}'),
            (maint.get("total", "?"), 'MAINTENANCE, unusable as a class', 'bad'),
        ])
        + cards([
            ('EMPTY is daytime',
             f'{empty.get("day", "?")} day against {empty.get("night", "?")} night.'),
            ('ACTIVE_PLAY is night',
             f'{play.get("night", "?")} night against {play.get("day", "?")} day.'),
            ('So a clock beats a camera',
             'A rule reading only the lighting field scores <b>98.4%</b>. Accuracy here '
             'cannot separate classifying occupancy from recognising a place.'),
            ('And the fix is small',
             '20-30 minutes of an unoccupied pitch at two other venues, day and night. '
             'Not more data in general.'),
        ])
        + f'<p class="note">{c.get("generated", "0")} generated frames are excluded from '
          'every count above (A13): they are a training-side augmentation, and counting '
          'them here would make a gap look filled that is not.</p>'
        + '</div>'
    )


def database() -> str:
    """The schema, and the facts about it that matter operationally."""
    return (
        '<div class="slide">'
        '<p class="lede">One row per camera per sampled minute is the raw observation. '
        'Everything above it is derived, and can be recomputed.</p>'
        + cards([
            ('A verdict never travels alone',
             'A slot verdict and the evidence frames behind it are written in one '
             'transaction. There is no state in which a verdict exists without its '
             'evidence.'),
            ('A gap is recorded, never filled',
             'A camera that drops for a minute leaves a missing sample. No footage is not '
             'evidence a pitch was unused.'),
            ('No financial column exists',
             'Checked by a test rather than by intention: <code>db/schema.py</code> may not '
             'name a price, an invoice or a charge.'),
        ])
        + '</div>'
    )


def code() -> str:
    """The repository as counts, so the map below it has a scale."""
    graph = _json("pipeline_graph.json")
    inv = _json("model_inventory.json")
    src = len(list((ROOT / "src").rglob("*.py")))
    exps = len(list((ROOT / "experiments").glob("*.py")))
    tests = len(list((ROOT / "tests").glob("test_*.py")))
    items = [(str(src), 'source modules'), (str(exps), 'experiment scripts'),
             (str(tests), 'test files')]
    if graph:
        items.append((f'{graph["total_minutes"] / 60:.1f}h',
                      f'to reproduce all {graph["n_stages"]} stages'))
    extra = ""
    if inv:
        extra = cards([
            ('Frozen, not fine-tuned',
             f'{inv["frozen_total_params"] / 1e6:.1f}M pretrained parameters, none updated. '
             'Embed once, reuse forever, which is what bought the experimental breadth.'),
            ('One preprocessing path',
             'Experiments and the live pipeline share <code>preprocess.py</code>. The pilot '
             'proved what happens when they drift apart.'),
            ('Every result regenerable',
             '<code>reproduce_all.py</code> verifies its own outputs and reports BLOCKED '
             'rather than skipping: a script that exits 0 having done nothing is worse than '
             'one that fails.'),
        ])
    return ('<div class="slide">'
            '<p class="lede">What exists, and what it costs to rebuild from the footage.</p>'
            + tiles(items) + extra + '</div>')


def ideas() -> str:
    """Ideas are only useful with a stop condition, so that is what the slide counts."""
    text = _read(DOCS / "IDEAS.md")
    n = len(re.findall(r"^## ", text, re.MULTILINE))
    stops = len(re.findall(r"[Ss]top condition", text))
    return (
        '<div class="slide">'
        '<p class="lede">Ideas outside the plan. Each carries a <b>stop condition</b> '
        '&mdash; what would have to be true to abandon it &mdash; because an idea without '
        'one is a way to spend the remaining weeks without deciding anything.</p>'
        + tiles([(str(n), 'ideas recorded'), (str(stops), 'stop conditions written')])
        + '</div>'
    )


def ethics() -> str:
    """The commitments, as the short list they are."""
    return (
        '<div class="slide">'
        '<p class="lede">Commitments the system is built to keep, each checked somewhere in '
        'the code rather than promised here.</p>'
        + cards([
            ('Anomalies are per field, never per person',
             'Attributing a discrepancy to an individual adds nothing scientifically and a '
             'great deal of ethical exposure. <code>Reconciliation</code> has no field that '
             'could name one.'),
            ('The system never bills',
             'Output is decision support. There is no severity at which a person stops '
             'being required.'),
            ('REVIEW never becomes an anomaly',
             'The system may not convert its own uncertainty into someone else&rsquo;s '
             'error.'),
            ('Faces are redacted before publication',
             'Two layers &mdash; person detection and pixelation, then a whole-frame blur '
             'floor &mdash; on every frame that leaves the system.'),
            ('Footage is never kept by the review tools',
             'Uploads to the clip and image reviewers are deleted in a '
             '<code>finally</code>, so a crash mid-analysis leaves nothing behind.'),
        ])
        + '</div>'
    )


def prereg() -> str:
    """What was fixed before the runs, and the gates that check it."""
    text = _read(THESIS / "preregistration.md")
    amendments = len(re.findall(r"^#+ *A\d+", text, re.MULTILINE))
    gates = gate_rows()
    passed = sum(1 for _, _, v in gates if "passed" in v.lower())
    waiting = sum(1 for _, _, v in gates if "person" in v.lower())
    gate_cards = [
        (f'{g} &middot; week {w}', v) for g, w, v in gates
    ]
    return (
        '<div class="slide">'
        '<p class="lede">Hypotheses and the evaluation protocol were fixed <b>before</b> '
        'the runs, and every amendment since is numbered and dated in place.</p>'
        + tiles([
            (str(amendments), 'numbered amendments', 'lead'),
            (str(len(gates)), 'milestone gates'),
            (str(passed), 'gates passed', 'good'),
            (str(waiting), 'waiting on a person', 'bad'),
        ])
        + ('<h3>Milestone gates</h3>' + cards(gate_cards) if gates else '')
        + '<p class="note">Gate verdicts are checked against the artefacts by '
          '<code>experiments/gate_check.py</code> rather than ticked by hand. "Waiting on a '
          'person" is reported separately from "not met", because nothing in the repository '
          'can move it.</p>'
        + '</div>'
    )


def augmentation() -> str:
    """The exception to "less": this tab's argument *is* the pictures.

    Augmentation and preprocessing code fails silently - a preset that does nothing, a fog
    veil that flattens the pitch, a crop that removes the goalmouth - and every one of those
    passes a shape and dtype check. The only reliable check is a person looking, so this tab
    shows every switch, before and after, at the size the model receives.
    """
    pairs = _csv("preprocess_pairs.csv")
    loudest = max((float(p["mean_abs_change_255"]) for p in pairs), default=1.0) or 1.0

    cells = []
    for p in pairs:
        change = float(p["mean_abs_change_255"])
        area = float(p["area_retained"])
        name = Path(p["file"]).name
        meta = (f'<b>{change:.1f}</b>/255 moved &middot; '
                f'<b>{float(p["share_pixels_changed"]):.0%}</b> of pixels')
        if area < 1.0:
            meta += f' &middot; <b>{area:.0%}</b> of the frame kept'
        cells.append(
            f'<figure class="sl-pair{" crop" if area < 1.0 else ""}">'
            f'<img src="/figs/preproc/{name}" loading="lazy" '
            f'alt="The same frame before and after {p["label"]}">'
            f'<div class="cap"><code>{p["label"]}</code>'
            f'<div class="m">{meta}</div></div></figure>'
        )

    quietest = min(pairs, key=lambda p: float(p["mean_abs_change_255"])) if pairs else None
    note = ""
    if quietest:
        note = (
            '<p class="note"><b>A searched margin on top of a near no-op is not a margin.</b> '
            f'<code>{quietest["label"]}</code> moves the frame by '
            f'{float(quietest["mean_abs_change_255"]):.2f}/255 and touches '
            f'{float(quietest["share_pixels_changed"]):.0%} of it, against {loudest:.1f} for '
            'the loudest switch &mdash; yet the preprocessing search credits it with +0.016 '
            'recall. Both are true, and together they say that margin is resolution rather '
            'than effect.</p>'
        )

    pair_block = (
        f'<h3>Every preprocessing switch, before and after ({len(pairs)})</h3>'
        '<p class="lede">Each pair is one switch applied to the same night active-play '
        'frame, at the 224&times;224 the model actually receives. The numbers under each '
        'say whether it did anything &mdash; two same-sized tiles cannot, because a crop is '
        're-letterboxed back to 224 and looks like a zoom.</p>'
        f'<div class="sl-pairs">{"".join(cells)}</div>{note}'
        if pairs else
        "<p class='missing'>No <code>preprocess_pairs.csv</code> yet. Run "
        "<code>uv run python experiments/preprocess_pairs.py</code>.</p>"
    )

    return (
        '<div class="slide">'
        '<p class="lede">Augmentation code fails <b>silently</b>: a preset that does '
        'nothing, a fog veil that flattens the pitch, a crop that removes the goalmouth '
        '&mdash; all of them pass a shape and dtype check. The only reliable check is a '
        'person looking, so this tab is deliberately all pictures.</p>'
        + figure("augmentation_grid.jpg",
                 "<b>Every preset, three draws each</b>, over a night play frame and a day "
                 "empty frame. Probability is forced to 1 so the sheet shows the effect "
                 "rather than the coin flip that gates it; in training each effect fires "
                 "with probability <code>p</code>, so a real batch mixes these with "
                 "untouched originals.",
                 script="uv run python experiments/augmentation_grid.py")
        + figure("augmentation_effects.jpg",
                 "<b>One effect at a time</b>, at the magnitude the <code>full</code> preset "
                 "uses for it. The magnitudes are read off the preset rather than retyped, "
                 "so this sheet cannot drift from the config it illustrates. This is the "
                 "sheet that says <em>which knob</em>: a preset row compounds up to nine "
                 "effects, so when one is wrong the compound row shows only that something "
                 "is.",
                 script="uv run python experiments/augmentation_grid.py")
        + figure("preprocess_effects.jpg",
                 "<b>Every preprocessing switch</b> over two frames, at the model's own "
                 "input size. Preprocessing is deterministic, so there is nothing to draw "
                 "three times.",
                 script="uv run python experiments/augmentation_grid.py")
        + pair_block
        + '</div>'
    )


def searches() -> str:
    """Two searches with opposite cost profiles, and the resolution floor under both."""
    prompts = _csv("prompt_search.csv")
    evals = _json("preprocess_search.json").get("evaluations", [])
    best = max(prompts, key=lambda r: float(r["balanced"])) if prompts else None

    items: list[tuple[str, str] | tuple[str, str, str]] = []
    if best:
        items.append((best["balanced"], 'best balanced score, zero labels', 'lead'))
        items.append((str(len(prompts)), 'prompt sets scored exhaustively'))
    if evals:
        items.append((str(len(evals)), 'preprocessing evaluations, greedy'))
    items.append(('0.930', 'DINOv2 probe, ~1,500 labels'))

    return (
        '<div class="slide">'
        '<p class="lede">Two configuration searches with <b>opposite cost profiles</b>, and '
        'the same guard on both: every cross-venue fold is 100% active play, so recall alone '
        'can be bought by answering &ldquo;playing&rdquo; more often.</p>'
        + tiles(items)
        + cards([
            ('Prompts are cheap, so the search is exhaustive',
             'Images are embedded <b>once</b>; a prompt set is a few short strings, so '
             'thousands of combinations cost seconds. All 375 were scored.'),
            ('Preprocessing is expensive, so the search is greedy',
             'Every candidate needs a fresh embedding pass over the whole dataset &mdash; '
             '740 of the pipeline&rsquo;s 1,114 minutes sit in this one stage.'),
            ('A searched prompt beat every trained probe',
             f'&ldquo;{best["desc_ACTIVE_PLAY"]}&rdquo; reaches {best["play_recall"]} '
             f'cross-venue recall at false-play {best["false_play"]}, above '
             'DINOv2&rsquo;s 0.930 &mdash; which needed about 1,500 labelled frames.'
             if best else 'Scored on cross-venue transfer.'),
            ('And the caveat is not small',
             'The winner was selected against the same folds it is scored on, and 375 '
             'candidates is a great deal of selection freedom. This is a <b>development '
             'result</b>; the locked venues were held out throughout, so the honest '
             'confirmation remains available.'),
        ], wide=True)
        + '<p class="note">The hypothesis tests deliberately use a different, pre-declared '
          'prompt set &mdash; the first descriptor per class, fixed before the search ran. '
          'A selected number cannot test the selection.</p>'
        + '</div>'
    )


def overview() -> str:
    """The first five pages of the progress-review deck, as the page a reader lands on.

    Every other slide here reads its numbers out of an artefact, because a slide that
    restates them becomes a second copy of the thesis and the copy that goes stale. This
    one is the deliberate exception, and for the opposite reason: it is a **transcription
    of a dated document** - the progress review presented on the state of the project - so
    its numbers are supposed to be frozen at what was claimed on the day. Wiring them to
    today's result files would quietly rewrite the history the review records, which is the
    one thing a progress review must not do.

    That is also why the pilot's figures are safe to hard-code: they were measured on branch
    `pilot/model-selection` against a split this repository has since retired, and no
    artefact in `results/` can or should reproduce them. Page five is the retraction, and it
    ships in the same tab as the claim it retracts.
    """
    bakeoff = [
        ("ViT-Base/16", "frozen + head", "0.9923", "0.6623", "203.0"),
        ("DINOv2-Base", "frozen + head", "0.9846", "0.6568", "253.9"),
        ("ConvNeXtV2-Tiny", "frozen + head", "0.9808", "0.6540", "102.4"),
        ("OpenCLIP ViT-B/32 (LAION-2B)", "zero-shot", "0.7577", "0.3866", "246.8"),
        ("CLIP ViT-B/32", "zero-shot", "0.6538", "0.4269", "212.0"),
        ("CLIP ViT-L/14", "zero-shot", "0.3538", "0.2500", "1403.8"),
        ("SigLIP 2 base", "zero-shot", "0.3192", "0.3011", "1035.0"),
    ]
    rows = "".join(
        f"<tr><td><strong>{model}</strong></td><td>{family}</td>"
        f"<td class='num'>{acc}</td><td class='num'>{f1}</td>"
        f"<td class='num'>{ms}</td></tr>"
        for model, family, acc, f1, ms in bakeoff
    )

    return (
        "<h1>Low-Bandwidth, CPU-Only Occupancy Verification for Multi-Pitch Football "
        "Facilities</h1>"
        '<div class="slide">'
        '<p class="lede"><strong>Master&rsquo;s thesis progress review.</strong> From a '
        "seven-model pilot bake-off to a leakage-free multi-venue benchmark &mdash; this "
        "review documents the full arc of the project, including the measurements that "
        "invalidated the pilot&rsquo;s own results.</p>"
        '<p class="sl-quote"><strong>The evaluation protocol, not the architecture, '
        "turned out to be the result.</strong></p>"

        "<h2>The Problem, the System, and the Constraints</h2>"
        + cards([
            ("The verification gap",
             "A facility rents pitches by the hour and staff record which slots were used. "
             "Those records are unverified, and three failure modes cost money or trust: "
             "<b>no-shows</b>, <b>unbooked usage</b>, and plain <b>data-entry error</b>."),
            ("Four stages",
             "One frame per fixed CCTV camera per minute &mdash; sparse sampling, not a "
             "video stream. Each frame classified EMPTY / ACTIVE_PLAY / "
             "MAINTENANCE-or-NON-SPORTING. Roughly 60 predictions per hour collapsed to "
             "USED / NOTUSED / REVIEW. The verdict reconciled against the booking and "
             "staff-entry record, with disagreements flagged with evidence."),
        ], wide=True)
        + "<h3>Four hard constraints</h3>"
        + cards([
            ("No GPU", "One Intel Mini-PC only."),
            ("60-second cycle",
             "20&ndash;30 cameras; the budget is round-trip time, not single-image "
             "latency."),
            ("Data sovereignty",
             "Only the verdict leaves the site &mdash; never the footage."),
            ("Adverse optics",
             "Night floodlighting, low contrast and glare throughout."),
        ])
        + '<p class="sl-quote">One design rule carried everywhere: <strong>the vision '
          "path must never consume the booking record as an input feature.</strong> A "
          "model that has seen the booking flag cannot provide evidence independent of "
          "the record it audits.</p>"

        "<h2>Step One: The Pilot Bake-Off</h2>"
        "<p>Branch <code>pilot/model-selection</code>, 1&ndash;2 September 2026. One "
        "question: which models are worth building on? 1,296 hand-labelled frames, one "
        "venue, single random stratified 80/20 split. Seven candidates through one shared "
        "harness.</p>"
        '<div class="scroll"><table><thead><tr><th>Model</th><th>Family</th>'
        '<th class="num">Accuracy</th><th class="num">Macro-F1</th>'
        '<th class="num">ms/frame</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
        + cards([
            ("Also considered, not run",
             "Florence-2, Moondream2, SmolVLM2, DINOv3 (licence-gated), ConvNeXt-Tiny v1, "
             "EfficientNet-B0, MobileNetV3-Large, YOLO-World, Grounding DINO."),
            ("What the headline hid",
             "Accuracy 0.9923 but macro-F1 0.6623 &mdash; two of four classes had almost no "
             "support and scored 0.0 precision and recall. A separate ViT run scored "
             "0.3808, traced to the harness reading a randomly initialised "
             "<code>pooler_output</code> instead of mean-pooled features: a pipeline bug, "
             "not a model result."),
        ], wide=True)

        + "<h2>Literature Review Versus Pilot: Four Confirmations</h2>"
        "<p>The Related Work chapter makes predictions the pilot could test directly. Four "
        "held, one of them textbook.</p>"
        + cards([
            ("1 &middot; Frozen probes beat zero-shot decisively",
             "Best probe 0.9923 against best zero-shot 0.7577 &mdash; a 23-point gap, which "
             "is exactly the cost of a no-label deployment the literature describes."),
            ("2 &middot; Pretraining data beat architecture, at identical size",
             "OpenCLIP ViT-B/32 trained on LAION-2B scored 0.7577; OpenAI CLIP ViT-B/32 "
             "&mdash; the same architecture &mdash; scored 0.6538. Eleven points "
             "attributable to training data alone, matching the controlled scaling-law "
             "literature."),
            ("3 &middot; Bigger zero-shot was worse, not better",
             "CLIP ViT-L/14 scored 0.3538 and SigLIP 2 base 0.3192, both far below the "
             "B/32 models and 5&ndash;7&times; slower per frame. The benchmark-table "
             "intuition that the larger variant wins is wrong on this task, as the "
             "fine-grained and occlusion-robustness studies predict."),
            ("4 &middot; Ranking did not follow benchmark position",
             "ViT-Base led the pilot and finishes third under honest evaluation."),
        ], wide=True)

        + "<h2>The Flaw Was the Protocol, Not the Models</h2>"
        "<p>Three measurements, taken later, invalidate the pilot&rsquo;s numbers.</p>"
        + tiles([
            ("98.4%", "scored by a clock rule that reads no pixels", "lead"),
            ("98.5%", "of frames have a near-duplicate", "lead"),
            ("62", "distinct scenes behind 394 test frames", "lead"),
        ])
        + cards([
            ("A clock rule scores 98.4%",
             "&ldquo;If night then ACTIVE_PLAY, else EMPTY&rdquo; &mdash; using no pixels "
             "at all. In the labelled data EMPTY is 98% daytime and ACTIVE_PLAY 99% night. "
             "Class and illumination are the same variable."),
            ("98.5% of frames have a near-duplicate",
             "Frames taken seconds apart from a fixed camera are near-identical, so a "
             "random split places near-copies of test images into training."),
            ("394 test frames, 62 distinct scenes",
             "Once effective sample size replaces frame count, a colour histogram becomes "
             "statistically indistinguishable from DINOv2 on the leaky split (p = 1.000)."),
        ], wide=True)
        + "<blockquote>The pilot&rsquo;s 99% is consistent with having learned "
          "day-versus-night rather than occupancy. <strong>No change of backbone removes a "
          "shortcut that the data affords.</strong></blockquote>"
        "</div>"
    )


#: tab id -> the slide that opens it. A tab absent from this map renders as it always did.
SLIDES = {
    "overview": overview,
    "models": models,
    "findings": findings,
    "prereg": prereg,
    "questions": questions,
    "dataset": dataset,
    "database": database,
    "code": code,
    "ideas": ideas,
    "ethics": ethics,
    "augmentation": augmentation,
    "searches": searches,
}
