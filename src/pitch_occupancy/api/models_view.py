"""The model comparison view (RQ2).

A table of numbers does not answer "which one should I use". This view is built around the
decision instead: the recommendation first, the evidence behind it second, and the
disagreement between protocols made visible rather than left for the reader to reconstruct
from three separate tables.

Everything is read from the result CSVs, so a rerun changes the page. Where a result is
missing the cell reads as absent rather than as a stale number.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"

LABELS = {
    "dinov2": "DINOv2", "convnextv2": "ConvNeXtV2", "vit": "ViT",
    "clock_rule": "Clock rule", "cheap_histogram": "Colour histogram",
    "cheap_intensity": "Mean intensity", "majority": "Majority class",
}
#: What each model is, in one line - the thing a table of numbers cannot say.
NATURE = {
    "dinov2": "self-supervised, no labels in pretraining",
    "convnextv2": "self-supervised convnet, smallest and fastest",
    "vit": "supervised on ImageNet labels",
    "clock_rule": "reads the clock, never the pixels",
}
TRAINED = ("dinov2", "convnextv2", "vit")


def _csv(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(v, d=3) -> str:
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def _ci(lo, hi, d: int = 3) -> str:
    """A 95% interval, or nothing when the source did not carry one.

    Rendered small and beside its estimate rather than in its own column: the point is that
    the reader cannot see the estimate without seeing its width.
    """
    try:
        return f'<span class="ci">[{float(lo):.{d}f}, {float(hi):.{d}f}]</span>'
    except (TypeError, ValueError):
        return ""


def _overlap(a: dict, b: dict, lo: str, hi: str) -> bool:
    """Do two intervals overlap? Used to stop "leads" being claimed on point estimates."""
    try:
        a_lo, a_hi = float(a[lo]), float(a[hi])
        b_lo, b_hi = float(b[lo]), float(b[hi])
    except (KeyError, TypeError, ValueError):
        return False
    return a_lo <= b_hi and b_lo <= a_hi


def _bar(value: float | None, *, lo: float = 0.0, hi: float = 1.0, tone: str = "accent") -> str:
    """A bar on a shared scale, so two rows are comparable by length alone."""
    if value is None:
        return '<span class="mbar"></span>'
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    return (f'<span class="mbar"><i class="t-{tone}" style="width:{frac * 100:.1f}%"></i></span>')


def collect() -> dict:
    h1h2 = _csv("h1_h2_baseline_floor.csv")
    h3 = {r["model"]: r for r in _csv("h3_cross_venue_recall.csv")
          if r["held_out_venue"] == "MEAN_ACROSS_FOLDS"}
    sens = {r["model"]: r for r in _csv("h3_sensitivity_merged_venues.csv")
            if r["held_out_venue"] == "MEAN_ACROSS_FOLDS"}
    lat = {r["backbone"]: r for r in _csv("efficiency_latency.csv")}
    # The column that reverses the ranking. Cross-venue recall is measured on folds that are
    # 100% ACTIVE_PLAY, so it can be earned by answering "playing" to everything - and two of
    # the three backbones very nearly do.
    fp = {r["model"]: r for r in _csv("h3_with_false_play.csv")
          if r["held_out_venue"] == "MEAN_ACROSS_FOLDS"}
    grouped = {r["model"]: r for r in h1h2 if r["split"].startswith("grouped")}
    random_ = {r["model"]: r for r in h1h2 if r["split"].startswith("random")}

    best_prompt = {}
    p = RESULTS / "prompt_search_best.json"
    if p.exists():
        best_prompt = json.loads(p.read_text(encoding="utf-8")).get("best", {})

    rows = []
    for key in (*TRAINED, "clock_rule"):
        cross = h3.get(key, {})
        rows.append({
            "key": key, "label": LABELS[key], "nature": NATURE.get(key, ""),
            "random": random_.get(key, {}).get("macro_f1"),
            "grouped": grouped.get(key, {}).get("macro_f1"),
            "cross": cross.get("play_recall"),
            # The intervals were loaded and thrown away. `preregistration.md` standing rule
            # 3 is "uncertainty on every number", and this is the page that makes the
            # production recommendation - the one place a bare point estimate does the most
            # damage. They also change what the table says: DINOv2's cross-venue interval
            # and ConvNeXtV2's overlap across most of their width.
            "cross_lo": cross.get("ci_low"),
            "cross_hi": cross.get("ci_high"),
            "worst": cross.get("worst_fold"),
            "merged": sens.get(key, {}).get("play_recall"),
            "merged_lo": sens.get(key, {}).get("ci_low"),
            "merged_hi": sens.get(key, {}).get("ci_high"),
            "random_lo": random_.get(key, {}).get("macro_f1_lo"),
            "random_hi": random_.get(key, {}).get("macro_f1_hi"),
            "grouped_lo": grouped.get(key, {}).get("macro_f1_lo"),
            "grouped_hi": grouped.get(key, {}).get("macro_f1_hi"),
            "false_play": fp.get(key, {}).get("false_play_rate"),
            "ms": lat.get(key, {}).get("single_median_ms"),
            "conc": lat.get(key, {}).get("round_wall_s"),
        })
    return {"rows": rows, "best_prompt": best_prompt}


def _rank(rows: list[dict], field: str, *, lower_is_better: bool = False) -> dict[str, int]:
    ranked = sorted(
        (r for r in rows if r["key"] in TRAINED and r.get(field)),
        key=lambda r: (1 if lower_is_better else -1) * float(r[field]),
    )
    return {r["key"]: i + 1 for i, r in enumerate(ranked)}


def _balanced(row: dict) -> float | None:
    """Recall minus false-play.

    Indicative rather than a single coherent metric, and the page says so: the two terms come
    from different fits, because no cross-venue fold contains a single EMPTY frame to measure
    false-play on. What makes the comparison fair is that every model faces the same pair.
    """
    if not row.get("cross") or row.get("false_play") in (None, ""):
        return None
    return float(row["cross"]) - float(row["false_play"])


def render() -> str:
    d = collect()
    rows = d["rows"]
    # A row exists for every known model whether or not it has been evaluated, so emptiness
    # has to be judged on the values. Without this the view renders a full table of dashes,
    # which reads as "measured and inconclusive" rather than "not run yet".
    trained = [
        r for r in rows
        if r["key"] in TRAINED and any(r.get(k) for k in ("random", "grouped", "cross"))
    ]
    if not trained:
        return (
            "<h1>Which model, and why</h1>"
            "<p class='missing'>No model results yet. Run "
            "<code>uv run python experiments/reproduce_all.py</code>.</p>"
        )

    ranks = {
        "random split (leaky)": _rank(rows, "random"),
        "grouped split": _rank(rows, "grouped"),
        "cross-venue recall": _rank(rows, "cross"),
        "false-play (lower is better)": _rank(rows, "false_play", lower_is_better=True),
    }
    winner = max(
        (r for r in trained if r.get("cross")), key=lambda r: float(r["cross"]), default=None
    )
    leaky_winner = max(
        (r for r in trained if r.get("random")), key=lambda r: float(r["random"]), default=None
    )
    fastest = min(
        (r for r in trained if r.get("ms")), key=lambda r: float(r["ms"]), default=None
    )
    inverted = [k for k in TRAINED
                if ranks["random split (leaky)"].get(k) == 1
                and ranks["cross-venue recall"].get(k) == len(TRAINED)]

    # --- the recommendation ---
    bp = d["best_prompt"]
    zero_recall = bp.get("play_recall")
    zero_line = ""
    if zero_recall and winner and float(zero_recall) > float(winner["cross"]):
        zero_line = (
            f"<p><b>And a zero-shot prompt beats all of them</b> at "
            f"{_num(zero_recall)} with no labels at all &mdash; a development result, since "
            f"the prompt was chosen on the same venues it is scored on, but it reframes what "
            f"onboarding a new site costs.</p>"
        )

    rec = ""
    if winner and fastest:
        same = winner["key"] == fastest["key"]
        # "Leads" was asserted from point estimates alone. The runner-up's interval overlaps
        # the winner's across most of its width, and a paired test on the grouped split
        # returns *inconclusive* (H4, macro-F1 difference CI [-0.231, +0.003]). The
        # recommendation stands - something has to be deployed, and this is the best
        # estimate - but it must not read as an established gap.
        rivals = [
            r for r in rows
            if r["key"] in TRAINED and r["key"] != winner["key"]
            and _overlap(winner, r, "cross_lo", "cross_hi")
        ]
        overlap_note = ""
        if rivals:
            names = " and ".join(r["label"] for r in rivals)
            overlap_note = (
                f"<p class=\"vcav\"><b>The lead is not established by the intervals.</b> "
                f"{names} overlap{'s' if len(rivals) == 1 else ''} this one's 95% interval, "
                f"and a paired test on the grouped split returns <i>inconclusive</i> rather "
                f"than a difference. This is the best available estimate and something has "
                f"to be deployed &mdash; it is not a measured gap.</p>"
            )
        rec = f"""
        <div class="verdict">
          <div class="vk">Use this one</div>
          <div class="vname">{winner['label']}</div>
          <p>Leads the honest protocols at <b>{_num(winner['cross'])}</b>
          {_ci(winner.get('cross_lo'), winner.get('cross_hi'))} cross-venue recall
          and holds <b>{_num(winner['merged'])}</b>
          {_ci(winner.get('merged_lo'), winner.get('merged_hi'))} when the two audited
          venues are merged.
          {"It is also the fastest." if same else
           f"It is the slowest of the three at {_num(winner['ms'], 0)} ms/frame &mdash; which "
           f"does not matter: 20 cameras take {_num(winner['conc'], 1)} s of a 60-second cycle, "
           f"so latency is not the binding constraint and the choice falls to accuracy."}</p>
          {overlap_note}
          {zero_line}
          <p class="vfall">Fall back to <b>{fastest['label']}</b> only if the target hardware
          proves far slower than the development machine.</p>
        </div>"""

    # --- rank strip: the inversion, seen rather than described ---
    strips = ""
    for protocol, rank in ranks.items():
        cells = "".join(
            f'<div class="rc r{rank.get(k, 0)}"><span class="rn">{rank.get(k, "—")}</span>'
            f'<span class="rl">{LABELS[k]}</span></div>'
            for k in sorted(TRAINED, key=lambda k: rank.get(k, 9))
        )
        strips += f'<div class="strip"><div class="sp">{protocol}</div>{cells}</div>'

    inversion_note = ""
    if inverted:
        name = LABELS[inverted[0]]
        inversion_note = (
            f'<div class="callout warn"><p><b>{name} is first under the leaky protocol and '
            f'last under the honest one.</b> The evaluation does not merely deflate scores, '
            f'it reverses the decision: anyone following the pilot&rsquo;s protocol would have '
            f'shipped the weakest generaliser of the three.</p></div>'
        )

    # --- the comparison table, with bars on a shared scale ---
    body = ""
    for r in rows:
        cross = float(r["cross"]) if r.get("cross") else None
        tone = "accent" if r["key"] in TRAINED else "muted"
        body += f"""<tr>
          <td><div class="mn">{r['label']}</div><div class="mt">{r['nature']}</div></td>
          <td class="num">{_num(r['random'])}{_ci(r.get('random_lo'), r.get('random_hi'))}</td>
          <td class="num">{_num(r['grouped'])}{_ci(r.get('grouped_lo'), r.get('grouped_hi'))}</td>
          <td class="cell">{_bar(cross, lo=0.0, hi=1.0, tone=tone)}
            <span class="cv">{_num(r['cross'])}</span>
            {_ci(r.get('cross_lo'), r.get('cross_hi'))}</td>
          <td class="num">{_num(r['worst'])}</td>
          <td class="num {'bad' if r.get('false_play') and float(r['false_play']) > 0.5 else ''}">{_num(r['false_play'])}</td>
          <td class="num">{_num(_balanced(r))}</td>
          <td class="num">{_num(r['ms'], 0)}</td>
        </tr>"""

    # --- the false-play inversion ---
    worst_fp = max(
        (r for r in trained if r.get("false_play")),
        key=lambda r: float(r["false_play"]), default=None,
    )
    clock = next((r for r in rows if r["key"] == "clock_rule"), {})
    false_play_note = ""
    if worst_fp and clock.get("false_play"):
        best_bal = max(
            (r for r in trained if _balanced(r) is not None), key=_balanced, default=None
        )
        false_play_note = f"""<div class="callout warn">
        <p><b>Cross-venue recall is measured on folds that contain no empty pitch at all</b>,
        so it can be earned by answering &ldquo;playing&rdquo; to everything &mdash; and
        {worst_fp['label']} very nearly does: it calls
        <b>{float(worst_fp['false_play']):.1%}</b> of held-out empty frames a match. The
        clock rule, the straw man that never looks at the image, calls
        <b>{float(clock['false_play']):.1%}</b>.</p>
        <p>Ranked on recall minus false-play the table inverts, and
        <b>{best_bal['label']}</b> is the only backbone that survives it. The two terms come
        from different fits &mdash; no cross-venue fold holds a single empty frame to measure
        against &mdash; so read them side by side rather than as one number; every model
        faces the identical pair.</p>
        <p><b>This is observed behaviour on one pitch, not an established ranking.</b> The
        243 held-out empty frames are consecutive views of a single camera and amount to
        three to ten <i>distinct scenes</i>; once that is accounted for, none of the six
        pairwise comparisons survives a Holm correction. The gap is large enough that the
        failure is a lack of power rather than evidence of equivalence &mdash; but settling
        it needs empty footage from more than one scene.</p></div>"""

    # --- latency against the budget ---
    budget = 60.0
    lat_rows = "".join(
        f"""<tr><td class="mn">{r['label']}</td>
        <td class="cell"><span class="mbar wide"><i class="t-ok"
          style="width:{min(float(r['conc']) / budget, 1) * 100:.1f}%"></i></span>
          <span class="cv">{_num(r['conc'], 1)}s</span></td>
        <td class="num">{_num(float(r['conc']) and budget / float(r['conc']), 1)}x</td></tr>"""
        for r in trained if r.get("conc")
    )

    return f"""
<h1>Which model, and why</h1>
<p>Four candidates under three evaluation protocols. The protocols disagree, so the order
they are read in decides the answer.</p>

{rec}

<h2>The protocols rank them differently</h2>
<div class="strips">{strips}</div>
{inversion_note}

{false_play_note}

<h2>Every score</h2>
<p>Macro-F1 for the split protocols, play recall for cross-venue. Bars share one scale, so
lengths are comparable down the column. <b>False-play</b> is how often a model calls a
held-out empty pitch a match, and <b>balanced</b> is recall minus that.</p>
<div class="scroll"><table>
  <thead><tr><th>Model</th><th class="num">Random split</th><th class="num">Grouped split</th>
  <th>Cross-venue recall</th><th class="num">Worst fold</th>
  <th class="num">False-play</th><th class="num">Balanced</th>
  <th class="num">ms/frame</th></tr></thead>
  <tbody>{body}</tbody>
</table></div>
<p class="foot">The clock rule uses no image data. It is competitive under both split
protocols and collapses across venues &mdash; which is how we know the split protocols were
measuring the dataset rather than the models.</p>

<h2>Latency is not the constraint</h2>
<p>Time for 20 cameras against the 60-second sampling cycle. Every model finishes with room
to spare, which is why the recommendation is decided on accuracy.</p>
<div class="scroll"><table>
  <thead><tr><th>Model</th><th>20 cameras, of a 60s cycle</th><th class="num">Headroom</th></tr></thead>
  <tbody>{lat_rows}</tbody>
</table></div>
<p class="foot">Measured on a 4-thread development laptop, not the target Mini-PC. Nothing
here settles the deployment claim.</p>
"""


STYLES = """
td.bad{color:var(--warn);font-weight:600}
.verdict{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--accent);
 border-radius:11px;padding:19px 22px;margin:20px 0}
.verdict .vk{font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.12em;
 text-transform:uppercase;color:var(--accent);margin-bottom:4px}
.verdict .vname{font-size:27px;font-weight:700;letter-spacing:-.02em;margin-bottom:8px}
.verdict p{margin:0 0 10px;max-width:72ch}
.verdict .vfall{color:var(--ink-3);font-size:13.5px;margin:0}
.strips{display:flex;flex-direction:column;gap:9px;margin:16px 0}
.strip{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.strip .sp{font:500 11px 'JetBrains Mono',monospace;color:var(--ink-3);
 width:170px;flex:none;text-transform:uppercase;letter-spacing:.07em}
.rc{display:flex;align-items:center;gap:7px;padding:6px 12px;border-radius:8px;
 background:var(--surface-2);border:1px solid var(--line)}
.rc .rn{font:700 12px 'JetBrains Mono',monospace;color:var(--ink-3)}
.rc .rl{font-size:13px;font-weight:600}
.rc.r1{background:var(--accent-soft);border-color:transparent}
.rc.r1 .rn,.rc.r1 .rl{color:var(--accent)}
.mbar{display:inline-block;width:96px;height:8px;border-radius:2px;background:var(--surface-2);
 vertical-align:middle;overflow:hidden}
.mbar.wide{width:180px}
.mbar i{display:block;height:100%;border-radius:2px}
.t-accent{background:var(--accent)} .t-muted{background:var(--ink-3)}
.t-ok{background:var(--up,#2c7a52)}
.cell{white-space:nowrap}
.cv{font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;font-size:13px;
.ci{display:block;font:500 10.5px 'JetBrains Mono',monospace;color:var(--ink-3);
 letter-spacing:-.01em;margin-top:2px;white-space:nowrap}
.vcav{font-size:13px;color:var(--ink-2);margin-top:10px}
 margin-left:9px;color:var(--ink);font-weight:600}
.mn{font-weight:600;color:var(--ink)}
.mt{font-size:12px;color:var(--ink-3);margin-top:1px}
.foot{font-size:13px;color:var(--ink-3);max-width:72ch}
.callout{border-radius:10px;padding:14px 18px;margin:14px 0;background:var(--surface-2);
 max-width:74ch}
.callout.warn{background:var(--warn-soft)}
.callout p{margin:0;color:var(--ink-2)}
.callout b{color:var(--ink)}
"""
