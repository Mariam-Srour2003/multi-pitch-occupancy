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
    return {"rows": rows}


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

    return f"""
<h1>Which model, and why</h1>
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
"""


STYLES = """
td.bad{color:var(--warn);font-weight:600}
.mbar{display:inline-block;width:96px;height:8px;border-radius:2px;background:var(--surface-2);
 vertical-align:middle;overflow:hidden}
.mbar i{display:block;height:100%;border-radius:2px}
.t-accent{background:var(--accent)} .t-muted{background:var(--ink-3)}
.cell{white-space:nowrap}
.cv{font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;font-size:13px;
 margin-left:9px;color:var(--ink);font-weight:600}
.ci{display:block;font:500 10.5px 'JetBrains Mono',monospace;color:var(--ink-3);
 letter-spacing:-.01em;margin-top:2px;white-space:nowrap}
.mn{font-weight:600;color:var(--ink)}
.mt{font-size:12px;color:var(--ink-3);margin-top:1px}
"""
