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


#: One column of the table: heading, what the number is, and which direction is better.
#: Kept beside the markup rather than in prose above it - a legend a reader has to hold in
#: their head while scanning a row is a legend they do not read.
COLUMNS = (
    ("Random split", "macro-F1, frames shuffled &mdash; the leaky protocol", "up"),
    ("Grouped split", "macro-F1, no venue in both train and test", "up"),
    ("Cross-venue recall", "playing frames found at an unseen venue", "up"),
    ("Worst fold", "the lowest single fold behind that recall", "up"),
    ("False-play", "held-out empty pitches called a match", "down"),
    ("Balanced", "recall minus false-play", "up"),
    ("ms/frame", "time to score one frame", "down"),
)


def _head() -> str:
    cells = "".join(
        f'<th class="num"><div class="th-n">{name}</div>'
        f'<div class="th-w">{what}</div>'
        f'<div class="th-d {d}">{"&uarr; higher" if d == "up" else "&darr; lower"} '
        f'is better</div></th>'
        for name, what, d in COLUMNS
    )
    return f"<thead><tr><th>Model</th>{cells}</tr></thead>"


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

    # --- the comparison table: one plain number per cell ---
    body = ""
    for r in rows:
        body += f"""<tr>
          <td><div class="mn">{r['label']}</div><div class="mt">{r['nature']}</div></td>
          <td class="num">{_num(r['random'])}</td>
          <td class="num">{_num(r['grouped'])}</td>
          <td class="num">{_num(r['cross'])}</td>
          <td class="num">{_num(r['worst'])}</td>
          <td class="num {'bad' if r.get('false_play') and float(r['false_play']) > 0.5 else ''}">{_num(r['false_play'])}</td>
          <td class="num">{_num(_balanced(r))}</td>
          <td class="num">{_num(r['ms'], 0)}</td>
        </tr>"""

    return f"""
<h1>Which model, and why</h1>
<div class="scroll"><table>
  {_head()}
  <tbody>{body}</tbody>
</table></div>
"""


STYLES = """
td.bad{color:var(--warn);font-weight:600}
.mn{font-weight:600;color:var(--ink)}
.mt{font-size:12px;color:var(--ink-3);margin-top:1px}
.th-n{font-weight:600;color:var(--ink)}
.th-w{font-weight:400;font-size:11.5px;color:var(--ink-3);margin-top:3px;
 max-width:15ch;white-space:normal;line-height:1.35}
.th-d{font:500 10.5px 'JetBrains Mono',monospace;margin-top:4px;white-space:nowrap}
.th-d.up{color:var(--up,#2c7a52)} .th-d.down{color:var(--down,#a8512f)}
"""
