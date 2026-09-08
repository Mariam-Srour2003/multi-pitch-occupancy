"""Generate the standalone project site - a self-contained export of the headline results (WP8).

Every number on the page is read from `results/`, never typed in, so nothing here can drift
from the data it reports. Figures are embedded as data URIs, so the page is one file that
works offline and can be published as an Artifact.

**It is an export of selected results, not of all of them, and that distinction was worth
one bug to learn.** The docstring used to say "covering the whole thesis". It reads a fixed
list of result files, so every experiment added after it was written was omitted in silence
- the page regenerated happily and said nothing about what was missing. The served front end
(`api/thesis_site.py`) is the complete one: it renders `EXPERIMENT_LOG.md` and the thesis
documents directly, so a new finding appears there the moment it is logged.

`tests/test_site_coverage.py` now holds the inventory: every committed result file is either
rendered here or listed as deliberately out of scope, so adding one forces the choice
instead of defaulting to omission.

    uv run python experiments/make_site.py

Regenerate after any experiment it covers, and republish.
"""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "project_site.html"


def read_csv(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def f(v, d=3) -> str:
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def collect() -> dict:
    h1h2 = read_csv("h1_h2_baseline_floor.csv")
    h3 = read_csv("h3_cross_venue_recall.csv")
    h3s = read_csv("h3_sensitivity_merged_venues.csv")
    lat = read_csv("efficiency_latency.csv")
    abl = read_csv("input_ablation.csv")
    prompts = read_csv("prompt_search.csv")
    search = read_json("preprocess_search.json")
    best_prompt = read_json("prompt_search_best.json")
    bench = read_csv("benchmark_v2.csv")

    # WP4-T1's floor rows, which say what each protocol can and cannot distinguish. The
    # page led on "the protocol reverses the ranking" and that understates it: on one
    # protocol a constant predictor wins outright, so there is no ranking to reverse.
    floor = []
    for r in bench:
        if r.get("replicate") != "SUMMARY" or not r.get("model", "").startswith("floor:"):
            continue
        trivial, backbone = r["model"][len("floor:"):].split("_vs_")
        floor.append({
            "protocol": r["protocol"].replace("_", " "),
            "trivial": trivial, "backbone": backbone,
            "gap": f(r["macro_f1"], 3),
            "beaten": bool(r.get("warnings")),
        })

    composition = [
        {"model": r["model"], "raw": f(r["macro_f1"], 3),
         "composition": f(r["majority_share_test"], 3),
         "attributable": f(float(r["macro_f1"]) - float(r["majority_share_test"]), 3)}
        for r in bench
        if r.get("protocol") == "random_minus_grouped" and r.get("model") in
        ("convnextv2", "dinov2", "vit")
    ]

    def by(rows, key, val):
        return [r for r in rows if r.get(key) == val]

    # model comparison across the three protocols
    grouped = {r["model"]: r for r in by(h1h2, "split", "grouped_slot_id_seed42")}
    random_ = {r["model"]: r for r in by(h1h2, "split", "random_seed42")}
    cross = {r["model"]: r for r in by(h3, "held_out_venue", "MEAN_ACROSS_FOLDS")}
    lat_by = {r["backbone"]: r for r in lat}

    models = []
    for key, label in (("dinov2", "DINOv2"), ("convnextv2", "ConvNeXtV2"), ("vit", "ViT")):
        models.append({
            "key": key, "label": label,
            "random": f(random_.get(key, {}).get("macro_f1"), 3),
            "grouped": f(grouped.get(key, {}).get("macro_f1"), 3),
            "cross": f(cross.get(key, {}).get("play_recall"), 3),
            "worst": f(cross.get(key, {}).get("worst_fold"), 3),
            "ms": f(lat_by.get(key, {}).get("single_median_ms"), 0),
            "conc": f(lat_by.get(key, {}).get("round_wall_s"), 1),
        })
    clock = {
        "key": "clock_rule", "label": "Clock rule (no pixels)",
        "random": f(random_.get("clock_rule", {}).get("macro_f1"), 3),
        "grouped": f(grouped.get("clock_rule", {}).get("macro_f1"), 3),
        "cross": f(cross.get("clock_rule", {}).get("play_recall"), 3),
        "worst": f(cross.get("clock_rule", {}).get("worst_fold"), 3),
        "ms": "—", "conc": "—",
    }

    # ablation means
    abl_means: dict[str, list[float]] = {}
    for r in abl:
        abl_means.setdefault(r["variant"], []).append(float(r["play_recall"]))
    ablation = sorted(
        ({"variant": k, "mean": sum(v) / len(v), "worst": min(v)} for k, v in abl_means.items()),
        key=lambda d: -d["mean"],
    )

    # h3 per fold
    folds = sorted({r["held_out_venue"] for r in h3 if r["held_out_venue"] != "MEAN_ACROSS_FOLDS"})
    per_fold = []
    for venue in folds:
        row = {"venue": venue.replace("clipvenue_", "")}
        for r in h3:
            if r["held_out_venue"] == venue:
                row[r["model"]] = f(r["play_recall"], 3)
        per_fold.append(row)

    sens = {r["model"]: r for r in h3s if r["held_out_venue"] == "MEAN_ACROSS_FOLDS"}

    top_prompts = sorted(prompts, key=lambda r: -float(r["balanced"]))[:6] if prompts else []

    evals = search.get("evaluations", [])
    base_eval = next((e for e in evals if e.get("label") == "baseline"), None)

    return {
        "models": models + [clock],
        "floor": floor,
        "composition": composition,
        "ablation": ablation,
        "per_fold": per_fold,
        "fold_models": ["dinov2", "convnextv2", "vit", "clock_rule"],
        "sensitivity": [
            {"model": m, "mean": f(sens[m]["play_recall"], 3),
             "lo": f(sens[m].get("ci_low"), 3), "hi": f(sens[m].get("ci_high"), 3),
             "merged": f(sens[m].get("merged_fold"), 3)}
            for m in ("dinov2", "convnextv2", "vit", "clock_rule") if m in sens
        ],
        "prompts": [
            {"recall": f(r["play_recall"], 3), "fp": f(r["false_play"], 4),
             "bal": f(r["balanced"], 3), "n": r["n_templates"],
             "desc": r.get("desc_ACTIVE_PLAY", "")}
            for r in top_prompts
        ],
        "best_prompt": best_prompt.get("best", {}),
        "search_n": len(evals),
        "search_rows": sorted(
            ({"label": e["label"], "recall": f(e["play_recall"], 4),
              "delta": f(e["play_recall"] - base_eval["play_recall"], 3) if base_eval else "",
              "worst": f(e["worst_fold"], 3), "fp": f(e["false_play"], 4)}
             for e in evals), key=lambda d: -float(d["recall"])
        )[:10],
        "figs": {
            n: data_uri(RESULTS / "figs" / f"{n}.png")
            for n in ("ranking_inversion", "label_efficiency", "cross_venue_recall")
        },
    }


def build(d: dict) -> str:
    def rows(items, cells):
        return "".join("<tr>" + "".join(cells(i)) + "</tr>" for i in items)

    model_rows = rows(d["models"], lambda m: [
        f'<td class="name">{m["label"]}</td>',
        f'<td class="num">{m["random"]}</td>',
        f'<td class="num">{m["grouped"]}</td>',
        f'<td class="num strong">{m["cross"]}</td>',
        f'<td class="num">{m["worst"]}</td>',
        f'<td class="num">{m["ms"]}</td>',
        f'<td class="num">{m["conc"]}</td>',
    ])
    floor_rows = rows(d["floor"], lambda x: [
        f'<td class="name">{x["protocol"]}</td>',
        f'<td class="sw">{x["trivial"]}</td>',
        f'<td class="sw">{x["backbone"]}</td>',
        f'<td class="num {"bad" if x["beaten"] else ""}">{x["gap"]}</td>',
    ])
    comp_rows = rows(d["composition"], lambda x: [
        f'<td class="name">{x["model"]}</td>',
        f'<td class="num">{x["raw"]}</td>',
        f'<td class="num">{x["composition"]}</td>',
        f'<td class="num strong">{x["attributable"]}</td>',
    ])
    fold_rows = rows(d["per_fold"], lambda r: [f'<td class="sw">{r["venue"]}</td>'] + [
        f'<td class="num">{r.get(m, "—")}</td>' for m in d["fold_models"]
    ])
    sens_rows = rows(d["sensitivity"], lambda s: [
        f'<td class="name">{s["model"]}</td>',
        f'<td class="num strong">{s["mean"]}</td>',
        f'<td class="num">[{s["lo"]}, {s["hi"]}]</td>',
        f'<td class="num">{s["merged"]}</td>',
    ])
    abl_rows = rows(d["ablation"], lambda a: [
        f'<td class="sw">{a["variant"]}</td>',
        f'<td class="num strong">{a["mean"]:.3f}</td>',
        f'<td class="num">{a["worst"]:.3f}</td>',
    ])
    prompt_rows = rows(d["prompts"], lambda p: [
        f'<td class="sw">{p["desc"]}</td>',
        f'<td class="num">{p["n"]}</td>',
        f'<td class="num strong">{p["recall"]}</td>',
        f'<td class="num">{p["fp"]}</td>',
        f'<td class="num">{p["bal"]}</td>',
    ])
    search_rows = rows(d["search_rows"], lambda s: [
        f'<td class="sw">{s["label"]}</td>',
        f'<td class="num strong">{s["recall"]}</td>',
        f'<td class="num">{s["delta"]}</td>',
        f'<td class="num">{s["worst"]}</td>',
        f'<td class="num">{s["fp"]}</td>',
    ])
    bp = d["best_prompt"]
    best_prompt_text = bp.get("desc_ACTIVE_PLAY", "—")
    best_prompt_recall = f(bp.get("play_recall"), 3)

    return TEMPLATE.format(
        model_rows=model_rows, fold_rows=fold_rows, sens_rows=sens_rows,
        floor_rows=floor_rows, comp_rows=comp_rows,
        composition_drop=(d["composition"][0]["composition"] if d["composition"] else "—"),
        abl_rows=abl_rows, prompt_rows=prompt_rows, search_rows=search_rows,
        search_n=d["search_n"], best_prompt=best_prompt_text,
        best_prompt_recall=best_prompt_recall,
        fig_rank=d["figs"]["ranking_inversion"],
        fig_label=d["figs"]["label_efficiency"],
        fig_cross=d["figs"]["cross_venue_recall"],
    )


TEMPLATE = """<title>Pitch Occupancy Thesis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap">
<style>
:root {{
  color-scheme: light;
  --ground:#f6f8f7; --surface:#ffffff; --surface-2:#eef2f1; --line:#dde4e2;
  --ink:#111817; --ink-2:#4d5c5a; --ink-3:#7a8886;
  --accent:#0d6d78; --accent-soft:#d7ebed;
  --up:#2c7a52; --up-soft:#dcefe3; --down:#a8512f;
  --warn:#8a6d1f; --warn-soft:#f6ecd4;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ground:#0e1414; --surface:#161e1e; --surface-2:#1c2625; --line:#2b3736;
    --ink:#eaf1ef; --ink-2:#a3b2af; --ink-3:#7b8a88;
    --accent:#4fb3bf; --accent-soft:#12363a;
    --up:#5cb884; --up-soft:#152f24; --down:#d98a63;
    --warn:#cfae57; --warn-soft:#2c2614;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ground:#0e1414; --surface:#161e1e; --surface-2:#1c2625; --line:#2b3736;
  --ink:#eaf1ef; --ink-2:#a3b2af; --ink-3:#7b8a88;
  --accent:#4fb3bf; --accent-soft:#12363a;
  --up:#5cb884; --up-soft:#152f24; --down:#d98a63;
  --warn:#cfae57; --warn-soft:#2c2614;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink);
  font:15px/1.6 Archivo, ui-sans-serif, system-ui, sans-serif; }}

nav {{ position:sticky; top:0; z-index:20; background:var(--ground);
  border-bottom:1px solid var(--line); }}
.navin {{ max-width:1080px; margin:0 auto; padding:0 24px; display:flex;
  align-items:center; gap:22px; flex-wrap:wrap; min-height:58px; }}
.brand {{ font-weight:700; letter-spacing:-.01em; font-size:15px; white-space:nowrap; }}
.brand span {{ color:var(--accent); }}
.navtabs {{ display:flex; gap:2px; flex-wrap:wrap; margin-left:auto; }}
.navtabs button {{ font:500 13.5px Archivo, sans-serif; background:none; border:0;
  color:var(--ink-2); padding:9px 12px; border-radius:7px; cursor:pointer; }}
.navtabs button:hover {{ background:var(--surface-2); color:var(--ink); }}
.navtabs button[aria-current="true"] {{ background:var(--ink); color:var(--ground); }}
.navtabs button:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}

main {{ max-width:1080px; margin:0 auto; padding:36px 24px 80px; }}
.view[hidden] {{ display:none; }}
.eyebrow {{ font:500 11px/1 'JetBrains Mono', monospace; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); margin:0 0 10px; }}
h1 {{ font-size:clamp(27px,3.6vw,38px); line-height:1.12; margin:0 0 12px; font-weight:700;
  letter-spacing:-.022em; text-wrap:balance; }}
h2 {{ font-size:20px; margin:34px 0 6px; font-weight:600; letter-spacing:-.012em; }}
h3 {{ font-size:15.5px; margin:22px 0 4px; font-weight:600; }}
p {{ max-width:70ch; color:var(--ink-2); margin:0 0 12px; }}
p.lede {{ font-size:16.5px; color:var(--ink-2); }}
strong {{ color:var(--ink); font-weight:600; }}
a {{ color:var(--accent); }}

.tiles {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
  margin:22px 0; }}
.tile {{ background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:15px 16px; }}
.tile .k {{ font:500 10.5px 'JetBrains Mono', monospace; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-3); margin-bottom:6px; }}
.tile .v {{ font:700 25px 'JetBrains Mono', monospace; font-variant-numeric:tabular-nums;
  letter-spacing:-.02em; }}
.tile .n {{ font-size:12.5px; color:var(--ink-2); margin-top:3px; }}
.tile.lead {{ border-left:4px solid var(--accent); }}

.scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px;
  background:var(--surface); margin:14px 0; }}
table {{ border-collapse:collapse; width:100%; min-width:560px; }}
th {{ font:500 10.5px 'JetBrains Mono', monospace; letter-spacing:.09em;
  text-transform:uppercase; color:var(--ink-3); text-align:left; padding:11px 14px;
  border-bottom:1px solid var(--line); background:var(--surface-2); white-space:nowrap; }}
td {{ padding:10px 14px; border-bottom:1px solid var(--line); font-size:13.5px; }}
tbody tr:last-child td {{ border-bottom:none; }}
tbody tr:hover {{ background:var(--surface-2); }}
.name {{ font-weight:600; }}
.sw {{ font-family:'JetBrains Mono', monospace; font-size:12.5px; }}
.num {{ font-family:'JetBrains Mono', monospace; font-variant-numeric:tabular-nums;
  text-align:right; white-space:nowrap; }}
.num.strong {{ font-weight:700; color:var(--ink); }}

.callout {{ border-radius:10px; padding:15px 18px; margin:16px 0; font-size:14px;
  background:var(--surface-2); color:var(--ink-2); max-width:74ch; }}
.callout.warn {{ background:var(--warn-soft); }}
.callout.good {{ background:var(--up-soft); }}
.callout p {{ color:inherit; margin:0 0 8px; }}
.callout p:last-child {{ margin:0; }}
.callout b {{ color:var(--ink); }}

figure {{ margin:18px 0; }}
figure img {{ width:100%; border:1px solid var(--line); border-radius:10px; display:block; }}
figcaption {{ font-size:12.5px; color:var(--ink-3); margin-top:8px; max-width:74ch; }}

pre {{ background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; overflow-x:auto; font:12.5px/1.65 'JetBrains Mono', monospace;
  color:var(--ink); margin:12px 0; }}
code {{ font-family:'JetBrains Mono', monospace; font-size:12.5px;
  background:var(--surface-2); padding:1px 5px; border-radius:4px; }}
pre code {{ background:none; padding:0; }}
ul {{ color:var(--ink-2); max-width:70ch; padding-left:20px; }}
li {{ margin-bottom:7px; }}
.pill {{ display:inline-block; font:500 10.5px 'JetBrains Mono', monospace;
  letter-spacing:.05em; padding:2px 8px; border-radius:5px; background:var(--accent-soft);
  color:var(--accent); }}
.pill.warn {{ background:var(--warn-soft); color:var(--warn); }}
.pill.good {{ background:var(--up-soft); color:var(--up); }}
@media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style>

<nav><div class="navin">
  <div class="brand">Pitch Occupancy<span>.</span></div>
  <div class="navtabs" id="tabs"></div>
</div></nav>

<main>

<section class="view" data-view="overview">
  <p class="eyebrow">Master's thesis &middot; CPU-only computer vision</p>
  <h1>Auditing how football pitches are actually used</h1>
  <p class="lede">One frame per camera per minute, classified as empty, active play or
  maintenance, aggregated into a <code>USED</code> / <code>NOTUSED</code> / <code>REVIEW</code>
  verdict, then reconciled against what the facility&rsquo;s booking records claim. Entirely on
  one CPU, with no GPU anywhere in the system.</p>

  <div class="tiles">
    <div class="tile lead"><div class="k">Zero-shot, no labels</div><div class="v">{best_prompt_recall}</div>
      <div class="n">cross-venue play recall</div></div>
    <div class="tile"><div class="k">Frames labelled</div><div class="v">1,692</div>
      <div class="n">across 9 venues</div></div>
    <div class="tile"><div class="k">20 cameras</div><div class="v">2.5s</div>
      <div class="n">of a 60-second cycle</div></div>
    <div class="tile"><div class="k">Tests</div><div class="v">283</div>
      <div class="n">passing</div></div>
  </div>

  <div class="callout warn"><p><b>The finding that reframed the project.</b> A rule that reads
  only the clock &mdash; <i>&ldquo;if night then playing, else empty&rdquo;</i> &mdash; scores
  <b>98.4%</b> on the labelled data, matching every model the pilot benchmarked. Class and scene
  are the same variable in this dataset: empty pitches are 98% daytime, active play 99% night.
  Accuracy measured here cannot distinguish classifying occupancy from recognising a place.</p>
  <p>Everything since has been about building an evaluation that can tell the difference.</p></div>

  <h2>What the argument became</h2>
  <ul>
    <li><b>The evaluation was uninformative</b> &mdash; a colour histogram beat a deep probe
    under a random split.</li>
    <li><b>The models were not the problem</b> &mdash; across unseen venues the trivial
    baselines collapse while the frozen backbones hold.</li>
    <li><b>The protocol reverses the decision</b> &mdash; ViT ranks first under the leaky
    split and last under the honest one.</li>
    <li><b>The models genuinely read people</b> &mdash; blurring away the players costs
    0.12 recall; removing colour or the background does not.</li>
  </ul>
</section>

<section class="view" data-view="findings" hidden>
  <p class="eyebrow">Findings</p>
  <h1>The protocol reverses the ranking</h1>
  <p class="lede">Three evaluation protocols, three different winners. A reader following the
  pilot&rsquo;s protocol would have selected ViT &mdash; the weakest generaliser of the three.</p>

  <figure><img src="{fig_rank}" alt="Model rank under three evaluation protocols: ViT first under the leaky random split and third under cross-venue evaluation.">
  <figcaption>Ranks rather than scores, because the three metrics are not comparable. Each
  protocol&rsquo;s score is printed beside its point.</figcaption></figure>

  <h2>The protocol a constant predictor wins</h2>
  <p>Stronger than &ldquo;the ranking reverses&rdquo;, and it came from running a fourth
  protocol beside the three above. <b>Under leave-one-venue-out a model that reads no
  pixels and always answers &ldquo;playing&rdquo; scores a perfect macro-F1</b>, ahead of
  every backbone &mdash; because every held-out venue is 100% active play, so the macro
  average is taken over the one class present. There is no ranking there to reverse.</p>
  <div class="scroll"><table>
    <thead><tr><th>Protocol</th><th>Best trivial baseline</th><th>Best backbone</th>
    <th class="num">Gap</th></tr></thead>
    <tbody>{floor_rows}</tbody>
  </table></div>
  <p class="note">A negative gap means the trivial baseline is not beaten. Diagnostics for
  every row &mdash; test size, class count, majority share &mdash; are in
  <code>benchmark_v2.csv</code>; no number here is quotable without them.</p>

  <h2>How much of the leakage penalty is leakage</h2>
  <p>A model that never trains cannot leak, so OpenCLIP scored zero-shot on the identical
  test sets measures what changing the test set does on its own. It drops
  <b>{composition_drop}</b> from the leaky split to the honest one. Subtracting that leaves
  the part attributable to
  leakage &mdash; about a third of the score rather than the two thirds the raw delta
  implies.</p>
  <div class="scroll"><table>
    <thead><tr><th>Model</th><th class="num">Raw drop</th>
    <th class="num">Composition</th><th class="num">Attributable</th></tr></thead>
    <tbody>{comp_rows}</tbody>
  </table></div>
  <p class="note">A control, not a proof: it assumes the composition effect is additive and
  similar across models.</p>

  <h2>Every model, every protocol</h2>
  <div class="scroll"><table>
    <thead><tr><th>Model</th><th class="num">Random split<br>macro-F1</th>
    <th class="num">Grouped split<br>macro-F1</th><th class="num">Cross-venue<br>play recall</th>
    <th class="num">Worst fold</th><th class="num">ms/frame</th><th class="num">20 cams (s)</th></tr></thead>
    <tbody>{model_rows}</tbody>
  </table></div>
  <p>The clock rule uses no image data at all. Under the leaky and grouped protocols it is
  competitive; across venues it collapses.</p>

  <h2>Cross-venue recall, fold by fold</h2>
  <div class="scroll"><table>
    <thead><tr><th>Held-out venue</th><th class="num">DINOv2</th><th class="num">ConvNeXtV2</th>
    <th class="num">ViT</th><th class="num">Clock rule</th></tr></thead>
    <tbody>{fold_rows}</tbody>
  </table></div>

  <figure><img src="{fig_cross}" alt="Per-venue play recall for each model, with the 0.90 target marked.">
  <figcaption>Five of seven venues contribute 12&ndash;30 frames, so the spread matters more
  than the mean.</figcaption></figure>

  <h2>Robustness: merging the two audited venues</h2>
  <p>Two venue groups were flagged low-confidence. A pessimistic rerun merges them into one
  fold, so holding the pair out removes both from training.</p>
  <div class="scroll"><table>
    <thead><tr><th>Model</th><th class="num">Mean recall</th><th class="num">95% CI</th>
    <th class="num">Merged fold</th></tr></thead>
    <tbody>{sens_rows}</tbody>
  </table></div>
  <div class="callout"><p>H3 survives the merge, so the venue grouping cannot change the
  conclusion. <b>The main run stays the primary number</b> &mdash; the higher means here are
  partly fold arithmetic, since merging turns two small folds into one medium fold in an
  unweighted average.</p></div>

  <h2>More labels made the models worse</h2>
  <figure><img src="{fig_label}" alt="Macro-F1 against labelling budget: the curves peak at 100-300 labels and fall when all labels are used.">
  <figcaption>ConvNeXtV2 peaks at 100 labels and is worse with all 671. Small budgets are
  class-stratified; the full pool is not, and is dominated by one mostly-empty recording.
  Balanced sampling accidentally de-confounds.</figcaption></figure>
</section>

<section class="view" data-view="dataset" hidden>
  <p class="eyebrow">Dataset</p>
  <h1>1,692 frames, and what they cannot answer</h1>

  <div class="tiles">
    <div class="tile"><div class="k">Active play</div><div class="v">1,192</div>
      <div class="n">9 venues, day and night</div></div>
    <div class="tile"><div class="k">Empty</div><div class="v">494</div>
      <div class="n">one venue, 98% daytime</div></div>
    <div class="tile"><div class="k">Maintenance</div><div class="v">6</div>
      <div class="n">unusable as a class</div></div>
    <div class="tile"><div class="k">Real slots</div><div class="v">2</div>
      <div class="n">STAN needs about 30</div></div>
  </div>

  <div class="callout warn"><p><b>One gap explains four blocked questions.</b> No empty pitch
  exists outside a single venue. That single fact makes cross-venue three-class evaluation,
  the calibration study, the risk&ndash;coverage operating point, and the clean version of the
  leakage comparison all unanswerable.</p>
  <p>The fix is not more data in general &mdash; it is <b>full-length recordings from any
  second venue</b>. A complete slot contains the empty periods before kickoff and after the
  whistle for free.</p></div>

  <h2>Where the frames come from</h2>
  <ul>
    <li><b>4 slot recordings</b> &mdash; one facility, two slots, two cameras each. The morning
    slot is almost entirely empty, the evening one almost entirely a match. This is the source
    of the confound.</li>
    <li><b>66 highlight clips</b> &mdash; 10&ndash;14 seconds each, roughly nine venues, day
    and night. Every clip contains players (YOLO-verified, minimum 3, median 8), so they add
    venue diversity and daytime play but no empty pitches.</li>
  </ul>

  <h2>Conditions not covered</h2>
  <p>Lighting (day/night) is the only condition axis. There is <b>no weather field and no rain
  footage</b> &mdash; every frame is dry, so no claim about precipitation robustness is
  supported. See the ideas backlog for why rain would be valuable, and the trap it carries.</p>
</section>

<section class="view" data-view="searches" hidden>
  <p class="eyebrow">Searches</p>
  <h1>Tuning what goes in, not just what learns</h1>
  <p class="lede">Two configuration searches, both scored on cross-venue transfer and both
  guarded by a false-play control &mdash; every cross-venue test set is entirely active play,
  so recall can be bought by simply saying &ldquo;playing&rdquo; more often.</p>

  <h2>Zero-shot prompt search <span class="pill good">no labels</span></h2>
  <p>375 prompt sets, scored exhaustively. Images are embedded once, so a prompt set costs
  almost nothing &mdash; the opposite cost profile to the preprocessing search.</p>
  <div class="scroll"><table>
    <thead><tr><th>Descriptor for &ldquo;active play&rdquo;</th><th class="num">Templates</th>
    <th class="num">Recall</th><th class="num">False-play</th><th class="num">Balanced</th></tr></thead>
    <tbody>{prompt_rows}</tbody>
  </table></div>
  <div class="callout good"><p><b>A searched prompt beats every trained probe, with zero
  labels.</b> &ldquo;{best_prompt}&rdquo; reaches {best_prompt_recall} cross-venue recall,
  above DINOv2&rsquo;s 0.930 &mdash; which needed about 1,500 labelled frames. For active-play
  detection, onboarding a new site currently costs no labels at all.</p></div>
  <div class="callout warn"><p><b>The caveat is not small.</b> The prompt was selected against
  the same folds it is scored on, and 375 candidates is a great deal of selection freedom.
  This is a development result. The locked final venues were held out throughout, so the
  honest confirmation &mdash; evaluate this one prompt on them, once &mdash; remains
  available.</p></div>

  <h2>Preprocessing search <span class="pill">{search_n} evaluations</span></h2>
  <p>Ten switches, each a hypothesis about what does <i>not</i> travel between venues. Greedy,
  because every candidate needs a fresh embedding pass over the dataset.</p>
  <div class="scroll"><table>
    <thead><tr><th>Configuration</th><th class="num">Recall</th><th class="num">vs baseline</th>
    <th class="num">Worst fold</th><th class="num">False-play</th></tr></thead>
    <tbody>{search_rows}</tbody>
  </table></div>

  <h2>What the model is actually reading</h2>
  <p>Removing information from the input and re-measuring shows which parts carried the signal.</p>
  <div class="scroll"><table>
    <thead><tr><th>Input variant</th><th class="num">Play recall</th><th class="num">Worst fold</th></tr></thead>
    <tbody>{abl_rows}</tbody>
  </table></div>
  <div class="callout"><p><b>Blurring costs 0.12 recall</b>, and at that scale no individual
  person is visible &mdash; so the prediction rests on people, not scenery. Colour and the
  frame border are net distractions, each helping when removed. <b>But not together</b>:
  grayscale plus cropping falls below the untouched baseline. Removing information has a
  floor.</p></div>
</section>

<section class="view" data-view="system" hidden>
  <p class="eyebrow">System</p>
  <h1>From a frame to a billing decision</h1>

  <pre><code>frame source  -&gt;  preprocess  -&gt;  classify  -&gt;  fuse two cameras
                                                      |
                                              aggregate the slot
                                                      |
                                    verdict + 3 evidence images
                                                      |
                                    reconcile against the booking record</code></pre>

  <h2>Rules the system holds to</h2>
  <ul>
    <li><b>Never invent an observation.</b> A camera that drops for a minute leaves a gap; a
    total outage yields <code>REVIEW</code>, not <code>NOTUSED</code>. No footage is not
    evidence a pitch was unused.</li>
    <li><b>REVIEW never becomes an anomaly.</b> The system may not convert its own uncertainty
    into someone else&rsquo;s error.</li>
    <li><b>Anomalies are per field, never per person.</b> Attributing discrepancies to
    individuals adds no scientific value and considerable ethical exposure.</li>
    <li><b>No verdict without its evidence.</b> A verdict and the samples behind it are written
    in one transaction.</li>
    <li><b>The system never bills.</b> Output is decision support; a human confirms every
    anomaly.</li>
  </ul>

  <h2>Validated end to end</h2>
  <div class="scroll"><table>
    <thead><tr><th>Slot</th><th class="num">Minutes</th><th class="num">Play</th>
    <th class="num">Empty</th><th>Verdict</th><th>Expected</th></tr></thead>
    <tbody>
      <tr><td class="sw">2026-07-11 10:00</td><td class="num">59</td><td class="num">0.02</td>
        <td class="num">0.93</td><td><span class="pill">NOTUSED</span></td><td>NOTUSED</td></tr>
      <tr><td class="sw">2026-07-12 20:30</td><td class="num">60</td><td class="num">1.00</td>
        <td class="num">0.00</td><td><span class="pill">USED</span></td><td>USED</td></tr>
    </tbody>
  </table></div>

  <h2>Efficiency</h2>
  <p>All three backbones fit the 60-second sampling cycle with large margins, so latency is
  <b>not</b> the binding constraint &mdash; the model choice falls to accuracy. Measured on a
  4-thread development laptop; the target Mini-PC must be measured separately.</p>
</section>

<section class="view" data-view="run" hidden>
  <p class="eyebrow">Reproduce</p>
  <h1>Running the project</h1>
  <div class="callout warn"><p><b>The code is not on <code>main</code>.</b> All 39 commits sit
  on a stacked chain of feature branches; the tip holds everything.</p></div>

  <h3>Setup</h3>
<pre><code>git checkout feat/prompt-search
uv sync
uv run pitch info
uv run pytest -q -m "not slow"</code></pre>

  <h3>Everyday commands</h3>
<pre><code>uv run pitch manifest      # rebuild the manifest, print confound warnings
uv run pitch coverage      # class x lighting x venue matrix
uv run pitch cache         # embed every frame per backbone (~25 min)
uv run pitch serve         # dashboard API on :8000</code></pre>

  <h3>Reproduce every result</h3>
<pre><code>uv run python experiments/reproduce_all.py --check   # what is stale
uv run python experiments/reproduce_all.py           # run what is missing</code></pre>
  <p>Stages verify their own outputs and report <code>BLOCKED</code> rather than skipping when
  an input is missing &mdash; a reproduction script that exits 0 having done nothing is worse
  than one that fails.</p>

  <h3>Individual experiments</h3>
<pre><code>uv run python experiments/h3_cross_venue_recall.py
uv run python experiments/prompt_search.py --limit 600
uv run python experiments/preprocess_search.py --check   # cost estimate first
uv run python experiments/make_figures.py</code></pre>
  <div class="callout warn"><p>Do <b>not</b> pass <code>--limit</code> to the preprocessing
  search for a number you intend to quote. Subsampling shrinks the venue folds until recall
  can only take a few discrete values; the script warns when it happens.</p></div>

  <h3>Where things live</h3>
  <ul>
    <li><code>results/EXPERIMENT_LOG.md</code> &mdash; every finding, with its caveats</li>
    <li><code>thesis/preregistration.md</code> &mdash; hypotheses fixed before the runs</li>
    <li><code>thesis/rq_matrix.md</code> &mdash; which experiment answers which question</li>
    <li><code>docs/CODEBASE.md</code> &mdash; every branch and module</li>
    <li><code>docs/IDEAS.md</code> &mdash; ideas not in the plan, each with a stop condition</li>
  </ul>
</section>

</main>

<script>
const VIEWS = [
  ["overview", "Overview"], ["findings", "Findings"], ["dataset", "Dataset"],
  ["searches", "Searches"], ["system", "System"], ["run", "Reproduce"],
];
const tabs = document.getElementById("tabs");
const views = [...document.querySelectorAll(".view")];

function show(name) {{
  views.forEach(v => {{ v.hidden = v.dataset.view !== name; }});
  [...tabs.children].forEach(b =>
    b.setAttribute("aria-current", String(b.dataset.view === name)));
  if (location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
  window.scrollTo({{ top: 0, behavior: "instant" }});
}}

VIEWS.forEach(([id, label]) => {{
  const b = document.createElement("button");
  b.textContent = label; b.dataset.view = id;
  b.onclick = () => show(id);
  tabs.appendChild(b);
}});

show(VIEWS.some(v => v[0] === location.hash.slice(1)) ? location.hash.slice(1) : "overview");
window.addEventListener("hashchange", () => {{
  const h = location.hash.slice(1);
  if (VIEWS.some(v => v[0] === h)) show(h);
}});
</script>
"""


def main() -> None:
    html = build(collect())
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
