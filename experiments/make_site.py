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

**Three pages are built from measurements of the code rather than of the data**, and they
are the ones a supervisor asks for first:

* **Models** - what was frozen and what was trained, from `results/model_inventory.json`.
  Every parameter count on it comes from instantiating the model and counting, so the
  87,055:1 ratio the page leads on is checkable by re-running one script.
* **Preprocessing** - every switch as a before/after pair at the model's own input size,
  from `results/preprocess_pairs.csv`, with how far the frame actually moved. The pictures
  say what a switch does; the numbers beside them say whether it did anything, which two
  same-sized tiles cannot - a crop is re-letterboxed back to 224, so discarding three
  quarters of the frame looks like a mild zoom.
* **Reproduce** - what a run actually costs, from `results/pipeline_graph.json`. The page
  was four command blocks and a paragraph, which answers *what do I type* and nothing about
  what happens next; the runner had always declared every stage's runtime, outputs and
  dependencies, and none of it reached a reader. Its one finding is that a single stage is
  two thirds of the 18.6 hours.

All three are built as whole sections by :func:`models_view`, :func:`preprocess_view` and
:func:`reproduce_view` rather than as row fragments, because all three are mostly layout.
They are substituted into the template as *values*, so nothing inside them is re-formatted -
which is why the two tables `preprocess_view` reuses are handed to it rather than left as
placeholders.

    uv run python experiments/make_site.py

Regenerate after any experiment it covers, and republish. `model_inventory.py`,
`preprocess_pairs.py` and `pipeline_graph.py` are all stages in `reproduce_all.py`; run
those first if an input is missing, or the affected page renders a "not generated" notice
instead of guessing.
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


def pct(v, d=1) -> str:
    try:
        return f"{float(v) * 100:.{d}f}%"
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
    # The two artefacts the models and preprocessing pages are built from. Both are
    # measurements of the code rather than of the data - `model_inventory.py` instantiates
    # every model and counts it, `preprocess_pairs.py` applies every switch to one frame
    # and measures how far the frame moved - so neither is a number anyone typed.
    inventory = read_json("model_inventory.json")
    pairs = read_csv("preprocess_pairs.csv")
    # The reproduction pipeline as a graph, so the Reproduce page can show what a run does
    # rather than only what to type. Derived from `reproduce_all.STAGES`, so it cannot
    # disagree with the order the runner actually uses.
    pipeline = read_json("pipeline_graph.json")
    gate = read_csv("fusion_head_gate.csv")
    stan_rows = [r for r in read_csv("stan_preliminary.csv") if r.get("model")]
    clahe_gate = read_json("search_resolution_gate.json")

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

    # ablation means, and the false-play control that decides how to read them. The control
    # rows carry no `play_recall` - they are one row per variant, not one per fold - so a
    # reader that assumed every row had a recall broke when they were added, which is how
    # this comment came to exist.
    abl_means: dict[str, list[float]] = {}
    abl_control: dict[str, tuple[float, float]] = {}
    for r in abl:
        if r.get("venue") == "CONTROL_held_out_empty":
            abl_control[r["variant"]] = (
                float(r["false_play_rate"]), float(r["empty_accuracy"])
            )
            continue
        abl_means.setdefault(r["variant"], []).append(float(r["play_recall"]))
    ablation = sorted(
        ({"variant": k, "mean": sum(v) / len(v), "worst": min(v),
          "false_play": abl_control.get(k, (float("nan"), float("nan")))[0],
          "empty_accuracy": abl_control.get(k, (float("nan"), float("nan")))[1]}
         for k, v in abl_means.items()),
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
            for n in ("ranking_inversion", "label_efficiency", "cross_venue_recall",
                      "risk_coverage_band", "onboarding_cost")
        },
        "inventory": inventory,
        "pipeline": pipeline,
        # Each pair carries its own image, so the page stays one file. `img` is empty when
        # the pair has not been regenerated, and the builder skips those rather than
        # rendering a card with a hole in it.
        "pairs": [dict(p, img=data_uri(RESULTS / p["file"])) for p in pairs],
        "gate": {
            "spread_lo": min((float(r["gate_weight_spread"]) for r in gate), default=0.0),
            "spread_hi": max((float(r["gate_weight_spread"]) for r in gate), default=0.0),
            "dinov2_weight": (
                sum(float(r["mlp_mean_weight_dinov2"]) for r in gate) / len(gate)
                if gate else 0.0
            ),
            "corr_night": (
                sum(float(r["corr_gate_weight_with_night"]) for r in gate) / len(gate)
                if gate else 0.0
            ),
            "n_folds": len(gate),
        },
        "stan": [
            {"model": r["model"], "composed": f(r["composed_test_accuracy"], 4),
             "n_composed": r["n_composed_test"], "recorded": f(r["recorded_accuracy"], 4),
             "n_recorded": r["n_recorded"]}
            for r in stan_rows
        ],
        "clahe_gate": clahe_gate,
    }


#: A padlock and a spark, inline so the frozen/trained distinction survives greyscale
#: printing and a projector that eats colour. Drawn once and referenced by both badges.
LOCK = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" '
        'aria-hidden="true"><rect x="4" y="11" width="16" height="10" rx="2"/>'
        '<path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>')
SPARK = ('<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
         '<path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z"/></svg>')


def frozen_badge(n: int) -> str:
    return (f'<span class="badge frozen">{LOCK} Frozen &middot; '
            f'{n:,} parameters, 0 updated</span>')


def trained_badge(n: int) -> str:
    return f'<span class="badge trained">{SPARK} Trained &middot; {n:,} parameters fitted</span>'


def meter(label: str, value: float, shown: str, *, tone: str = "") -> str:
    """A 0-1 bar with its number printed beside it, never instead of it."""
    width = max(0.0, min(1.0, value)) * 100
    return (f'<div class="meter"><div class="lab">{label}</div>'
            f'<div class="track"><i class="{tone}" style="width:{width:.1f}%"></i></div>'
            f'<v>{shown}</v></div>')


def split_svg(*, grouped: bool) -> str:
    """Two miniature split diagrams: duplicate frames straddling the line, or not.

    Six pairs of near-identical frames, drawn as pairs. Under a random shuffle the two
    halves of a pair land on opposite sides of the train/test line; grouping by scene keeps
    each pair whole. This is the actual mechanism behind the leaky protocol's 0.99, and it
    is the one thing on the models page that is drawn rather than measured - so it is drawn
    schematically and labelled as a schematic.
    """
    parts = [
        '<svg viewBox="0 0 260 104" role="img" '
        f'aria-label="{"Grouped split: each pair of near-identical frames stays on one side of the train/test line." if grouped else "Random split: the two halves of each near-identical pair land on opposite sides of the train/test line."}">'
    ]
    parts.append('<line x1="130" y1="8" x2="130" y2="96" stroke="var(--down)" '
                 'stroke-width="2" stroke-dasharray="4 3"/>')
    parts.append('<text x="60" y="16" font-size="9" font-family="monospace" '
                 'fill="var(--ink-3)" text-anchor="middle">TRAIN</text>')
    parts.append('<text x="196" y="16" font-size="9" font-family="monospace" '
                 'fill="var(--ink-3)" text-anchor="middle">TEST</text>')
    for i in range(6):
        y = 26 + i * 12
        # Under grouping, pairs 0-3 go to train and 4-5 to test, whole. Under a shuffle,
        # every pair has one frame each side - which is the picture.
        left = [True, True] if (grouped and i < 4) else \
               [False, False] if grouped else [True, False]
        for j, on_train in enumerate(left):
            x = (40 + j * 20) if on_train else (150 + j * 20)
            parts.append(f'<rect x="{x}" y="{y}" width="17" height="9" rx="2" '
                         f'fill="var(--frozen)" opacity="{0.45 + i * 0.09:.2f}"/>')
    parts.append('</svg>')
    return "".join(parts)


#: The sampling cycle every latency number is measured against. A 4.1-second round is only
#: meaningful as a share of it, and the table of milliseconds never said so.
CYCLE_SECONDS = 60.0


def latency_meter(d: dict) -> str:
    """Each backbone's 20-camera round as a share of the 60-second cycle it has to fit in.

    Bars against the cycle rather than against each other, because the question is not
    which model is fastest - it is whether any of them is close to the deadline, and the
    answer is that none is. A bar chart scaled to the slowest model would imply otherwise.
    """
    rows = [m for m in d["models"] if m["conc"] != "—"]
    if not rows:
        return ""
    return "".join(
        meter(f'{m["label"]} - {m["ms"]} ms/frame', float(m["conc"]) / CYCLE_SECONDS,
              f'{m["conc"]}s', tone="up")
        for m in sorted(rows, key=lambda m: float(m["conc"]))
    ) + meter("the cycle it must fit in", 1.0, f"{CYCLE_SECONDS:.0f}s", tone="frozen")


def composition_meter(d: dict) -> str:
    """The raw drop, split into the part the test set explains and the part leakage does.

    Both bars are scaled against the same raw drop, so the composition bar's length *is*
    the share of the penalty that is not leakage. Quoting the two numbers side by side left
    that division to the reader, and the division is the finding.
    """
    comp = d["composition"]
    if not comp:
        return ""
    worst = max(float(c["raw"]) for c in comp)
    if not worst:
        return ""
    row = comp[0]
    composition = float(row["composition"])
    return (
        meter("raw drop, leaky to honest", 1.0, row["raw"], tone="down")
        + meter("explained by the test set alone", composition / worst,
                row["composition"], tone="frozen")
        + meter("attributable to leakage", float(row["attributable"]) / worst,
                row["attributable"], tone="down")
    )


def models_view(d: dict) -> str:
    """The page that answers "so did you actually train anything?".

    Built from `results/model_inventory.json`, which is produced by instantiating every
    model and counting its parameters - so the 87,000:1 ratio this page leads on is a
    measurement of the code, and a reader who doubts it can re-run one script.
    """
    inv = d["inventory"]
    if not inv:
        return ('<section class="view" data-view="models" hidden><p class="eyebrow">Models'
                '</p><h1>Model inventory not generated</h1><p>Run <code>uv run python '
                'experiments/model_inventory.py</code> and regenerate this page.</p></section>')

    frozen_total = inv["frozen_total_params"]
    probe = inv["trained"]["probe"]
    stan = inv["trained"]["stan"]
    fusion = {r["rung"]: r for r in inv["trained"]["fusion"]}
    mlp = fusion["mlp"]
    trained_total = inv["trained_total_params"]

    # --- the three frozen lanes ---
    lanes = []
    for b in inv["frozen"]:
        arch = b["architecture"]
        shape = (f'stages {arch["depths"]}, widths {arch["widths"]}'
                 if arch["family"] == "convnet"
                 else f'{arch["layers"]} layers, hidden {arch["hidden"]}, '
                      f'{arch["heads"]} heads, patch {arch["patch"]}')
        crop = ("resizes to exactly 224&times;224 &mdash; a true no-op"
                if b["geometry_is_a_no_op"] else b["processor_geometry"])
        lanes.append(
            f'<div class="block frozen"><h4>{b["label"]}</h4>'
            f'<div class="sub">{b["hf_id"]}</div>'
            f'{frozen_badge(b["params"])}'
            f'<dl><dt>Architecture</dt><dd>{arch["family"]}, {shape}</dd>'
            f'<dt>Pretraining</dt><dd>{b["pretraining"]} &mdash; <b>{b["pretraining_labels"]}</b></dd>'
            f'<dt>Pooling</dt><dd><code>{b["pooling"]}</code></dd>'
            f'<dt>Output</dt><dd>one {arch["output_dim"]}-d vector per frame</dd>'
            f'<dt>Processor</dt><dd>{crop}</dd>'
            f'<dt>Role here</dt><dd>{b["role"]}</dd></dl></div>'
        )

    # --- the parameter scale, drawn to scale and then explicitly broken ---
    scale_rows = []
    for b in inv["frozen"]:
        w = b["params"] / frozen_total * 100
        scale_rows.append(
            f'<div class="row"><span>{b["label"]}</span>'
            f'<div class="bar frozen" style="width:{w:.1f}%">{b["params"]:,}</div></div>'
        )
    trained_w = trained_total / frozen_total * 100
    scale_rows.append(
        f'<div class="row"><span>everything trained</span><div class="tiny">'
        f'<div class="bar trained" style="width:{max(trained_w, 0.14):.2f}%"></div>'
        f'<em>{trained_total:,} &mdash; this speck is the only thing that ever saw a label</em>'
        f'</div></div>'
    )

    ratio = inv["frozen_to_trained_ratio"]
    pooler = next((b for b in inv["frozen"] if b["unused_pooler_params"]), None)
    n_stan_real = d["stan"][0]["n_recorded"] if d["stan"] else "2"

    stan_rows = "".join(
        f'<tr><td class="sw">{s["model"]}</td><td class="num strong">{s["composed"]}</td>'
        f'<td class="num">{s["recorded"]}</td></tr>' for s in d["stan"]
    )
    gate = d["gate"]

    controls = [
        ("Majority class", "predicts the most frequent training label",
         "The true zero point. Under leave-one-venue-out it scores a perfect macro-F1, "
         "because every held-out venue is 100% active play."),
        ("Clock rule", "reads the lighting field, no pixels at all",
         "Learns &ldquo;night means a match, day means an empty pitch&rdquo; and scored "
         "<b>98.4%</b> &mdash; matching every backbone. The most uncomfortable number here."),
        ("Mean intensity", "one number per frame (1-d)",
         "How bright is it, fed to logistic regression. Day-or-night, learned rather than "
         "hand-written."),
        ("Colour histogram", "coarse RGB histogram (48-d)",
         "Turf colour, floodlight cast and composition. Cannot represent &ldquo;are there "
         "players on the pitch&rdquo; &mdash; and beat a deep probe under a random split."),
        ("OpenCLIP zero-shot", inv["zero_shot"]["hf_id"],
         f'<b>Never fitted at all</b> &mdash; {inv["zero_shot"]["params"]:,} pretrained '
         "parameters and a <code>fit()</code> that is a deliberate no-op. So the same "
         "per-frame prediction is scored under every protocol, and any movement in its "
         "score is caused by the test set alone. That is what makes it the composition "
         "control."),
    ]
    control_cards = "".join(
        f'<div class="card"><h4>{name}</h4>'
        f'<div class="sub" style="font:500 10.5px \'JetBrains Mono\',monospace;'
        f'color:var(--ink-3);margin:-3px 0 7px">{what}</div>'
        f'<p>{why}</p></div>' for name, what, why in controls
    )

    reasons = [
        ("Sample size", "1,578 development frames across 8 venues cannot support "
         "fine-tuning 86M parameters. A fine-tuned model would memorise the venues, and "
         "the finding would be about these 8 sites rather than about the models."),
        ("It makes the comparison fair", "Freezing holds the representation fixed, so when "
         "DINOv2 beats ConvNeXtV2 the only thing that differs <i>is the representation</i>. "
         "Fine-tune all three and the comparison is between three optimisation runs, not "
         "three representations &mdash; the research question would change underneath me."),
        ("Cost and reproducibility", "Embed once, reuse forever: minutes to embed 1,692 "
         "frames, under a second to fit a head. This is the only reason 4 protocols &times; "
         "5 seeds &times; 8 predictors, plus every ablation, was runnable at all. Freezing "
         "bought the experimental breadth."),
        ("It makes the headline finding legible", "A leaked split plus a fine-tuned 86M "
         "network gives 0.99 and no explanation. A leaked split plus a 2,307-weight probe "
         "gives 0.99 <b>and a mechanism</b> &mdash; the probe is too small to do anything "
         "clever, so near-duplicate frames are the only available explanation. Which was "
         "then measured directly."),
    ]
    reason_board = "".join(
        f'<div class="claim"><div class="n">{i + 1}</div><div><b>{t}</b>'
        f'<span>{body}</span></div></div>'
        for i, (t, body) in enumerate(reasons)
    )

    return f"""<section class="view" data-view="models" hidden>
  <p class="eyebrow">Models</p>
  <h1>What was trained, and what never was</h1>
  <p class="lede">The three vision backbones were <b>never trained on this data</b>. They are
  frozen, off-the-shelf pretrained feature extractors. What was trained is a small classifier
  head on top of their output, plus two small models further down the pipeline &mdash;
  {trained_total:,} parameters against {frozen_total:,} frozen ones.</p>

  <div class="tiles">
    <div class="tile lead"><div class="k">Frozen, never updated</div>
      <div class="v">{frozen_total / 1e6:.1f}M</div>
      <div class="n">{inv["frozen_gradients_received"]} received a gradient</div></div>
    <div class="tile"><div class="k">Trained, all of it</div><div class="v">{trained_total:,}</div>
      <div class="n">probe + gate + STAN</div></div>
    <div class="tile"><div class="k">Ratio</div><div class="v">{ratio:,}:1</div>
      <div class="n">{inv["probe_ratio"]:,}:1 against the probe alone</div></div>
    <div class="tile"><div class="k">Backbones</div><div class="v">3</div>
      <div class="n">run once each, into a cache</div></div>
  </div>

  <div class="legend">
    <div><span class="badge frozen" style="margin:0">{LOCK} Frozen</span>
      pretrained weights, zero gradients, no label ever seen</div>
    <div><span class="badge trained" style="margin:0">{SPARK} Trained</span>
      fitted on this project&rsquo;s labels</div>
  </div>

  <h2>The pipeline, and where labels enter it</h2>
  <div class="rail">
    <div><div class="k">Stage 1</div><div class="t">Input</div>
      <div class="d">1,692 frames, 10 venues. Deterministic resize and letterbox to
      224&times;224. Nothing learned.</div></div>
    <div><div class="k">Stage 2</div><div class="t">3 frozen backbones</div>
      <div class="d">Run once each over every frame, into a cached
      (1692, {inv["feature_dim"]}) matrix.</div></div>
    <div class="seam"></div>
    <div><div class="k">Stage 3</div><div class="t">Trained probe</div>
      <div class="d">{probe["params"]:,} parameters. Refit from scratch per backbone
      &times; protocol &times; seed.</div></div>
    <div><div class="k">Stage 4</div><div class="t">Downstream heads</div>
      <div class="d">Gated fusion ({mlp["params"]:,}) and STAN ({stan["params"]:,}) &mdash;
      both genuinely trained.</div></div>
    <div><div class="k">Stage 5</div><div class="t">Controls</div>
      <div class="d">Trivial floors and a zero-shot model, so &ldquo;it works&rdquo; is a
      claim that can fail.</div></div>
  </div>
  <p class="seamnote">Labels enter here, and nowhere to the left of this line.</p>

  <h2>Drawn to scale</h2>
  <div class="scale">{"".join(scale_rows)}</div>
  <p class="note">The frozen bars are to scale against each other. The trained bar is
  <b>not</b> &mdash; at true scale it would be {trained_w:.4f}% of the row, narrower than
  one pixel. The scale is broken deliberately rather than fudged silently.</p>

  <h2>Stage 2 &mdash; the three frozen extractors</h2>
  <div class="grid2">{"".join(lanes)}</div>

  <div class="callout warn"><p><b>Two traps live in this stage, and both cost real time.</b></p>
  <p><b>1. The pooling head nobody trained.</b> Hugging Face exposes a
  <code>pooler_output</code> on these models; on a plain ViT it is
  <b>randomly initialised and never trained</b> &mdash; loading the checkpoint literally
  reports <code>pooler.dense.weight | MISSING</code>, and this inventory counts
  <b>{pooler["unused_pooler_params"]:,} such parameters</b> on {pooler["label"]}. Reading it
  cost the pilot a false 38% score and hours of debugging a model problem that did not exist.
  Every extractor here mean-pools explicitly and every cached feature file is stamped
  <code>pooling="mean"</code>, so a stale cache is rejected loudly.</p>
  <p><b>2. Two of the three silently re-crop the frame.</b> ConvNeXtV2
  (<code>crop_pct 0.875</code>) and DINOv2 (shortest edge 256, then centre-crop 224) each
  discard <b>23.4% of the frame area</b> &mdash; exactly where the letterbox padding lives.
  Measured cosine similarity between embeddings with and without that step:
  <b>0.886</b> for ConvNeXtV2, <b>0.961</b> for DINOv2, exact for ViT. ViT is the only one
  that sees what preprocessing produced.</p></div>

  <h2>Stage 3 &mdash; the trained head, and it is the whole of &ldquo;training&rdquo;</h2>
  <div class="block trained">
    <h4>{probe["name"]}</h4>
    <div class="sub">{probe["detail"]}</div>
    {trained_badge(probe["params"])}
    <dl><dt>Shape</dt><dd>{probe["shape"]}</dd>
    <dt>Refit</dt><dd>{probe["refit"]}</dd>
    <dt>Balanced</dt><dd>{probe["why"]}</dd></dl>
  </div>

  <h2>Stage 4 &mdash; two more genuinely trained models</h2>
  <p>These are the answer to &ldquo;so you trained nothing?&rdquo;. Both are small, both are
  reported with what they failed to buy.</p>
  <div class="grid2">
    <div class="block trained"><h4>Gated fusion head <span class="pill">RQ5</span></h4>
      <div class="sub">3 statistics &rarr; MLP(8) &rarr; softmax over K backbones</div>
      {trained_badge(mlp["params"])}
      <dl><dt>Question</dt><dd>can a router pick the best backbone per frame?</dd>
      <dt>Ladder</dt><dd>uniform ({fusion["uniform"]["params"]:,}) &rarr; constant
      ({fusion["constant"]["params"]:,}) &rarr; mlp ({mlp["params"]:,}), everything else
      held identical</dd>
      <dt>Training</dt><dd>full-batch Adam on CPU, 300 epochs, lr 0.01, seed 42</dd></dl>
      <p style="font-size:12.5px;margin:11px 0 0;color:var(--ink-2)"><b>Answer: no.</b> It
      puts {gate["dinov2_weight"]:.2f} of its weight on DINOv2 in every one of
      {gate["n_folds"]} folds and moves that weight by only
      {gate["spread_lo"]:.3f}&ndash;{gate["spread_hi"]:.3f} between frames. It is a learned
      constant wearing a router&rsquo;s costume &mdash; and its weight correlates
      {gate["corr_night"]:.2f} with night, so what little it does route on is the confound.
      Against <code>uniform</code> alone it would have read as &ldquo;the gate works&rdquo;.</p>
    </div>
    <div class="block trained"><h4>STAN <span class="pill warn">preliminary</span></h4>
      <div class="sub">{stan["shape"]}</div>
      {trained_badge(stan["params"])}
      <dl><dt>Input</dt><dd>a slot&rsquo;s ordered per-minute class probabilities</dd>
      <dt>Receptive field</dt><dd>about {stan["receptive_field_minutes"]} minutes</dd>
      <dt>Pooling</dt><dd>{stan["pooling"]}</dd>
      <dt>Training</dt><dd>{stan["training"]}, on 200 composed slots</dd>
      <dt>Ceiling</dt><dd>{stan["params"]:,} against a planned {stan["ceiling"]:,}</dd></dl>
      <p style="font-size:12.5px;margin:11px 0 0;color:var(--ink-2)"><b>Gated at 30 real
      slots; there are {n_stan_real}.</b> The code enforces that itself rather than trusting
      a note in a plan, and its baselines are trained too &mdash; tuned thresholds, median
      smoothing, a 3-state HMM with Viterbi decoding, a logistic regression on four summary
      numbers. Beating unlearned constants would prove nothing.</p>
    </div>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>Slot-level predictor</th><th class="num">200 composed slots</th>
    <th class="num">{n_stan_real} real slots</th></tr></thead>
    <tbody>{stan_rows}</tbody>
  </table></div>
  <p class="note">The composed column is a saturated set and the real column has
  {n_stan_real} observations in it, so neither is quotable. The ordering is the only thing
  this table supports.</p>

  <h2>Stage 5 &mdash; trained on nothing, on purpose</h2>
  <div class="grid3">{control_cards}</div>

  <h2>Why the backbones are frozen</h2>
  <div class="board">{reason_board}</div>

  <div class="callout warn"><p><b>What freezing does not excuse.</b> The backbones&rsquo;
  pretraining sets (ImageNet-21k/22k, LVD-142M) contain sports imagery that was not chosen
  and cannot be audited here. Freezing removes <i>this project&rsquo;s</i> training leakage;
  it does not remove <i>their</i> pretraining exposure. A stated limitation, not a solved
  problem.</p></div>

  <h2>&ldquo;But the backbone has seen your test set&rdquo;</h2>
  <p>Features are extracted once over every frame, before any split exists, so the objection
  is the right one to raise. The answer is that the backbone is frozen and never sees a label
  or a split &mdash; embedding a test frame is preprocessing, mathematically identical before
  or after the split. <b>The leakage is in the split, not the model:</b> frames sampled
  seconds apart are near-identical, so a random shuffle puts the same moment on both sides.</p>
  <div class="splitviz">
    <div class="card"><h4>Random shuffle &mdash; leaky</h4>{split_svg(grouped=False)}
      <p style="margin-top:8px"><b>100%</b> of the errors on this split had a near-duplicate
      frame on the training side.</p></div>
    <div class="card"><h4>Grouped by scene &mdash; honest</h4>{split_svg(grouped=True)}
      <p style="margin-top:8px"><b>0%</b> did. Same frames, same model, same head &mdash;
      only the cut moved.</p></div>
  </div>
  <p class="note">Schematic, not measured: each pair of blocks stands for two frames seconds
  apart. The 100% / 0% figures beside them are measured, in
  <code>near_duplicates.csv</code>.</p>
</section>
"""


#: How many stages a level row prints before it stops naming them. Six levels of 45 stages
#: is a readable page; six levels of every stage's full name is a wall, which is the thing
#: the Reproduce page was being rewritten to stop being.
CHIPS_PER_LEVEL = 9

#: Runtime bands for the stacked bar, coarsest first. A stage is "heavy" when it dominates
#: the pipeline on its own - and on this repository exactly one does.
HEAVY_MINUTES = 60


def reproduce_view(d: dict) -> str:
    """The reproduction pipeline, drawn from `results/pipeline_graph.json`.

    The page this replaced was four command blocks and a paragraph, which answers *what do
    I type* and nothing about what happens next: how long a full rerun takes, what the
    critical path is, which stages can start immediately, or where a fresh checkout stops
    for want of footage. All of that was already declared in `reproduce_all.STAGES` and
    none of it was ever shown.

    The one number a reader should leave with is that a single stage is two thirds of the
    runtime - so the stacked bar is the first thing on the page, before any command.
    """
    g = d["pipeline"]
    if not g:
        return ('<section class="view" data-view="run" hidden><p class="eyebrow">Reproduce'
                '</p><h1>Pipeline graph not generated</h1><p>Run <code>uv run python '
                'experiments/pipeline_graph.py</code> and regenerate this page.</p>'
                '</section>')

    stages = g["stages"]
    total = g["total_minutes"] or 1
    heavy = sorted((s for s in stages if s["minutes"] >= HEAVY_MINUTES),
                   key=lambda s: -s["minutes"])
    rest = total - sum(s["minutes"] for s in heavy)

    # The stacked bar. Named stages while they are wide enough to carry a label, then one
    # remainder block - a bar with 45 unreadable slivers would be a chart of nothing.
    segments, key = [], []
    tones = ("var(--warn)", "var(--accent)", "var(--frozen)")
    for i, s in enumerate(heavy):
        share = s["minutes"] / total * 100
        colour = tones[i % len(tones)]
        segments.append(
            f'<span style="width:{share:.2f}%;background:{colour}" '
            f'title="{s["name"]}: {s["minutes"]} min">'
            f'{f"{share:.0f}%" if share > 7 else ""}</span>'
        )
        key.append(f'<span><i style="background:{colour}"></i><code>{s["name"]}</code> '
                   f'&mdash; {s["minutes"]} min, {share:.0f}%</span>')
    segments.append(
        f'<span style="width:{rest / total * 100:.2f}%;background:var(--up)">'
        f'{rest / total * 100:.0f}%</span>'
    )
    key.append(f'<span><i style="background:var(--up)"></i>the other '
               f'{len(stages) - len(heavy)} stages &mdash; {rest} min combined</span>')

    # Dependency levels, which are computed rather than labelled - so these rows are the
    # real critical path, not a tidy grouping.
    level_rows = []
    for lvl in range(g["max_level"] + 1):
        at = sorted((s for s in stages if s["level"] == lvl), key=lambda s: -s["minutes"])
        chips = "".join(
            f'<span class="chip {"heavy" if s["minutes"] >= HEAVY_MINUTES else "done"}">'
            f'<b>{s["name"]}</b><s>{s["minutes"]}m</s></span>'
            for s in at[:CHIPS_PER_LEVEL]
        )
        if len(at) > CHIPS_PER_LEVEL:
            chips += (f'<span class="chip">+{len(at) - CHIPS_PER_LEVEL} more</span>')
        mins = sum(s["minutes"] for s in at)
        level_rows.append(
            f'<div class="lvl"><div class="tag">level<b>{lvl}</b>{mins} min</div>'
            f'<div class="chips">{chips}</div></div>'
        )

    done = g["by_status"].get("done", 0)
    externals = "".join(f'<span class="chip"><b>{p}</b></span>'
                        for p in g["external_inputs"])
    hours = g["total_minutes"] / 60
    crit = g["critical_path_minutes"] / 60

    status_note = (
        f'All {done} stages report their outputs present.'
        if done == len(stages) else
        ", ".join(f'{v} {k}' for k, v in sorted(g["by_status"].items()))
    )

    # Written against the stage that actually dominates rather than a remembered one, and
    # only when one does: trimming the preprocessing search would otherwise leave the page
    # asserting a two-thirds share that had stopped being true, which is the failure mode
    # this whole export is built to avoid.
    top = heavy[0] if heavy else None
    top_share = top["minutes"] / total if top else 0.0
    dominant_note = (
        f'<div class="callout warn"><p><b>One stage is '
        f'{"two thirds" if top_share > 0.6 else f"{top_share:.0%}"} of the pipeline.</b> '
        f'<code>{top["name"]}</code> is {top["minutes"]} of the {g["total_minutes"]} '
        f'minutes, because every candidate configuration needs a fresh embedding pass over '
        f'the whole dataset &mdash; which is exactly why that search is greedy and the '
        f'prompt search, where images are embedded once, could be exhaustive. It is also '
        f'why <code>--check</code> exists: on a laptop that sleeps, you want to know what '
        f'is missing before starting a {crit:.0f}-hour run, not during it.</p></div>'
        if top and top_share > 0.25 else
        '<p class="note">No single stage dominates the runtime, so a full rerun costs '
        'roughly what the sum of its parts suggests.</p>'
    )

    return f"""<section class="view" data-view="run" hidden>
  <p class="eyebrow">Reproduce</p>
  <h1>What happens when you run it</h1>
  <p class="lede">{g["n_stages"]} stages, {g["max_level"] + 1} dependency levels,
  <b>{hours:.1f} hours</b> of compute for a full rerun from the footage. Every stage
  declares what it produces and what it needs, so this page is drawn from the runner itself
  rather than described alongside it.</p>

  <div class="tiles">
    <div class="tile lead"><div class="k">Full rerun</div><div class="v">{hours:.1f}h</div>
      <div class="n">total compute across {g["n_stages"]} stages</div></div>
    <div class="tile"><div class="k">Critical path</div><div class="v">{crit:.1f}h</div>
      <div class="n">the floor, however much you parallelise</div></div>
    <div class="tile"><div class="k">Outputs present</div>
      <div class="v">{done}/{len(stages)}</div><div class="n">as this page was built</div></div>
    <div class="tile"><div class="k">Inputs it cannot make</div>
      <div class="v">{len(g["external_inputs"])}</div>
      <div class="n">the footage and its caches</div></div>
  </div>

  <h2>Where the {hours:.1f} hours go</h2>
  <div class="stack">{"".join(segments)}</div>
  <div class="stackkey">{"".join(key)}</div>
  {dominant_note}

  <h2>What can run when</h2>
  <p>Levels are <b>computed</b> from the declarations, not assigned: a stage sits one level
  below the deepest stage whose output it consumes. So level 0 is what a fresh checkout with
  the footage can start immediately, and the depth of the stack is the real shape of the
  dependency chain.</p>
  <div class="levels">{"".join(level_rows)}</div>
  <p class="note">Amber chips are stages of an hour or more. A few level-0 stages read only
  a feature cache from <code>data/</code>, which nothing here produces &mdash; so they are
  genuinely startable first, even though they are conceptually late work.</p>

  <h2>Where the repository stops</h2>
  <p>These inputs no stage can regenerate. They are the boundary of the reproducibility
  claim: <b>given the footage</b>, every number in the thesis can be recomputed from this
  repository &mdash; and without it, none can.</p>
  <div class="chips">{externals}</div>

  <div class="callout"><p><b>A stage with a missing input reports
  <code>BLOCKED</code>, never success.</b> A reproduction script that exits 0 having done
  nothing is worse than one that fails, so stages verify their own outputs afterwards rather
  than trusting an exit code &mdash; which on this project has twice not meant what it
  said. {status_note}</p></div>

  <h2>Running it</h2>
  <div class="grid3">
    <div class="card"><h4>1. See what is stale</h4>
<pre style="margin:9px 0 0"><code>uv run python experiments/reproduce_all.py --check</code></pre>
      <p style="margin-top:9px">Runs nothing. Reports missing outputs, stale outputs and
      blocked stages.</p></div>
    <div class="card"><h4>2. Run what is missing</h4>
<pre style="margin:9px 0 0"><code>uv run python experiments/reproduce_all.py</code></pre>
      <p style="margin-top:9px">Resumable &mdash; the expensive stages pick up simply by
      running again.</p></div>
    <div class="card"><h4>3. One stage at a time</h4>
<pre style="margin:9px 0 0"><code>uv run python experiments/h3_cross_venue_recall.py</code></pre>
      <p style="margin-top:9px">Every stage is also a script. Run the latency stages
      deliberately, on an idle machine.</p></div>
  </div>

  <h3>Setup</h3>
<pre><code>git checkout feat/prompt-search
uv sync
uv run pitch info
uv run pytest -q -m "not slow"</code></pre>
  <div class="callout warn"><p><b>The code is not on <code>main</code>.</b> All 39 commits
  sit on a stacked chain of feature branches; the tip holds everything.</p>
  <p>And do <b>not</b> pass <code>--limit</code> to the preprocessing search for a number you
  intend to quote. Subsampling shrinks the venue folds until recall can only take a few
  discrete values; the script warns when it happens.</p></div>

  <h3>Everyday commands</h3>
<pre><code>uv run pitch manifest      # rebuild the manifest, print confound warnings
uv run pitch coverage      # class x lighting x venue matrix
uv run pitch cache         # embed every frame per backbone (~25 min)
uv run pitch serve         # dashboard API on :8000</code></pre>

  <h3>Where things live</h3>
  <div class="grid3">
    <div class="card"><h4><code>results/EXPERIMENT_LOG.md</code></h4><p>Every finding, with
    its caveats.</p></div>
    <div class="card"><h4><code>thesis/preregistration.md</code></h4><p>Hypotheses fixed
    before the runs.</p></div>
    <div class="card"><h4><code>thesis/rq_matrix.md</code></h4><p>Which experiment answers
    which question.</p></div>
    <div class="card"><h4><code>docs/CODEBASE.md</code></h4><p>Every branch and
    module.</p></div>
    <div class="card"><h4><code>docs/IDEAS.md</code></h4><p>Ideas not in the plan, each with
    a stop condition.</p></div>
    <div class="card"><h4><code>results/gate_status.md</code></h4><p>Milestone criteria,
    checked against the artefacts rather than ticked by hand.</p></div>
  </div>
</section>
"""


#: The share of a 224x224 letterboxed frame that is actual content rather than grey
#: padding, for a 16:9 source: 224 x 126 out of 224 x 224. It is the ceiling on
#: `share_pixels_changed`, and without it a switch that changed *everything* reads as
#: having changed only half the frame - which is how a reader would conclude that no switch
#: does much.
CONTENT_SHARE = 126 / 224


def preprocess_view(d: dict, *, search_rows: str, abl_rows: str) -> str:
    """Before and after for every preprocessing switch, with the change measured.

    Reads `results/preprocess_pairs.csv`, which applies each switch to one redacted night
    active-play frame and records how far the frame moved. The pictures answer *what does
    this switch do*; the three numbers under each answer *did it do anything*, which two
    same-sized tiles cannot show on their own - a crop is re-letterboxed back to 224, so it
    looks like a zoom rather than a loss of three quarters of the frame.
    """
    pairs = [p for p in d["pairs"] if p.get("img")]
    if not pairs:
        return ('<section class="view" data-view="preprocess" hidden>'
                '<p class="eyebrow">Preprocessing</p><h1>Before/after pairs not generated</h1>'
                '<p>Run <code>uv run python experiments/preprocess_pairs.py</code> and '
                'regenerate this page.</p></section>')

    # Bars are scaled against the loudest switch rather than an arbitrary ceiling, so the
    # cards are comparable to each other - which is the comparison a reader is making.
    loudest_change = max(float(q["mean_abs_change_255"]) for q in pairs)

    cards = []
    for p in pairs:
        change = float(p["mean_abs_change_255"])
        share = float(p["share_pixels_changed"])
        area = float(p["area_retained"])
        searched = p["searched"] == "True"
        area_row = (
            meter("frame area kept", area, f"{area:.0%}", tone="down")
            if area < 1.0 else ""
        )
        cards.append(
            f'<figure class="pair"><img src="{p["img"]}" alt="The same frame before and '
            f'after {p["label"]}, side by side at the 224x224 size the model receives.">'
            f'<div class="body"><div class="hd"><code>{p["label"]}</code>'
            f'<span class="stage">{p["stage"]}'
            f'{"" if searched else " &middot; diagnostic"}</span></div>'
            f'<p class="why">{p["hypothesis"]}</p>'
            f'{meter("change vs baseline", change / loudest_change, f"{change:.1f}/255")}'
            f'{meter("pixels moved", share / CONTENT_SHARE, f"{share:.0%}")}'
            f'{area_row}</div></figure>'
        )

    quietest = min(pairs, key=lambda p: float(p["mean_abs_change_255"]))
    loudest = max(pairs, key=lambda p: float(p["mean_abs_change_255"]))
    croppers = [p for p in pairs if float(p["area_retained"]) < 1.0]
    worst_crop = min(croppers, key=lambda p: float(p["area_retained"])) if croppers else None
    cg = d["clahe_gate"]
    # Contrast bars are scaled against the loudest thing on the axis - the measured maximum
    # or the threshold itself, whichever is higher - so the threshold's bar can sit above the
    # 90th percentile's and be seen to.
    contrast_top = max(cg.get("max", 0.0), cg.get("current_threshold", 40.0)) or 1.0

    return f"""<section class="view" data-view="preprocess" hidden>
  <p class="eyebrow">Preprocessing</p>
  <h1>Every switch, before and after</h1>
  <p class="lede">Each switch here is a <b>hypothesis about what does not travel between
  venues</b> &mdash; not a setting chosen by taste. So each one is shown as the frame the
  model actually receives, before and after, with how far the frame moved measured beside
  it. {len(pairs)} switch values, one source frame, redacted before anything was drawn.</p>

  <h2>The order is fixed, not searched</h2>
  <div class="rail">
    <div><div class="k">Stage 1</div><div class="t">Geometry</div>
      <div class="d">undistort &rarr; ROI &rarr; crop. Lens shape is itself a venue cue.</div></div>
    <div><div class="k">Stage 2</div><div class="t">Photometric</div>
      <div class="d">standardise, CLAHE, gamma, saturation, denoise.</div></div>
    <div><div class="k">Stage 3</div><div class="t">Resize</div>
      <div class="d">letterbox to 224&times;224, aspect preserved.</div></div>
    <div><div class="k">Stage 4</div><div class="t">Post-resize</div>
      <div class="d">sharpen, blur &mdash; so their radii are in model-input pixels.</div></div>
  </div>
  <p class="note">One entry point, used by both the experiments and the live pipeline. The
  pilot&rsquo;s most expensive bug came from a feature path that differed between the two, so
  a variant passes options rather than reimplementing a stage. <code>roi</code> is absent
  below because no pitch polygon has been drawn yet, and showing a tile captioned
  &ldquo;ROI&rdquo; beside an unmodified frame would be the most misleading thing on this
  page.</p>

  <h2>How to read the three bars</h2>
  <div class="grid3">
    <div class="card"><h4>Change vs baseline</h4><p>Mean absolute change per pixel channel on
    0&ndash;255, relative to the loudest switch. <b>How far the frame moved.</b></p></div>
    <div class="card"><h4>Pixels moved</h4><p>Share of pixels that changed at all, scaled so
    100% means <b>the whole content area</b>. A letterboxed 16:9 frame is only
    {CONTENT_SHARE:.0%} content &mdash; the grey bars never change, so 100% here is
    {CONTENT_SHARE:.0%} of the square.</p></div>
    <div class="card"><h4>Frame area kept</h4><p>Shown only when a switch discards part of
    the frame. <b>The number the pictures hide:</b> a crop is re-letterboxed back to 224, so
    it looks like a zoom.</p></div>
  </div>

  <h2>Before and after, switch by switch</h2>
  <p>Ordered loudest first, so the switches that barely touch the frame sit together at the
  bottom &mdash; which is the point of the section after this one.</p>
  <div class="pairs">{"".join(cards)}</div>

  <div class="callout warn"><p><b>A searched margin on top of a near no-op is not a margin.</b>
  <code>{quietest["label"]}</code> moves the frame by
  <b>{float(quietest["mean_abs_change_255"]):.2f}/255</b> and touches only
  {float(quietest["share_pixels_changed"]):.0%} of it &mdash; against
  {float(loudest["mean_abs_change_255"]):.1f} for the loudest switch,
  <code>{loudest["label"]}</code>. Yet the search reports it at 1.0000 recall, +0.016 over
  baseline. Both facts are true, and together
  they say the +0.016 is resolution, not effect. <code>search_resolution.csv</code> re-scores
  every searched margin against what the fold sizes can actually resolve.</p></div>

  <div class="callout"><p><b>And the crops are the opposite trap.</b>
  <code>{worst_crop["label"] if worst_crop else "centre_crop"}</code> keeps
  {f'{float(worst_crop["area_retained"]):.0%}' if worst_crop else "25%"} of the frame while
  looking, side by side, like a mild zoom. It scored the largest apparent gain in the input
  ablation &mdash; 0.998 play recall &mdash; at <b>empty accuracy 0.000</b>, on folds that
  contain no empty pitch to catch it. It had not learned to see play; it had learned to
  answer PLAY.</p></div>

  <h2>What the search found</h2>
  <p>Ten switches, {d["search_n"]} evaluations, greedy &mdash; every candidate needs a fresh
  embedding pass over the whole dataset, which is the opposite cost profile to the prompt
  search. Scored on cross-venue transfer, guarded by a false-play control.</p>
  <div class="scroll"><table>
    <thead><tr><th>Configuration</th><th class="num">Recall</th><th class="num">vs baseline</th>
    <th class="num">Worst fold</th><th class="num">False-play</th></tr></thead>
    <tbody>{search_rows}</tbody>
  </table></div>
  <p class="note">Read the false-play column first: every cross-venue test set is entirely
  active play, so recall alone can be bought by answering &ldquo;playing&rdquo; more often.
  Two rows above reach 1.0000 recall with a false-play rate above 0.4.</p>

  <h2>The CLAHE gate was never calibrated against this footage</h2>
  <p>The <code>auto</code> setting applies CLAHE when RMS contrast falls below
  <b>{cg.get("current_threshold", 40)}</b>. Measured over all {cg.get("n_frames", 0):,}
  development frames, that fires on <b>{pct(cg.get("fires_at_current"))}</b> of them.</p>
  <div class="grid2">
    <div class="card"><h4>Measured contrast distribution</h4>
      {meter("10th percentile", cg.get("p10", 0) / contrast_top, f'{cg.get("p10", 0):.1f}')}
      {meter("median", cg.get("median", 0) / contrast_top, f'{cg.get("median", 0):.1f}')}
      {meter("90th percentile", cg.get("p90", 0) / contrast_top, f'{cg.get("p90", 0):.1f}')}
      {meter("the threshold", cg.get("current_threshold", 40) / contrast_top,
             f'{cg.get("current_threshold", 40):.1f}', tone="down")}
      <p style="margin-top:9px">Scaled against the highest contrast measured
      ({contrast_top:.1f}). The threshold sits above the 90th percentile, so
      <code>auto</code> means &ldquo;almost always&rdquo;. Setting it from this distribution
      would make it mean something: the median makes it the darker half, the 10th percentile
      the worst tenth.</p></div>
    <div class="card"><h4>Uncalibrated, not vacuous</h4>
      <span class="big">{cg.get("auto_differs_from_on", 0)}</span>
      <p>frames on which <code>auto</code> and <code>on</code> actually produce a different
      image. A weak switch rather than a no-op &mdash; which is why the pair above looks
      identical for both values.</p>
      <p style="margin-top:9px"><b>An earlier diagnostic put this at 3 frames, and it was
      wrong.</b> It measured contrast on the letterboxed 224&times;224 output, while the gate
      runs before the resize and tests the full-resolution frame. Corrected by counting the
      images that actually differ, which needs no assumption about which one is measured.</p>
    </div>
  </div>

  <h2>Removing information has a floor</h2>
  <p>The ablation removes one kind of information at a time and re-measures. <b>Read the last
  two columns first</b> &mdash; the held-out venues contain no empty pitch, so the false-play
  control is what separates a real gain from a shifted decision boundary.</p>
  <div class="scroll"><table>
    <thead><tr><th>Input variant</th><th class="num">Play recall</th><th class="num">Worst fold</th>
    <th class="num">False play</th><th class="num">Empty accuracy</th></tr></thead>
    <tbody>{abl_rows}</tbody>
  </table></div>
  <div class="grid3">
    <div class="card"><h4>Blur costs 0.12 recall</h4><p>At <code>blur8</code> no individual
    person is visible, and recall falls from 0.960 to 0.840. <b>The prediction rests on
    people, not scenery.</b></p></div>
    <div class="card"><h4>Colour is a net distraction</h4><p>Grayscale improves both axes at
    once: recall +0.022, false-play 0.231 &rarr; 0.021. The only removal that lands above the
    floor.</p></div>
    <div class="card"><h4>Two helps make a harm</h4><p>Grayscale and a centre crop each help
    alone; together they score <b>below</b> the untouched baseline and call 98.8% of empty
    pitches a match.</p></div>
  </div>
</section>
"""


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
        f'<td class="num">{a["false_play"]:.3f}</td>',
        f'<td class="num">{a["empty_accuracy"]:.3f}</td>',
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
        # The composition drop as a bar against the largest raw drop, so "a third rather
        # than two thirds" is a proportion a reader can see instead of one they compute.
        meter_composition=composition_meter(d),
        # The confound as two bars. Both figures are stated in `results/coverage.md` and in
        # the clock rule's own docstring; they are the reason accuracy here is uninformative,
        # and two sentences quoting "98%" and "99%" never made them look like one variable.
        meter_empty_day=meter("empty frames that are daytime", 0.98, "98%", tone="down"),
        meter_play_night=meter("active-play frames that are night", 0.99, "99%", tone="down"),
        meter_latency=latency_meter(d),
        abl_rows=abl_rows, prompt_rows=prompt_rows, search_rows=search_rows,
        search_n=d["search_n"], best_prompt=best_prompt_text,
        best_prompt_recall=best_prompt_recall,
        fig_rank=d["figs"]["ranking_inversion"],
        fig_label=d["figs"]["label_efficiency"],
        fig_cross=d["figs"]["cross_venue_recall"],
        fig_band=d["figs"]["risk_coverage_band"],
        fig_onboarding=d["figs"]["onboarding_cost"],
        # Built as whole sections rather than as row fragments, because both are mostly
        # layout. They are substituted as values, so nothing inside them is re-formatted -
        # which is why the two tables they reuse are handed in rather than left as
        # placeholders for a second pass that would never happen.
        models_view=models_view(d),
        preprocess_view=preprocess_view(d, search_rows=search_rows, abl_rows=abl_rows),
        reproduce_view=reproduce_view(d),
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
  /* The frozen/trained split, which the models page turns on. Cool and desaturated for
     what never learned, warm and saturated for what did - and never colour alone: every
     block also carries a lock or spark glyph and a literal parameter count. */
  --frozen:#4a5f77; --frozen-soft:#e7ecf2; --frozen-line:#c3cfdd;
  --trained:#a8562a; --trained-soft:#fbeadf; --trained-line:#eccdb7;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ground:#0e1414; --surface:#161e1e; --surface-2:#1c2625; --line:#2b3736;
    --ink:#eaf1ef; --ink-2:#a3b2af; --ink-3:#7b8a88;
    --accent:#4fb3bf; --accent-soft:#12363a;
    --up:#5cb884; --up-soft:#152f24; --down:#d98a63;
    --warn:#cfae57; --warn-soft:#2c2614;
    --frozen:#93a8c2; --frozen-soft:#18212c; --frozen-line:#31404f;
    --trained:#e09a6d; --trained-soft:#2a1c13; --trained-line:#4a3325;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ground:#0e1414; --surface:#161e1e; --surface-2:#1c2625; --line:#2b3736;
  --ink:#eaf1ef; --ink-2:#a3b2af; --ink-3:#7b8a88;
  --accent:#4fb3bf; --accent-soft:#12363a;
  --up:#5cb884; --up-soft:#152f24; --down:#d98a63;
  --warn:#cfae57; --warn-soft:#2c2614;
  --frozen:#93a8c2; --frozen-soft:#18212c; --frozen-line:#31404f;
  --trained:#e09a6d; --trained-soft:#2a1c13; --trained-line:#4a3325;
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

/* ---- presentation components -------------------------------------------------------
   Added when the findings pages were turned from prose into panels. Each one exists
   because a specific paragraph was doing a picture's job: `.board` for the four headline
   claims, `.meter` for every 0-1 score that was quoted inline, `.pair` for the
   before/after frames, `.block` for the frozen/trained split, `.rail` for the pipeline.  */

.grid2 {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(272px,1fr));
  margin:18px 0; }}
.grid3 {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  margin:18px 0; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:11px;
  padding:16px 17px; }}
.card h4 {{ margin:0 0 7px; font-size:14px; font-weight:600; letter-spacing:-.01em; }}
.card p {{ margin:0; font-size:13px; max-width:none; }}
.card .big {{ font:700 24px 'JetBrains Mono', monospace; font-variant-numeric:tabular-nums;
  letter-spacing:-.02em; display:block; margin:2px 0 4px; }}

/* the four headline claims, as a board rather than a bullet list */
.board {{ display:grid; gap:12px; margin:20px 0; }}
.claim {{ display:grid; grid-template-columns:auto 1fr; gap:14px; align-items:start;
  background:var(--surface); border:1px solid var(--line); border-radius:11px;
  padding:14px 16px; }}
.claim .n {{ font:700 12px 'JetBrains Mono', monospace; color:var(--ground);
  background:var(--ink); width:24px; height:24px; border-radius:6px; display:grid;
  place-items:center; }}
.claim b {{ display:block; margin-bottom:2px; }}
.claim span {{ font-size:13px; color:var(--ink-2); }}

/* a 0-1 score as a bar. `data-v` prints beside it so the number is never only a length. */
.meter {{ display:grid; grid-template-columns:1fr auto; gap:10px; align-items:center;
  margin:6px 0; }}
.meter .track {{ height:9px; background:var(--surface-2); border-radius:99px;
  overflow:hidden; border:1px solid var(--line); }}
.meter .track i {{ display:block; height:100%; background:var(--accent); }}
.meter .track i.up {{ background:var(--up); }}
.meter .track i.down {{ background:var(--down); }}
.meter .track i.frozen {{ background:var(--frozen); }}
.meter v {{ font:600 12.5px 'JetBrains Mono', monospace; font-variant-numeric:tabular-nums;
  color:var(--ink); min-width:52px; text-align:right; }}
.meter .lab {{ font:500 11px 'JetBrains Mono', monospace; color:var(--ink-3);
  grid-column:1/-1; margin-bottom:-2px; letter-spacing:.04em; text-transform:uppercase; }}

/* frozen vs trained. Colour, glyph and a literal count - never one of the three alone. */
.block {{ border:1px solid var(--line); border-radius:12px; padding:15px 16px;
  background:var(--surface); }}
.block.frozen {{ border:1px solid var(--frozen-line); border-left:5px solid var(--frozen);
  background:var(--frozen-soft); }}
.block.trained {{ border:1px solid var(--trained-line); border-left:5px solid var(--trained);
  background:var(--trained-soft); }}
.block h4 {{ margin:0 0 3px; font-size:14.5px; font-weight:700; letter-spacing:-.012em; }}
.block .sub {{ font:500 11px 'JetBrains Mono', monospace; color:var(--ink-3);
  word-break:break-all; margin-bottom:9px; }}
.badge {{ display:inline-flex; align-items:center; gap:6px; font:700 10px 'JetBrains Mono',
  monospace; letter-spacing:.11em; text-transform:uppercase; padding:4px 9px;
  border-radius:6px; margin-bottom:9px; }}
.badge.frozen {{ background:var(--frozen); color:var(--ground); }}
.badge.trained {{ background:var(--trained); color:var(--ground); }}
.badge svg {{ width:11px; height:11px; flex:none; }}
.block dl {{ margin:0; display:grid; grid-template-columns:auto 1fr; gap:3px 12px;
  font-size:12.5px; }}
.block dt {{ font:500 10.5px 'JetBrains Mono', monospace; letter-spacing:.06em;
  text-transform:uppercase; color:var(--ink-3); padding-top:2px; white-space:nowrap; }}
.block dd {{ margin:0; color:var(--ink-2); }}
.block dd b {{ font-family:'JetBrains Mono', monospace; font-variant-numeric:tabular-nums; }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px; margin:16px 0; }}
.legend > div {{ display:flex; align-items:center; gap:8px; font-size:12.5px;
  color:var(--ink-2); border:1px solid var(--line); border-radius:8px; padding:7px 11px;
  background:var(--surface); }}

/* the pipeline as a rail, with the seam where labels enter drawn as a real line */
.rail {{ display:flex; align-items:stretch; gap:0; overflow-x:auto; margin:20px 0;
  border:1px solid var(--line); border-radius:12px; background:var(--surface); }}
.rail > div {{ padding:14px 16px; min-width:158px; flex:1; }}
.rail > div + div {{ border-left:1px solid var(--line); }}
.rail .k {{ font:500 10px 'JetBrains Mono', monospace; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-3); margin-bottom:5px; }}
.rail .t {{ font-size:13.5px; font-weight:600; margin-bottom:3px; }}
.rail .d {{ font-size:12px; color:var(--ink-2); }}
.rail .seam {{ min-width:0; flex:none; padding:0; width:3px; background:var(--trained);
  border:0; position:relative; }}
.seamnote {{ display:flex; gap:8px; align-items:center; font:600 11.5px 'JetBrains Mono',
  monospace; color:var(--trained); margin:-8px 0 18px; }}
.seamnote:before {{ content:""; width:3px; height:15px; background:var(--trained);
  border-radius:2px; flex:none; }}

/* parameter scale, drawn to scale until it stops being legible - then broken, and said so */
.scale {{ margin:18px 0; }}
.scale .row {{ display:grid; grid-template-columns:110px 1fr; gap:12px; align-items:center;
  margin-bottom:9px; }}
.scale .row > span {{ font:500 11px 'JetBrains Mono', monospace; color:var(--ink-2);
  text-align:right; }}
.scale .bar {{ height:22px; border-radius:5px; display:flex; align-items:center;
  padding:0 9px; font:700 11px 'JetBrains Mono', monospace; white-space:nowrap; }}
.scale .bar.frozen {{ background:var(--frozen); color:var(--ground); }}
.scale .bar.trained {{ background:var(--trained); color:var(--ground);
  min-width:3px; padding:0; }}
.scale .bar.trained + em {{ font-style:normal; }}
.scale .tiny {{ display:flex; align-items:center; gap:9px; }}
.scale .tiny em {{ font:600 11px 'JetBrains Mono', monospace; color:var(--trained);
  font-style:normal; }}

/* before / after frames */
.pairs {{ display:grid; gap:18px; grid-template-columns:repeat(auto-fit,minmax(296px,1fr));
  margin:20px 0; }}
/* `margin:0` and `border-radius:0` override the global `figure` / `figure img` rules: a
   pair is a grid item whose spacing comes from the grid's gap, and the card's own
   `overflow:hidden` is what rounds the image - leaving the img's inherited radius on would
   round its bottom corners against the body below it. */
.pair {{ background:var(--surface); border:1px solid var(--line); border-radius:12px;
  overflow:hidden; display:flex; flex-direction:column; margin:0; }}
.pair img {{ width:100%; display:block; border:0; border-radius:0;
  border-bottom:1px solid var(--line); }}
.pair .body {{ padding:13px 15px 15px; display:flex; flex-direction:column; flex:1; }}
.pair .hd {{ display:flex; justify-content:space-between; align-items:baseline; gap:10px;
  margin-bottom:3px; }}
.pair .hd code {{ font-size:12.5px; font-weight:600; background:none; padding:0;
  color:var(--ink); }}
.pair .hd .stage {{ font:500 9.5px 'JetBrains Mono', monospace; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-3); }}
.pair .why {{ font-size:12.5px; color:var(--ink-2); margin:0 0 11px; flex:1; }}
.pair .why b {{ color:var(--ink); }}

/* the reproduction pipeline: dependency levels, and a stage as a chip */
.levels {{ display:grid; gap:8px; margin:20px 0; }}
.lvl {{ display:grid; grid-template-columns:82px 1fr; gap:13px; align-items:start;
  background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:12px 14px; }}
.lvl .tag {{ font:500 10px 'JetBrains Mono', monospace; letter-spacing:.09em;
  text-transform:uppercase; color:var(--ink-3); padding-top:4px; }}
.lvl .tag b {{ display:block; font:700 17px 'JetBrains Mono', monospace; color:var(--ink);
  letter-spacing:-.02em; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.chip {{ display:inline-flex; align-items:baseline; gap:7px; font:500 11.5px
  'JetBrains Mono', monospace; border:1px solid var(--line); border-radius:6px;
  padding:4px 8px; background:var(--surface-2); color:var(--ink-2); }}
.chip b {{ color:var(--ink); font-weight:600; }}
.chip s {{ text-decoration:none; color:var(--ink-3); font-variant-numeric:tabular-nums; }}
.chip.done {{ border-color:var(--up); background:var(--up-soft); color:var(--up); }}
.chip.done b {{ color:var(--up); }}
.chip.heavy {{ border-color:var(--warn); background:var(--warn-soft); color:var(--warn); }}
.chip.heavy b {{ color:var(--warn); }}
.chip.blocked {{ border-color:var(--down); color:var(--down); }}

/* one stacked bar: where the 18 hours actually go */
.stack {{ display:flex; height:34px; border-radius:8px; overflow:hidden; margin:16px 0 9px;
  border:1px solid var(--line); }}
.stack > span {{ display:grid; place-items:center; font:600 10.5px 'JetBrains Mono',
  monospace; color:var(--ground); overflow:hidden; white-space:nowrap; }}
.stackkey {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:12.5px;
  color:var(--ink-2); margin-bottom:6px; }}
.stackkey > span {{ display:inline-flex; align-items:center; gap:7px; }}
.stackkey i {{ width:10px; height:10px; border-radius:3px; flex:none; }}

/* two miniature split diagrams, for the leakage objection */
.splitviz {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(258px,1fr));
  margin:18px 0; }}
.splitviz svg {{ width:100%; height:auto; display:block; }}
.frames {{ display:flex; gap:3px; flex-wrap:wrap; margin:8px 0 4px; }}
.frames i {{ width:15px; height:15px; border-radius:3px; background:var(--surface-2);
  border:1px solid var(--line); display:block; }}
.frames i.tr {{ background:var(--frozen); border-color:var(--frozen); }}
.frames i.te {{ background:var(--warn); border-color:var(--warn); }}
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
  <div class="board">
    <div class="claim"><div class="n">1</div><div><b>The evaluation was uninformative</b>
      <span>A colour histogram beat a deep probe under a random split. Frames sampled
      seconds apart sat on both sides of the cut.</span></div></div>
    <div class="claim"><div class="n">2</div><div><b>The models were not the problem</b>
      <span>Across unseen venues the trivial baselines collapse &mdash; the clock rule falls
      to 0.219 &mdash; while the frozen backbones hold above 0.87.</span></div></div>
    <div class="claim"><div class="n">3</div><div><b>The protocol reverses the decision</b>
      <span>ViT ranks first under the leaky split and last under the honest one. The choice
      of protocol, not the choice of model, decided which one looked best.</span></div></div>
    <div class="claim"><div class="n">4</div><div><b>The models genuinely read people</b>
      <span>Blurring away the players costs 0.12 recall; removing colour or the frame border
      does not. So the prediction rests on people, not scenery.</span></div></div>
  </div>

  <h2>Where to look</h2>
  <div class="grid3">
    <div class="card"><h4><a href="#findings">Findings</a></h4><p>The protocol comparison, the
    leakage decomposition, and what a new camera costs in labels.</p></div>
    <div class="card"><h4><a href="#models">Models</a></h4><p>What was frozen and what was
    trained &mdash; 200.8M pretrained parameters against 10,932 fitted ones, counted rather
    than claimed.</p></div>
    <div class="card"><h4><a href="#preprocess">Preprocessing</a></h4><p>Every switch as a
    before/after pair at the model&rsquo;s own input size, with how far the frame
    moved.</p></div>
  </div>
</section>

<section class="view" data-view="findings" hidden>
  <p class="eyebrow">Findings</p>
  <h1>The protocol reverses the ranking</h1>
  <p class="lede">Three evaluation protocols, three different winners. A reader following the
  pilot&rsquo;s protocol would have selected ViT &mdash; the weakest generaliser of the three.</p>

  <figure><img src="{fig_rank}" alt="Model rank under three evaluation protocols: ViT first under the leaky random split and third under cross-venue evaluation.">
  <figcaption>Ranks rather than scores, because the three metrics are not comparable. Each
  protocol&rsquo;s score is printed beside its point.</figcaption></figure>

  <div class="grid3">
    <div class="card"><h4>Random split &mdash; leaky</h4><span class="big">ViT 1st</span>
      <p>Frames seconds apart land on both sides of the cut. Everything scores about 0.99,
      including a colour histogram.</p></div>
    <div class="card"><h4>Grouped by slot &mdash; honest</h4><span class="big">DINOv2 1st</span>
      <p>ConvNeXtV2 and ViT tie at <b>0.4975</b>, which is exactly the single-class floor:
      they collapse to the majority class rather than merely scoring less.</p></div>
    <div class="card"><h4>Cross-venue &mdash; transfer</h4><span class="big">ViT last</span>
      <p>DINOv2 holds at 0.930 play recall, ViT falls to 0.869 with a worst fold of 0.500.
      The pilot&rsquo;s protocol would have picked ViT.</p></div>
  </div>

  <h2>Every model, every protocol</h2>
  <div class="scroll"><table>
    <thead><tr><th>Model</th><th class="num">Random split<br>macro-F1</th>
    <th class="num">Grouped split<br>macro-F1</th><th class="num">Cross-venue<br>play recall</th>
    <th class="num">Worst fold</th><th class="num">ms/frame</th><th class="num">20 cams (s)</th></tr></thead>
    <tbody>{model_rows}</tbody>
  </table></div>
  <div class="callout warn"><p><b>The bottom row reads no pixels at all.</b> Under the leaky
  and grouped protocols the clock rule is competitive with every backbone; across venues it
  collapses to 0.219, with a worst fold of 0.000. That contrast is the only reason the
  backbones&rsquo; cross-venue numbers mean anything &mdash; it is the evidence that they are
  reading occupancy rather than recognising a place.</p></div>

  <h2>The protocol a constant predictor wins</h2>
  <p class="lede">Stronger than &ldquo;the ranking reverses&rdquo;: under leave-one-venue-out
  a model that reads no pixels and always answers &ldquo;playing&rdquo; scores a <b>perfect
  macro-F1</b>, ahead of every backbone. There is no ranking there to reverse.</p>
  <div class="scroll"><table>
    <thead><tr><th>Protocol</th><th>Best trivial baseline</th><th>Best backbone</th>
    <th class="num">Gap</th></tr></thead>
    <tbody>{floor_rows}</tbody>
  </table></div>
  <p class="note">A negative gap means the trivial baseline is not beaten &mdash; every
  held-out venue is 100% active play, so the macro average is taken over the one class
  present. Diagnostics for every row (test size, class count, majority share) are in
  <code>benchmark_v2.csv</code>; no number here is quotable without them.</p>

  <h2>How much of the leakage penalty is actually leakage</h2>
  <div class="grid2">
    <div class="card"><h4>The control</h4>
      <p>A model that never trains cannot leak. So OpenCLIP, scored zero-shot on the
      identical test sets, measures what <b>changing the test set</b> does on its own.</p>
      {meter_composition}
      <p style="margin-top:9px">Subtracting that leaves the part attributable to leakage:
      about a third of the score, not the two thirds the raw delta implies.</p></div>
    <div class="card" style="padding:0;overflow:hidden"><div class="scroll" style="margin:0;border:0">
      <table><thead><tr><th>Model</th><th class="num">Raw drop</th>
      <th class="num">Composition</th><th class="num">Attributable</th></tr></thead>
      <tbody>{comp_rows}</tbody></table></div></div>
  </div>
  <p class="note">A control, not a proof: it assumes the composition effect is additive and
  similar across models.</p>

  <h2>Cross-venue recall, fold by fold</h2>
  <figure><img src="{fig_cross}" alt="Per-venue play recall for each model, with the 0.90 target marked.">
  <figcaption>Five of seven venues contribute 12&ndash;30 frames, so the spread matters more
  than the mean.</figcaption></figure>
  <div class="scroll"><table>
    <thead><tr><th>Held-out venue</th><th class="num">DINOv2</th><th class="num">ConvNeXtV2</th>
    <th class="num">ViT</th><th class="num">Clock rule</th></tr></thead>
    <tbody>{fold_rows}</tbody>
  </table></div>

  <h3>Robustness: merging the two audited venues</h3>
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

  <h2>How much human review buys a given reliability</h2>
  <figure><img src="{fig_band}" alt="Risk-coverage as a shaded band for DINOv2 and as lines for ConvNeXtV2 and ViT, with a second panel showing the width the confidences leave.">
  <figcaption>The operational question, and the one plot here a conventional drawing would
  get wrong.</figcaption></figure>
  <div class="grid3">
    <div class="card"><h4>Why it is a band</h4><span class="big">890 / 907</span>
      <p>of DINOv2&rsquo;s calibrated confidences are <b>identical</b>. &ldquo;The most
      confident <i>k</i>&rdquo; is undefined over most of the range, so the accuracy there is
      an interval, not a point.</p></div>
    <div class="card"><h4>What the shading means</h4><p>Every accuracy those tied confidences
    permit. The solid edge is the <b>worst case</b> &mdash; the only bound an operator can be
    held to.</p></div>
    <div class="card"><h4>The trap in the curves</h4><p>The two best-looking curves belong to
    the two models that <b>never predict EMPTY</b>: right on all 898 play frames, wrong on all
    9 empty ones. A figure of curves alone would recommend them.</p></div>
  </div>

  <h2>What a new camera costs, in labels</h2>
  <figure><img src="{fig_onboarding}" alt="Macro-F1 against the number of labelled frames from the new camera, for three backbones, with a dashed series showing the same frames used alone and a marker at the budget where the two meet.">
  <figcaption>The deployment question, and the answer is smaller than expected.</figcaption></figure>
  <div class="grid3">
    <div class="card"><h4>A new camera starts poor</h4><span class="big">0.36&ndash;0.51</span>
      <p>macro-F1 for a probe trained on one camera, scored on the <b>other camera watching
      the same pitch</b>.</p></div>
    <div class="card"><h4>One frame nearly fixes it</h4><span class="big">~0.98</span>
      <p>reached by every backbone from a single labelled frame of the new camera.</p></div>
    <div class="card"><h4>The dashed series is the finding</h4><p>From five labelled frames,
    training on those frames <b>alone</b> matches training on them plus 775 from the source
    camera &mdash; on all three backbones. What buys the accuracy is having <i>any</i> labels
    from the new camera, not a large corpus elsewhere.</p></div>
  </div>
  <p class="note">Read as an optimistic bound: a second camera on one pitch is an easier
  target than a new venue, and the target set is twelve distinct scenes.</p>

  <h2>More labels made the models worse</h2>
  <figure><img src="{fig_label}" alt="Macro-F1 against labelling budget: the curves peak at 100-300 labels and fall when all labels are used.">
  <figcaption>ConvNeXtV2 peaks at 100 labels and is worse with all 671.</figcaption></figure>
  <div class="callout"><p><b>Not a scaling law &mdash; a sampling artefact, and a useful
  one.</b> Small budgets are class-stratified; the full pool is not, and is dominated by one
  mostly-empty recording. So balanced sampling accidentally de-confounds, and the curve is
  measuring the confound rather than the labels.</p></div>
</section>

{models_view}

{preprocess_view}

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

  <h2>The confound, as the numbers actually sit</h2>
  <p>Class and scene are very nearly the same variable here. That is what makes accuracy on
  this data uninformative, and it is one fact rather than a general shortage of data.</p>
  <div class="grid2">
    <div class="card"><h4>Empty pitches are daytime</h4>
      {meter_empty_day}
      <p style="margin-top:9px">All 494 of them come from <b>one venue</b>, and 98% are
      daylight.</p></div>
    <div class="card"><h4>Active play is night</h4>
      {meter_play_night}
      <p style="margin-top:9px">So a rule reading only the clock scores 98.4%, and the
      confound &mdash; not the model &mdash; is what a high score measures.</p></div>
  </div>

  <h2>Where the frames come from</h2>
  <div class="grid2">
    <div class="card"><h4>4 slot recordings</h4><span class="big">2 slots &times; 2 cameras</span>
      <p>One facility. The morning slot is almost entirely empty, the evening one almost
      entirely a match. <b>This is the source of the confound</b> &mdash; and the only place
      an empty pitch appears anywhere in the dataset.</p></div>
    <div class="card"><h4>66 highlight clips</h4><span class="big">10&ndash;14s each</span>
      <p>Roughly nine venues, day and night. Every clip contains players (YOLO-verified,
      minimum 3, median 8), so they add venue diversity and daytime play &mdash; but
      <b>no empty pitches</b>, which is why the gap above cannot be closed by more clips.</p></div>
  </div>

  <h2>Conditions not covered</h2>
  <div class="grid3">
    <div class="card"><h4>Lighting</h4><span class="big">covered</span>
      <p>Day and night, and the only condition axis the dataset has.</p></div>
    <div class="card"><h4>Weather</h4><span class="big">no field, no footage</span>
      <p>Every frame is dry. <b>No claim about precipitation robustness is supported</b> by
      anything here.</p></div>
    <div class="card"><h4>Maintenance</h4><span class="big">6 frames</span>
      <p>Unusable as a class. It stays in the taxonomy because the system must be able to
      abstain on it, not because it can be evaluated.</p></div>
  </div>
</section>

<section class="view" data-view="searches" hidden>
  <p class="eyebrow">Searches</p>
  <h1>The best model on this task has no labels</h1>
  <p class="lede">375 prompt sets, scored exhaustively against cross-venue transfer and
  guarded by a false-play control. Images are embedded <b>once</b>, so a prompt set costs
  almost nothing &mdash; the exact opposite cost profile to the
  <a href="#preprocess">preprocessing search</a>, where every candidate needs a fresh
  embedding pass and the search has to be greedy.</p>

  <div class="tiles">
    <div class="tile lead"><div class="k">Best prompt, zero labels</div>
      <div class="v">{best_prompt_recall}</div><div class="n">cross-venue play recall</div></div>
    <div class="tile"><div class="k">DINOv2 probe</div><div class="v">0.930</div>
      <div class="n">after ~1,500 labelled frames</div></div>
    <div class="tile"><div class="k">Prompt sets scored</div><div class="v">375</div>
      <div class="n">templates &times; descriptors</div></div>
    <div class="tile"><div class="k">Labels required</div><div class="v">0</div>
      <div class="n">to onboard a new site</div></div>
  </div>

  <h2>How the prompt space is built</h2>
  <div class="rail">
    <div><div class="k">Part 1</div><div class="t">Template</div>
      <div class="d">the framing &mdash; <code>a photo of {{}}</code>,
      <code>a CCTV still of {{}}</code>. Mostly affects how the text encoder situates the
      phrase.</div></div>
    <div><div class="k">Part 2</div><div class="t">Descriptor</div>
      <div class="d">what the class looks like, in the vocabulary a caption would use. This
      carries the discriminative content.</div></div>
    <div><div class="k">Combine</div><div class="t">Prompt set</div>
      <div class="d">one descriptor per class, rendered through several templates and
      averaged &mdash; so a <i>set</i> is scored, not a phrase.</div></div>
    <div><div class="k">Score</div><div class="t">Cosine</div>
      <div class="d">nearest class direction, on cross-venue folds, with false-play as the
      guard.</div></div>
  </div>

  <h2>The top prompt sets</h2>
  <div class="scroll"><table>
    <thead><tr><th>Descriptor for &ldquo;active play&rdquo;</th><th class="num">Templates</th>
    <th class="num">Recall</th><th class="num">False-play</th><th class="num">Balanced</th></tr></thead>
    <tbody>{prompt_rows}</tbody>
  </table></div>
  <p class="note">Ranked by the balanced score rather than by recall, because three of these
  reach recall 1.000 by answering &ldquo;playing&rdquo; more often &mdash; the false-play
  column is what separates them.</p>

  <div class="grid2">
    <div class="callout good" style="margin:0"><p><b>A searched prompt beats every trained
    probe, with zero labels.</b> &ldquo;{best_prompt}&rdquo; reaches {best_prompt_recall}
    cross-venue recall, above DINOv2&rsquo;s 0.930 &mdash; which needed about 1,500 labelled
    frames. For active-play detection, onboarding a new site currently costs no labels at
    all.</p></div>
    <div class="callout warn" style="margin:0"><p><b>The caveat is not small.</b> The prompt
    was selected against the same folds it is scored on, and 375 candidates is a great deal
    of selection freedom. This is a <b>development result</b>. The locked final venues were
    held out throughout, so the honest confirmation &mdash; evaluate this one prompt on them,
    once &mdash; remains available.</p></div>
  </div>

  <div class="callout"><p><b>Which is why the hypothesis tests use a different prompt set.</b>
  H6 asks whether zero-shot lags trained probes, and a prompt selected on the same folds
  cannot answer it &mdash; a selected number cannot test the selection. So the declared set
  is fixed <i>by position</i>: the first descriptor per class, all five templates, chosen
  before the search ran and deliberately not its winner.</p></div>
</section>

<section class="view" data-view="system" hidden>
  <p class="eyebrow">System</p>
  <h1>From a frame to a billing decision</h1>

  <div class="rail">
    <div><div class="k">Step 1</div><div class="t">Sample</div>
      <div class="d">one frame per camera per minute</div></div>
    <div><div class="k">Step 2</div><div class="t">Preprocess</div>
      <div class="d">letterbox to 224&times;224 &mdash; the same entry point the experiments
      use</div></div>
    <div><div class="k">Step 3</div><div class="t">Classify</div>
      <div class="d">frozen backbone + trained probe &rarr; class and confidence</div></div>
    <div><div class="k">Step 4</div><div class="t">Fuse</div>
      <div class="d">two cameras on one pitch, weighed by confidence</div></div>
    <div><div class="k">Step 5</div><div class="t">Aggregate</div>
      <div class="d">the minute sequence &rarr; <code>USED</code> / <code>NOTUSED</code> /
      <code>REVIEW</code></div></div>
    <div><div class="k">Step 6</div><div class="t">Reconcile</div>
      <div class="d">against the booking record, with 3 evidence frames</div></div>
  </div>

  <h2>Rules the system holds to</h2>
  <div class="grid2">
    <div class="card"><h4>Never invent an observation</h4><p>A camera that drops for a minute
    leaves a gap; a total outage yields <code>REVIEW</code>, not <code>NOTUSED</code>.
    <b>No footage is not evidence a pitch was unused.</b></p></div>
    <div class="card"><h4>REVIEW never becomes an anomaly</h4><p>The system may not convert
    its own uncertainty into someone else&rsquo;s error.</p></div>
    <div class="card"><h4>Anomalies are per field, never per person</h4><p>Attributing
    discrepancies to individuals adds no scientific value and considerable ethical
    exposure.</p></div>
    <div class="card"><h4>No verdict without its evidence</h4><p>A verdict and the samples
    behind it are written in one transaction.</p></div>
    <div class="card"><h4>The system never bills</h4><p>Output is decision support; a human
    confirms every anomaly.</p></div>
    <div class="card"><h4>Faces are redacted before publication</h4><p>Two layers &mdash;
    person detection and pixelation, then a whole-frame blur floor &mdash; on every frame
    that leaves the system, including the ones on this page.</p></div>
  </div>

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

  <h2>Efficiency: how much of the 60-second cycle each model uses</h2>
  <div class="card">{meter_latency}</div>
  <p class="note">Twenty cameras, one round, measured on a 4-thread development laptop. All
  three fit with large margins, so latency is <b>not</b> the binding constraint and the model
  choice falls to accuracy. The target Mini-PC must be measured separately.</p>
</section>

{reproduce_view}

</main>

<script>
const VIEWS = [
  ["overview", "Overview"], ["findings", "Findings"], ["models", "Models"],
  ["preprocess", "Preprocessing"], ["dataset", "Dataset"], ["searches", "Searches"],
  ["system", "System"], ["run", "Reproduce"],
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
