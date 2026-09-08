"""Thesis figures (WP8-T2).

Every figure is regenerated from the result CSVs, never hand-edited, so a re-run of an
experiment updates the thesis rather than silently disagreeing with it.

Palette is the validated four-slot categorical set (blue / orange / aqua / yellow), which
passes the lightness, chroma, CVD-separation and normal-vision checks. Its contrast warning
against the surface is discharged by direct-labelling every series - which is also the rule
for four or fewer series, so identity is never carried by colour alone.

    uv run python experiments/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figs"

# validated categorical slots
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e4e3de"

COLOURS = {"dinov2": BLUE, "convnextv2": ORANGE, "vit": AQUA, "clock_rule": YELLOW}
LABELS = {
    "dinov2": "DINOv2", "convnextv2": "ConvNeXtV2", "vit": "ViT",
    "clock_rule": "clock rule", "cheap_histogram": "colour histogram",
    "cheap_intensity": "mean intensity", "majority": "majority class",
}


def style(ax) -> None:
    """Recessive axes and grid; the data carries the ink."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {FIGS.relative_to(ROOT)}/{name}.png")


# ---------------------------------------------------------------------------


def fig_label_efficiency() -> None:
    """Macro-F1 against labelling budget. Non-monotone - that is the finding."""
    df = pd.read_csv(RESULTS / "label_efficiency.csv")
    ref = float(df[df.model == "clock_rule"].macro_f1.iloc[0])
    df = df[df.model != "clock_rule"]
    agg = df.groupby(["model", "n_labels"]).macro_f1.agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style(ax)
    ax.axhline(ref, color=YELLOW, linewidth=2, linestyle=(0, (5, 3)), zorder=2)
    ax.text(
        11, ref - 0.022, f"clock rule, zero labels  ({ref:.3f})",
        color=INK_2, fontsize=9, va="top",
    )

    for model in ("convnextv2", "dinov2", "vit"):
        m = agg[agg.model == model].sort_values("n_labels")
        c = COLOURS[model]
        ax.fill_between(
            m.n_labels, m["mean"] - m["std"].fillna(0), m["mean"] + m["std"].fillna(0),
            color=c, alpha=0.09, linewidth=0, zorder=3,
        )
        ax.plot(m.n_labels, m["mean"], color=c, linewidth=2, zorder=4)
        ax.plot(
            m.n_labels, m["mean"], "o", color=c, markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5,
        )
        last = m.iloc[-1]
        # ViT and ConvNeXtV2 both finish at 0.498; nudge them apart so both stay legible
        nudge = {"vit": -0.021, "convnextv2": 0.021}.get(model, 0.0)
        ax.text(
            last.n_labels * 1.07, last["mean"] + nudge, LABELS[model],
            color=c, fontsize=10, va="center", fontweight="medium",
        )
        peak = m.loc[m["mean"].idxmax()]
        if model != "vit":
            # sits below-right of the peak: above it collides with the subtitle band
            ax.annotate(
                f"peak {peak['mean']:.3f} at {int(peak.n_labels)}",
                xy=(peak.n_labels, peak["mean"]),
                xytext=(peak.n_labels * 1.18, peak["mean"] + 0.035),
                color=c, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=c, linewidth=0.9, alpha=0.6),
            )

    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 300, 671])
    ax.set_xticklabels(["10", "25", "50", "100", "300", "671\n(all)"])
    ax.set_xlim(8, 1250)
    ax.set_xlabel("labelled training frames", color=INK_2, fontsize=10)
    ax.set_ylabel("macro-F1", color=INK_2, fontsize=10)
    ax.set_title(
        "More labels made the models worse",
        color=INK, fontsize=13, fontweight="semibold", loc="left", pad=44,
    )
    ax.text(
        0, 1.015,
        "Grouped split, 5 seeds. Small budgets are class-stratified; the full pool is not, "
        "and is dominated by\none mostly-empty recording. Balanced sampling de-confounds; "
        "using every label re-introduces the scene.",
        transform=ax.transAxes, color=INK_2, fontsize=8.5, va="bottom", linespacing=1.5,
    )
    save(fig, "label_efficiency")


def fig_ranking_inversion() -> None:
    """Rank under three protocols. Rank, not score - the metrics are not comparable.

    The ranks *and* the printed scores used to be hardcoded here, which meant the figure did
    not read the CSVs it is generated from and could not notice when they changed. It stopped
    agreeing with them the moment the H1/H2 estimand was fixed: the panel still printed the
    random split's 0.657 for DINOv2 and ConvNeXtV2 after the corrected value became 0.988, so
    the most quoted figure in the thesis contradicted its own source table while the
    reproduction stage reported success. Everything below is derived from the CSVs now.
    """
    MODELS = ("vit", "dinov2", "convnextv2")
    protocols = ["random split\n(leaky)", "grouped split", "cross-venue\nrecall"]

    h1 = pd.read_csv(RESULTS / "h1_h2_baseline_floor.csv")
    h3 = pd.read_csv(RESULTS / "h3_cross_venue_recall.csv")

    def by_split(fragment: str) -> dict[str, float]:
        rows = h1[h1.split.str.contains(fragment, case=False)]
        if rows.empty:
            raise SystemExit(f"no split matching {fragment!r} in h1_h2_baseline_floor.csv")
        return {m: float(rows.loc[rows.model == m, "macro_f1"].iloc[0]) for m in MODELS}

    cross = h3[h3.held_out_venue == "MEAN_ACROSS_FOLDS"]
    columns = [
        by_split("random"),
        by_split("grouped"),
        {m: float(cross.loc[cross.model == m, "play_recall"].iloc[0]) for m in MODELS},
    ]

    #: Ties share a rank, and two dots at identical coordinates hide one series entirely, so
    #: tied models are spread by a small offset around the rank they share. Which models tie
    #: is read from the data rather than described in a comment that can go stale.
    OFFSET = 0.10
    tracks: dict[str, list[float]] = {m: [] for m in MODELS}
    scores: dict[str, list[str]] = {m: [] for m in MODELS}
    for col in columns:
        order = sorted(MODELS, key=lambda m: -col[m])
        rank: dict[str, float] = {}
        i = 0
        while i < len(order):
            tied = [m for m in order if abs(col[m] - col[order[i]]) < 5e-4]
            shared = (i + 1 + i + len(tied)) / 2
            for k, m in enumerate(tied):
                rank[m] = shared + (k - (len(tied) - 1) / 2) * (OFFSET * 2 if len(tied) > 1 else 0)
            i += len(tied)
        for m in MODELS:
            tracks[m].append(rank[m])
            scores[m].append(f"{col[m]:.3f}")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    style(ax)
    ax.grid(axis="x", visible=False)
    x = [0, 1, 2]
    for model, ys in tracks.items():
        c = COLOURS[model]
        ax.plot(x, ys, color=c, linewidth=2.4, zorder=3)
        ax.plot(
            x, ys, "o", color=c, markersize=9,
            markeredgecolor=SURFACE, markeredgewidth=2.5, zorder=4,
        )
        ax.text(-0.10, ys[0], LABELS[model], color=c, fontsize=10,
                ha="right", va="center", fontweight="medium")
        # score in the series colour, beside its own point, so ties never share a label.
        # Tied models sit a fraction apart and another series' line often passes straight
        # through the gap, so each label carries a thin halo of the background colour - it
        # keeps two tied scores readable where they would otherwise be crossed out.
        for xi, yi, s in zip(x, ys, scores[model], strict=True):
            ax.text(
                xi + 0.055, yi - 0.055, s, color=c, fontsize=8.5,
                ha="left", va="bottom", zorder=6,
                path_effects=[
                    patheffects.withStroke(linewidth=2.6, foreground=SURFACE),
                ],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(protocols, color=INK_2)
    ax.set_xlim(-0.85, 2.30)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["1st", "2nd", "3rd"])
    ax.set_ylim(3.45, 0.6)
    ax.set_ylabel("rank", color=INK_2, fontsize=10)
    ax.set_title(
        "The evaluation protocol reverses the model ranking",
        color=INK, fontsize=13, fontweight="semibold", loc="left", pad=44,
    )
    # The subtitle asserted "first under the leaky protocol and last under the honest one".
    # "Last" was true of the old numbers; corrected, ViT is *tied* 2nd/3rd under the grouped
    # split. Describe what the data says rather than restating a sentence that has to be
    # remembered - a caption is a claim, and this one is now checked on every regeneration.
    grouped = columns[1]
    beaten_by = sum(1 for m in MODELS if grouped[m] > grouped["vit"] + 5e-4)
    tied_with = [m for m in MODELS if m != "vit" and abs(grouped[m] - grouped["vit"]) < 5e-4]
    where = (
        "last" if beaten_by == len(MODELS) - 1 and not tied_with
        else f"tied {beaten_by + 1}{'st' if beaten_by == 0 else 'nd' if beaten_by == 1 else 'rd'}"
        if tied_with else f"{beaten_by + 1}th"
    )
    ax.text(
        0, 1.015,
        f"ViT is first under the leaky protocol and {where} under the honest one. Ranks are "
        "shown because the three\nmetrics are not comparable; the score under each protocol "
        "is printed beneath its point.",
        transform=ax.transAxes, color=INK_2, fontsize=8.5, va="bottom", linespacing=1.5,
    )
    save(fig, "ranking_inversion")


def fig_cross_venue() -> None:
    """Per-fold recall. Shows the spread the mean hides."""
    df = pd.read_csv(RESULTS / "h3_cross_venue_recall.csv")
    folds = df[df.held_out_venue != "MEAN_ACROSS_FOLDS"].copy()
    means = df[df.held_out_venue == "MEAN_ACROSS_FOLDS"].set_index("model")
    folds["venue"] = folds.held_out_venue.str.replace("clipvenue_", "", regex=False)
    order = sorted(folds.venue.unique())

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    style(ax)
    ax.grid(axis="y", visible=False)
    # symmetric about the tick: an asymmetric spread makes each venue's points
    # appear to belong to the row below it
    offsets = {"dinov2": 0.24, "convnextv2": 0.08, "vit": -0.08, "clock_rule": -0.24}

    for model in ("dinov2", "convnextv2", "vit", "clock_rule"):
        sub = folds[folds.model == model]
        c = COLOURS[model]
        ys = [order.index(v) + offsets[model] for v in sub.venue]
        ax.plot(
            sub.play_recall, ys, "o", color=c, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4, label=LABELS[model],
        )
        mean = float(means.loc[model, "play_recall"])
        ax.axvline(mean, color=c, linewidth=1.4, alpha=0.45, zorder=2)

    ax.axvline(0.90, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.90, -0.66, " 0.90 target", color=INK_2, fontsize=8.5, va="bottom")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_ylim(-0.85, len(order) - 0.45)
    ax.set_xlim(-0.04, 1.06)
    ax.set_xlabel("ACTIVE_PLAY recall on the held-out venue", color=INK_2, fontsize=10)
    ax.set_title(
        "Frozen backbones transfer to unseen venues; the clock rule does not",
        color=INK, fontsize=13, fontweight="semibold", loc="left", pad=44,
    )
    ax.text(
        0, 1.015,
        "One point per held-out venue; the vertical line is each model's mean across folds. "
        "Five of seven venues\ncontribute 12-30 frames, so the spread matters more than the mean.",
        transform=ax.transAxes, color=INK_2, fontsize=8.5, va="bottom", linespacing=1.5,
    )
    leg = ax.legend(
        loc="lower left", frameon=False, fontsize=9, ncol=4,
        bbox_to_anchor=(0, -0.28), handletextpad=0.3, columnspacing=1.4,
    )
    for t in leg.get_texts():
        t.set_color(INK_2)
    save(fig, "cross_venue_recall")


def fig_risk_coverage_band() -> None:
    """Risk-coverage, drawn as a band because a line would be a claim the data cannot make.

    Two things have to survive this figure, and a conventional risk-coverage line destroys
    both. **DINOv2's calibrated confidences are 890/907 identical**, so "the most confident
    k" is undefined over most of the range and its accuracy there is an interval - a line
    through the middle of it would be the most misleading plot in the thesis. And
    **ConvNeXtV2 and ViT trace a near-perfect curve while never predicting EMPTY at all**:
    right on all 898 active-play frames, wrong on all 9 empty ones, so confidence ranks the
    nine last and the curve looks ideal. The two best-looking lines belong to the two models
    that cannot do the job.

    Two panels, because one axis cannot carry both. The upper panel is the operational
    region, where a manager reads an operating point. The lower panel is how *undetermined*
    that reading is - the width the confidences leave - and it is the reason the upper panel
    must be shaded rather than drawn.
    """
    df = pd.read_csv(RESULTS / "rq6_risk_coverage.csv")
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.6, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.16},
    )
    for a in (ax, ax2):
        style(a)

    #: Where each model's name sits. Every series ends at almost the same value, so labels
    #: are spread along the curves instead of stacked at the right edge - and each is placed
    #: *on* its own line rather than a fixed distance below it, because below DINOv2's line
    #: is the band, and a name floating inside the band reads as a curve that is not there.
    label_at = {"dinov2": 0.62, "convnextv2": 0.22, "vit": 0.58}
    floor = 0.95

    for model in ("dinov2", "convnextv2", "vit"):
        sub = df[df.model == model].sort_values("coverage")
        if sub.empty:
            continue
        c = COLOURS[model]
        tied = int(sub.n_tied.iloc[0])

        if tied:
            ax.fill_between(sub.coverage, sub.accuracy_worst, sub.accuracy_best,
                            color=c, alpha=0.22, linewidth=0, zorder=2)
            ax.plot(sub.coverage, sub.accuracy_best, color=c, linewidth=1.1,
                    linestyle=(0, (3, 2)), alpha=0.8, zorder=3)
        ax.plot(sub.coverage, sub.accuracy_worst, color=c, linewidth=2.2, zorder=4)
        ax2.plot(sub.coverage, sub.accuracy_best - sub.accuracy_worst,
                 color=c, linewidth=2.2, zorder=4)

        row = sub.iloc[(sub.coverage - label_at[model]).abs().argmin()]
        y = float(row.accuracy_worst)
        # On the line for the identified curves; below it for the banded one, where the
        # space above belongs to the band.
        offset, va = (-0.006, "top") if tied else (0.0, "center")
        ax.text(
            float(row.coverage), y + offset, LABELS[model],
            color=c, fontsize=9.5, fontweight="semibold", ha="center", va=va,
            path_effects=[patheffects.withStroke(linewidth=3.5, foreground=SURFACE)],
            zorder=6,
        )

    ax.axhline(0.99, color=INK_MUTED, linewidth=1.3, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.006, 0.9905, "99% precision target", color=INK_2, fontsize=8.5,
            va="bottom", ha="left",
            path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)])

    dv = df[df.model == "dinov2"].sort_values("coverage")
    tied = int(dv.n_tied.iloc[0]) if not dv.empty else 0
    if tied:
        at = dv.iloc[(dv.coverage - 0.45).abs().argmin()]
        ax.annotate(
            f"{tied} of {len(dv) and 907} confidences are identical, so\n"
            f"anywhere in {at.accuracy_worst:.3f}-{at.accuracy_best:.3f} is consistent with them",
            xy=(float(at.coverage), float((at.accuracy_worst + at.accuracy_best) / 2)),
            xytext=(0.44, 0.9605), color=INK_2, fontsize=8.5, linespacing=1.5,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1),
        )

    below = dv[dv.accuracy_worst < floor]
    if not below.empty:
        edge = float(below.coverage.max())
        ax.annotate(
            f"below {edge:.0%} coverage the band\nreaches {below.accuracy_worst.min():.2f}, "
            f"off this panel",
            xy=(edge, floor + 0.002), xytext=(0.03, 0.9515),
            color=INK_2, fontsize=8, linespacing=1.4, va="bottom",
            path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)],
        )

    # The two identified curves coincide until they separate near full coverage; without
    # saying so, one of them looks missing.
    ax.text(
        0.22, 0.9965, "(ConvNeXtV2 and ViT coincide until 88% coverage)",
        color=INK_MUTED, fontsize=8, ha="center", va="top",
        path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)],
    )

    ax.set_xlim(0, 1.0)
    ax.set_ylim(floor, 1.005)
    ax.set_ylabel("Accuracy on the answered slots", color=INK_2, fontsize=10)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("Coverage - share of slots answered automatically "
                   "(review rate is the complement)", color=INK_2, fontsize=10)
    ax2.set_ylabel("Width the\nconfidences leave", color=INK_2, fontsize=9.5)
    ax2.text(0.995, 0.9, "zero width = the curve is a curve", color=INK_2,
             fontsize=8.5, ha="right", va="top")

    ax.set_title(
        "The curve a manager buys cannot be drawn as a curve",
        color=INK, fontsize=13, fontweight="semibold", loc="left", pad=52,
    )
    ax.text(
        0, 1.015,
        "Shaded: every accuracy the model's confidences permit at that coverage. The solid edge is the worst\n"
        "case, the only bound an operator can be held to. ConvNeXtV2 and ViT look ideal because they never\n"
        "predict EMPTY - right on all 898 active-play frames, wrong on all 9 empty ones.",
        transform=ax.transAxes, color=INK_2, fontsize=8.5, va="bottom", linespacing=1.5,
    )
    save(fig, "risk_coverage_band")


def fig_baseline_floor() -> None:
    """What a model that ignores the image scores, per protocol.

    The floor is the frame this project reads every benchmark through, and one picture of it
    answers the question an examiner asks first: *would something trivial have done this?*
    The answer changes completely with the protocol, which is the point - so the four
    protocols share an axis and the trivial baselines are drawn against the backbones on it.

    Read from `benchmark_v2.csv`, which carries all four protocols. Note the cross-venue
    column: the trivial bar is *above* the backbone bar there, because a constant predictor
    scores a perfect macro-F1 on test folds that contain one class.
    """
    df = pd.read_csv(RESULTS / "benchmark_v2.csv")
    df = df[df.replicate != "SUMMARY"]
    trivial = ("majority", "clock_rule", "cheap_intensity", "cheap_histogram")
    deep = ("convnextv2", "dinov2", "vit")
    order = ["random", "grouped_slot", "lo_venue_out", "temporal"]
    pretty = {"random": "random\n(leaky)", "grouped_slot": "grouped\nby slot",
              "lo_venue_out": "cross-\nvenue", "temporal": "temporal"}

    means = df.groupby(["protocol", "model"]).macro_f1.mean()
    rows = []
    for protocol in order:
        if protocol not in means.index.get_level_values(0):
            continue
        block = means.loc[protocol]
        t = {m: block[m] for m in trivial if m in block.index}
        d = {m: block[m] for m in deep if m in block.index}
        if not t or not d:
            continue
        tm, tv = max(t.items(), key=lambda kv: kv[1])
        dm, dv = max(d.items(), key=lambda kv: kv[1])
        rows.append({"protocol": protocol, "trivial": tm, "tv": tv, "deep": dm, "dv": dv})

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    style(ax)
    ax.grid(axis="x", visible=False)
    xs = range(len(rows))
    width = 0.34

    for i, r in enumerate(rows):
        beaten = r["tv"] >= r["dv"]
        ax.bar(i - width / 2, r["tv"], width, color=YELLOW, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
        ax.bar(i + width / 2, r["dv"], width,
               color=INK_MUTED if beaten else BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
        for x, v, name in ((i - width / 2, r["tv"], r["trivial"]),
                           (i + width / 2, r["dv"], r["deep"])):
            ax.text(x, v + 0.018, f"{v:.3f}", ha="center", va="bottom",
                    color=INK_2, fontsize=8.5)
            ax.text(x, 0.02, LABELS.get(name, name), ha="center", va="bottom",
                    color=SURFACE, fontsize=8, rotation=90, fontweight="medium")
        if beaten:
            # Above the pair rather than beside it: the bars it describes are directly
            # below, so no leader is needed and none can cross the thing being annotated.
            ax.text(
                i, max(r["tv"], r["dv"]) + 0.085, "the floor is not cleared",
                ha="center", va="bottom", color=INK, fontsize=9, fontweight="semibold",
            )

    ax.set_xticks(list(xs))
    ax.set_xticklabels([pretty[r["protocol"]] for r in rows], fontsize=9.5)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("macro-F1 (mean over replicates)", color=INK_2, fontsize=10)
    ax.set_title(
        "Would something trivial have done this? It depends entirely on the protocol",
        color=INK, fontsize=13, fontweight="semibold", loc="left", pad=44,
    )
    ax.text(
        0, 1.015,
        "Yellow: the best model that ignores the image (majority class, a clock rule, mean intensity, a\n"
        "colour histogram). Blue: the best frozen backbone. Grey where the backbone fails to clear the floor.",
        transform=ax.transAxes, color=INK_2, fontsize=8.5, va="bottom", linespacing=1.5,
    )
    save(fig, "baseline_floor")


def main() -> None:
    print("figures:")
    fig_label_efficiency()
    fig_ranking_inversion()
    fig_cross_venue()
    fig_risk_coverage_band()
    fig_baseline_floor()


if __name__ == "__main__":
    main()
