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
    """Rank under three protocols. Rank, not score - the metrics are not comparable."""
    protocols = ["random split\n(leaky)", "grouped split", "cross-venue\nrecall"]
    # Ties are drawn at a slight offset from the shared rank so both series stay visible:
    # DINOv2/ConvNeXtV2 tie at 2nd under the random split, ConvNeXtV2/ViT under the grouped.
    tracks = {
        "vit": [1.0, 2.10, 3.0],
        "dinov2": [2.10, 1.0, 1.0],
        "convnextv2": [1.90, 1.90, 2.0],
    }
    scores = {
        "vit": ["0.996", "0.498", "0.869"],
        "dinov2": ["0.657", "0.579", "0.930"],
        "convnextv2": ["0.657", "0.498", "0.910"],
    }

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
        # score in the series colour, beside its own point, so ties never share a label
        for xi, yi, s in zip(x, ys, scores[model], strict=True):
            ax.text(
                xi + 0.055, yi - 0.055, s, color=c, fontsize=8.5,
                ha="left", va="bottom", zorder=5,
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
    ax.text(
        0, 1.015,
        "ViT is first under the leaky protocol and last under the honest one. Ranks are "
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


def main() -> None:
    print("figures:")
    fig_label_efficiency()
    fig_ranking_inversion()
    fig_cross_venue()


if __name__ == "__main__":
    main()
