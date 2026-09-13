"""Dataset coverage reporting (WP2-T1).

Answers one question: *which cells of the collection matrix are empty?* Class x lighting x
venue is the design space; a cell with no frames is a claim the thesis cannot make, and it
is far cheaper to see that in a table than to discover it when a split turns out
degenerate.

The output is written to ``results/coverage.md`` and is meant to go into the thesis
roughly as-is.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date

from pitch_occupancy.data.manifest import ManifestRow
from pitch_occupancy.data.taxonomy import CLASS3_ORDER, Class3

__all__ = ["CoverageCell", "coverage_cells", "empty_cells", "render_report"]

#: Working target per class per condition cell, from the thesis data plan.
TARGET_PER_CELL = 100


@dataclass(frozen=True, slots=True)
class CoverageCell:
    class3: str
    lighting: str
    venue: str
    n: int


def coverage_cells(rows: list[ManifestRow]) -> list[CoverageCell]:
    counts = Counter((r.class3, r.lighting, r.venue) for r in rows)
    return [CoverageCell(c, l, v, n) for (c, l, v), n in sorted(counts.items())]


def empty_cells(rows: list[ManifestRow]) -> list[tuple[str, str]]:
    """(class, lighting) combinations with no frames anywhere.

    These are the claims the dataset cannot support at all - distinct from cells that are
    merely thin.
    """
    seen = {(r.class3, r.lighting) for r in rows}
    lightings = sorted({r.lighting for r in rows if r.lighting})
    return [
        (c.value, light)
        for c in CLASS3_ORDER
        for light in lightings
        if (c.value, light) not in seen
    ]


def _table(rows: list[ManifestRow], col_attr: str) -> str:
    cols = sorted({str(getattr(r, col_attr)) for r in rows if getattr(r, col_attr)})
    counts = Counter((r.class3, str(getattr(r, col_attr))) for r in rows)
    head = f"| class | {' | '.join(cols)} | total |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [head, sep]
    for c in CLASS3_ORDER:
        cells = [counts.get((c.value, col), 0) for col in cols]
        marks = ["-" if n == 0 else str(n) for n in cells]
        lines.append(f"| {c.name} | {' | '.join(marks)} | {sum(cells)} |")
    totals = [sum(counts.get((c.value, col), 0) for c in CLASS3_ORDER) for col in cols]
    lines.append(f"| **total** | {' | '.join(str(t) for t in totals)} | {sum(totals)} |")
    return "\n".join(lines)


#: Rows with this `source` were generated, not recorded (A13).
SYNTHETIC_SOURCE = "synthetic"


def recorded(rows: list[ManifestRow]) -> list[ManifestRow]:
    """Only frames that came off a camera.

    Every headline table in this report counts these and these only. A coverage report that
    silently mixed the two would answer "how many empty pitches do we have at other venues?"
    with a number made partly of pictures - and that question is the one
    `thesis/data_requests.md` exists to ask, the one the supervisor will ask, and the one
    the M2 gate measures. Generated frames get their own section below, never a share of the
    main tables.
    """
    return [r for r in rows if getattr(r, "source", "") != SYNTHETIC_SOURCE]


def render_report(rows: list[ManifestRow]) -> str:
    """Full markdown report. Headline tables count recorded frames only; see `recorded`."""
    real = recorded(rows)
    gen = [r for r in rows if r not in real]
    n_venues = len({r.venue for r in real})
    per_class = Counter(r.class3 for r in real)
    gaps = empty_cells(real)

    banner = (
        f"**{len(gen)} generated frame(s) are excluded from every table above the "
        f"Generated section** (A13). They are a training-side augmentation and counting "
        f"them here would make a gap look filled that is not."
    ) if gen else ""

    out = [
        "# Dataset coverage",
        "",
        f"Generated {date.today().isoformat()} | `uv run pitch coverage` | "
        f"{len(real)} recorded frames | {n_venues} venues"
        + (f" | +{len(gen)} generated (excluded)" if gen else ""),
        "",
        *( [banner, ""] if banner else [] ),
        "## Class x lighting",
        "",
        _table(real, "lighting"),
        "",
        "## Class x venue",
        "",
        _table(real, "venue"),
        "",
        "## Provenance",
        "",
        _table(real, "labeled_by"),
        "",
        "## Gaps",
        "",
    ]

    if gaps:
        out.append("Combinations with **no frames at all** - claims this dataset cannot support:")
        out.append("")
        for cls, light in gaps:
            out.append(f"- `{Class3(cls).name}` x `{light}`")
        out.append("")

    thin = sorted(
        (c for c in coverage_cells(real) if c.n < TARGET_PER_CELL),
        key=lambda c: c.n,
    )
    starved = [c for c in CLASS3_ORDER if per_class.get(c.value, 0) < TARGET_PER_CELL]
    if starved:
        out.append(
            f"Classes below the {TARGET_PER_CELL}-frame working target **in total**: "
            + ", ".join(f"`{c.name}` ({per_class.get(c.value, 0)})" for c in starved)
        )
        out.append("")
    out.append(f"{len(thin)} of {len(coverage_cells(rows))} populated cells are below target.")
    out.append("")

    # concentration: the check that actually predicts a degenerate split
    out.append("## Concentration")
    out.append("")
    out.append(
        "A class drawn overwhelmingly from one venue or one lighting condition cannot be "
        "separated from that condition by any split, so accuracy on it measures scene "
        "recognition. Share of each class held by its single largest source:"
    )
    out.append("")
    out.append("| class | top venue | share | top lighting | share |")
    out.append("|---|---|---|---|---|")
    for c in CLASS3_ORDER:
        members = [r for r in rows if r.class3 == c.value]
        if not members:
            out.append(f"| {c.name} | - | - | - | - |")
            continue
        v, vn = Counter(r.venue for r in members).most_common(1)[0]
        li, ln = Counter(r.lighting for r in members).most_common(1)[0]
        out.append(
            f"| {c.name} | `{v}` | {vn / len(members):.0%} | `{li}` | {ln / len(members):.0%} |"
        )
    out.append("")

    if gen:
        out.append("## Generated frames (A13) - not counted above")
        out.append("")
        out.append(
            "Training-side augmentation, listed apart from the recorded corpus because a "
            "gap they appear to fill is still a gap: the test set stays real, so a venue "
            "whose only empty pitch is a generated one still has no empty pitch."
        )
        out.append("")
        out.append(_table(gen, "venue"))
        out.append("")
        out.append("| recorded defect | frames |")
        out.append("|---|---|")
        for k, n in sorted(Counter(getattr(r, "quality", "") for r in gen).items()):
            out.append(f"| `{k or 'unrecorded'}` | {n} |")
        out.append("")
    return "\n".join(out)
