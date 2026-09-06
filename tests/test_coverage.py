"""Coverage reporting. Its job is to make an absent cell visible before a split turns
out degenerate, so the tests are mostly about what it refuses to hide."""

from __future__ import annotations

import pytest

from pitch_occupancy.data.coverage import (
    coverage_cells,
    empty_cells,
    render_report,
)
from pitch_occupancy.data.manifest import ManifestRow

EMPTY, PLAY, MAINT = "C1_EMPTY", "C2_ACTIVE_PLAY", "C3_MAINTENANCE_NON_SPORTING"


def row(cls: str, lighting: str, venue: str, i: int = 0) -> ManifestRow:
    return ManifestRow(
        file=f"{cls}/{venue}{lighting}{i}.jpg", class4="1_empty", class3=cls, venue=venue,
        camera="c", slot_date="2026-07-11", slot_time="10:00", slot_id=f"{venue}_s", t_s=i,
        source="regular", labeled_by="human", lighting=lighting, quality="unknown",
        split_role="",
    )


@pytest.fixture
def rows() -> list[ManifestRow]:
    return (
        [row(EMPTY, "day", "v1", i) for i in range(10)]
        + [row(PLAY, "night", "v1", i) for i in range(10)]
        + [row(PLAY, "day", "v2", i) for i in range(5)]
    )


def test_cells_are_counted_per_class_lighting_venue(rows) -> None:
    cells = {(c.class3, c.lighting, c.venue): c.n for c in coverage_cells(rows)}
    assert cells[(EMPTY, "day", "v1")] == 10
    assert cells[(PLAY, "day", "v2")] == 5


def test_empty_cells_names_what_is_missing(rows) -> None:
    gaps = empty_cells(rows)
    assert (EMPTY, "night") in gaps
    assert (MAINT, "day") in gaps
    assert (MAINT, "night") in gaps
    assert (EMPTY, "day") not in gaps


def test_no_gaps_when_every_combination_is_present() -> None:
    rows = [
        row(cls, light, "v1")
        for cls in (EMPTY, PLAY, MAINT)
        for light in ("day", "night")
    ]
    assert empty_cells(rows) == []


def test_report_marks_absent_cells_rather_than_printing_zero(rows) -> None:
    """A zero reads as a measurement; a dash reads as a gap."""
    report = render_report(rows)
    assert "| MAINTENANCE_NON_SPORTING | - | - | 0 |" in report


def test_report_flags_concentration(rows) -> None:
    """The check that predicts a degenerate split before one is built."""
    report = render_report(rows)
    assert "## Concentration" in report
    assert "100%" in report  # EMPTY is entirely v1/day here


def test_report_names_starved_classes(rows) -> None:
    assert "MAINTENANCE_NON_SPORTING" in render_report(rows)
    assert "below the 100-frame working target" in render_report(rows)


def test_report_is_ascii_only(rows) -> None:
    """Windows consoles mangle non-ASCII, and this report is meant to be read there."""
    report = render_report(rows)
    assert all(ord(c) < 128 for c in report), [c for c in report if ord(c) > 127]


def test_report_handles_an_empty_dataset() -> None:
    assert "# Dataset coverage" in render_report([])
