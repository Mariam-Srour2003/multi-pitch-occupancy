"""What the standalone site exports, and what it deliberately leaves out.

`make_site.py` reads a fixed list of result files. Its docstring claimed the page covered
"the whole thesis", and for a while it did - then eleven experiments were added and every
one of them was omitted in silence. Regenerating produced a byte-identical page and said
nothing, which is the same failure mode as the branch reference that named a stale tip and
the search viewer that served a withdrawn run.

The fix is not to render everything: the served front end (`api/thesis_site.py`) already
does, by rendering `EXPERIMENT_LOG.md` directly. The fix is that omission has to be a
*decision*. Every committed result file is either read by the exporter or named below, so
adding one fails this test until someone says which it is.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAKE_SITE = ROOT / "experiments" / "make_site.py"

#: Result files the standalone export deliberately does not render, with the reason. The
#: served site shows all of them; this page is a curated export.
OUT_OF_SCOPE = {
    # Method and audit artefacts - they qualify the headline numbers rather than being them,
    # and the reasoning that makes them meaningful does not survive being cut to a table.
    "camera_fingerprint.csv": "audit: which physical camera took each frame",
    "camera_health.csv": "audit: per-camera frame quality",
    "frame_quality.csv": "audit: per-frame quality measures",
    "near_duplicates.csv": "audit: near-duplicate pairs behind the effective-sample counts",
    "effective_sample_audit.csv": "audit: distinct scenes per reported test",
    "class_balancing.csv": "method: why class weighting stays on",
    "coverage.md": "method: the class x lighting x venue matrix",
    # Hypothesis reports whose value is the argument, not the row. They are written up in
    # EXPERIMENT_LOG.md, which the served site renders in full.
    "h4_model_equivalence.csv": "H4: refuted; the reasoning is the result",
    "h6_zero_shot_gap.csv": "H6: inconclusive; the prompt-space distribution is the result",
    "false_play_significance.csv": "significance that does not survive the effective sample",
    "false_play_rescored.csv": "repair of the search's false-play control",
    "h3_with_false_play.csv": "H3 re-reported with its control",
    "geometry_convention_probe.csv": "WP3-T3 probe, superseded by input_path_protocol.csv",
    "input_path_protocol.csv": "WP3-T3(b): the finding did not survive; a table would imply it did",
    "logit_average_baseline.csv": "WP5-T9 on raw caches; refuted",
    "logit_average_baseline_preproc.csv": "WP5-T9 on letterboxed caches",
    "rq6_calibration.csv": "RQ6: blocked by the class mix",
    "rq6_risk_coverage.csv": "RQ6: a band, and it needs its own figure (WP4-T9)",
    "label_efficiency.csv": "shown as a figure rather than a table",
    "end_to_end_slots.csv": "two real slots; too few to tabulate",
    "benchmark_v2_protocols.json": "diagnostics behind benchmark_v2.csv",
    "preprocess_search_500frame_UNTRUSTWORTHY.json": "withdrawn run, kept for the record",
}


def rendered() -> set[str]:
    """Result files the exporter actually reads, parsed from its source."""
    source = MAKE_SITE.read_text(encoding="utf-8")
    return set(re.findall(r'read_(?:csv|json)\("([^"]+)"\)', source))


def committed() -> set[str]:
    return {
        p.name for p in RESULTS.iterdir()
        if p.is_file() and p.suffix in {".csv", ".json"} and not p.name.startswith(".")
    }


def test_the_exporter_reads_the_files_it_claims_to() -> None:
    for name in rendered():
        assert (RESULTS / name).exists(), f"make_site.py reads a missing {name}"


def test_every_result_file_is_either_rendered_or_deliberately_out_of_scope() -> None:
    """Omission must be a decision, not a default.

    If this fails after you add an experiment, the choice is: render it in `make_site.py`,
    or add it to `OUT_OF_SCOPE` with the reason it does not belong on the export.
    """
    unaccounted = committed() - rendered() - set(OUT_OF_SCOPE)
    assert not unaccounted, (
        "these results are neither on the standalone site nor listed as out of scope: "
        + ", ".join(sorted(unaccounted))
    )


def test_the_out_of_scope_list_does_not_rot() -> None:
    """A reason attached to a file that no longer exists is a reason nobody will re-read."""
    stale = {name for name in OUT_OF_SCOPE if not (RESULTS / name).exists()}
    assert not stale, f"OUT_OF_SCOPE names files that no longer exist: {sorted(stale)}"


def test_the_summary_line_no_longer_claims_to_cover_everything() -> None:
    """The claim that made the omission invisible, on the line a reader actually reads.

    Checked on the summary line rather than the whole docstring, because the docstring now
    quotes the old wording in order to explain what went wrong - and a test that forbade the
    string outright would forbid recording the history.
    """
    summary = MAKE_SITE.read_text(encoding="utf-8").splitlines()[0]
    assert "whole thesis" not in summary
    assert "export" in summary


def test_the_wp4t1_result_reached_the_page() -> None:
    """Asserted on the generated HTML, not on the builder.

    The lesson from the search viewer: a safeguard tested through the function that produces
    it, rather than the page that serves it, certifies something no reader ever sees. This
    result contradicts the page's own headline - "the protocol reverses the ranking" - by
    showing a protocol with no ranking to reverse, so it has to be on the page itself.
    """
    page = RESULTS / "project_site.html"
    if not page.exists():
        return
    html = page.read_text(encoding="utf-8")
    assert "The protocol a constant predictor wins" in html
    assert "How much of the leakage penalty is leakage" in html
