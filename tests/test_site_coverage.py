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

import csv
import json
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
    "error_taxonomy.csv": "WP4-T6: one row per misclassification, too granular for the export",
    "error_taxonomy_summary.csv": "WP4-T6: the per-protocol summary and its derived findings",
    "class_balancing.csv": "method: why class weighting stays on",
    "person_count_clip_venues.csv": (
        "Person counts at all nine clip venues, every frame ACTIVE_PLAY. Confirms the "
        "rule does not miss play; cannot confirm it does not invent it, since no clip "
        "venue has a recorded empty pitch."
    ),
    "ball_detection_rule.csv": (
        "A17: per-frame ball detections inside the boundary. The venue_01 row alone would "
        "read as the best signal in the project (+0.967); the cross-venue row, where a ball "
        "is found in 40% of genuine play frames, is what says it cannot be a rule."
    ),
    "person_count_rule.csv": (
        "Per-frame person counts inside the boundary. The distribution is the result - "
        "a single row would hide that 1-4 people does not mean not-playing."
    ),
    "three_class_on_video.csv": (
        "C3 on three real person-minutes. Three positives and a unanimous negative - "
        "the argument is the result, and a row would read as a measured class score."
    ),
    "distinct_scenes_rerun.csv": (
        "A15: every fit-dependent number, full against pruned. The pair is the result - "
        "one row would hide that the in-domain figure falls while the rest improve."
    ),
    "motion_override_on_video.csv": "motion rule transferred to one unseen clip; 16 hand-labelled samples",
    "video_verdicts_none.csv": "boundary diagnostic on one unseen clip; hand-labelled, 24 samples",
    "video_verdicts_cam2.csv": "boundary diagnostic on one unseen clip; hand-labelled, 24 samples",
    "video_verdicts_derived.csv": "boundary diagnostic on one unseen clip; hand-labelled, 24 samples",
    "video_verdicts.csv": "scratch output of the last classify_video run",
    "probe_regularisation.csv": (
        "Sweep of the probe penalty and of ROI pooling. Both negative, and the second "
        "only after a confound was removed - the argument is the result."
    ),
    "rq6_real_class_mix.csv": (
        "RQ6 risk-coverage on venue_01 camera B. Real answer, one venue - a row beside "
        "the headline figures would read as a transferable operating point."
    ),
    "motion_feature_ablation.csv": (
        "A14 probe: motion as a feature and as an override rule. Negative on both, and "
        "the reason is the dataset gap rather than the idea - the argument is the result."
    ),
    "motion_cue_probe.csv": (
        "Diagnostic: whether frame-to-frame motion separates the classes. Measured at "
        "8-second spacing where deployment samples once a minute, so the AUC is not a "
        "deployment number and a row would be read as one."
    ),
    "h3_with_generated_empty.csv": (
        "H3 re-run with a false-play control beside the recall. It qualifies H3 rather "
        "than replacing it, and the pair of columns is the result - a single row would "
        "read as a competing headline."
    ),
    "a13_false_play_repair.csv": (
        "A13 condition 3: whether the generated EMPTY frames repair the false-play "
        "collapse. One camera at one venue, and the answer differs by backbone, so a "
        "row on the export would read as a headline it cannot carry."
    ),
    "a13_false_play_repair_convnextv2.json": "A13 condition 3, per-backbone detail",
    "a13_false_play_repair_vit.json": "A13 condition 3, per-backbone detail",
    "a13_false_play_repair.json": "A13 condition 3, per-backbone detail",
    "a13_real_vs_generated.json": (
        "A13 gate: whether generated frames are separable from recorded ones. A "
        "pass licenses the augmentation and says nothing about the results, so a "
        "row on the export would read as a finding about the pitch."
    ),
    "coverage.md": "method: the class x lighting x venue matrix",
    "roi_pooling_leak.csv": (
        "WP3-T1 audit: what a pitch boundary keeps out of the model, and what survives it. "
        "The row that matters is a 1.0 meaning 'the neighbouring pitch cannot reach this' - "
        "a table of cosines would read as a result about occupancy rather than a check that "
        "a masking feature does what its docstring says. The argument is in the log."
    ),
    # Hypothesis reports whose value is the argument, not the row. They are written up in
    # EXPERIMENT_LOG.md, which the served site renders in full.
    "h4_model_equivalence.csv": "H4: refuted; the reasoning is the result",
    "h6_zero_shot_gap.csv": "H6: inconclusive; the prompt-space distribution is the result",
    "h5_preprocessing_switches.csv": "H5: one clause unrunnable, one refuted - the clauses are the result",
    "false_play_significance.csv": "significance that does not survive the effective sample",
    "false_play_rescored.csv": "repair of the search's false-play control",
    "h3_with_false_play.csv": "H3 re-reported with its control",
    "geometry_convention_probe.csv": "WP3-T3 probe, superseded by input_path_protocol.csv",
    "input_path_protocol.csv": "WP3-T3(b): the finding did not survive; a table would imply it did",
    "logit_average_baseline.csv": "WP5-T9 on raw caches; refuted",
    "logit_average_baseline_preproc.csv": "WP5-T9 on letterboxed caches",
    "fusion_head_ablation.csv": "WP5-T2: the gate does not earn its place - the argument is the result",
    "augmentation_transfer.csv": "WP3-T6: per-draw rows behind a retracted headline - the spread is the result",
    "augmentation_transfer_spread.csv": "WP3-T6: the retraction as a table - `light` moves 0.3479-0.8550 across five draws that differ in nothing else, and the argument for why that ends the claim is the result",
    "onboarding_cost.csv": "WP4-T3: the onboarding curve - the argument and its caveats are the result",
    "empty_recognition.csv": "WP4-T13: what the false-play control measures - the argument is the result",
    "xai_evidence_focus.csv": "WP4-T5: evidence-on-people per frame; the summary is in the log",
    "fusion_head_comparisons.csv": "WP5-T2: paired tests, none of which the design could make significant",
    "stan_draw_spread.csv": "WP5-T1: five construction draws - the 1.0000 holds in three of them and the ordering in all five; the argument is the result",
    "rq6_calibration.csv": "RQ6: blocked by the class mix",
    "rq6_reliability.csv": "RQ6: the reliability bins - 905 of 907 frames land in one, so there is no diagram to draw",
    "rq6_risk_coverage.csv": "RQ6: shown as figs/risk_coverage_band rather than as a table",
    "label_efficiency.csv": "shown as a figure rather than a table",
    "end_to_end_slots.csv": "two real slots; too few to tabulate",
    "end_to_end_model_slots.csv": "the same two slots with the model in the loop; in-sample, so a wiring check rather than a number worth exporting",
    "reconciliation_value.csv": "WP6-T10: a break-even under stated assumptions, not a measurement",
    "benchmark_v2_protocols.json": "diagnostics behind benchmark_v2.csv",
    "gate_status.json": "milestone gate criteria; the readable form is results/gate_status.md",
    "search_resolution.csv": "WP3-T8: re-scoring that says which searched margins mean nothing",
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
    assert "How much of the leakage penalty is actually leakage" in html


def test_the_risk_coverage_band_reached_the_page_as_a_figure() -> None:
    """`OUT_OF_SCOPE` says this result is shown as a figure rather than a table, so the
    figure has to actually be on the page - otherwise the reason is a fiction and the
    result is simply missing."""
    page = RESULTS / "project_site.html"
    if not page.exists():
        return
    html = page.read_text(encoding="utf-8")
    assert "How much human review buys a given reliability" in html
    # The tie count, which is the reason the figure is a band rather than a curve. Asserted
    # as two numbers rather than as the sentence that used to carry them: the page now sets
    # "890 / 907" as a statistic beside the word "identical", and a test pinned to one
    # phrasing of a caption fails on rewording while the result is still there. What must
    # not disappear is the count.
    assert "890" in html and "907" in html


def test_the_models_page_states_the_frozen_and_trained_counts() -> None:
    """The models page exists to answer "did you actually train anything?".

    Asserted on the page rather than on `model_inventory.json`, because the inventory being
    correct is not the same as the answer reaching a reader. Both totals are checked: a page
    that printed only the 200M would imply nothing was trained, and one that printed only
    the trained count would leave the ratio - which is the actual answer - unstated.
    """
    page = RESULTS / "project_site.html"
    if not page.exists():
        return
    html = page.read_text(encoding="utf-8")
    inventory = json.loads((RESULTS / "model_inventory.json").read_text(encoding="utf-8"))
    assert "What was trained, and what never was" in html
    assert f'{inventory["trained_total_params"]:,}' in html
    assert f'{inventory["frozen_to_trained_ratio"]:,}:1' in html
    # The seam is the diagram's one load-bearing line: if labels appear to enter earlier,
    # every frozen claim on the page becomes unreadable.
    assert "Labels enter here" in html


def test_every_preprocessing_switch_reached_the_page_with_a_before_and_after() -> None:
    """A switch on the page without its pair is a claim with no picture behind it.

    `preprocess_pairs.csv` is the inventory of what was rendered, so this checks the page
    against it rather than against a list typed here - which would be a second list to keep
    in step with the first.
    """
    page = RESULTS / "project_site.html"
    pairs = RESULTS / "preprocess_pairs.csv"
    if not page.exists() or not pairs.exists():
        return
    html = page.read_text(encoding="utf-8")
    with pairs.open(newline="", encoding="utf-8") as fh:
        labels = [r["label"] for r in csv.DictReader(fh)]
    assert labels, "preprocess_pairs.csv is empty"
    missing = [lab for lab in labels if f"<code>{lab}</code>" not in html]
    assert not missing, f"switches with no before/after card on the page: {missing}"


def test_the_crops_carry_the_area_they_discard() -> None:
    """The number two same-sized tiles cannot show.

    Both crops are re-letterboxed back to 224, so the pair of pictures looks like a zoom.
    `centre_crop=0.5` keeps a quarter of the frame, and it is the variant whose apparent
    0.998 recall came with empty accuracy 0.000 - so the page has to say what it threw away.
    """
    page = RESULTS / "project_site.html"
    if not page.exists():
        return
    html = page.read_text(encoding="utf-8")
    assert "frame area kept" in html
    assert "25%" in html


def test_the_reproduce_page_is_drawn_from_the_runner() -> None:
    """The Reproduce page used to be four command blocks and a paragraph.

    It answered "what do I type" and nothing about what a run costs, which is the question
    a reader with a twelve-hour pipeline actually has. Asserted against
    `pipeline_graph.json` rather than against typed figures, because the point of that
    artefact is that the page cannot drift from `reproduce_all.STAGES` - a stage added
    without regenerating shows up here as a stage count the page does not carry.
    """
    page = RESULTS / "project_site.html"
    graph_file = RESULTS / "pipeline_graph.json"
    if not page.exists() or not graph_file.exists():
        return
    html = page.read_text(encoding="utf-8")
    graph = json.loads(graph_file.read_text(encoding="utf-8"))
    assert "What happens when you run it" in html
    assert f'{graph["n_stages"]} stages' in html
    # The two totals answer different questions - how much compute, and how long until the
    # thesis is back - so a page carrying only one of them is the misleading version.
    assert f'{graph["total_minutes"] / 60:.1f}h' in html
    assert f'{graph["critical_path_minutes"] / 60:.1f}h' in html


def test_the_dominant_stage_is_named_on_the_reproduce_page() -> None:
    """Two thirds of the runtime sits in one stage, and that is the page's one real finding.

    Checked by recomputing which stage dominates rather than by looking for
    `preprocess-search`: if the search is ever trimmed and something else becomes the
    expensive stage, this should keep passing while the page names the new one - and fail if
    the page still names the old one.
    """
    page = RESULTS / "project_site.html"
    graph_file = RESULTS / "pipeline_graph.json"
    if not page.exists() or not graph_file.exists():
        return
    html = page.read_text(encoding="utf-8")
    graph = json.loads(graph_file.read_text(encoding="utf-8"))
    top = max(graph["stages"], key=lambda s: s["minutes"])
    if top["minutes"] / graph["total_minutes"] <= 0.25:
        return  # nothing dominates; the page says so instead, and says it without a name
    assert f'<code>{top["name"]}</code>' in html
    assert f'{top["minutes"]} of the {graph["total_minutes"]}' in html
