"""The two front ends.

Two audiences: the thesis site is for the researcher and supervisor, the dashboard for the
client. The tests below guard the properties that would otherwise fail silently - a page
rendering an empty shell, or a document quietly missing from the nav."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api.app import app
from pitch_occupancy.api.markdown import render
from pitch_occupancy.api.thesis_site import DOCUMENTS


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# --- routing ----------------------------------------------------------------


def test_root_serves_the_thesis_site(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "thesis" in r.text.lower()


def test_client_route_serves_the_operator_dashboard(client) -> None:
    r = client.get("/client")
    assert r.status_code == 200
    assert "Operator dashboard" in r.text


def test_the_two_front_ends_link_to_each_other(client) -> None:
    assert 'href="/client"' in client.get("/").text
    assert 'href="/"' in client.get("/client").text


def test_api_still_works_alongside_them(client) -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert "slots_evaluated" in client.get("/api/v1/meters").json()


# --- the thesis site --------------------------------------------------------


def test_every_declared_document_gets_a_tab_and_a_view(client) -> None:
    html = client.get("/").text
    for key, (label, _) in DOCUMENTS.items():
        assert f'data-view="{key}"' in html, f"{key} has no view"
        assert label in html, f"{label} is missing from the nav"


def test_the_searches_view_exists_beyond_the_documents(client) -> None:
    """Searches are assembled from result files rather than a Markdown source."""
    assert 'data-view="searches"' in client.get("/").text


def test_documents_actually_render_rather_than_appearing_empty(client) -> None:
    """A view that silently renders nothing looks the same as one that is not there."""
    html = client.get("/").text
    for key in DOCUMENTS:
        body = re.search(
            rf'data-view="{key}"[^>]*><div class="doc">(.*?)</div></section>', html, re.S
        )
        assert body, f"{key} view not found"
        assert len(body.group(1)) > 200, f"{key} rendered almost nothing"


def test_missing_document_degrades_to_a_notice(tmp_path, monkeypatch) -> None:
    """A document that has not been generated should say so, not vanish."""
    from pitch_occupancy.api import thesis_site

    monkeypatch.setitem(
        thesis_site.DOCUMENTS, "ghost", ("Ghost", tmp_path / "nope.md")
    )
    try:
        html = thesis_site.page()
        assert "Not generated yet" in html
    finally:
        thesis_site.DOCUMENTS.pop("ghost", None)


# --- the markdown renderer --------------------------------------------------


def test_renders_headings_and_paragraphs() -> None:
    out = render("# Title\n\nSome text here.\n")
    assert "<h1>Title</h1>" in out
    assert "<p>Some text here.</p>" in out


def test_renders_tables_with_a_scroll_container() -> None:
    """Wide result tables must scroll inside their own box, never the page body."""
    out = render("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert '<div class="scroll">' in out
    assert "<th>a</th>" in out and "<td>1</td>" in out


def test_renders_inline_marks() -> None:
    out = render("**bold** and `code` and [link](http://x)")
    assert "<strong>bold</strong>" in out
    assert "<code>code</code>" in out
    assert '<a href="http://x">link</a>' in out


def test_renders_fenced_code_without_interpreting_it() -> None:
    out = render("```\nuv run pitch seed\n**not bold**\n```")
    assert "<pre><code>" in out
    assert "**not bold**" in out  # untouched inside the fence


def test_escapes_html_in_source_text() -> None:
    assert "<script>" not in render("a <script>alert(1)</script> b")


def test_renders_lists_and_blockquotes() -> None:
    out = render("- one\n- two\n\n> a caveat\n")
    assert out.count("<li>") == 2
    assert "<blockquote>" in out


def test_horizontal_rule_separates_sections() -> None:
    assert "<hr>" in render("a\n\n---\n\nb\n")


def test_the_real_experiment_log_renders(client) -> None:
    """The log is the substance of the thesis; if it fails to render, the site is empty."""
    from pitch_occupancy.api.thesis_site import RESULTS

    log = RESULTS / "EXPERIMENT_LOG.md"
    if not log.exists():
        pytest.skip("no experiment log yet")
    out = render(log.read_text(encoding="utf-8"))
    assert out.count("<h2>") > 3
    assert "<table>" in out


# --- the models view --------------------------------------------------------


def test_models_view_leads_with_a_recommendation(client) -> None:
    """A table of numbers does not answer "which one should I use"."""
    html = client.get("/").text
    assert 'data-view="models"' in html
    assert "Use this one" in html


def test_models_view_ranks_each_protocol_separately(client) -> None:
    """One strip per protocol, and each is named - a bare count would pass while the page
    silently lost the protocol that reverses the decision."""
    from pitch_occupancy.api.models_view import render as render_models

    out = render_models()
    for protocol in ("random split (leaky)", "grouped split", "cross-venue recall",
                     "false-play (lower is better)"):
        assert protocol in out, protocol
    assert out.count('class="strip"') == 4


def test_models_view_flags_a_ranking_inversion(client) -> None:
    """The finding is that the protocol reverses the decision - it must not need a reader
    to reconstruct that from three separate tables."""
    from pitch_occupancy.api.models_view import render as render_models

    assert "reverses the decision" in render_models()


def test_models_view_survives_missing_results(monkeypatch, tmp_path) -> None:
    """An absent CSV should read as absent, never as a stale or invented number."""
    from pitch_occupancy.api import models_view

    monkeypatch.setattr(models_view, "RESULTS", tmp_path)
    out = models_view.render()
    assert "No model results yet" in out


def test_models_view_marks_the_clock_rule_as_using_no_pixels(client) -> None:
    from pitch_occupancy.api.models_view import render as render_models

    assert "never the pixels" in render_models()


# --- diagrams ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["confound_matrix", "blocked_questions", "pipeline", "protocols", "schema",
     "augmentation_axes", "empty_blindness"],
)
def test_every_diagram_stays_inside_its_viewbox(name) -> None:
    """Content drawn past the viewBox is clipped, and clipping is invisible in code."""
    from pitch_occupancy.api import diagrams

    svg = getattr(diagrams, name)()
    if not svg:
        pytest.skip("diagram needs data that is not present")
    vb = [float(x) for x in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    rects = re.findall(r'x="([-\d.]+)"[^>]*width="([-\d.]+)"', svg)
    rightmost = max((float(x) + float(w) for x, w in rects), default=0.0)
    lowest = max((float(m) for m in re.findall(r'y="([-\d.]+)"', svg)), default=0.0)
    assert rightmost <= vb[2], f"{name} draws to x={rightmost} past {vb[2]}"
    assert lowest <= vb[3], f"{name} draws to y={lowest} past {vb[3]}"


@pytest.mark.parametrize(
    "name",
    ["confound_matrix", "blocked_questions", "pipeline", "protocols", "schema",
     "augmentation_axes", "empty_blindness"],
)
def test_every_diagram_is_captioned_and_labelled(name) -> None:
    """A diagram nobody can read is worse than a sentence."""
    from pitch_occupancy.api import diagrams

    svg = getattr(diagrams, name)()
    if not svg:
        pytest.skip("diagram needs data that is not present")
    assert 'role="img"' in svg
    assert "aria-label=" in svg
    assert "<figcaption>" in svg
    assert svg.count("<svg") == svg.count("</svg>") == 1


def test_confound_diagram_reads_the_real_manifest() -> None:
    """It must show the dataset's actual shape, not an illustration of it."""
    from pitch_occupancy.api.diagrams import MANIFEST, confound_matrix

    if not MANIFEST.exists():
        pytest.skip("no manifest")
    svg = confound_matrix()
    # per-cell counts, not class totals - the split across lighting is the whole finding
    assert ">485<" in svg  # empty, daylight
    assert ">9<" in svg  # empty, floodlit - the near-absent cell
    assert ">970<" in svg  # active play, floodlit


def test_confound_diagram_degrades_when_there_is_no_manifest(monkeypatch, tmp_path) -> None:
    from pitch_occupancy.api import diagrams

    monkeypatch.setattr(diagrams, "MANIFEST", tmp_path / "absent.csv")
    assert diagrams.confound_matrix() == ""


def test_diagrams_lead_the_tabs_they_explain(client) -> None:
    html = client.get("/").text
    assert html.count("<svg") >= 4
    assert html.count("<figcaption>") >= 4


def test_schema_diagram_shows_live_row_counts() -> None:
    """A schema picture that does not match the database is worse than none."""
    from pitch_occupancy.api.diagrams import DB_PATH, schema

    if not DB_PATH.exists():
        pytest.skip("no database seeded")
    svg = schema()
    assert "frame_samples" in svg
    assert "rows</text>" in svg


def test_schema_diagram_survives_a_missing_database(monkeypatch, tmp_path) -> None:
    from pitch_occupancy.api import diagrams

    monkeypatch.setattr(diagrams, "DB_PATH", tmp_path / "absent.db")
    svg = diagrams.schema()
    assert "frame_samples" in svg  # structure still drawn
    assert "rows</text>" not in svg  # but no counts invented


def test_database_tab_is_present(client) -> None:
    html = client.get("/").text
    assert 'data-view="database"' in html
    assert "Database" in html


# --- augmentation tab and figure serving -------------------------------------


def test_augmentation_tab_exists_and_renders(client) -> None:
    html = client.get("/").text
    assert 'data-view="augmentation"' in html
    assert html.count('data-view="augmentation"') >= 2  # a tab button and a view section
    assert "discards nothing" in html


def test_augmentation_tab_states_what_is_not_measured(client) -> None:
    """The module is built but unbenchmarked. A page that showed the presets without saying
    so would imply a result that does not exist."""
    html = client.get("/").text
    assert "Not measured yet" in html
    assert "no footage in this dataset" in html.lower()


def test_figures_are_served(client) -> None:
    r = client.get("/figs/augmentation_grid.jpg")
    if r.status_code == 404:
        pytest.skip("grid not generated in this checkout")
    assert r.headers["content-type"] == "image/jpeg"
    assert len(r.content) > 1000


@pytest.mark.parametrize(
    "name",
    ["../../.gitignore", "..%2F..%2Fpyproject.toml", "venue_check/x.jpg", ".env",
     "EXPERIMENT_LOG.md"],
)
def test_figure_route_refuses_anything_but_a_figure_filename(client, name) -> None:
    """`results/figs/venue_check/` holds cropped pitch frames - operator footage that must
    never be published. A subpath or a traversal must not reach them, and nor must any
    non-image file."""
    assert client.get(f"/figs/{name}").status_code in (404, 405)


def test_the_empty_blindness_diagram_shows_both_outcomes() -> None:
    """It exists to contrast two training sets, so a version showing one is broken."""
    from pitch_occupancy.api.diagrams import empty_blindness

    svg = empty_blindness()
    assert "23%" in svg and "100%" in svg
    assert "venue_01 camera A" in svg and "clip venues" in svg
    # the marker has to be inside the svg or every connector loses its head
    body = svg[svg.index("<svg"):svg.index("</svg>")]
    assert "<defs>" in body and 'marker-end="url(#eb-arrow)"' in body


def test_the_findings_tab_leads_with_the_empty_blindness_diagram(client) -> None:
    """The most consequential finding in the project should not be buried in prose."""
    html = client.get("/").text
    assert "every empty pitch called a match" in html


def test_the_search_table_never_shows_a_broken_false_play_as_a_number() -> None:
    """Entries written before the control was repaired carry false_play = 0.0 for
    everything. That is not a low rate, it is a dead measurement, and rendering it as
    0.0000 would read as the best possible result."""
    from pitch_occupancy.api.thesis_site import _search_summary

    html = _search_summary()
    if "not re-scored" in html:
        assert "predate the repair" in html
    assert "Balanced" in html


def test_the_search_safeguard_is_actually_rendered(client) -> None:
    """The bug this pins: the assertion above passed while nothing rendered it.

    `_search_summary` was called by that test and by nothing else - `page()` never
    included it - so a test certified a safeguard no reader ever saw, while the panel that
    *was* rendered served the same un-rescored entries as clean top results. Testing a
    function is not testing a page.
    """
    from pitch_occupancy.api.thesis_site import _search_summary

    summary = _search_summary()
    page = client.get("/").text

    marker = next(
        (m for m in ("not re-scored", "Balanced", "balanced = recall") if m in summary),
        None,
    )
    assert marker, "the summary rendered nothing identifiable to look for"
    assert marker in page, (
        f"_search_summary() produces {marker!r} but the served page does not contain it - "
        f"the safeguard is dead code again"
    )


def test_the_model_view_reports_false_play_not_only_recall() -> None:
    """Cross-venue recall is measured on folds with no empty pitch in them, so a page that
    ranks on it alone recommends the model that says PLAY most often."""
    from pitch_occupancy.api.models_view import collect, render

    if not any(r.get("false_play") for r in collect()["rows"]):
        pytest.skip("h3_with_false_play.csv not generated")
    html = render()
    assert "False-play" in html and "Balanced" in html
    assert "answering" in html and "playing" in html


def test_the_model_view_names_the_backbone_that_survives_the_control() -> None:
    from pitch_occupancy.api.models_view import _balanced, collect, render

    rows = [r for r in collect()["rows"] if _balanced(r) is not None]
    if not rows:
        pytest.skip("no false-play data")
    best = max((r for r in rows if r["key"] != "clock_rule"), key=_balanced)
    assert best["label"] in render()


# --- markdown entities ------------------------------------------------------


def test_typographic_entities_render_as_characters_not_as_text() -> None:
    """`html.escape` broke entities the author meant to be rendered.

    Six `&minus;` and three `&mdash;` were displaying as raw text on the site, several of
    them the sign of a signed number - a reader saw "&minus;0.2000" where the entire point
    was that the value is negative.
    """
    from pitch_occupancy.api.markdown import _inline

    assert _inline("a &minus;0.2 drop") == "a −0.2 drop"
    assert _inline("built &mdash; untested") == "built — untested"


def test_restoring_entities_does_not_reopen_html_injection() -> None:
    """The escaping exists to stop markdown source injecting HTML. It still does.

    Every restored entity is purely typographic, so none can begin a tag or an attribute.
    """
    from pitch_occupancy.api.markdown import _inline

    for hostile in (
        "<script>alert(1)</script>",
        "&amp;lt;img src=x onerror=y&amp;gt;",
        '&amp;quot; onload=x',
        "<iframe src=evil>",
    ):
        out = _inline(hostile)
        assert "<script" not in out
        assert "<img" not in out
        assert "<iframe" not in out


def test_an_unknown_entity_is_left_escaped() -> None:
    """Only the whitelist is restored; anything else stays visible as source."""
    from pitch_occupancy.api.markdown import _inline

    assert _inline("&nosuchentity;") == "&amp;nosuchentity;"


def test_the_served_page_contains_no_double_escaped_entities(client) -> None:
    """The end-to-end version, which is what the reader actually gets."""
    import re

    leftovers = re.findall(r"&amp;[a-z]{2,10};", client.get("/").text)
    assert not leftovers, f"literal entity text on the page: {sorted(set(leftovers))}"


# --- the models tab must show uncertainty ----------------------------------
#
# `preregistration.md` standing rule 3 is "uncertainty on every number: 95% bootstrap CI".
# This is the page that makes the production recommendation, and it rendered bare point
# estimates while the source CSVs carried ci_low/ci_high and macro_f1_lo/hi all along -
# collect() loaded those files and dropped the columns.


def test_the_models_table_renders_confidence_intervals() -> None:
    import re

    from pitch_occupancy.api.models_view import render

    intervals = re.findall(r"\[\d\.\d{3}, \d\.\d{3}\]", render())
    assert len(intervals) >= 6, f"expected intervals beside the estimates, found {intervals}"


def test_the_recommendation_carries_its_interval() -> None:
    from pitch_occupancy.api.models_view import collect, render

    winner = max(
        (r for r in collect()["rows"] if r.get("cross")), key=lambda r: float(r["cross"])
    )
    html = render()
    assert f"{float(winner['cross_lo']):.3f}" in html
    assert f"{float(winner['cross_hi']):.3f}" in html


def test_a_lead_is_not_claimed_when_the_intervals_overlap() -> None:
    """DINOv2's cross-venue interval overlaps both rivals', and H4 returns inconclusive.

    Something still has to be deployed, so the recommendation stands - but it must not read
    as a measured gap when a paired test does not find one.
    """
    from pitch_occupancy.api.models_view import collect, render, _overlap, TRAINED

    rows = collect()["rows"]
    winner = max(
        (r for r in rows if r["key"] in TRAINED and r.get("cross")),
        key=lambda r: float(r["cross"]),
    )
    overlapping = [
        r for r in rows
        if r["key"] in TRAINED and r["key"] != winner["key"]
        and _overlap(winner, r, "cross_lo", "cross_hi")
    ]
    html = render()
    if overlapping:
        assert "not established by the intervals" in html
        for r in overlapping:
            assert r["label"] in html
    else:  # pragma: no cover - would mean the data changed materially
        assert "not established by the intervals" not in html


def test_overlap_detection_is_symmetric_and_handles_missing_bounds() -> None:
    from pitch_occupancy.api.models_view import _overlap

    a = {"lo": "0.10", "hi": "0.50"}
    b = {"lo": "0.40", "hi": "0.90"}
    far = {"lo": "0.80", "hi": "0.95"}
    assert _overlap(a, b, "lo", "hi") and _overlap(b, a, "lo", "hi")
    assert not _overlap(a, far, "lo", "hi")
    assert not _overlap(a, {"lo": "", "hi": ""}, "lo", "hi")
    assert not _overlap(a, {}, "lo", "hi")


def test_a_missing_interval_renders_nothing_rather_than_a_dash() -> None:
    """An absent CI must not look like a measured one."""
    from pitch_occupancy.api.models_view import _ci

    assert _ci(None, None) == ""
    assert _ci("", "") == ""
    assert "0.100" in _ci("0.1", "0.9")
