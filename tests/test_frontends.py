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
from pitch_occupancy.api.thesis_site import DOCUMENTS, HIDDEN


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
        if key in HIDDEN:  # built, not placed - covered by the hidden-renderer test
            continue
        assert f'data-view="{key}"' in html, f"{key} has no view"
        assert label in html, f"{label} is missing from the nav"


def test_the_searches_view_exists_beyond_the_documents(client) -> None:
    """Searches are assembled from result files rather than a Markdown source."""
    assert 'data-view="searches"' in client.get("/").text


def test_documents_actually_render_rather_than_appearing_empty(client) -> None:
    """A view that silently renders nothing looks the same as one that is not there."""
    html = client.get("/").text
    for key in DOCUMENTS:
        if key in HIDDEN:
            continue
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


def test_the_input_path_challenge_reaches_the_served_page(client) -> None:
    """Asserted on the HTML the reader receives, not on the helper that builds it.

    A safeguard once lived in a function called by one test and nothing else, while the page
    that was actually served ranked by recall alone. Testing the function certified something
    no reader ever saw. So this goes through the client: the recommendation was challenged and
    the challenge was tested, and a reader deciding what to deploy must see both.
    """
    from pitch_occupancy.api import models_view

    if not (models_view.RESULTS / "input_path_protocol.csv").exists():
        pytest.skip("input-path protocol results not present")
    html = client.get("/").text
    assert "This recommendation was challenged" in html
    assert "under the swap" in html


def test_the_input_path_verdict_is_read_from_the_csv_not_written_in(monkeypatch, tmp_path) -> None:
    """The most-quoted figure in this project was once a plot with its ranks hardcoded,
    contradicting its own source table. A verdict paragraph can fail the same way, so it is
    checked by feeding it the opposite data and requiring the opposite word."""
    from pitch_occupancy.api import models_view

    same = {
        (k, f"train_{cam}"): {"estimate": "-0.5000"}
        for k in models_view.TRAINED for cam in ("camera_A", "camera_B")
    }
    assert "reverses" not in models_view._input_path_note(same)
    assert "same direction" in models_view._input_path_note(same)

    flipped = dict(same)
    flipped[("dinov2", "train_camera_B")] = {"estimate": "+0.5000"}
    assert "reverses" in models_view._input_path_note(flipped)
    assert "DINOv2 reverses outright" in models_view._input_path_note(flipped)


def test_the_input_path_rates_in_the_prose_come_from_the_csv_too() -> None:
    """The paragraph quotes a before/after pair and an other-direction rate. Those are the
    numbers most likely to be typed in once and left behind when the run changes."""
    from pitch_occupancy.api import models_view

    made_up = {
        ("convnextv2", "train_camera_A"):
            {"estimate": "-0.7000", "raw": "0.8100", "preproc": "0.1100"},
        ("convnextv2", "train_camera_B"):
            {"estimate": "-0.0100", "raw": "0.0420", "preproc": "0.0320"},
    }
    out = models_view._input_path_note(made_up)
    assert "from 0.81 to 0.11" in out
    assert "already 0.042" in out
    assert "0.99" not in out and "0.028" not in out


def test_the_input_path_panel_is_absent_rather_than_invented_when_unmeasured() -> None:
    from pitch_occupancy.api import models_view

    assert models_view._input_path_note({}) == ""


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
    # 970 until A25 relabelled 216 clip frames from `day` to `night`: a brightness threshold
    # had filed floodlit night football as daylight. The correction makes the confound worse,
    # which is the point of the diagram.
    assert ">1186<" in svg  # active play, floodlit


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


def test_augmentation_tab_never_shows_the_headline_without_its_retraction(client) -> None:
    """This page said "Not measured yet" for as long as that was true, and the test pinned
    the sentence. Then it was measured, the headline was retracted the same day, and the
    page kept saying it - so the guard is now on the pairing rather than on a phrase.

    `light` reached 0.8550 in one draw of five and 0.3479-0.4136 in the other four. Either
    number alone misleads: the first is the maximum of five, the second hides that the
    effect can appear at all. If the page shows one it must show the spread."""
    html = client.get("/").text
    if "0.8550" in html or "0.687" in html:
        assert "0.2206" in html, "the headline is on the page without its standard deviation"
        assert "maximum of five" in html
        assert "0.4406" in html, "nothing says four of five draws are below no augmentation"
    assert "no footage in this dataset" in html.lower()  # the wet-weather caveat stands


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


# --- the findings summary ---------------------------------------------------


def test_the_findings_tab_is_counts_not_an_archive(client) -> None:
    """The tab is four counts. It was the whole log open, then three collapsed layers.

    Both earlier versions put the archive on the page - 86 entries rendered open, then 34
    claims and 86 collapsed rows. Collapsed or not, that is an archive wearing a page's
    clothes, and this tab is where the project is presented. Asserted on the served page
    rather than on the builder, because the page is what a reader got.

    The cost is real and this test pins it too: the log is no longer reachable from the
    site. If a later change puts it back, that should be a decision someone makes on
    purpose, not a regression that slips in - so the absences are asserted, not assumed.
    """
    html = client.get("/").text
    assert '<div class="ftiles">' in html, "the counts are the tab now"
    for gone in ("What the project found", "What was withdrawn", "The full log",
                 '<details class="fentry"', '<details class="fmonth"'):
        assert gone not in html, f"the archive is back on the page: {gone!r}"


def test_the_hidden_renderers_still_build() -> None:
    """`ideas`, and the three findings layers, are off the page and still implemented.

    Taking them off the page was a presentation decision, not a decision to lose them - so
    each is called here. Without this they are code nothing runs, which is exactly the
    untested-branch defect this project has twice removed from its own guards.
    """
    from pitch_occupancy.api import findings_summary as fs
    from pitch_occupancy.api.slides import SLIDES
    from pitch_occupancy.api.thesis_site import _doc

    assert fs.render_claims().count('class="fclaim"') > 20
    assert "<h2>What was withdrawn</h2>" in fs.render_retractions()
    assert fs.render_log().count('<details class="fentry"') > 20

    for tab in HIDDEN:
        assert tab in DOCUMENTS, f"{tab} is hidden by deletion, not by HIDDEN"
        assert '<div class="slide">' in SLIDES[tab](), f"{tab}'s slide stopped building"
        assert "<details" in _doc(DOCUMENTS[tab][1], tab), f"{tab}'s document stopped building"


def test_no_log_entry_is_dropped_by_the_summary() -> None:
    """Summarising must not mean omitting: every `##` heading in the log gets an entry.

    This is the property that makes collapsing honest. A page that showed the best entries
    and quietly lost the rest would be the same failure as the export that stopped covering
    the thesis - and would be just as invisible.
    """
    from pitch_occupancy.api.findings_summary import LOG, log_entries

    if not LOG.exists():
        pytest.skip("no experiment log yet")
    headings = sum(
        1 for line in LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    )
    entries = log_entries()
    # One extra for the preamble, which has no heading and is kept as its own entry.
    assert len(entries) == headings + 1


def test_entry_dates_are_found_wherever_the_heading_puts_them() -> None:
    """A third of the log's headings put the date last: `## Title (WP3-T8) - 2026-09-07`.

    A pattern anchored to the front of the heading read 19 of 76 as undated and dropped
    them into one unsorted pile, titles and all. So this asserts the weaker, real property:
    if the heading contains a date, the entry carries it.
    """
    import re

    from pitch_occupancy.api.findings_summary import log_entries

    for entry in log_entries():
        source = f'{entry["date"]} {entry["title"]}'
        if re.search(r"\d{4}-\d{2}-\d{2}", source) and not entry["date"]:
            pytest.fail(f"date left in the title: {entry['title']!r}")


def test_every_retraction_link_points_at_an_entry_that_exists() -> None:
    """The retraction strip is the reason to keep the log reachable at all.

    A number that was published and then withdrawn is part of the method, so those entries
    stay as written - this strip only makes them findable. A link into an id the page does
    not contain would make the most valuable entries the least reachable, which is worse
    than not linking them.
    """
    import re

    from pitch_occupancy.api.findings_summary import render_log, render_retractions

    # Against the renderers, not the page: both are off the tab now, and pointed at the
    # served HTML this found no links and skipped itself - a test that can no longer fail
    # is worse than no test, because it still reports as coverage.
    html = render_retractions() + render_log()
    targets = set(re.findall(r'href="#(entry-\d+)"', html))
    if not targets:
        pytest.skip("no retractions detected in this log")
    for target in targets:
        assert f'id="{target}"' in html, f"retraction links to a missing {target}"


def test_the_claims_summary_carries_every_ledger_claim() -> None:
    """The summary is the ledger, so a claim missing from it is a claim nobody reads.

    Counted against `claims.toml` rather than a fixed 34: adding a claim should not need a
    test edit, but losing one silently should fail.
    """
    import tomllib

    from pitch_occupancy.api.findings_summary import LEDGER, claim_groups

    if not LEDGER.exists():
        pytest.skip("no claims ledger")
    declared = len(tomllib.loads(LEDGER.read_text(encoding="utf-8"))["claim"])
    shown = sum(len(g["claims"]) for g in claim_groups())
    assert shown == declared


# --- the slides -------------------------------------------------------------


def _tab_bodies(html: str) -> dict[str, str]:
    """Each tab's rendered body.

    Split on the view boundaries rather than matched lazily to the next `</section>`: the
    findings summary nests `<section class="fgroup">` inside its tab, so a lazy match stops
    at the first inner close and silently measures a prefix. That is how a first attempt at
    this reported the findings tab as "0% collapsed" while it was 91%.
    """
    import re

    parts = re.split(
        r'<section class="view" data-view="([a-z]+)" hidden><div class="doc">', html
    )
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_every_tab_opens_with_a_slide(client) -> None:
    """A tab used to open with its source Markdown rendered in full - the right evidence and
    a poor page. Each now leads with counts and figures, and this asserts that on the served
    HTML rather than on the builder, because the wall is what a reader got."""
    bodies = _tab_bodies(client.get("/").text)
    for tab in ("overview", "models", "findings", "prereg", "questions", "dataset",
                "database", "code", "ethics", "augmentation", "searches"):
        assert tab in bodies, f"{tab} tab is missing entirely"
        assert '<div class="slide">' in bodies[tab], f"{tab} has no slide"


def test_the_document_is_collapsed_rather_than_dropped(client) -> None:
    """Summarising must not mean omitting.

    The whole argument for leading with a slide is that the evidence stays one click below
    it. `<details>`, not a CSS toggle - find-in-page and a saved copy still reach it.
    """
    bodies = _tab_bodies(client.get("/").text)
    for tab in ("prereg", "questions", "code", "ethics", "dataset", "database"):
        assert "<details" in bodies[tab], f"{tab} dropped its document instead of collapsing"


def test_the_long_documents_are_mostly_collapsed(client) -> None:
    """The page is judged on what it shows before anything is expanded.

    `docs/CODEBASE.md` is 40 KB and `preregistration.md` 33 KB; if either arrives visible,
    the slide has been added in front of the wall rather than in place of it.
    """
    import re

    bodies = _tab_bodies(client.get("/").text)
    for tab in ("code", "prereg"):
        body = bodies[tab]
        visible = re.sub(r'<div class="sl-detail-body">.*?</details>', "", body, flags=re.S)
        assert len(visible) < len(body) * 0.2, (
            f"{tab} shows {len(visible)} of {len(body)} bytes before expanding"
        )


def test_the_augmentation_tab_is_deliberately_image_heavy(client) -> None:
    """The one tab that is more, not less.

    Augmentation and preprocessing code fails silently - a preset that does nothing, a crop
    that removes the goalmouth - and every one of those passes a shape and dtype check. The
    only reliable check is a person looking, so this tab has to carry the pictures.
    """
    body = _tab_bodies(client.get("/").text)["augmentation"]
    assert body.count("<img") >= 10, "the augmentation tab lost its sheets"
    assert "/figs/preproc/" in body, "the before/after pairs are not on the page"


def test_every_preprocessing_pair_reaches_the_augmentation_tab(client) -> None:
    """Checked against the CSV that records what was generated, not a list typed here -
    which would be a second inventory, and the one that goes stale."""
    import csv

    from pitch_occupancy.api.slides import RESULTS

    pairs = RESULTS / "preprocess_pairs.csv"
    if not pairs.exists():
        pytest.skip("no preprocessing pairs generated")
    with pairs.open(newline="", encoding="utf-8") as fh:
        labels = [r["label"] for r in csv.DictReader(fh)]

    body = _tab_bodies(client.get("/").text)["augmentation"]
    missing = [lab for lab in labels if f"<code>{lab}</code>" not in body]
    assert not missing, f"switches with no before/after card: {missing}"


def test_the_rq_statuses_are_parsed_from_the_matrix_not_restated() -> None:
    """The statuses move as the work moves, so a copy in the page would be right the day it
    was written and wrong afterwards with nothing to catch it."""
    from pitch_occupancy.api.slides import rq_status

    rows = rq_status()
    assert len(rows) >= 6, f"only parsed {len(rows)} research questions"
    assert all(rid.startswith("RQ") for rid, _, _ in rows)
    # The matrix records more than one outcome; a parser that collapsed them all to one
    # value would render a board of identical chips and look fine.
    assert len({status for _, _, status in rows}) > 1


def test_a_slide_reports_a_missing_artefact_rather_than_rendering_a_hole(tmp_path) -> None:
    """A page that silently shows nothing where its argument should be is the failure
    `augmentation_grid.py` was written to prevent, and it applies to the page too."""
    from pitch_occupancy.api import slides

    assert "missing" in slides.figure("no_such_figure.png", script="uv run something")


def test_the_figure_route_serves_the_pairs_but_still_refuses_venue_check(client) -> None:
    """`results/figs/venue_check/` holds cropped pitch frames - operator footage that must
    never be published. Opening `figs/preproc/` for the before/after pairs must not have
    opened that with it."""
    assert client.get("/figs/preproc/gamma_0p7.jpg").status_code == 200
    for blocked in ("venue_check/x.jpg", "preproc/../venue_check/x.jpg",
                    "preproc/sub/x.jpg", "../../.gitignore", ".env"):
        assert client.get(f"/figs/{blocked}").status_code in (404, 405), blocked


def test_the_dataset_slide_counts_venues_not_lighting_columns() -> None:
    """The corpus's whole problem is that exactly one venue has an empty pitch.

    `coverage.md` opens with a class-by-*lighting* table whose EMPTY row is
    `| EMPTY | 485 | 9 | 494 |`, so a pattern matching "the EMPTY row" finds that one first
    and counts day and night as venues - which reported "all from 2 venues" for the fact the
    whole data request is about. Scoped to the class-by-venue section instead.
    """
    from pitch_occupancy.api.slides import RESULTS, coverage_counts

    if not (RESULTS / "coverage.md").exists():
        pytest.skip("no coverage report generated")
    counts = coverage_counts()
    assert counts["venues_with_empty"] == 1
    assert counts["venues_counted"] > 2, "the lighting table was matched again"
