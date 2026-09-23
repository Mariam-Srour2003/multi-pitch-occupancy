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
from pitch_occupancy.api.talk import SECTIONS


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


def test_every_section_of_the_talk_gets_a_tab_and_a_view(client) -> None:
    """The site is the talk: one tab per section, in order, and each one reachable."""
    html = client.get("/").text
    for key, (label, _) in SECTIONS.items():
        assert f'data-view="{key}"' in html, f"{key} has no view"
        assert label in html, f"{label} is missing from the nav"


def test_the_nav_is_exactly_the_eight_parts_of_the_report(client) -> None:
    """The site follows the report: summary, plan, introduction, four chapters, conclusion.

    Asserted as an exact list rather than a subset. A ninth tab is not a small regression
    here - the whole point of the structure is that the nav *is* the report's table of
    contents, and an extra entry means a part exists that the report does not have.
    """
    import re as _re

    html = client.get("/").text
    nav = _re.findall(r'<button data-view="(\w+)">([^<]+)</button>', html)
    assert [k for k, _ in nav] == list(SECTIONS), f"nav drifted: {nav}"
    assert len(nav) == 8, f"{len(nav)} tabs, expected 8"
    for gone in ("prereg", "questions", "database", "code", "ethics", "findings",
                 "dataset", "ideas", "overview", "models", "rules", "searches",
                 "augmentation", "start", "topic", "roadmap", "purpose", "how", "problem",
                 "solution", "thanks"):
        assert f'data-view="{gone}"' not in html, f"{gone} tab is still on the page"


def test_every_section_renders_blocks_rather_than_appearing_empty(client) -> None:
    """A view that silently renders nothing looks the same as one that is not there."""
    bodies = _tab_bodies(client.get("/").text)
    for key in SECTIONS:
        assert key in bodies, f"{key} view not found"
        body = bodies[key]
        assert len(body) > 200, f"{key} rendered almost nothing"
        assert '<header class="tk-hero' in body, f"{key} has no hero"
        assert '<section class="tk-block' in body, f"{key} has no blocks"


def test_the_page_carries_no_delivery_notes(client) -> None:
    """The site is what the room looks at while you talk, so it holds no script.

    Each tab briefly ended with a collapsed "What to say here". It was removed on purpose:
    delivery notes on the screen compete with the person delivering them. The spoken
    version lives in `thesis/presentation/SPEAKER_SCRIPT.md`. Asserted rather than
    assumed, so putting it back is a decision someone makes rather than a regression that
    slips in.
    """
    html = client.get("/").text
    for gone in ("tk-say", "What to say"):
        assert gone not in html, f"delivery notes are back on the page: {gone!r}"


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


def test_the_model_comparison_reaches_chapter_two(client) -> None:
    """The comparison is collapsed under Chapter 2, which is where the report makes its
    claim about which backbone leads."""
    body = _tab_bodies(client.get("/").text)["ch2"]
    assert "The full model comparison" in body, "the comparison left the page"
    assert "DINOv2" in body


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
    ["blocked_questions", "pipeline", "protocols", "schema", "augmentation_axes"],
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
    ["blocked_questions", "pipeline", "protocols", "schema", "augmentation_axes"],
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


def test_the_rule_table_reaches_the_report(client) -> None:
    """The decision table is the one thing Chapter 3 cannot be read without.

    This test used to guard the storage schema too. The schema block was taken off the site
    on 2026-09-22, deliberately: the chapters were cut back to the model, the detector and
    the data work, and a database diagram is not any of those. `diagrams.schema()` and the
    test that it never invents a row count both stand, so the drawing is still correct if it
    is ever wanted back - it is simply not on a tab. The rule table is a different case: it
    is what the chapter argues, not evidence filed beneath it.
    """
    bodies = _tab_bodies(client.get("/").text)
    assert "rule table" in bodies["ch3"], "the decision table left the page"
    assert "Preprocessing" in bodies["ch4"], "the preprocessing list left the page"


# --- augmentation tab and figure serving -------------------------------------


def test_the_augmentation_argument_reaches_chapter_four(client) -> None:
    """Augmentation is Chapter 4, with the full argument collapsed beneath it."""
    body = _tab_bodies(client.get("/").text)["ch4"]
    assert "augmentation" in body.lower()
    assert "The augmentation argument in full" in body
    assert "discards nothing" in body


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

    The counts came off the site on 2026-09-22 with the rest of the conclusion tab, which
    was emptied to the two things that happen last - questions, and the demo. That is a
    decision, not a regression: `thesis/claims.md` still records every claim and every
    retraction, `verify_claims` still re-derives them from their artefacts on every run, and
    `render_findings()` still builds the view for anyone who wants it back on a tab. What
    this test still guards is the part that was never wanted: the archive itself.
    """
    html = client.get("/").text
    for gone in ("The full log", '<details class="fentry"', '<details class="fmonth"'):
        assert gone not in html, f"the archive is back on the page: {gone!r}"
    # and it is evidence under a slide now, not a tab of its own
    assert 'data-view="findings"' not in html


def test_the_hidden_renderers_still_build() -> None:
    """The three findings layers are off the page and still implemented.

    Taking them off the page was a presentation decision, not a decision to lose them - so
    each is called here. Without this they are code nothing runs, which is exactly the
    untested-branch defect this project has twice removed from its own guards.
    """
    from pitch_occupancy.api import findings_summary as fs

    assert fs.render_claims().count('class="fclaim"') > 20
    assert "<h2>What was withdrawn</h2>" in fs.render_retractions()
    assert fs.render_log().count('<details class="fentry"') > 20


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
        r'<section class="view" data-view="([a-z0-9]+)" hidden>'
        r'<div class="doc">', html
    )
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_every_tab_is_a_scrollable_page_not_a_deck(client) -> None:
    """Each tab reads top to bottom. No slides, no pager, nothing to press.

    The deck was replaced on purpose: a reader who has to page through 51 slides to find one
    number is worse served than one who can scroll and use find-in-page.
    """
    bodies = _tab_bodies(client.get("/").text)
    for tab in SECTIONS:
        assert tab in bodies, f"{tab} tab is missing entirely"
        assert '<div class="tk">' in bodies[tab], f"{tab} is not a talk page"
        for deck in ("data-deck", "dk-s", "dk-bar", "dk-notes"):
            assert deck not in bodies[tab], f"{tab} still carries deck markup: {deck}"


def test_the_evidence_is_served_open_rather_than_dropped(client) -> None:
    """Replacing the documents with the talk must not mean losing what they showed.

    The model chapter kept its comparison one click below the page, in a `<details>`, until
    2026-09-23. Nothing on the site is collapsed now: the comparison is the whole of the
    evidence for the chapter's claim about which backbone leads, and a reader who has to
    click to reach it is a reader who does not look - the same reasoning that already put
    the augmentation sheet in Chapter 4's body.

    The claims ledger was served the same way under the conclusion until 2026-09-22, when
    that tab was emptied to the two things that actually happen last - questions, and the
    demo. The ledger itself is unaffected: `verify_claims` still re-derives every claim from
    its artefact, `thesis/claims.md` still records the retractions, and `render_findings()`
    still builds the view. It is no longer on a tab.
    """
    body = _tab_bodies(client.get("/").text)["ch2"]
    # By what is served under the heading, not by a count: a count passes while the wrong
    # thing is on the page, and breaks on a cosmetic change that costs nothing.
    assert "The full model comparison" in body, "ch2 dropped its evidence"
    assert "<summary>" not in body, "the comparison is behind a toggle again"
    for model in ("DINOv2", "ConvNeXtV2", "ViT", "Clock rule"):
        assert model in body, f"the comparison lost {model}"


def test_the_page_is_slides_first_and_the_walls_are_gone(client) -> None:
    """The page is judged on what it shows before anything is expanded.

    The whole point of replacing the document tabs was to put the argument in front of the
    wall rather than behind it. Two properties say that happened: the visible page is made
    of slides, and the rendered Markdown documents - 40 KB of codebase map, 33 KB of
    pre-registration - are no longer served at all.
    """
    html = client.get("/").text
    # Was 30, then 24, 22, 21. The chapters were cut back to the model, the detector and
    # the data work on 2026-09-22, and the conclusion was emptied to questions and the demo;
    # on 2026-09-23 the augmentation retraction, the generated-data slide and the
    # probe-versus-fine-tuning slide were withdrawn, then the preset cards, the two
    # searches, the applications beats, the detector settings table and the YOLOv8n model
    # cards. The number is a tripwire against the wall coming back, not a content floor - it
    # moves down with a deliberate cut and never up on its own. What is asserted is that the
    # page is still made of blocks of the same kind.
    assert html.count('<section class="tk-block') >= 20, "the page is not made of blocks"
    for wall in ("The full document &mdash;", "preregistration.md", "CODEBASE.md",
                 "data_layout.md", "rq_matrix.md"):
        assert wall not in html, f"a source document is still being served: {wall!r}"
    # It used to be ~347 KB, most of it rendered Markdown nobody read.
    assert len(html) < 260_000, f"page is {len(html) // 1024} KB - a wall has come back"


def test_the_augmentation_tab_is_deliberately_image_heavy(client) -> None:
    """The one tab that is more, not less.

    Augmentation and preprocessing code fails silently - a preset that does nothing, a crop
    that removes the goalmouth - and every one of those passes a shape and dtype check. The
    only reliable check is a person looking, so this tab has to carry the pictures.
    """
    html = client.get("/").text
    # Was 10, when the before/after sheet was all seventeen switches. It is six of them plus
    # the three full sheets now, and the floor moved with it - the property is that the tab
    # still argues in pictures, not that it carries a particular number of them.
    assert html.count("<img") >= 8, "the talk lost its augmentation sheets"
    assert "/figs/preproc/" in html, "the before/after pairs are not on the page"


def test_the_chosen_preprocessing_pairs_reach_the_tab_each_with_its_phrase(client) -> None:
    """The sheet was every row of the CSV until 2026-09-23; it is now the six of `_SHEET`.

    Seventeen tiles are scrolled past, and the second value of a switch shows the same thing
    less strongly - so the tab keeps six that are read over seventeen that are not, and all
    seventeen are still drawn on `preprocess_effects.jpg` in the same chapter.

    What is asserted is the property that makes the cut defensible: every tile that survived
    carries the phrase saying what it does. A tile with a name and two numbers and no
    explanation is the wall this was cutting back, one card at a time.
    """
    from pitch_occupancy.api.talk import RESULTS, _SHEET

    if not (RESULTS / "preprocess_pairs.csv").exists():
        pytest.skip("no preprocessing pairs generated")
    html = client.get("/").text
    for label, blurb in _SHEET:
        assert f"<code>{label}</code>{{}}".format(f'<div class="w">{blurb}</div>') in html, (
            f"{label} has no before/after card, or lost the phrase explaining it"
        )
    assert html.count('<figure class="tk-pair') == len(_SHEET) == 6


def test_the_rq_statuses_are_parsed_from_the_matrix_not_restated() -> None:
    """The statuses move as the work moves, so a copy in the page would be right the day it
    was written and wrong afterwards with nothing to catch it."""
    from pitch_occupancy.api.talk import rq_status

    rows = rq_status()
    assert len(rows) >= 6, f"only parsed {len(rows)} research questions"
    assert all(rid.startswith("RQ") for rid, _, _ in rows)
    # The matrix records more than one outcome; a parser that collapsed them all to one
    # value would render a board of identical chips and look fine.
    assert len({status for _, _, status in rows}) > 1


def test_a_slide_reports_a_missing_artefact_rather_than_rendering_a_hole(tmp_path) -> None:
    """A page that silently shows nothing where its argument should be is the failure
    `augmentation_grid.py` was written to prevent, and it applies to the page too."""
    from pitch_occupancy.api import talk

    assert "missing" in talk.figure("no_such_figure.png", script="uv run something")


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
    from pitch_occupancy.api.talk import RESULTS, coverage_counts

    if not (RESULTS / "coverage.md").exists():
        pytest.skip("no coverage report generated")
    counts = coverage_counts()
    assert counts["venues_with_empty"] == 1
    assert counts["venues_counted"] > 2, "the lighting table was matched again"
