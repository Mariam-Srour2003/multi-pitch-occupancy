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
