"""The shared page script: one ``esc`` for every page, escaping both quotes."""

from __future__ import annotations

from pathlib import Path

import pytest

from pitch_occupancy.api import layout
from pitch_occupancy.api.clip_pair_page import PAIR_HTML
from pitch_occupancy.api.dashboard import DASHBOARD_HTML
from pitch_occupancy.api.image_page import IMAGE_HTML

API = Path(layout.__file__).parent


def test_the_shared_esc_escapes_both_quotes() -> None:
    assert """/[&<>"']/g""" in layout.JS_COMMON and "&#39;" in layout.JS_COMMON


def test_page_puts_the_shared_script_first() -> None:
    html = layout.page("t", "s", "<p>b</p>", script="go();")
    assert f"<script>\n{layout.JS_COMMON}go();" in html
    assert layout.JS_COMMON not in layout.page("t", "s", "<p>b</p>")


@pytest.mark.parametrize("html", [PAIR_HTML, DASHBOARD_HTML, IMAGE_HTML],
                         ids=["clip_pair", "dashboard", "image"])
def test_stand_alone_pages_carry_it_once(html: str) -> None:
    assert html.count("const esc=") == 1 and layout.JS_COMMON in html


def test_no_page_defines_its_own_esc() -> None:
    files = [*API.glob("*.py"), *(API / "templates").iterdir()]
    own = [p.name for p in files if "const esc" in p.read_text(encoding="utf-8")]
    assert not own
