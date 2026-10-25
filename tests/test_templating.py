"""Page markup read from `api/templates/` and `api/static/`, slots filled in one pass."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pitch_occupancy.api import templating

API = Path(templating.__file__).parent


@pytest.fixture
def one_template(monkeypatch, tmp_path: Path) -> Path:
    (tmp_path / "templates").mkdir()
    monkeypatch.setattr(templating, "_HERE", tmp_path)
    templating.template.cache_clear()
    yield tmp_path / "templates" / "t.html"
    templating.template.cache_clear()


def test_slots_are_filled_once(one_template: Path) -> None:
    one_template.write_text("<p>{{a}}</p>{{b}} {x:1}", encoding="utf-8")
    assert templating.render("t.html", a="{{b}}", b="B") == "<p>{{b}}</p>B {x:1}"


def test_slots_and_parts_must_match(one_template: Path) -> None:
    one_template.write_text("{{a}}", encoding="utf-8")
    with pytest.raises(KeyError):
        templating.render("t.html")
    with pytest.raises(KeyError):
        templating.render("t.html", a="1", b="2")


def test_every_template_is_used_and_every_slot_filled() -> None:
    sources = "".join(p.read_text(encoding="utf-8") for p in API.glob("*.py"))
    for path in (API / "templates").iterdir():
        assert f'"{path.name}"' in sources, f"{path.name} is not loaded by any module"
    for path in (API / "static").iterdir():
        assert f'"{path.name}"' in sources, f"static/{path.name} is not loaded by any module"


def test_no_page_leaves_a_slot_unfilled() -> None:
    from pitch_occupancy.api import (
        clip_page,
        clip_pair_page,
        dashboard,
        image_page,
        results_page,
        roi_page,
    )

    pages = [clip_page.CLIP_HTML, clip_pair_page.PAIR_HTML, dashboard.DASHBOARD_HTML,
             image_page.IMAGE_HTML, results_page.BODY, roi_page.ROI_HTML]
    assert not [p for p in pages if re.search(r"\{\{\w+\}\}", p)]
