"""The shared page frame: one style sheet, one navigation bar, one way to build a page.

The links live in `nav.py`; the style sheet, fonts and shared scripts live in `static/` and
are read once here. Pages inline them rather than linking them, so a page is one response
and its text does not depend on how static files are served.
"""

from __future__ import annotations

from html import escape

from pitch_occupancy.api.nav import NAV, NAV_CSS, nav_html
from pitch_occupancy.api.templating import static_text

__all__ = ["CALL_JS", "FONTS", "JS_COMMON", "NAV", "STYLE", "nav_html", "page"]

#: The font links.
FONTS = static_text("fonts.html")
#: The style sheet: colour tokens, light/dark, cards, tiles, tables.
STYLE = "<style>" + static_text("base.css") + "</style>"

#: JavaScript every page gets first: ``$`` (element by id) and ``esc`` (text made safe for
#: HTML, attribute values in either quote included). `page` adds it; the stand-alone pages
#: put it at the top of their own ``<script>``.
JS_COMMON = static_text("common.js")

#: ``call(url, opts)``: fetch JSON, throwing the server's ``detail`` (or the HTTP status).
CALL_JS = static_text("call.js")

_EXTRA = "<style>" + static_text("page.css") + "</style>"


def page(title: str, subtitle: str, body: str, *, active: str = "", css: str = "",
         script: str = "") -> str:
    """A whole page: head, header with navigation, ``body`` inside ``<main>``, ``script``.

    ``css`` is extra style rules (without the ``<style>`` tag); ``script`` is JavaScript
    (without the ``<script>`` tag), run after `JS_COMMON` - so it must not redeclare ``$``
    or ``esc``.
    """
    return (
        '<!doctype html>\n<html lang="en"><head>\n<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n{FONTS}{STYLE}{NAV_CSS}{_EXTRA}"
        + (f"<style>{css}</style>" if css else "")
        + "\n</head><body>\n<header><div class=\"hin\">"
        '<div class="brand">Pitch <span>Occupancy</span></div>'
        f'<div class="sub">{escape(subtitle)}</div>{nav_html(active)}</div></header>\n'
        f"<main>\n{body}\n</main>\n"
        + (f"<script>\n{JS_COMMON}{script}\n</script>\n" if script else "")
        + "</body></html>\n"
    )
