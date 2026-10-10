"""The site's navigation bar, in one place.

Kept apart from `layout.py` because the clip page, whose style sheet `layout.py` borrows, needs
the bar too - and a module both imports would otherwise be a cycle.
"""

from __future__ import annotations

from html import escape

__all__ = ["NAV", "NAV_CSS", "nav_html"]

#: The site's sections, in the order the header shows them: ``(key, label, href)``.
NAV: tuple[tuple[str, str, str], ...] = (
    ("dashboard", "Dashboard", "/"),
    ("testing", "Testing", "/testing"),
    ("cameras", "Cameras", "/cameras"),
    ("bookings", "Bookings", "/bookings"),
    ("roi", "Boundaries", "/roi"),
)

#: The bar's own rules. Pages that build their head by hand include this after their sheet.
NAV_CSS = """<style>
.hin nav{margin-left:auto;display:flex;gap:16px;flex-wrap:wrap}
.hin nav a{margin-left:0!important;color:var(--accent);font-size:13px;text-decoration:none}
.hin nav a.on{color:var(--ink);font-weight:600;text-decoration:underline;
  text-underline-offset:5px;text-decoration-color:var(--accent)}
</style>"""


def nav_html(active: str = "") -> str:
    """The header's section links, with ``active`` marked."""
    links = "".join(
        f'<a href="{href}"{" class=on" if key == active else ""}>{escape(label)}</a>'
        for key, label, href in NAV)
    return f"<nav>{links}</nav>"
