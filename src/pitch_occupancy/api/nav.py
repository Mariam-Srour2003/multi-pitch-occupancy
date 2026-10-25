"""The site's navigation bar, in one place.

Kept apart from `layout.py` so the pages that build their own head can take the bar alone.
"""

from __future__ import annotations

from html import escape

from pitch_occupancy.api.templating import static_text

__all__ = ["NAV", "NAV_CSS", "SECTIONS", "nav_html"]

#: The site's sections, in the order the header shows them: ``(key, label, href)``.
#: The everyday pages (`simple_pages.py`) come first, in the order a new site uses them; the
#: rest of the original pages are one click away under More.
NAV: tuple[tuple[str, str, str], ...] = (
    ("home", "Home", "/"),
    ("playgrounds", "Playgrounds", "/playgrounds"),
    ("cameras-simple", "Cameras", "/cameras/add"),
    ("results", "Results", "/results"),
    ("records", "Records", "/records"),
    ("more", "More", "/more"),
)

#: The original sections, now under More. On one of their pages the bar adds the section after
#: More ("More · Testing"), so a reviewer page still leads back to its hub.
SECTIONS: dict[str, tuple[str, str]] = {
    "dashboard": ("Pitch verdicts", "/client"),
    "reports": ("Reports", "/reports"),
    "testing": ("Testing", "/testing"),
    "cameras": ("Camera setup", "/cameras"),
    "bookings": ("Bookings", "/bookings"),
    "roi": ("Boundaries", "/roi"),
    "alerts": ("Alerts", "/alerts"),
}

#: The bar's own rules. Pages that build their head by hand include this after their sheet.
NAV_CSS = "<style>" + static_text("nav.css") + "</style>"


def nav_html(active: str = "") -> str:
    """The header's section links, with ``active`` marked.

    The original pages still name their old sections ("testing", "roi", ...); those live
    under More now, so More lights up for them and the section follows it as a link back.
    """
    section = SECTIONS.get(active)
    if active and active not in {key for key, _, _ in NAV}:
        active = "more"
    links = "".join(
        f'<a href="{href}"{" class=on" if key == active else ""}>{escape(label)}</a>'
        for key, label, href in NAV)
    if section is not None:
        label, href = section
        links += f'<a href="{href}" class=on>· {escape(label)}</a>'
    return f"<nav>{links}</nav>"
