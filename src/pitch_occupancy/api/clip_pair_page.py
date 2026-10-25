"""The two-camera reviewer page. Served at `/clip/pair`; posts to `/api/v1/clip/pair/analyse`.

Its own page rather than a mode of `/clip`, so the single-camera reviewer keeps doing exactly
what it did. It shares the clip page's style sheet and fonts (`layout.STYLE`, `layout.FONTS`).

The layout puts the three answers on top of each other: camera A's own verdict, camera B's
own verdict, and the pitch's from the two together. That stacking is the point of the page.
A reader sees a moment where A says "3 people, not playing" and B says "2 people, not
playing" and the pitch says *playing*, and the row beneath says why - five people added, the
ball on B's side - and whether that answer would survive the two views overlapping.
"""

from __future__ import annotations

from pitch_occupancy.api.layout import FONTS, JS_COMMON, STYLE
from pitch_occupancy.api.nav import NAV_CSS, nav_html
from pitch_occupancy.api.templating import render, template

__all__ = ["PAIR_HTML"]

_EXTRA = template("clip_pair_page_extra.html")

PAIR_HTML = render("clip_pair_page.html", FONTS=FONTS, STYLE=STYLE, EXTRA=_EXTRA,
                   NAV_CSS=NAV_CSS, nav_html=nav_html("testing"), JS_COMMON=JS_COMMON)
