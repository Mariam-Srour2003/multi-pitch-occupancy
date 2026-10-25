"""The clip reviewer page. Served at `/clip`; posts to `/api/v1/clip/analyse`.

Kept apart from `dashboard.py` because it is a different job for a different moment: the
dashboard is the daily operator view over database state, this is a one-off "tell me what is
in this file" with nothing persisted on either side.

The layout is built around the one thing the reviewer has to judge and the model cannot:
**which corrections to accept.** Smoothing turns an isolated disagreeing sample into its
neighbours, which is right when the sample was a misread and wrong when it was a real brief
event. Both occur in the footage this was built against. So corrected samples are not
quietly replaced - the timeline marks them, the table strikes the raw prediction through
beside the corrected one, and a toggle switches the whole view back to raw. If the reviewer
never looks, they still see a count rather than a clean answer that hid its edits.
"""

from __future__ import annotations

from pitch_occupancy.api.layout import FONTS, STYLE
from pitch_occupancy.api.nav import NAV_CSS, nav_html
from pitch_occupancy.api.templating import render

__all__ = ["CLIP_HTML"]

CLIP_HTML = render("clip_page.html", FONTS=FONTS, STYLE=STYLE, NAV_CSS=NAV_CSS,
                   nav_html=nav_html("testing"))
