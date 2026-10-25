"""The image reviewer page. Served at `/images`; posts to `/api/v1/images/walkthrough`.

The still-image counterpart to `clip_page.py`, and the layout differs from it in one way that
matters. The clip page is built around *time*: a timeline, segments, and the reviewer's
judgement about which smoothing corrections to accept. None of that exists for a folder of
stills - they have no order, no neighbours and no duration - so building the same page with
the time parts greyed out would be a worse page, not a smaller one.

What a batch of stills has instead is **comparison**, so that is what this is built around.
Every image is checked the same way, and the gallery at the end sorts them by confidence
with the least confident first - because on a batch the useful question is not "what is the
verdict" but "which of these should I look at myself". A grid in upload order would bury
exactly the images the reader opened the page for.

The live stage above it is the clip walkthrough's, unchanged in spirit: one image at a time,
the detector's boxes and the rule path underneath, arriving as the detector finishes each
one.
"""

from __future__ import annotations

from pitch_occupancy.api.layout import JS_COMMON
from pitch_occupancy.api.nav import NAV_CSS, nav_html
from pitch_occupancy.api.templating import render

__all__ = ["IMAGE_HTML"]

IMAGE_HTML = render("image_page.html", NAV_CSS=NAV_CSS, nav_html=nav_html("testing"),
                    JS_COMMON=JS_COMMON)
