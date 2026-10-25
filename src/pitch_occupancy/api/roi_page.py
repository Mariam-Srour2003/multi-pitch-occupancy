"""The pitch boundary editor. Served at `/roi`; posts to `/api/v1/roi/*`.

Click the corners of the pitch, save it under a camera, and only people standing inside the
outline are counted. The reason it exists is the neighbouring
pitch: five-a-side pitches are built in rows, so a camera watching pitch 2 sees pitches 1 and
3 down the sides of its frame, and a match on the next pitch over puts real players into
frames where *this* pitch is empty.

Two decisions shape the layout.

**The preview runs the rule twice and shows both answers.** The failure a boundary prevents
is invisible in the verdict - the detector is right about what is in frame and wrong about
what was asked - so the editor puts the inside-the-outline and whole-frame verdicts side by
side with the detector's drawing of each. That is how an operator sees the boundary working
rather than being told it does, and it is also how they notice the opposite: a boundary that
changes the verdict on a frame with no neighbour in it is cutting off part of the pitch.

**Drawing is instant, classifying is not.** Every click redraws the outline and updates the
coverage figure locally, on a canvas, with no request at all. The model runs only when asked,
because a detector pass per click would make the editor unusable and teach the operator to
avoid the one feature that tells them whether the outline is right.
"""

from __future__ import annotations

from pitch_occupancy.api.nav import NAV_CSS, nav_html
from pitch_occupancy.api.templating import render

__all__ = ["ROI_HTML"]

ROI_HTML = render("roi_page.html", NAV_CSS=NAV_CSS, nav_html=nav_html("roi"))
