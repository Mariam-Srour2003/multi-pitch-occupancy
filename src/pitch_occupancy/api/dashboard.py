"""The operator dashboard, served at `/`.

The tool a facility manager opens. It reads live database state through the same `/api/v1`
endpoints an integrator would use - nothing here has privileged access, so if the page can
show it, the API can serve it.

The flow it is shaped around: land on what needs
attention, open a slot's evidence, agree or correct in one click. Anything that does not
serve that flow is not here.

At the top, a "Getting started" card lists the setup steps from `GET /api/v1/setup/checklist`
(`setup_check`). It opens while a required step is missing and collapses itself once they are
all done; if the endpoint is not there the card simply stays hidden.
"""

from __future__ import annotations

from fastapi.responses import HTMLResponse

from pitch_occupancy.api.layout import JS_COMMON
from pitch_occupancy.api.nav import NAV_CSS, nav_html
from pitch_occupancy.api.templating import render

__all__ = ["DASHBOARD_HTML"]

DASHBOARD_HTML = render("dashboard.html", NAV_CSS=NAV_CSS,
                        nav_html=nav_html("dashboard"), JS_COMMON=JS_COMMON)


def dashboard_response() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)
