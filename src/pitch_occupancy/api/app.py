"""FastAPI application.

This is the *only* part of the system uvicorn serves. The occupancy pipeline itself is a
scheduled batch process (``pitch_occupancy.worker``) that runs independently, so a
dashboard restart never interrupts sampling.

Run it with::

    uv run uvicorn pitch_occupancy.api.app:app --reload

Endpoints arrive with their work packages:
  - WP6-T1  simulator API   ``/api/v1/cameras/{camera_id}/snapshot``
  - WP6-T6  dashboard       fields, slots, evidence inspector, overrides
  - WP6-T5  reconciliation  anomaly list and adjudication
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pitch_occupancy import __version__
from pitch_occupancy.api.dashboard import dashboard_response
from pitch_occupancy.api.thesis_site import page as thesis_page
from pitch_occupancy.api.routes import router
from pitch_occupancy.api.search_control import router as search_router
from pitch_occupancy.config import settings

app = FastAPI(
    title="Pitch Occupancy",
    version=__version__,
    summary="Occupancy verdicts and booking reconciliation for multi-pitch facilities",
)
app.include_router(router)
app.include_router(search_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def thesis() -> HTMLResponse:
    """The thesis frontend - every finding, experiment, search and document.

    Read from disk on each request, so rerunning an experiment and reloading is enough;
    there is no build step to forget and no copy that can drift from the results.
    """
    return HTMLResponse(thesis_page())


@app.get("/client", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    """The operator dashboard - the client-facing view of live slot state.

    It calls the same public `/api/v1` endpoints an integrator would, so nothing on the
    page has privileged access: anything it can show, the API can serve.
    """
    return dashboard_response()


class Health(BaseModel):
    status: str
    version: str
    source_type: str


@app.get("/health", response_model=Health, tags=["meta"])
def health() -> Health:
    """Liveness probe, and a cheap way to confirm which frame source is configured."""
    return Health(
        status="ok",
        version=__version__,
        source_type=settings.source_type.value,
    )
