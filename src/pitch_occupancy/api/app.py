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
from pitch_occupancy.api.routes import router
from pitch_occupancy.config import settings

app = FastAPI(
    title="Pitch Occupancy",
    version=__version__,
    summary="Occupancy verdicts and booking reconciliation for multi-pitch facilities",
)
app.include_router(router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    """The operator dashboard.

    It calls the same public `/api/v1` endpoints an integrator would - nothing on the page
    has privileged access, so anything it can show, the API can serve.
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
