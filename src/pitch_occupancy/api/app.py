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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from pitch_occupancy import __version__
from pitch_occupancy.api.dashboard import dashboard_response
from pitch_occupancy.api.routes import router
from pitch_occupancy.api.schedule_editor import router as schedule_router
from pitch_occupancy.api.search_control import router as search_router
from pitch_occupancy.api.simulator import router as simulator_router
from pitch_occupancy.api.thesis_site import page as thesis_page
from pitch_occupancy.config import settings

app = FastAPI(
    title="Pitch Occupancy",
    version=__version__,
    summary="Occupancy verdicts and booking reconciliation for multi-pitch facilities",
)
app.include_router(router)
app.include_router(search_router)
# WP6-T1. Disabled unless PITCH_SIMULATOR_TOKEN is set - mounting it is not enabling it.
app.include_router(simulator_router)
# WP6-T6. The API's only write path, and the module docstring says why that is defensible
# here when bookings.py refuses to have one at all.
app.include_router(schedule_router)


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


#: Only these load from `results/figs`. An allowlist by extension rather than a static mount,
#: because `results/figs/venue_check/` holds cropped **pitch frames** - operator-supplied
#: footage that must never be published (see `thesis/ethics.md`). A blanket mount would put
#: them one guessed URL away from anyone who reaches the server.
FIGURE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml"}


@app.get("/figs/{name}", include_in_schema=False)
def figure(name: str) -> FileResponse:
    """Serve one generated figure from `results/figs`, by filename only.

    No subpaths: `name` must be a bare filename, so a traversal like `../../data/...`
    cannot escape the directory even before the resolved path is re-checked against it.
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(404)
    figs = (settings.results_dir / "figs").resolve()
    path = (figs / name).resolve()
    if path.parent != figs or path.suffix.lower() not in FIGURE_TYPES or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type=FIGURE_TYPES[path.suffix.lower()])


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
