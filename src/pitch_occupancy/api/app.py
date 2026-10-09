"""FastAPI application.

This is the *only* part of the system uvicorn serves. The occupancy pipeline itself is a
scheduled batch process (``pitch_occupancy.worker``) that runs independently, so a
dashboard restart never interrupts sampling.

Run it with::

    uv run python -m uvicorn pitch_occupancy.api.app:app --reload

Pages:
  - ``/``            the operator dashboard (also at ``/client``)
  - ``/clip``        one video in, a timeline of what the pitch was doing
  - ``/clip/pair``   both cameras of one pitch, judged together
  - ``/images``      one still or a batch, each explained
  - ``/roi``         draw the pitch boundary for a camera
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pitch_occupancy import __version__
from pitch_occupancy.api.clip_page import CLIP_HTML
from pitch_occupancy.api.clip_pair import router as pair_router
from pitch_occupancy.api.clip_pair_page import PAIR_HTML
from pitch_occupancy.api.clip_review import router as clip_router
from pitch_occupancy.api.clip_walkthrough import router as walkthrough_router
from pitch_occupancy.api.dashboard import dashboard_response
from pitch_occupancy.api.image_page import IMAGE_HTML
from pitch_occupancy.api.image_walkthrough import router as image_router
from pitch_occupancy.api.roi_editor import router as roi_router
from pitch_occupancy.api.roi_page import ROI_HTML
from pitch_occupancy.api.routes import router
from pitch_occupancy.api.schedule_editor import router as schedule_router
from pitch_occupancy.api.simulator import router as simulator_router
from pitch_occupancy.config import settings

app = FastAPI(
    title="Pitch Occupancy",
    version=__version__,
    summary="Occupancy verdicts and booking reconciliation for multi-pitch facilities",
)
app.include_router(router)
# Camera simulator. Disabled unless PITCH_SIMULATOR_TOKEN is set - mounting it is not
# enabling it.
app.include_router(simulator_router)
# The API's only write path, and the module docstring says why that is defensible here when
# bookings.py refuses to have one at all.
app.include_router(schedule_router)
# Analyses an uploaded clip and keeps none of it - see the module docstring for why a POST
# that stores nothing sits beside /schedule/validate rather than the override.
app.include_router(clip_router)
# Two videos of the same pitch, one timeline: the cameras' counts added and the ball shared
# before the rule is applied, as `slots/fusion.fuse_pitch` does for the worker.
app.include_router(pair_router)
# The same clip analysis streamed step by step, with the evidence map `vision/explain.py`
# produces.
app.include_router(walkthrough_router)
# The same explanation over still images. A separate surface rather than a clip of length
# one: stills have no neighbours, so none of the temporal correction a clip gets applies to
# them, and the page is built to say so rather than to imply otherwise.
app.include_router(image_router)
# Pitch boundaries. The preview runs the model twice so a boundary can be checked rather
# than trusted.
app.include_router(roi_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/client", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    """The operator dashboard - the client-facing view of live slot state.

    It calls the same public `/api/v1` endpoints an integrator would, so nothing on the
    page has privileged access: anything it can show, the API can serve.
    """
    return dashboard_response()


@app.get("/clip", response_class=HTMLResponse, include_in_schema=False)
def clip_reviewer() -> HTMLResponse:
    """The clip reviewer - hand it a video, get a timeline of what the pitch was doing.

    Separate from the dashboard because there is no slot, no booking and no database row
    behind it: the input is a file somebody has in their hand, and nothing is kept.
    """
    return HTMLResponse(CLIP_HTML)


@app.get("/clip/pair", response_class=HTMLResponse, include_in_schema=False)
def pair_reviewer() -> HTMLResponse:
    """The two-camera reviewer - both cameras of one pitch, judged together.

    One camera sees half a pitch, so its own verdict is not the pitch's. This page takes
    both videos, adds the people the two see, takes the ball from either, and applies the
    decision table once - and shows each camera's own answer beside the pitch's.
    """
    return HTMLResponse(PAIR_HTML)


@app.get("/images", response_class=HTMLResponse, include_in_schema=False)
def image_reviewer() -> HTMLResponse:
    """The image reviewer - hand it one picture or a batch, see where the score came from.

    The clip reviewer's counterpart for footage nobody has, which is the common case when a
    question arrives as a screenshot or an exported still. It explains each image the same
    way and then sorts the batch by confidence, least confident first: on a set of stills
    the useful question is which ones to look at, not what the majority verdict was.
    """
    return HTMLResponse(IMAGE_HTML)


@app.get("/roi", response_class=HTMLResponse, include_in_schema=False)
def boundary_editor() -> HTMLResponse:
    """The pitch boundary editor - draw what counts as this pitch, and ignore the rest.

    Pitches are built in rows, so a camera watching one sees its neighbours down the sides
    of the frame, and a match on the next pitch puts real players into frames where this
    pitch is empty. No amount of training fixes that: the evidence genuinely is there and
    the question was under-specified. A boundary specifies it.
    """
    return HTMLResponse(ROI_HTML)


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
