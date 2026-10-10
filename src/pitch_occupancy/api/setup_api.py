"""The "Getting started" checklist over HTTP, for the dashboard's setup card.

Read-only: it reports what `setup_check` finds in this machine's configuration and changes
nothing. Like every other endpoint the dashboard uses, it is the same answer an integrator
would get - the page has no privileged access.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from pitch_occupancy import setup_check

__all__ = ["router"]

router = APIRouter(tags=["setup"])


@router.get("/api/v1/setup/checklist")
def setup_checklist() -> dict[str, Any]:
    """Each setup step's state (done, todo, optional, warning, unknown), a one-line detail
    and where to fix it, plus ``required_done``. Never fails on a broken config file: that
    step is reported as a warning carrying the error instead."""
    return setup_check.summary()
