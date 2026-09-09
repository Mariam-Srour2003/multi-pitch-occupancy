"""The simulator snapshot API (WP6-T1).

``GET /api/v1/cameras/{camera_id}/snapshot?mode=simulation`` returns one JPEG, the way a real
camera's HTTP snapshot endpoint would. It exists so the ingestion path can be exercised over a
network boundary without a camera: the pipeline that will one day poll a facility's cameras
polls this instead, and a bug that only appears over HTTP is a bug nobody finds until
deployment.

**It fails closed.** The bearer token comes from ``PITCH_SIMULATOR_TOKEN`` and is empty by
default, and empty means *disabled* rather than *open*. An unset secret that degrades to "no
authentication required" is the same defect as a threshold set where it can never fire, and
this project has found several of those; here the consequence would be serving frames of
identifiable people to anyone who asks. Unconfigured returns 503, wrong or missing token
returns 401, and both are tested.

**Only ``mode=simulation`` is implemented, and ``mode=live`` refuses rather than falling back.**
There are no cameras to proxy. A live mode that quietly served recorded footage would be the
worst possible failure of this endpoint - a deployment check that passes against a recording.

Metadata rides on ``X-`` response headers rather than being wrapped around the image, so the
body is a plain JPEG that any client, ``curl -o`` included, can save and open. ``format=json``
returns the same metadata as a JSON object with no image, for a caller that wants to know what
is available before fetching it.
"""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response

from pitch_occupancy.config import settings

__all__ = ["router", "SnapshotSource", "set_source", "RecordedSnapshots"]

router = APIRouter(prefix="/api/v1", tags=["simulator"])


def require_token(authorization: str | None = Header(default=None)) -> None:
    """Bearer auth, failing closed when unconfigured.

    Raises:
        HTTPException: 503 when no token is configured, 401 when one is configured and the
            request does not present it.
    """
    expected = settings.simulator_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(
                "the simulator API is disabled: no PITCH_SIMULATOR_TOKEN is configured. It "
                "serves frames of identifiable people, so an unset token disables it rather "
                "than opening it"
            ),
        )
    presented = ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    # Compared in constant time: the token is short and an endpoint that leaks it a character
    # at a time through response timing is not much better than one with no token.
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@dataclass(frozen=True, slots=True)
class Snapshot:
    camera_id: str
    minute_index: int
    jpeg: bytes
    source_path: str | None
    captured_at: str


class SnapshotSource:
    """What the endpoint serves. Replaced in tests; a real deployment has no live variant."""

    def cameras(self) -> list[str]:
        raise NotImplementedError

    def snapshot(self, camera_id: str, minute_index: int) -> Snapshot | None:
        raise NotImplementedError


class RecordedSnapshots(SnapshotSource):
    """Serves frames from a recorded slot, encoding each to JPEG on the way out.

    Encoding per request rather than caching: at one snapshot per camera per minute the cost
    is irrelevant, and a cache would be one more thing that can serve a stale frame - which is
    exactly the failure a simulator exists to avoid producing.
    """

    def __init__(self, source, *, quality: int = 90) -> None:
        self._source = source
        self._quality = quality

    def cameras(self) -> list[str]:
        return self._source.cameras()

    def snapshot(self, camera_id: str, minute_index: int) -> Snapshot | None:
        import cv2

        frame = self._source.read(camera_id, minute_index)
        if frame is None:
            return None
        ok, buffer = cv2.imencode(
            ".jpg", frame.image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
        )
        if not ok:  # pragma: no cover - imencode failing on a valid array
            return None
        return Snapshot(
            camera_id=camera_id,
            minute_index=minute_index,
            jpeg=buffer.tobytes(),
            source_path=frame.source_path,
            captured_at=datetime.now().isoformat(timespec="seconds"),
        )


_source: SnapshotSource | None = None


def set_source(source: SnapshotSource | None) -> None:
    """Install the snapshot backend. Called at startup, and by the tests."""
    global _source
    _source = source


def _current() -> SnapshotSource:
    if _source is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "no snapshot source is configured. Call simulator.set_source() with a "
                "RecordedSnapshots wrapping a VideoSlotSource"
            ),
        )
    return _source


@router.get("/cameras", dependencies=[Depends(require_token)])
def list_cameras() -> dict[str, list[str]]:
    """Which cameras this simulator can serve."""
    return {"cameras": _current().cameras()}


@router.get(
    "/cameras/{camera_id}/snapshot",
    dependencies=[Depends(require_token)],
    responses={200: {"content": {"image/jpeg": {}}}},
)
def snapshot(
    camera_id: str,
    mode: str = Query("simulation", pattern="^(simulation|live)$"),
    minute: int = Query(0, ge=0, description="minute index within the slot"),
    format: str = Query("jpeg", pattern="^(jpeg|json)$"),
) -> Response:
    """One frame, as a real camera's snapshot endpoint would serve it.

    ``mode=live`` is rejected rather than falling back to the recording: an endpoint that
    quietly served footage when asked for a camera would make a deployment check pass against
    a file.
    """
    if mode == "live":
        raise HTTPException(
            status_code=501,
            detail=(
                "live mode is not implemented and does not fall back to the recording. "
                "There are no cameras to proxy yet (WP7-T3); use mode=simulation"
            ),
        )
    source = _current()
    if camera_id not in source.cameras():
        raise HTTPException(
            status_code=404,
            detail=f"unknown camera {camera_id!r}; have {', '.join(source.cameras())}",
        )
    shot = source.snapshot(camera_id, minute)
    if shot is None:
        # 404 rather than a placeholder image: a gap must stay a gap all the way through, and
        # a substituted frame here would be a fabricated observation with a 200 on it.
        raise HTTPException(
            status_code=404,
            detail=f"no frame for {camera_id} at minute {minute}",
        )

    metadata = {
        "camera_id": shot.camera_id,
        "minute_index": str(shot.minute_index),
        "captured_at": shot.captured_at,
        "mode": "simulation",
        "source": shot.source_path or "",
    }
    if format == "json":
        return Response(
            content=json.dumps({**metadata, "bytes": len(shot.jpeg)}),
            media_type="application/json",
        )
    return Response(
        content=shot.jpeg,
        media_type="image/jpeg",
        headers={f"X-{k.replace('_', '-').title()}": v for k, v in metadata.items()},
    )


def recorded_source_for_today(day: Date | None = None):  # pragma: no cover - wiring helper
    """A `RecordedSnapshots` over the first recorded slot, for running the simulator locally."""
    from pitch_occupancy.frame_source import VideoSlotSource, discover_slots

    found = discover_slots(settings.raw_dir / "venue_01")
    if not found:
        return None
    key = sorted(found)[0]
    return RecordedSnapshots(VideoSlotSource(found[key]))
