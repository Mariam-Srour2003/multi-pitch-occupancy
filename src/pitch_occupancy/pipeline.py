"""The one seam every surface classifies through.

Every surface that classifies footage - the worker, the scheduler and the review pages -
gets its inference from here, so they cannot disagree about the same footage by having
assembled it differently.

:func:`assemble` takes a detector key and
returns a :class:`Pipeline`: the detector-first classifier, its rule and the boundary lookup,
together. The worker, the scheduler, `/clip`, `/images` and `/roi` get theirs from here,
and a test that proves `run_due` passes the pipeline through is a test that the deployed
path is the measured one.

**One path.** A burst of frames per camera, the YOLO detector's count inside the boundary,
the motion cue, the decision table in `configs/rules.json` (`vision/rules.py`), and a
pitch-level fusion that sums the cameras' counts. A key in `vision/detector.DETECTORS` picks
the detector; an unknown key - an old database row or schedule may still name a model that
no longer exists - falls back to the default detector with a warning.

**The boundary is mandatory on the deployed path and reported on the interactive ones.**
`worker.run_slot` given a pipeline records a minute whose camera has no resolvable boundary
as UNCERTAIN and moves on; the slot's capture floor then does what it already did for a
camera that produced no frame. The review pages, which a person is looking at, score the
whole frame and say so in the trace - the honest failure there is a labelled one, not a
refusal. `require_boundary` is the switch, and `shared()` sets it the interactive way.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.slots.fusion import PitchVerdict, fuse_pitch
from pitch_occupancy.vision import roi
from pitch_occupancy.vision.pitch_classifier import CameraObservation, DetectorFirstClassifier
from pitch_occupancy.vision.rules import FrameVerdict, MinuteState, RuleConfig

__all__ = ["Pipeline", "assemble", "resolve_model_key", "shared", "reset_shared"]

LOG = logging.getLogger(__name__)


@dataclass
class Pipeline:
    """The detector-first classifier, its rule, and its boundary lookup, assembled once.

    ``classify`` satisfies `worker.Classifier` and additionally accepts ``polygon=``, as
    `vision/pitch_classifier.DetectorFirstClassifier` does. ``boundaries`` is
    `roi.resolve_with_key` by default and is a field so a test can hand a pipeline a boundary
    without a store.
    """

    model_key: str
    classify: Callable[..., tuple[object, float]]
    #: Refuse to classify a camera with no boundary (the deployed path), or score the whole
    #: frame and say so in the trace (the interactive pages).
    require_boundary: bool = True
    boundaries: Callable[..., tuple[roi.Polygon | None, str | None]] = roi.resolve_with_key
    #: The decision rule's numbers (`configs/rules.json`).
    rules: RuleConfig | None = None
    #: The `DetectorFirstClassifier` behind ``classify``, for `observe`.
    classifier: DetectorFirstClassifier | None = None

    # --- the boundary ---------------------------------------------------------------

    def boundary(self, camera_id: str, *, venue: str | None = None,
                 slot_key: str | None = None) -> tuple[roi.Polygon | None, str | None]:
        """The camera's boundary and the key it was found under, or ``(None, None)``."""
        return self.boundaries(camera_id, venue=venue, slot_key=slot_key)

    def polygon(self, camera_id: str, *, venue: str | None = None,
                slot_key: str | None = None) -> roi.Polygon | None:
        return self.boundary(camera_id, venue=venue, slot_key=slot_key)[0]

    def polygon_for(self, *, venue: str | None = None,
                    slot_key: str | None = None) -> Callable[[str], roi.Polygon | None]:
        """A per-camera lookup in the shape `worker.run_slot` takes, bound to one slot."""
        return lambda camera_id: self.polygon(camera_id, venue=venue, slot_key=slot_key)

    @property
    def burst(self) -> tuple[int, float]:
        """``(frames, spacing_s)`` per camera-minute: the rule's burst, or one frame."""
        if self.rules is not None:
            return max(1, self.rules.burst_frames), self.rules.burst_spacing_s
        return 1, 0.0

    # --- one frame ------------------------------------------------------------------

    def classify_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        camera_id: str | None = None,
        polygon: roi.Polygon | None = None,
        previous: np.ndarray | None = None,
        venue: str | None = None,
        slot_key: str | None = None,
    ) -> FrameVerdict:
        """Classify one frame the way the deployed path does, and record how.

        ``polygon`` wins over a lookup; otherwise the boundary is resolved from
        ``camera_id`` (with ``venue`` and ``slot_key`` narrowing the search). ``previous`` is
        the same camera's previous frame, for the motion cue; without one there is no cue,
        as it is on a camera's first minute.
        """
        polygon, key = self._resolve(polygon, camera_id, venue, slot_key)
        return self._observation(camera_id or "still", [frame_bgr], polygon, key,
                                 previous).verdict

    # --- one minute -----------------------------------------------------------------

    def observe_minute(
        self,
        camera_id: str,
        frames: Sequence[np.ndarray],
        previous: np.ndarray | None = None,
        *,
        polygon: roi.Polygon | None = None,
        venue: str | None = None,
        slot_key: str | None = None,
    ) -> CameraObservation:
        """One camera's burst for one minute, as an observation the pitch can be fused from.

        Every frame of the burst is counted.
        """
        frames = [f for f in frames if f is not None]
        if not frames:
            raise ValueError("an observation needs at least one frame")
        polygon, key = self._resolve(polygon, camera_id, venue, slot_key)
        return self._observation(camera_id, frames, polygon, key, previous)

    def fuse(self, observations: dict[str, CameraObservation]) -> PitchVerdict:
        """The pitch's minute from its cameras' observations: `slots/fusion.fuse_pitch` -
        the counts are summed and the table is applied once."""
        assert self.rules is not None
        return fuse_pitch({cam: obs.verdict for cam, obs in observations.items()},
                          self.rules)

    def describe(self) -> str:
        """One line for a log: what is deployed, and what it rests on."""
        boundary = "required" if self.require_boundary else "reported"
        if self.rules is None:
            return f"detector {self.model_key}: no rule loaded; boundary {boundary}"
        cfg = self.rules
        frozen = (f"frozen {cfg.frozen_at} ({cfg.frozen_commit or '?'}) on {cfg.tuned_on}"
                  if cfg.frozen else "UNFROZEN - not for deployment")
        n, spacing = self.burst
        return (f"detector {self.model_key} @{cfg.imgsz} x{cfg.tiles}: rules {frozen}; "
                f"burst {n} x {spacing:g}s; boundary {boundary}")

    # --- internals ------------------------------------------------------------------

    def _resolve(self, polygon, camera_id, venue, slot_key):
        if polygon is not None:
            return polygon, "given"
        if camera_id is not None:
            return self.boundary(camera_id, venue=venue, slot_key=slot_key)
        return None, None

    def _observation(self, camera_id, frames, polygon, key, previous):
        assert self.classifier is not None
        if polygon is None and self.require_boundary:
            verdict = FrameVerdict(
                MinuteState.UNCERTAIN, 0.0, model_key=self.model_key, rule=2,
                trace=(f"row 2: no boundary for camera {camera_id!r} - the whole frame is "
                       f"not this pitch",))
            return CameraObservation(camera_id, verdict, frames=len(frames))
        observation = self.classifier.observe(camera_id, list(frames), polygon,
                                              previous=previous,
                                              require_boundary=self.require_boundary)
        verdict = observation.verdict
        if key is not None:
            steps = (f"boundary {key} keeps {roi.coverage(polygon):.0%} of the frame",
                     *verdict.trace) if polygon is not None else verdict.trace
            verdict = replace(verdict, boundary_key=key, trace=steps)
        return replace(observation, verdict=verdict)


def resolve_model_key(model_key: str | None = None) -> str:
    """The detector key to assemble: ``model_key`` if it names a detector, else the default.

    The default is the detector `configs/rules.json` names, or
    `vision.detector.DEFAULT_DETECTOR` when that is not a known detector either. An unknown
    key is logged and replaced rather than raised on, because old database rows and saved
    schedules can still carry the name of a model this product no longer has.
    """
    from pitch_occupancy.vision.detector import DEFAULT_DETECTOR, DETECTORS

    if model_key and model_key in DETECTORS:
        return model_key
    try:
        default = RuleConfig.load(settings.rules_path).detector
    except Exception:  # noqa: BLE001 - an unreadable rule file is reported by `assemble`
        default = DEFAULT_DETECTOR
    if default not in DETECTORS:
        default = DEFAULT_DETECTOR
    if model_key and model_key != default:
        LOG.warning("unknown model key %r; using the default detector %r", model_key, default)
    return default


def assemble(model_key: str | None = None, *, require_boundary: bool = True,
             rules: RuleConfig | None = None) -> Pipeline:
    """Build the deployed pipeline for ``model_key``, defaulting to the configured detector.

    The rule comes from `settings.rules_path` (or ``rules``, for a caller that varies it).
    An unknown key falls back to the default detector with a warning (`resolve_model_key`).
    """
    from pitch_occupancy.vision.detector import Detector

    key = resolve_model_key(model_key or settings.default_model_key)
    cfg = rules or RuleConfig.load(settings.rules_path)
    if cfg.detector != key:
        cfg = replace(cfg, detector=key)
    detector_first = DetectorFirstClassifier(Detector.load(key), cfg)
    return Pipeline(
        model_key=key, classify=detector_first,
        require_boundary=require_boundary, rules=cfg, classifier=detector_first,
    )


_SHARED: Pipeline | None = None
_SHARED_LOCK = threading.Lock()


def shared() -> Pipeline:
    """The process-wide pipeline the API surfaces and the playground checks share.

    Assembled on first use and reused - the first call pays for loading the detector, later
    ones pay nothing - and assembled *reporting* rather than requiring a boundary, because a page
    a person is reading should show the whole-frame answer labelled as such rather than
    refuse. Locked: it is called from worker threads, and two first callers arriving together
    would otherwise each load the detector. The worker builds its own with `assemble`.
    """
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = assemble(require_boundary=False)
        return _SHARED


def reset_shared() -> None:
    """Forget the shared pipeline, so a test or a config change can rebuild it."""
    global _SHARED
    with _SHARED_LOCK:
        _SHARED = None
