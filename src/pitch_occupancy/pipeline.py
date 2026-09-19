"""The one seam every surface classifies through (A36, WP9-T1).

The defect this closes was found four times before it was named. A35 found the clip review
page answering ACTIVE_PLAY at confidence 1.000, 24 times out of 24, on an empty floodlit
pitch: the page had built its own inference from the probe alone, with no boundary and no
gates. One layer up, `scheduler.run_due` had been calling `worker.run_slot` without
`polygon_for`, `motion_gate` or `person_gate` since either existed, so
``worker --source video`` and the live path ran the bare probe too - the configuration
measured at 0.62-0.77 false-play, never the one measured at 0.012. And `roi.get` was being
handed camera ids the store had never heard of (``file0``, ``camera_A``), so even a caller
that asked for a boundary got None. Every surface assembled inference independently, and
they disagreed about the same footage.

So there is now one place that assembles it. :func:`assemble` takes a model key and returns
a :class:`Pipeline`: the classifier, its gates, and the boundary lookup, together. The
worker, the scheduler, `/clip`, `/images`, `/roi` and every experiment get theirs from here,
and a test that proves `run_due` passes the pipeline through is a test that the deployed
path is the measured one.

**What a pipeline is, in this phase.** The probe path exactly as A14-A22 measured it: DINOv2
features pooled inside the camera's boundary, a logistic probe, then the motion gate and the
person gate, each allowed only to weaken. `assemble` dispatches on the key - anything in
`vision/backbones.BACKBONES` is this path - and the detector-first path arrives behind the
same seam with WP9-T3, keyed from `vision/detector.DETECTORS`, so switching the deployment is
one setting and comparing the two is one argument.

**The boundary is mandatory on the deployed path and reported on the interactive ones.**
`worker.run_slot` given a pipeline records a minute whose camera has no resolvable boundary
as UNCERTAIN and moves on; the slot's capture floor then does what it already did for a
camera that produced no frame. The review pages, which a person is looking at, score the
whole frame and say so in the trace - the honest failure there is a labelled one, not a
refusal. `require_boundary` is the switch, and `shared()` sets it the interactive way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.vision import roi
from pitch_occupancy.vision.rules import FrameVerdict, MinuteState, from_class3

__all__ = ["Pipeline", "assemble", "default_gates", "shared", "reset_shared"]


def default_gates() -> tuple[object, object]:
    """The two gates the deployed probe path applies, constructed the way production does.

    A factory rather than module constants so a test about neighbour smoothing can swap them
    for ``(None, None)`` and see the smoothing, instead of watching the person gate turn every
    generated frame EMPTY - correctly, there is nobody in it.
    """
    from pitch_occupancy.vision.motion import MotionGate
    from pitch_occupancy.vision.people import PersonGate

    return MotionGate(), PersonGate()


@dataclass
class Pipeline:
    """A classifier, its gates and its boundary lookup, assembled once and shared.

    ``classify`` satisfies `worker.Classifier` and additionally accepts ``polygon=``, as
    `vision/classifier.ProbeClassifier` does. ``boundaries`` is `roi.resolve_with_key` by
    default and is a field so a test can hand a pipeline a boundary without a store.
    """

    model_key: str
    #: ``"probe"`` for a frozen backbone with a linear head; ``"detector"`` from WP9-T3.
    kind: str
    classify: Callable[..., tuple[object, float]]
    motion_gate: object | None = None
    person_gate: object | None = None
    #: Refuse to classify a camera with no boundary (the deployed path), or score the whole
    #: frame and say so in the trace (the interactive pages).
    require_boundary: bool = True
    boundaries: Callable[..., tuple[roi.Polygon | None, str | None]] = roi.resolve_with_key
    #: Development frames the probe was fitted on; 0 for a path that fits nothing.
    n_train: int = 0

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

    def gates(self) -> tuple[object | None, object | None]:
        return self.motion_gate, self.person_gate

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
        the same camera's previous frame, for the motion gate; without one the gate is silent,
        as it is on a camera's first minute.
        """
        trace: list[str] = []
        key: str | None
        if polygon is not None:
            key = "given"
        elif camera_id is not None:
            polygon, key = self.boundary(camera_id, venue=venue, slot_key=slot_key)
        else:
            key = None

        if polygon is None:
            if self.require_boundary:
                return FrameVerdict(
                    MinuteState.UNCERTAIN, 0.0, model_key=self.model_key,
                    trace=(f"no boundary for camera {camera_id!r}: the whole frame is not "
                           f"this pitch, and the deployed path does not score it as one",),
                )
            trace.append("no boundary: the whole frame was scored, neighbours included")
        else:
            trace.append(f"boundary {key} keeps {roi.coverage(polygon):.0%} of the frame")

        state, confidence = self.classify(frame_bgr, polygon=polygon)
        probed = state
        trace.append(f"{self.kind} {self.model_key}: {state.name} {confidence:.2f}")

        motion: float | None = None
        people: int | None = None
        ball: bool | None = None
        ball_confidence = 0.0

        if self.motion_gate is not None:
            before = state
            state, motion = self.motion_gate.apply(state, previous, frame_bgr, polygon)
            if motion is not None and state is not before:
                trace.append(f"motion gate: cue {motion:.3f} -> {state.name}")
        if self.person_gate is not None:
            before = state
            state, counted = self.person_gate.inspect(state, frame_bgr, polygon)
            if counted is not None:
                people, ball, ball_confidence = (
                    counted.people, counted.ball, counted.ball_confidence)
                trace.append(f"detector: {people} inside the boundary, "
                             f"ball {'seen' if ball else 'not seen'}")
            if state is not before:
                trace.append(f"person gate -> {state.name}")

        return FrameVerdict(
            from_class3(state), float(confidence), trace=tuple(trace), polygon=polygon,
            boundary_key=key, people=people, ball=ball, ball_confidence=ball_confidence,
            motion=motion, probed=probed, gated=state is not probed, model_key=self.model_key,
        )

    def describe(self) -> str:
        """One line for a log: what is deployed, and what it was fitted on."""
        gates = [name for name, gate in (("motion", self.motion_gate),
                                          ("person", self.person_gate)) if gate is not None]
        fitted = (f"probe fitted on {self.n_train} development frames (the locked venues are "
                  f"not among them)" if self.kind == "probe" else "no fitting")
        return (f"{self.kind} {self.model_key}: {fitted}; gates: "
                f"{', '.join(gates) or 'none'}; boundary "
                f"{'required' if self.require_boundary else 'reported'}")


def assemble(model_key: str | None = None, *, gates: bool = True,
             require_boundary: bool = True) -> Pipeline:
    """Build the deployed pipeline for ``model_key``, defaulting to the configured one.

    Raises on an unknown key rather than guessing: a deployment that fell back to some
    other model would report verdicts no table describes.
    """
    key = model_key or settings.default_model_key
    from pitch_occupancy.vision.backbones import BACKBONES

    if key in BACKBONES:
        from pitch_occupancy.vision.classifier import load_classifier

        classifier = load_classifier(key)
        motion_gate, person_gate = default_gates() if gates else (None, None)
        return Pipeline(
            model_key=key, kind="probe", classify=classifier,
            motion_gate=motion_gate, person_gate=person_gate,
            require_boundary=require_boundary, n_train=classifier.n_train,
        )
    raise KeyError(
        f"unknown model key {key!r}. Known backbones: {sorted(BACKBONES)}. Detector-first "
        f"keys arrive with WP9-T3 (vision/detector.DETECTORS)."
    )


_SHARED: Pipeline | None = None


def shared() -> Pipeline:
    """The process-wide pipeline the API surfaces share.

    Assembled on first use and reused - the first call pays for the probe fit, later ones
    pay nothing - and assembled *reporting* rather than requiring a boundary, because a page
    a person is reading should show the whole-frame answer labelled as such rather than
    refuse. The worker builds its own with `assemble` and the default.
    """
    global _SHARED
    if _SHARED is None:
        _SHARED = assemble(require_boundary=False)
    return _SHARED


def reset_shared() -> None:
    """Forget the shared pipeline, so a test or a config change can rebuild it."""
    global _SHARED
    _SHARED = None
