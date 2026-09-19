"""The one seam every surface classifies through (A36, WP9-T1, WP9-T3).

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
a :class:`Pipeline`: the classifier, its gates or its rule, and the boundary lookup,
together. The worker, the scheduler, `/clip`, `/images`, `/roi` and every experiment get
theirs from here, and a test that proves `run_due` passes the pipeline through is a test that
the deployed path is the measured one.

**Two kinds behind one seam.** A key in `vision/backbones.BACKBONES` assembles the probe path
exactly as A14-A22 measured it: DINOv2 features pooled inside the camera's boundary, a
logistic probe, then the motion gate and the person gate, each allowed only to weaken. A key
in `vision/detector.DETECTORS` assembles the detector-first path (A36): a burst of frames per
camera, a count inside the boundary, the decision table, and a pitch-level fusion that sums
the cameras' counts. Switching the deployment is one setting; comparing the two is one
argument to `assemble`.

**The boundary is mandatory on the deployed path and reported on the interactive ones.**
`worker.run_slot` given a pipeline records a minute whose camera has no resolvable boundary
as UNCERTAIN and moves on; the slot's capture floor then does what it already did for a
camera that produced no frame. The review pages, which a person is looking at, score the
whole frame and say so in the trace - the honest failure there is a labelled one, not a
refusal. `require_boundary` is the switch, and `shared()` sets it the interactive way.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.slots.fusion import PitchVerdict, fuse, fuse_pitch
from pitch_occupancy.vision import roi
from pitch_occupancy.vision.pitch_classifier import CameraObservation, DetectorFirstClassifier
from pitch_occupancy.vision.rules import FrameVerdict, MinuteState, RuleConfig, from_class3

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
    """A classifier, its gates or rule, and its boundary lookup, assembled once and shared.

    ``classify`` satisfies `worker.Classifier` and additionally accepts ``polygon=``, as both
    `vision/classifier.ProbeClassifier` and `vision/pitch_classifier.DetectorFirstClassifier`
    do. ``boundaries`` is `roi.resolve_with_key` by default and is a field so a test can hand
    a pipeline a boundary without a store.
    """

    model_key: str
    #: ``"probe"`` for a frozen backbone with a linear head; ``"detector"`` for A36's path.
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
    #: The detector-first path's rule numbers; None on the probe path.
    rules: RuleConfig | None = None
    #: The `DetectorFirstClassifier` behind ``classify`` on the detector path, for `observe`.
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

    def gates(self) -> tuple[object | None, object | None]:
        return self.motion_gate, self.person_gate

    @property
    def burst(self) -> tuple[int, float]:
        """``(frames, spacing_s)`` per camera-minute: the rule's burst, or one frame."""
        if self.kind == "detector" and self.rules is not None:
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
        the same camera's previous frame, for the motion gate; without one the gate is silent,
        as it is on a camera's first minute.
        """
        polygon, key = self._resolve(polygon, camera_id, venue, slot_key)
        if self.kind == "detector":
            return self._detector_verdict(camera_id or "still", [frame_bgr], polygon, key,
                                          previous)
        return self._probe_verdict(frame_bgr, polygon, key, camera_id, previous)

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

        On the probe path the burst's last frame is classified as `classify_frame` does; the
        probe has no use for the rest. On the detector path every frame is counted.
        """
        frames = [f for f in frames if f is not None]
        if not frames:
            raise ValueError("an observation needs at least one frame")
        polygon, key = self._resolve(polygon, camera_id, venue, slot_key)
        if self.kind == "detector":
            return self._detector_observation(camera_id, frames, polygon, key, previous)
        verdict = self._probe_verdict(frames[-1], polygon, key, camera_id, previous)
        return CameraObservation(camera_id, verdict, frames=len(frames),
                                 evidence_index=len(frames) - 1)

    def fuse(self, observations: dict[str, CameraObservation]) -> PitchVerdict:
        """The pitch's minute from its cameras' observations.

        Detector path: `slots/fusion.fuse_pitch` - the counts are summed and the table is
        applied once. Probe path: `slots/fusion.fuse` - strongest activity wins - over the
        cameras that decided, wrapped in the same `PitchVerdict` so the worker has one loop.
        """
        if self.kind == "detector":
            assert self.rules is not None
            return fuse_pitch({cam: obs.verdict for cam, obs in observations.items()},
                              self.rules)
        per_camera = tuple((cam, obs.verdict.state, obs.verdict.confidence)
                           for cam, obs in sorted(observations.items()))
        decided = {cam: obs.verdict for cam, obs in observations.items()
                   if obs.verdict.decided}
        if not decided:
            return PitchVerdict(MinuteState.UNCERTAIN, 0.0,
                                ("no camera could be scored",), per_camera, disagreed=False)
        fused = fuse({cam: (v.class3, v.confidence) for cam, v in decided.items()})
        trace = tuple(f"{cam}: {v.state.name} {v.confidence:.2f}"
                      for cam, v in sorted(decided.items()))
        trace += (f"max-activity fusion -> {from_class3(fused.state).name}",)
        motions = [v.motion for v in decided.values() if v.motion is not None]
        return PitchVerdict(from_class3(fused.state), fused.confidence, trace, per_camera,
                            fused.disagreed, motion=max(motions) if motions else None,
                            n_scored=len(decided))

    def describe(self) -> str:
        """One line for a log: what is deployed, and what it rests on."""
        if self.kind == "detector" and self.rules is not None:
            cfg = self.rules
            frozen = (f"frozen {cfg.frozen_at} ({cfg.frozen_commit or '?'}) on {cfg.tuned_on}"
                      if cfg.frozen else "UNFROZEN - not for deployment")
            n, spacing = self.burst
            return (f"detector {self.model_key} @{cfg.imgsz} x{cfg.tiles}: rules {frozen}; "
                    f"burst {n} x {spacing:g}s; boundary "
                    f"{'required' if self.require_boundary else 'reported'}")
        gates = [name for name, gate in (("motion", self.motion_gate),
                                          ("person", self.person_gate)) if gate is not None]
        return (f"probe {self.model_key}: probe fitted on {self.n_train} development frames "
                f"(the locked venues are not among them); gates: {', '.join(gates) or 'none'}; "
                f"boundary {'required' if self.require_boundary else 'reported'}")

    # --- internals ------------------------------------------------------------------

    def _resolve(self, polygon, camera_id, venue, slot_key):
        if polygon is not None:
            return polygon, "given"
        if camera_id is not None:
            return self.boundary(camera_id, venue=venue, slot_key=slot_key)
        return None, None

    def _probe_verdict(self, frame_bgr, polygon, key, camera_id, previous) -> FrameVerdict:
        trace: list[str] = []
        if polygon is None:
            if self.require_boundary:
                return FrameVerdict(
                    MinuteState.UNCERTAIN, 0.0, model_key=self.model_key, rule=2,
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

    def _detector_observation(self, camera_id, frames, polygon, key, previous):
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

    def _detector_verdict(self, camera_id, frames, polygon, key, previous) -> FrameVerdict:
        return self._detector_observation(camera_id, frames, polygon, key, previous).verdict


def assemble(model_key: str | None = None, *, gates: bool = True,
             require_boundary: bool = True, rules: RuleConfig | None = None) -> Pipeline:
    """Build the deployed pipeline for ``model_key``, defaulting to the configured one.

    A backbone key assembles the probe path; a detector key assembles the detector-first
    path with the rule in `settings.rules_path` (or ``rules``, for an experiment that varies
    it). Raises on an unknown key rather than guessing: a deployment that fell back to some
    other model would report verdicts no table describes.
    """
    key = model_key or settings.default_model_key
    from pitch_occupancy.vision.backbones import BACKBONES
    from pitch_occupancy.vision.detector import DETECTORS, Detector

    if key in BACKBONES:
        from pitch_occupancy.vision.classifier import load_classifier

        classifier = load_classifier(key)
        motion_gate, person_gate = default_gates() if gates else (None, None)
        return Pipeline(
            model_key=key, kind="probe", classify=classifier,
            motion_gate=motion_gate, person_gate=person_gate,
            require_boundary=require_boundary, n_train=classifier.n_train,
        )
    if key in DETECTORS:
        cfg = rules or RuleConfig.load(settings.rules_path)
        if cfg.detector != key:
            cfg = replace(cfg, detector=key)
        detector_first = DetectorFirstClassifier(Detector.load(key), cfg)
        return Pipeline(
            model_key=key, kind="detector", classify=detector_first,
            require_boundary=require_boundary, rules=cfg, classifier=detector_first,
        )
    raise KeyError(
        f"unknown model key {key!r}. Known backbones: {sorted(BACKBONES)}; known detectors: "
        f"{sorted(DETECTORS)}."
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
