"""Step through a clip one frame at a time, showing what the model looked at.

`clip_analysis` answers *what* the model decided and when. This answers *why*, frame by
frame, and yields each step as soon as it is computed so a caller can render it while the
next one is still being worked out. Watching it arrive at one frame a second is the point:
a timeline that appears all at once says "trust me", and this says "here is the frame, here
is where the evidence was, here is the score it added up to".

**One forward pass, not two.** The obvious implementation calls the classifier for the
prediction and `spatial_features` for the explanation, which runs the backbone twice and
doubles the cost of every step. The probe is a logistic regression on *mean-pooled* features,
so the pooled vector the classifier would have used is exactly ``features.mean(axis=0)`` -
the same numbers, one pass. Checked rather than assumed: `test_walkthrough.py` asserts the
derived prediction matches the deployed classifier, and it agrees to about 3e-9.

**The evidence map is exact, and says so in its own output.** `class_evidence_map` is not a
saliency heuristic - the score is the mean of the per-position contributions plus a constant,
so the map *sums to the score*. Every step carries ``score_from_map``, ``score_direct`` and
the error between them, because a decomposition claiming to be exact should be checkable by
the person reading it rather than trusted on the strength of a docstring.

**Redaction is off by default here, and that is deliberate rather than careless.** The rule
in `explain.py` - faces pixelated, not optional, not a flag - governs frames *written to
disk*, which in this project means into git history, permanently, under a data-release
decision that is still open. Nothing here is written anywhere: these frames are streamed to
an operator who is already authorised to watch this footage, which is the same category as
`api/routes.evidence_image` serving unredacted evidence to the same person. Turn it on with
``redact=True`` for a screen share, a published figure, or anything that leaves the room - it
costs about 640 ms a frame, which is why it is not simply always on.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = ["Step", "Shot", "walk_clip", "walk_images", "explain_frame", "EXPLAIN_ALL",
           "RedactionUnavailable"]

#: ``explain_n=EXPLAIN_ALL`` explains every sampled frame. Explaining costs roughly twice a
#: bare prediction, so a caller that wants the first few steps illustrated and the rest merely
#: classified passes a small number instead.
EXPLAIN_ALL: int = -1


class RedactionUnavailable(RuntimeError):
    """``redact=True`` was asked for and the detector could not run, so nothing was shown.

    Raised rather than degraded. A frame the caller asked to have redacted, served with the
    people in it untouched because the detector failed to load, is the one failure a
    redaction flag exists to rule out; the API's streams turn this into an ``error`` record,
    which is the honest answer to "show me this footage safely".
    """


class _Explained:
    """Derived quantities shared by anything carrying an evidence map.

    A mixin rather than a base dataclass so :class:`Step` and :class:`Shot` can each declare
    their own fields - a clip step is placed in *time*, an image shot is placed by *name* -
    while the two numbers that need explaining are defined once. `focus_ratio` in particular
    carries a rule about zero-area denominators that is worth exactly one copy.
    """

    __slots__ = ()

    @property
    def explained(self) -> bool:
        return self.evidence is not None

    @property
    def reconstruction_error(self) -> float | None:
        if self.score_from_map is None or self.score_direct is None:
            return None
        return abs(self.score_from_map - self.score_direct)

    @property
    def focus_ratio(self) -> float | None:
        """Positive evidence on people, over the area they occupy.

        The number that says whether the model is reading *players* rather than the scene
        around them: far above 1 means the evidence concentrates on people, near 1 means it
        is spread as though they were not there. None when nobody was detected, because a
        ratio over zero area is not a small number - it is not a number.
        """
        if not self.people_area:
            return None
        return (self.evidence_on_people or 0.0) / self.people_area


@dataclass(frozen=True, slots=True)
class Step(_Explained):
    """One sampled frame, and what the model made of it."""

    index: int
    t_s: float
    predicted: str
    confidence: float
    elapsed_ms: float
    #: What the probe said before a gate overruled it, or None when none did. ``predicted``
    #: is the verdict *after* the gates, because that is what the system answers - while the
    #: evidence map below still decomposes the probe's own score, which is the thing it is a
    #: decomposition of. A page showing one of those without the other is how the review page
    #: and the worker came to disagree, and showing the probe's verdict alone is how it
    #: happened again on this page.
    probed: str | None = None
    #: People inside the boundary and whether a ball was there, from the person gate's own
    #: detector pass. None means the gate did not run - a different fact from zero.
    n_inside: int | None = None
    ball: bool | None = None
    #: The motion cue against the previous sampled frame; None on the first, which has no
    #: predecessor and therefore no cue.
    motion: float | None = None
    #: The frame the reader is shown: masked to the boundary, outlined, and redacted if the
    #: caller asked. None once explaining stops. Not the decoded frame - see ``raw_bgr``.
    frame_bgr: np.ndarray | None = None
    #: The frame as decoded: unmasked, unoutlined, unredacted, and only on explained steps.
    #: What a second model must be run on - `overlay.detector_pane` on ``frame_bgr`` cannot
    #: find anyone outside a boundary that has been blacked out (it reported 0 outside on
    #: every frame), and its count moved with the Pixelate toggle (audit UP-14, RX-12).
    #: **Never serve it to a page as-is** when redaction was asked for.
    raw_bgr: np.ndarray | None = None
    #: The per-position contribution map, in score units, at the backbone's patch grid.
    evidence: np.ndarray | None = None
    grid: tuple[int, int] | None = None
    score_from_map: float | None = None
    score_direct: float | None = None
    n_people: int | None = None
    evidence_on_people: float | None = None
    people_area: float | None = None
    #: Share of positive evidence outside the pitch boundary. None when there is no boundary;
    #: zero by construction when there is one, because those positions are pooled out rather
    #: than filled in. It is reported instead of assumed - see `explain.evidence_outside`.
    evidence_outside: float | None = None
    #: The boundary this frame was analysed under, in frame fractions, for the caller that
    #: wants to draw it on something other than ``frame_bgr``.
    polygon: list[list[float]] | None = None


@dataclass(frozen=True, slots=True)
class Shot(_Explained):
    """One still image, and what the model made of it.

    The same fields as :class:`Step` minus ``t_s``, plus the file's name. That absence is
    the whole difference between the two pages: a clip's samples have neighbours, so
    `clip_analysis` can overrule an isolated misread with the ones either side of it. A set
    of stills has no such thing. Each prediction here stands on its own evidence, and the
    page says so rather than implying the same machinery is at work.
    """

    index: int
    name: str
    predicted: str
    confidence: float
    elapsed_ms: float
    #: As on :class:`Step`. A still has no predecessor, so the motion gate cannot run here
    #: and only the person gate speaks - stated rather than left to be inferred from a
    #: missing number.
    probed: str | None = None
    n_inside: int | None = None
    ball: bool | None = None
    width: int = 0
    height: int = 0
    frame_bgr: np.ndarray | None = None
    #: As on :class:`Step`: the image as read, before masking, outlining or redaction.
    raw_bgr: np.ndarray | None = None
    evidence: np.ndarray | None = None
    grid: tuple[int, int] | None = None
    score_from_map: float | None = None
    score_direct: float | None = None
    n_people: int | None = None
    evidence_on_people: float | None = None
    people_area: float | None = None
    #: Share of positive evidence outside the pitch boundary. None when there is no boundary;
    #: zero by construction when there is one, because those positions are pooled out rather
    #: than filled in. It is reported instead of assumed - see `explain.evidence_outside`.
    evidence_outside: float | None = None
    #: The boundary this frame was analysed under, in frame fractions, for the caller that
    #: wants to draw it on something other than ``frame_bgr``.
    polygon: list[list[float]] | None = None


def _gate(state, confidence, frame, polygon, *, previous=None,
          motion_gate=None, person_gate=None):
    """Apply the deployed gates to one frame's verdict. Returns what they made of it.

    The same two overrules `clip_analysis.analyse_clip` applies, in the same order, for the
    same reason: the probe alone is not the system, and a page showing the probe alone is
    showing something nobody deploys. **This page showed exactly that until 2026-09-19** -
    reported from use, a clip of an empty floodlit pitch stepped through as ACTIVE_PLAY at
    confidence 0.999 on almost every frame, while the Analyse tab on the same footage and the
    same boundary answered EMPTY and said *58 verdicts were weakened by the gates*. Two tabs,
    one clip, opposite answers. The analyse path was fixed first, and the walkthrough
    endpoints were where it surfaced next.

    Returns ``(state, probed, n_inside, ball, motion)`` where ``probed`` is the pre-gate
    verdict when a gate changed it and None when none did, so a caller can show the
    correction rather than silently replacing one answer with another.

    **``frame`` is the decoded frame, not the masked one, and ``polygon`` does the
    confining** - exactly what `clip_analysis.analyse_clip` and `worker.run_slot` pass. This
    page used to hand the gates the frame `roi.apply` had blacked out, so a player whose
    feet were inside the line and whose body crossed it was half erased before the detector
    looked: on ten real frames the two routes counted differently on seven (12 against 7),
    and 5 against 3 crosses `play_min` - the two review tabs disagreeing again (audit UP-13).
    """
    from pitch_occupancy.data.taxonomy import Class3

    before = Class3(state)
    state = before
    motion = None
    n_inside = ball = None
    if motion_gate is not None and previous is not None:
        state, motion = motion_gate.apply(state, previous, frame, polygon)
    if person_gate is not None:
        state, counted = person_gate.inspect(state, frame, polygon)
        if counted is not None:
            n_inside, ball = counted.people, counted.ball
    return state, (str(before) if state is not before else None), n_inside, ball, motion


def _classify(classifier, frame, polygon):
    """Classify one frame, confining the pooling to ``polygon`` if the classifier can.

    `worker.Classifier` is "callable with a frame, returns a class and a confidence", and
    several callers - every test stub, and anything wrapping a model this project does not
    own - satisfy exactly that and nothing more. Passing ``polygon=`` unconditionally would
    make the boundary a breaking change to a published seam, so it is passed only when there
    is one to pass, and a classifier that cannot take it still sees a frame that `roi.apply`
    has already masked. That degrades to the old behaviour rather than to an exception.
    """
    if polygon is None:
        return classifier(frame)
    return classifier(frame, polygon=polygon)


def explain_frame(frame, classifier, *, redact: bool = False,
                  polygon: list[list[float]] | None = None, raw=None) -> dict:
    """Predict one BGR frame and decompose the score over the backbone's patch grid.

    Returns the fields :class:`Step` and :class:`Shot` share, so the clip walkthrough and
    the image walkthrough cannot drift apart. That is not tidiness: this project's most
    expensive bug was a feature path that differed between two callers, and the subtlety
    here is exactly the kind that gets reimplemented slightly wrong - the pooled vector
    below **is** what `embed_batch` would have produced, so the prediction is the deployed
    one rather than a lookalike computed a second way.

    One forward pass, not two. The probe is a logistic regression on mean-pooled features,
    so the vector the classifier would have used is ``features.mean(axis=0)``.

    ``polygon`` does here what it does in `classifier.classify_batch`, and it has to be the
    same arithmetic or this function explains a different model from the deployed one - the
    weighted pool below is written against `backbones.embed_batch`, CLS weight included, and
    `test_walkthrough.py` asserts the two agree on a real frame rather than by inspection.

    With a boundary, the positions outside it carry weight zero through **both** the pooling
    and the decomposition, so they contribute nothing to the score and exactly zero to the
    map. ``evidence_outside`` reports that as a number the reader can watch.

    **People are found once, at 1280 px, on the unmasked frame.** ``raw`` is the frame
    before `roi.apply` (``frame`` itself when not given); the detector runs on it at
    `people.DETECT_IMGSZ` with the gate's own weights. Those boxes serve two purposes:

    * ``redact=True`` hands **all** of them to `explain.redact_frame` - pixelated people, then
      the floor of blur over the whole frame. This used to pixelate only the boxes a 640-px
      pass found and skip the floor, and on a real frame 5 of the 8 people the 1280-px pass
      finds were left untouched (audit RX-1). Redacting with every box, not only the ones
      inside the outline, is deliberate: a ``blur`` or ``mean`` fill leaves people outside
      partly visible, and the cost of over-redacting is nil.
    * ``n_people`` and ``evidence_on_people`` read only the people standing inside the
      boundary (by the foot, as `counting.inside` does), since the evidence outside it is
      zero by construction and their area would only dilute the ratio.

    **A detector that cannot run is "not checked", never a crash and never a leak.**
    Without ``redact`` the counts come back None (audit RX-9: ``len(None)`` used to kill
    the stream). With it, :class:`RedactionUnavailable` is raised rather than an
    unredacted frame returned.
    """
    import numpy as np

    from pitch_occupancy.vision import explain as X
    from pitch_occupancy.vision import roi
    from pitch_occupancy.vision.counting import inside, polygon_mask
    from pitch_occupancy.vision.people import DETECT_IMGSZ, gate_weights

    # The probe the deployed model uses for this polygon (`ProbeClassifier.probe_for`):
    # with a boundary that is the boundary-pooled one, and explaining the other would
    # explain a model that is not serving.
    probe = (classifier.probe_for(polygon) if hasattr(classifier, "probe_for")
             else classifier.probe)
    classes = list(probe.classes_)
    feats, grid = X.spatial_features(
        classifier.model, classifier.processor, classifier.spec, frame
    )
    drop_first = classifier.spec.kind != "convnet"
    cells = roi.grid_weights(polygon, (frame.shape[1], frame.shape[0]), grid,
                             classifier.processor)

    if cells is None:
        pooled = feats.mean(axis=0, keepdims=True)
    else:
        flat = cells.reshape(-1).astype(np.float64)
        if drop_first:
            flat = np.concatenate([[1.0], flat])
        pooled = (feats * flat[:, None]).sum(axis=0, keepdims=True) / flat.sum()
    proba = probe.predict_proba(pooled)[0]
    predicted = classes[int(proba.argmax())]

    weights, constant = X.probe_weights(probe, predicted)
    evidence, score_from_map = X.class_evidence_map(
        feats, weights, constant, grid, drop_first=drop_first, cell_weights=cells
    )
    direct = probe._model.decision_function(pooled)[0]  # noqa: SLF001
    score_direct = float(direct[classes.index(predicted)] if np.ndim(direct) else direct)

    boxes = X.detect_people(frame if raw is None else raw, imgsz=DETECT_IMGSZ,
                            model_name=gate_weights())
    if boxes is None and redact:
        raise RedactionUnavailable(
            "redaction was requested but the person detector is unavailable; refusing to "
            "show the frame unredacted")
    on_people = area = n_people = None
    if boxes is not None:
        mask = polygon_mask(polygon, frame.shape[:2])
        standing = [b for b in boxes if inside(mask, ((b[0] + b[2]) // 2, b[3]))]
        n_people = len(standing)
        if standing:
            on_people, area = X.evidence_on_people(evidence, standing, frame.shape[:2])

    shown = X.redact_frame(frame, boxes) if redact else frame
    return {
        "predicted": str(predicted),
        "confidence": float(proba.max()),
        # The outline is drawn on the frame the reader sees, not on the frame the model saw:
        # a yellow line is pixels, and putting it in before the backbone would make the
        # boundary itself part of the evidence it is supposed to delimit.
        "frame_bgr": roi.outline(shown, polygon),
        "evidence": evidence,
        "grid": grid,
        "score_from_map": score_from_map,
        "score_direct": score_direct,
        "n_people": n_people,
        "evidence_on_people": on_people,
        "people_area": area,
        "evidence_outside": X.evidence_outside(evidence, cells),
        "polygon": polygon,
    }


def walk_clip(
    path: Path | str,
    classifier,
    *,
    interval_s: float = 10.0,
    explain_n: int = EXPLAIN_ALL,
    redact: bool = False,
    max_samples: int = 360,
    polygon: list[list[float]] | None = None,
    roi_fill: str = "black",
    motion_gate: object | None = None,
    person_gate: object | None = None,
) -> Iterator[Step]:
    """Yield one :class:`Step` per sampled frame, as soon as each is computed.

    ``classifier`` is a `vision.classifier.ProbeClassifier` - this needs its backbone and
    probe, not merely something callable, because the explanation decomposes the probe's own
    weights. ``explain_n`` bounds how many frames get the full treatment; the rest are
    predicted and yielded without an evidence map, which is about twice as fast.

    A frame that cannot be decoded is skipped rather than yielded as a guess, exactly as
    `clip_analysis.analyse_clip` records a gap instead of inventing one.

    ``polygon`` applies a pitch boundary to **every sampled frame**, which is the
    whole reason a boundary is drawn on a still and then used on a clip: a fixed camera's
    pitch does not move between frame 0 and frame 3,000, so one outline drawn once is
    correct for the entire recording. It is applied to the full frame before the backbone's
    own resize, because the outline is expressed in fractions of the original frame.

    It does two things, and it needed both. `roi.apply` fills the outside so the neighbouring
    pitch is not in the pixels; passing the polygon on to the classifier and to
    :func:`explain_frame` drops those positions from the pooling so the *fill* is not scored
    either. With only the first, a boundary replaced one distraction with another - a large
    uniform region the probe still averaged in and the evidence map still coloured.

    ``motion_gate`` and ``person_gate`` are the deployed overrules, and passing neither
    reproduces this function's behaviour before 2026-09-19 - which was the probe alone, and
    was a bug on every page that called it (see :func:`_gate`). They cannot be applied by
    wrapping ``classifier``: the motion gate compares a frame to the *previous sample*, and
    this loop is the only thing that sees the samples in order. The same reasoning, and the
    same parameters, as `clip_analysis.analyse_clip`.

    **The step's ``predicted`` is the gated verdict and its evidence map is the probe's.**
    Those are two different objects and both belong on the page: the map is an exact
    decomposition of the probe's score, so it explains what the probe saw - including on a
    frame where a gate then overruled it, which is precisely the frame worth looking at.
    ``probed`` carries the pre-gate verdict so the page can show the correction.
    """
    import time

    import cv2

    from pitch_occupancy.vision.roi import apply as apply_roi

    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open {source.name} as video")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration_s = frames / fps if fps > 0 and frames > 0 else 0.0
        if duration_s > 0 and duration_s / interval_s > max_samples:
            interval_s = duration_s / max_samples

        index, t = 0, 0.0
        previous = None
        while duration_s <= 0 or t < duration_s:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                if duration_s <= 0:
                    break
                t += interval_s
                continue

            began = time.perf_counter()
            # The decoded frame goes to the gates with the polygon, the masked one to the
            # probe - the split `clip_analysis` and the worker make (see `_gate`).
            raw = frame
            masked = apply_roi(raw, polygon, fill=roi_fill)
            explaining = explain_n == EXPLAIN_ALL or index < explain_n

            if not explaining:
                state, confidence = _classify(classifier, masked, polygon)
                state, probed, n_inside, ball, motion = _gate(
                    state, confidence, raw, polygon, previous=previous,
                    motion_gate=motion_gate, person_gate=person_gate)
                yield Step(index=index, t_s=t, predicted=str(state), confidence=confidence,
                           elapsed_ms=(time.perf_counter() - began) * 1000,
                           probed=probed, n_inside=n_inside, ball=ball, motion=motion,
                           polygon=polygon)
            else:
                explained = explain_frame(masked, classifier, redact=redact,
                                          polygon=polygon, raw=raw)
                state, probed, n_inside, ball, motion = _gate(
                    explained["predicted"], explained["confidence"], raw, polygon,
                    previous=previous, motion_gate=motion_gate, person_gate=person_gate)
                explained["predicted"] = str(state)
                yield Step(
                    index=index, t_s=t,
                    elapsed_ms=(time.perf_counter() - began) * 1000,
                    probed=probed, n_inside=n_inside, ball=ball, motion=motion,
                    raw_bgr=raw, **explained,
                )
            previous = raw
            index += 1
            t += interval_s
            # On both branches. The bound used to sit after the explained branch only, and
            # a stream of unknown length (raw MJPEG, a CCTV export) with ``explain_n=0``
            # never reached it: 2000 steps from a 1.2 MB upload, and a clock reading 19990
            # minutes (audit UP-10). Known-length clips are already bounded by widening
            # the interval above; this is the cap for the ones that are not.
            if index >= max_samples:
                break
    finally:
        capture.release()


def walk_images(
    paths: Sequence[Path | str],
    classifier,
    *,
    explain_n: int = EXPLAIN_ALL,
    redact: bool = False,
    max_images: int = 64,
    polygon: list[list[float]] | None = None,
    polygons: Sequence[list[list[float]] | None] | None = None,
    wholes: Sequence[bool] | None = None,
    roi_fill: str = "black",
    person_gate: object | None = None,
) -> Iterator[Shot]:
    """Yield one :class:`Shot` per readable image, as soon as each is computed.

    The still-image counterpart to :func:`walk_clip`, and deliberately the same shape: the
    caller streams these to a page that renders shot *k* while the backbone is on *k+1*, so
    what a reader watches is the model working rather than an animation over a finished
    answer.

    **There is no smoothing here, and that is not an omission.** A clip's samples have
    neighbours, so `clip_analysis` can overrule an isolated misread using the samples either
    side of it. A folder of stills has no neighbours - they may be minutes or venues apart,
    and nothing in the upload says which. Every prediction here therefore stands on its own
    evidence, which makes the evidence map the only thing a reader can judge it by.

    An unreadable file yields nothing rather than a guess, exactly as a frame that will not
    decode is skipped in :func:`walk_clip`. The caller is told how many were skipped so the
    count on the page and the number of files chosen can be reconciled.

    ``polygon`` applies a pitch boundary before anything else, so what the model
    sees is only what the camera is responsible for. It is applied to the *full* frame, ahead
    of the backbone's own resize, because the outline is expressed in fractions of the
    original frame - masking after a crop would mask the wrong region. ``width`` and
    ``height`` still report the image as it arrived, since that is what the operator chose.

    As in :func:`walk_clip`, the polygon is both *filled* and *pooled out*: the fill removes
    the neighbouring pitch from the pixels, and the pooling removes the filled positions from
    the average, so what is outside the outline stops reaching the probe rather than reaching
    it as a black rectangle.

    ``person_gate`` is the one deployed overrule a still can carry. **The motion gate cannot
    run here at all**, and that is the honest limit of this page rather than an oversight:
    it compares a frame to the previous *sample of the same camera*, and a folder of stills
    has no such thing - the same absence that rules out neighbour smoothing. On the footage
    this project has, the motion gate is what catches most empty frames (on one clip it
    settled 56 of 59 before the person gate was consulted), so a still is judged with the
    weaker half of the deployed path and the page has to say so.
    """
    import time

    import cv2

    from pitch_occupancy.vision.roi import apply as apply_roi

    # One outline per image when `polygons` is given, falling back to the batch's `polygon`
    # for any entry that is None. A folder of stills is routinely a folder of *different
    # cameras* - the page has no way to know otherwise - and one outline across all of them
    # is right only when they happen to share a view. Indexed rather than zipped, so a short
    # list is a partial answer rather than a silently truncated batch.
    # `wholes[i]` is the operator having pressed *Use the whole frame* for this image, and
    # it beats every outline including the batch's. It exists because "no outline" cannot
    # carry that meaning: an image nobody has drawn on yet looks identical, and the caller
    # answers *that* case by supplying one. Declining a boundary and never being offered one
    # are different, and only one of them should end in a boundary.
    def outline_for(i: int):
        if wholes is not None and i < len(wholes) and wholes[i]:
            return None
        if polygons is not None and i < len(polygons) and polygons[i] is not None:
            return polygons[i]
        return polygon

    for index, raw in enumerate(list(paths)[:max_images]):
        shot_polygon = outline_for(index)
        source = Path(raw)
        began = time.perf_counter()
        frame = cv2.imread(str(source))
        if frame is None:
            continue

        height, width = frame.shape[:2]
        # Gate on the image as read and let the polygon confine it, as in `walk_clip`.
        raw = frame
        frame = apply_roi(raw, shot_polygon, fill=roi_fill)
        explaining = explain_n == EXPLAIN_ALL or index < explain_n
        if not explaining:
            state, confidence = _classify(classifier, frame, shot_polygon)
            state, probed, n_inside, ball, _ = _gate(
                state, confidence, raw, shot_polygon, person_gate=person_gate)
            yield Shot(
                index=index, name=source.name, predicted=str(state),
                confidence=confidence, width=width, height=height,
                probed=probed, n_inside=n_inside, ball=ball,
                elapsed_ms=(time.perf_counter() - began) * 1000, polygon=shot_polygon,
            )
            continue

        explained = explain_frame(frame, classifier, redact=redact, polygon=shot_polygon,
                                  raw=raw)
        state, probed, n_inside, ball, _ = _gate(
            explained["predicted"], explained["confidence"], raw, shot_polygon,
            person_gate=person_gate)
        explained["predicted"] = str(state)
        yield Shot(
            index=index, name=source.name, width=width, height=height,
            elapsed_ms=(time.perf_counter() - began) * 1000,
            probed=probed, n_inside=n_inside, ball=ball, raw_bgr=raw,
            **explained,
        )
