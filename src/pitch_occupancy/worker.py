"""The sampling scheduler (WP6-T2).

Runs as its own process, independent of the API. For each slot it pulls one frame per
camera per minute, classifies it, fuses the two halves, and at slot end aggregates a
verdict, selects evidence and reconciles against the booking record.

Keeping this out of the web layer is deliberate: a dashboard restart must never drop a
sample, and inference must never happen inside a request handler.

Run against recorded footage:

    uv run python -m pitch_occupancy.worker --source video

The classifier is injected rather than constructed here, so the pipeline can be exercised
with a stub in tests and with a real probe in production without the orchestration knowing
which it has.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from urllib.parse import urlparse

import numpy as np

from pitch_occupancy.config import CONFIGS_DIR, settings
from pitch_occupancy.data.taxonomy import Class3

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pitch_occupancy.pipeline import Pipeline
    from pitch_occupancy.vision.motion import MotionGate
    from pitch_occupancy.vision.people import PersonGate
from pitch_occupancy.db.store import Sample
from pitch_occupancy.frame_source import FrameSource
from pitch_occupancy.retention import has_room
from pitch_occupancy.slots.aggregate import SlotVerdict, Thresholds, aggregate_slot
from pitch_occupancy.slots.conditions import SlotConditions, summarise_conditions
from pitch_occupancy.slots.evidence import (
    EvidenceFrame,
    Transition,
    find_transitions,
    select_evidence_around_transitions,
)
from pitch_occupancy.slots.fusion import fuse
from pitch_occupancy.vision.rules import MinuteState

__all__ = ["Classifier", "SlotRun", "run_slot", "main"]


@runtime_checkable
class Classifier(Protocol):
    """Anything that turns one frame into a class and a confidence.

    Runtime-checkable so that a caller assembling the pipeline can assert it has one. The
    check is shallow - it sees a `__call__` and nothing about its signature - so a test
    that means to pin the contract should compare signatures, as `test_classifier.py` does.
    """

    def __call__(self, image_bgr: np.ndarray) -> tuple[Class3, float]: ...


@dataclass(frozen=True, slots=True)
class SlotRun:
    slot_id: str
    verdict: SlotVerdict
    samples: list[Sample]
    evidence: list[EvidenceFrame]
    minutes_captured: int
    minutes_missed: int
    conditions: SlotConditions | None = None
    transitions: tuple[Transition, ...] = ()
    #: What the gates saw, per minute they ran. These were accumulated inside the loop and
    #: then dropped on the floor, which made "the count is recorded" a claim about a local
    #: variable. They are cheap - one float and one int per minute - and they are the only
    #: trace of *why* a verdict was overruled, which is the first thing an operator disputing
    #: one will ask.
    motion_cues: tuple[float, ...] = ()
    people_counts: tuple[int, ...] = ()
    #: Minutes where a ball was seen inside the boundary. Evidence, not a rule: a ball is
    #: found in 40% of genuine play frames at unseen venues, so its absence means nothing.
    ball_minutes: tuple[int, ...] = ()
    #: Minutes in which no camera could be scored because none had a boundary (A36). They
    #: are also counted in `minutes_missed`, since that is what the capture floor reads;
    #: this says *why*, which "missed" alone does not.
    minutes_uncertain: int = 0
    #: The cameras that had no resolvable boundary. The first thing to fix at a new site.
    uncertain_cameras: tuple[str, ...] = ()

    @property
    def capture_rate(self) -> float:
        total = self.minutes_captured + self.minutes_missed
        return self.minutes_captured / total if total else 0.0


def _write_evidence(slot_dir: Path, minute: int, camera: str, image) -> Path | None:
    """Write one minute's evidence frame. Returns the path, or None if it could not be saved.

    A failed write must not take the slot down: the verdict is still valid without a picture,
    and losing an hour's classification because a disk was full would be a much worse outcome
    than losing the illustration of it.
    """
    import cv2

    path = slot_dir / f"minute_{minute:03d}_{camera}.jpg"
    try:
        ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    except Exception:  # noqa: BLE001 - any write failure degrades to "no picture"
        return None
    return path if ok else None


def run_slot(
    slot_id: str,
    source: FrameSource,
    classify: Classifier | None,
    *,
    thresholds: Thresholds | None = None,
    on_minute: Callable[[int, Class3], None] | None = None,
    evidence_dir: Path | None = None,
    polygon_for: Callable[[str], list[list[float]] | None] | None = None,
    motion_gate: MotionGate | None = None,
    person_gate: PersonGate | None = None,
    pipeline: Pipeline | None = None,
    venue: str | None = None,
    slot_key: str | None = None,
) -> SlotRun:
    """Sample, classify, fuse and aggregate one slot.

    ``pipeline`` is the assembled deployment (`pipeline.assemble`, A36): it supplies the
    classifier, both gates and the per-camera boundary lookup in one object, so a caller
    cannot pass one and forget the others - which is exactly what `scheduler.run_due` did
    for as long as the gates existed. ``venue`` and ``slot_key`` narrow the boundary lookup
    to the slot being run (`roi.resolve`). The older keyword arguments still work and still
    win when both are given, so a test can hand this a scripted gate.

    **With a pipeline, the boundary is mandatory.** A camera whose boundary cannot be
    resolved contributes no observation: its minute is recorded as UNCERTAIN, and if no
    camera on the pitch had one the minute counts as missed, which the capture floor turns
    into REVIEW. Without a boundary the model scores the neighbouring pitch and the car park
    as if they were this one, and A19 measured that at 0.74 false-play against 0.38 - a
    verdict produced that way is not one this system should record as its own.

    Minutes where no camera produced a frame are counted as missed rather than filled in.
    They lower the capture rate, which is what a verdict's trustworthiness should depend
    on - a slot evaluated from ten of sixty minutes is not the same claim as one evaluated
    from all sixty.

    ``evidence_dir`` turns on **saving the evidence images**, and until 2026-09-09 nothing
    did. `EvidenceFrame.image_path` was populated with a literal ``None`` on every minute, so
    the selection machinery ran, chose three good frames, and recorded three paths to nothing.
    The dashboard's evidence inspector said "no evidence images bound" and was right; the
    override harvest (WP6-T7) would have found every frame missing. A verdict an operator
    cannot see the evidence for is the one thing this system must not produce, since a human
    confirming every anomaly is what `slots/authority.py` rests on.

    The winning camera's frame is written for **every** observed minute and the unselected ones
    are deleted once the choice is made, because which three minutes matter is not knowable
    until the whole slot has been seen. Sixty small JPEGs written and fifty-seven removed is
    cheaper than holding sixty full-resolution frames in memory, which at 1080p is most of a
    gigabyte per slot.

    Left off by default: writing frames of identifiable people to disk is a decision a caller
    makes, not something that happens because a function was called.
    """
    require_boundary = False
    if pipeline is not None:
        classify = classify or pipeline.classify
        if motion_gate is None:
            motion_gate = pipeline.motion_gate
        if person_gate is None:
            person_gate = pipeline.person_gate
        if polygon_for is None:
            polygon_for = pipeline.polygon_for(venue=venue, slot_key=slot_key)
        require_boundary = pipeline.require_boundary
    if classify is None:
        raise TypeError("run_slot needs a classifier or a pipeline that carries one")

    cameras = source.cameras()
    samples: list[Sample] = []
    fused_states: list[Class3] = []
    fused_conf: list[float] = []
    evidence_rows: list[tuple[int, Class3, float, str | None]] = []
    disagreements: list[bool] = []
    written: list[Path] = []
    missed = 0

    slot_dir = None
    if evidence_dir is not None:
        # Runbook row 7. `retention.py` bounds what this system keeps and says nothing about
        # a disk filled by something else, and nothing checked before writing. A slot that
        # runs out of space mid-write leaves a verdict backed by a *partial* set of evidence
        # images, which is worse than one backed by none: the inspector cannot tell a slot
        # that was never configured to save frames from one whose frames stopped at minute
        # nineteen. So the choice is made once, before the first write, and the verdict is
        # produced either way - a full disk must not cost an hour of observation.
        room, why = has_room(Path(evidence_dir))
        if not room:
            evidence_dir = None
            if on_minute is None:  # the loop below has no other channel to report on
                print(f"  evidence images disabled for {slot_id}: {why}")
        else:
            slot_dir = Path(evidence_dir) / slot_id
            slot_dir.mkdir(parents=True, exist_ok=True)


    # One frame per camera, the previous minute's. Held only until the next frame replaces
    # it, so the memory cost is one frame per camera rather than one per minute.
    previous: dict[str, object] = {}
    motion_seen: list[float] = []
    people_seen: list[int] = []
    # Minutes where a ball was seen inside the boundary. Recorded, never consulted: across
    # nine unseen venues a ball is found in 40% of genuine play frames against the person
    # count's 100%, so its absence carries no information and a rule using it would be a rule
    # about venue_01. See `vision/people.py` (A17).
    ball_minutes: list[int] = []
    uncertain_minutes = 0
    uncertain_cameras: set[str] = set()

    for minute in range(source.n_minutes):
        observations: dict[str, tuple[Class3, float]] = {}
        images: dict[str, object] = {}
        abstained = False
        for camera in cameras:
            frame = source.read(camera, minute)
            if frame is None:
                continue  # a gap; never a fabricated observation
            # The pitch boundary (WP3-T1). Passed per camera because a boundary belongs to
            # one, and applying another camera's outline is measurably worse than none at
            # all: on an unseen clip the wrong outline bought 0.06 of false-play where the
            # right one bought 0.42.
            polygon = polygon_for(camera) if polygon_for is not None else None
            if polygon is None and require_boundary:
                # A36: no boundary, no verdict. Recorded as UNCERTAIN rather than dropped,
                # so the database shows a camera that was read and not scored, which is a
                # different fact from a camera that produced nothing.
                abstained = True
                uncertain_cameras.add(camera)
                samples.append(
                    Sample(
                        camera_id=camera,
                        minute_index=minute,
                        predicted=MinuteState.UNCERTAIN.value,
                        confidence=0.0,
                        captured_at=datetime.now(UTC).isoformat(timespec="seconds"),
                        image_path=frame.source_path,
                    )
                )
                continue
            if polygon_for is not None:
                state, confidence = classify(frame.image_bgr, polygon=polygon)
            else:
                state, confidence = classify(frame.image_bgr)

            # The motion gate (A14). This loop is the only place in the system that sees one
            # camera's frames in order, so it is the only place the rule can live.
            if motion_gate is not None:
                state, cue = motion_gate.apply(
                    state, previous.get(camera), frame.image_bgr, polygon,
                )
                if cue is not None:
                    motion_seen.append(cue)
            # The person gate (A16). After the motion gate, because it is the more expensive
            # of the two - it runs a detector - and the cheaper one may already have settled
            # the answer. Both only ever turn ACTIVE_PLAY into EMPTY, so the order changes
            # cost and not the verdict.
            if person_gate is not None:
                state, counted = person_gate.inspect(state, frame.image_bgr, polygon)
                if counted is not None:
                    people_seen.append(counted.people)
                    if counted.ball:
                        ball_minutes.append(minute)

            previous[camera] = frame.image_bgr

            observations[camera] = (state, confidence)
            if slot_dir is not None:
                images[camera] = frame.image_bgr
            samples.append(
                Sample(
                    camera_id=camera,
                    minute_index=minute,
                    predicted=state.value,
                    confidence=confidence,
                    captured_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    image_path=frame.source_path,
                )
            )

        if not observations:
            missed += 1
            if abstained:
                uncertain_minutes += 1
            continue

        fused = fuse(observations)
        fused_states.append(fused.state)
        fused_conf.append(fused.confidence)
        disagreements.append(fused.disagreed)

        # The frame from the camera whose observation won the fusion - the one that actually
        # justifies the minute's state. Saving an averaged or arbitrary camera would hand an
        # operator a picture of an empty half to explain a verdict of "play on the other one".
        path = None
        if slot_dir is not None:
            winner = max(
                (c for c, _, _ in fused.per_camera if c in images),
                key=lambda c: observations[c][1],
                default=None,
            )
            if winner is not None:
                path = _write_evidence(slot_dir, minute, winner, images[winner])
                if path is not None:
                    written.append(path)
        evidence_rows.append((minute, fused.state, fused.confidence,
                              str(path) if path else None))
        if on_minute is not None:
            on_minute(minute, fused.state)

    # The slot's real length, so a slot that lost most of its minutes is downgraded to
    # REVIEW rather than decided from the fragment that survived (WP7-T5).
    verdict = aggregate_slot(
        fused_states, fused_conf, thresholds, minutes_expected=source.n_minutes
    )
    evidence = select_evidence_around_transitions(
        evidence_rows, fused_states, verdict.status
    )

    # Which three minutes matter is only knowable once the whole slot has been seen, so every
    # observed minute was written and the rest are removed now. Deleting only files this call
    # created, by identity rather than by pattern: a glob would also sweep up a frame an
    # operator had already been shown and disputed.
    if slot_dir is not None:
        keep = {e.image_path for e in evidence if e.image_path}
        for path in written:
            if str(path) not in keep:
                path.unlink(missing_ok=True)

    return SlotRun(
        slot_id=slot_id,
        verdict=verdict,
        samples=samples,
        # a slot that changed state is better explained by the moment it changed than by
        # three frames from its thirds; thirds still apply when nothing changed
        evidence=evidence,
        minutes_captured=len(fused_states),
        minutes_missed=missed,
        conditions=summarise_conditions(
            minutes_expected=source.n_minutes,
            confidences=fused_conf,
            disagreements=disagreements,
            cameras_seen=len({s.camera_id for s in samples}),
        ),
        transitions=tuple(
            find_transitions(fused_states, minutes=[r[0] for r in evidence_rows])
        ),
        motion_cues=tuple(motion_seen),
        people_counts=tuple(people_seen),
        ball_minutes=tuple(ball_minutes),
        minutes_uncertain=uncertain_minutes,
        uncertain_cameras=tuple(sorted(uncertain_cameras)),
    )


CAMERA_CONFIG: Path = CONFIGS_DIR / "cameras.json"


def load_camera_urls(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Read `configs/cameras.json`: venue id -> camera id -> RTSP URL.

    Keys beginning with an underscore are dropped, so the example file's `_comment` block
    can explain the format in the file people actually open rather than in a docstring they
    would have to go looking for.

    Raises with the copy-this instruction rather than returning an empty mapping: a live run
    that finds no cameras and proceeds would report every slot as unobserved, and "no camera
    was configured" and "no camera responded" are different facts that must not arrive as the
    same verdict.
    """
    path = path or CAMERA_CONFIG
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Copy configs/cameras.example.json to it and fill in one "
            f"RTSP URL per camera. It is gitignored, because an RTSP URL usually carries the "
            f"credentials for a camera watching identifiable people."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    urls = {
        venue: {cam: url for cam, url in cameras.items() if not cam.startswith("_")}
        for venue, cameras in raw.items()
        if not venue.startswith("_") and isinstance(cameras, dict)
    }
    if not urls:
        raise SystemExit(f"{path} defines no cameras")
    return urls


def _confirm_live(urls: dict[str, dict[str, str]], schedule, evidence_dir: Path | None) -> None:
    """Make opening a real camera a deliberate act.

    Nothing in this project has ever connected to one. The live path is tested against an
    injected capture opener, which is the right way to test it and is not the same as having
    run it - so the first person to do so should see what it is about to open, and say yes.

    Credentials are not printed. A terminal scrollback is a place URLs leak from, and the
    host is enough to tell you whether you are pointed at the right camera.
    """
    print("\n=== live mode: this opens real camera streams ===")
    for venue, cameras in sorted(urls.items()):
        for cam, url in sorted(cameras.items()):
            host = urlparse(url).hostname or "?"
            print(f"  {venue}/{cam:<10} {urlparse(url).scheme}://{host}  (credentials hidden)")
    print(f"  {len(schedule.slots)} scheduled slot(s) from configs/slots_schedule.json")
    if evidence_dir is not None:
        print(f"  evidence images WILL be written to {evidence_dir} - these are frames of "
              f"identifiable people (thesis/ethics.md)")
    else:
        print("  evidence images: off (pass --evidence-dir to save them)")
    print("  the runbook's ladder has never met a real outage: docs/runbook.md")
    if input("\ntype 'yes' to connect: ").strip().lower() != "yes":
        raise SystemExit("not confirmed; nothing was opened")


def _run_live(args) -> None:
    """Follow the real schedule against real cameras (WP6-T3, WP7-T3).

    The path itself is not new — `scheduler.live_sources` and `frame_source.RTSPSource` have
    existed since 2026-09-09 and are tested against an injected capture opener. What was
    missing was any way to *invoke* it: `--source` accepted only `video`, and the camera URLs
    had nowhere to live, so pointing this at a camera meant writing Python.

    **It has still never been run against a camera.** Everything below is the first honest
    attempt, and the runbook's ladder is a prediction until it has met a real outage. The two
    things that make that survivable are already true: one dead camera is reported and skipped
    rather than taking the loop down, and a slot with too few minutes becomes REVIEW rather
    than a verdict.
    """
    from pitch_occupancy.db.schema import connect, initialise
    from pitch_occupancy.scheduler import describe, live_sources, load_schedule, run_forever

    urls = load_camera_urls(args.cameras)
    schedule = load_schedule()
    if not schedule.slots:
        raise SystemExit("configs/slots_schedule.json defines no slots")

    # Every configured venue must appear in the schedule and vice versa. A camera with no
    # slot is never read, and a slot with no camera raises inside `live_sources` an hour
    # later - both are better found now, before anything connects.
    scheduled = {s.venue_id for s in schedule.slots}
    for venue in sorted(set(urls) - scheduled):
        print(f"  note: {venue} has cameras configured and no scheduled slot")
    for venue in sorted(scheduled - set(urls)):
        raise SystemExit(
            f"slot venue {venue!r} has no cameras in the config; add it or remove the slot. "
            f"A slot with no reachable camera has no state, and defaulting to one invents it"
        )

    # And every camera a slot names, not just its venue. `live_sources` does check this, but
    # it checks at slot time: a config that names camera_A for a slot wanting A and B starts
    # cleanly, runs for an hour, and dies at the first slot boundary. Checking it here is the
    # difference between a typo caught in the first second and one caught after an evening.
    for slot in schedule.slots:
        missing = [c for c in slot.cameras if c not in urls.get(slot.venue_id, {})]
        if missing:
            raise SystemExit(
                f"slot {slot.venue_id} {slot.start} expects camera(s) {', '.join(missing)} "
                f"which have no URL in the camera config. A silently dropped camera is half a "
                f"pitch reported as the whole one, so this refuses rather than covering half"
            )

    if args.dry_run:
        print(f"{len(schedule.slots)} scheduled slot(s), cameras for {len(urls)} venue(s)")
        for line in describe(schedule, datetime.now(), days=1):
            print(f"  {line}")
        print("\ndry run: no stream was opened and nothing was written")
        return

    if not args.yes:
        _confirm_live(urls, schedule, args.evidence_dir)

    from pitch_occupancy.pipeline import assemble

    pipeline = assemble(args.model)
    print(f"\n{pipeline.describe()}")

    # Every camera's boundary, before a stream is opened. A camera without one is scored
    # UNCERTAIN every minute (A36), and finding that out at the first slot boundary is an
    # hour late. `--derive-roi` measures one from the stream now and stores it under the
    # production id; `/roi` is where a person confirms or redraws it.
    missing = _report_boundaries(
        pipeline,
        [(slot.venue_id, cam, f"{slot.venue_id}/{cam}", urls[slot.venue_id][cam], None)
         for slot in schedule.slots for cam in slot.cameras],
        derive=args.derive_roi,
    )
    if missing and not args.derive_roi:
        print(f"  {len(missing)} camera(s) have no boundary and will read UNCERTAIN; pass "
              f"--derive-roi to measure one from the stream, or draw one at /roi")

    connection = connect(settings.db_path)
    initialise(connection)
    try:
        ran = run_forever(
            schedule,
            source_for=live_sources(urls),
            pipeline=pipeline,
            connection=connection,
            model_key=args.model,
            evidence_dir=args.evidence_dir,
            iterations=args.iterations,
            on_slot=_report,
        )
    except KeyboardInterrupt:
        print("\ninterrupted; slots already finished are written")
        return
    finally:
        connection.close()
    print(f"\n{len(ran)} slot(s) written to the database")


def _report_boundaries(
    pipeline: Pipeline,
    cameras: list[tuple[str, str, str, object, str | None]],
    *,
    derive: bool,
) -> list[str]:
    """Print where each camera's boundary comes from; derive and store the missing ones.

    ``cameras`` is ``(venue, camera_id, store_key, footage, slot_key)`` per camera: the id
    the worker will read the camera under, the key a derived boundary is saved under, and
    the recording or stream a boundary can be measured from. Returns the ids still without
    one. Printed rather than logged because this runs once, at start, for a person.
    """
    from pitch_occupancy.vision import roi

    missing: list[str] = []
    for venue, camera, store_key, footage, slot_key in cameras:
        polygon, key = pipeline.boundary(camera, venue=venue, slot_key=slot_key)
        if polygon is not None:
            print(f"  boundary {venue}/{camera:<10} stored as {key!r}, keeps "
                  f"{roi.coverage(polygon):.0%}")
            continue
        if derive and footage is not None:
            polygon = roi.derive_from_video(footage)
            if polygon is not None:
                roi.save_derived(store_key, polygon)
                print(f"  boundary {venue}/{camera:<10} DERIVED now from the footage and "
                      f"stored as {store_key!r} ({len(polygon)} points, keeps "
                      f"{roi.coverage(polygon):.0%}) - confirm it at /roi")
                continue
            print(f"  boundary {venue}/{camera:<10} could not be derived: no turf found")
        else:
            print(f"  boundary {venue}/{camera:<10} MISSING - minutes will read UNCERTAIN")
        missing.append(f"{venue}/{camera}")
    return missing


def _report(slot_id: str, outcome: object) -> None:
    """Print one line per slot, whichever way it went.

    `run_due` hands this callback **the exception** when a slot fails, because one dead
    camera must not stop the other pitches being observed. A callback that assumed a
    `SlotRun` therefore crashed inside the handler for the failure it existed to report,
    which turns a skipped slot into a dead run. Found by running it: the schedule
    derived from recordings names four slot instances and only two were ever exported.
    """
    if isinstance(outcome, BaseException):
        print(f"  {slot_id:<34} SKIPPED  {type(outcome).__name__}: {outcome}")
        return
    uncertain = (f"  UNCERTAIN {outcome.minutes_uncertain} (no boundary: "
                 f"{', '.join(outcome.uncertain_cameras)})"
                 if getattr(outcome, "minutes_uncertain", 0) else "")
    print(f"  {slot_id:<34} {outcome.verdict.status:<8} "
          f"{outcome.minutes_captured:>3}/"
          f"{outcome.minutes_captured + outcome.minutes_missed} minutes  "
          f"capture {outcome.capture_rate:.0%}  "
          f"play {outcome.verdict.play_ratio:.2f} empty {outcome.verdict.empty_ratio:.2f}"
          f"{uncertain}")


def main() -> None:
    """Run the sampler over recorded slots, with the real classifier (WP6-T2).

    This is the acceptance criterion the work package was written around - *"runs
    continuously; DB fills; verdicts correct on recorded slots"* - and until
    `vision/classifier.py` existed it could not be met, because nothing in the repository
    could turn a frame into a class outside an experiment. This function raised
    `NotImplementedError` instead, and its message named the missing piece.

    ``--source video`` replays footage through the scheduler rather than pretending to be
    live: the same `run_due` a deployment calls, given recordings instead of cameras. The
    live path is `scheduler.live_sources` and it has still never been pointed at a camera
    (WP7-T3).

    ``--dry-run`` lists what would run and exits before the backbone is loaded, which is the
    shape `pitch retention` and `pitch schedule` already use: deciding is separable from
    doing, and a run that writes to the database should be something you asked for twice.
    """
    from pitch_occupancy.db.schema import connect, initialise
    from pitch_occupancy.scheduler import (
        describe,
        run_due,
        schedule_from_recordings,
        sources_from_recordings,
    )

    parser = argparse.ArgumentParser(description="Sample and evaluate slots.")
    parser.add_argument(
        "--source", choices=["video", "live"], default="video",
        help="video replays the recordings under --raw-dir; live opens the RTSP streams in "
             "configs/cameras.json and follows configs/slots_schedule.json in real time",
    )
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir / "venue_01")
    parser.add_argument("--model", default=settings.default_model_key)
    parser.add_argument("--dry-run", action="store_true",
                        help="list the slots and exit, loading no model and writing nothing")
    parser.add_argument("--evidence-dir", type=Path, default=None,
                        help="save evidence frames here; off by default, because these are "
                             "images of identifiable people")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many slots")
    parser.add_argument("--cameras", type=Path, default=None,
                        help="live only: path to the camera URL file (default "
                             "configs/cameras.json)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="live only: stop after this many scheduler ticks. The default "
                             "runs until interrupted")
    parser.add_argument("--yes", action="store_true",
                        help="live only: skip the confirmation prompt. For a service unit, "
                             "not for the first run")
    parser.add_argument("--derive-roi", action="store_true",
                        help="measure a boundary from the footage for any camera that has "
                             "none stored, and save it to configs/roi_derived.json under the "
                             "production id. Without it such cameras read UNCERTAIN (A36)")
    args = parser.parse_args()

    if args.source == "live":
        _run_live(args)
        return

    schedule, days = schedule_from_recordings(args.raw_dir)
    if not schedule.slots:
        raise SystemExit(f"no recorded slots under {args.raw_dir}")

    # Slots are per-day, so the total is the product. Printing the per-day count above a
    # list of every day's slots read as a miscount.
    print(f"{len(schedule.slots) * len(days)} slot(s): {len(schedule.slots)} per day "
          f"over {len(days)} day(s) under {args.raw_dir}")
    for day in days:
        for line in describe(schedule, datetime.combine(day, time.min), days=1):
            print(f"  {line}")
    if args.dry_run:
        print("\ndry run: nothing was classified and nothing was written")
        return

    from pitch_occupancy.frame_source import discover_slots
    from pitch_occupancy.pipeline import assemble
    from pitch_occupancy.scheduler import _recording_key
    from pitch_occupancy.vision import roi

    pipeline = assemble(args.model)
    print(f"\n{pipeline.describe()}")

    # The boundary for every recording's cameras, resolved the way `run_slot` will resolve
    # it - through the recording key, so `file0` finds `slot_..._camA` (`roi.FILE_KEY_CAMERA`).
    # A recording with no boundary reads UNCERTAIN; `--derive-roi` measures one from the
    # file first, which is what a first night at a new site would need.
    found = discover_slots(args.raw_dir)
    wanted = []
    for day in days:
        for slot in schedule.slots:
            key = _recording_key(slot, day)
            for cam in slot.cameras:
                store_key = f"{key}_{roi.FILE_KEY_CAMERA.get(cam, cam)}"
                wanted.append((slot.venue_id, cam, store_key, found.get(key, {}).get(cam), key))
    _report_boundaries(pipeline, wanted, derive=args.derive_roi)

    source_for = sources_from_recordings(args.raw_dir)

    # `run_due` persists only when it is given a connection - without one it classifies the
    # whole slot and stores nothing. The first version of this function did exactly that and
    # then printed "2 slot(s) written to the database", which is the shape of failure this
    # project keeps finding: not a crash, a confident sentence about something that did not
    # happen. `pitch info` names the file this opens.
    connection = connect(settings.db_path)
    initialise(connection)
    ran: list[str] = []
    seen: set[str] = set()
    try:
        for day in days:
            for slot in schedule.slots:
                if args.limit is not None and len(ran) >= args.limit:
                    break
                # `run_due` runs everything due at that moment, so two slots whose windows
                # overlap are both reached by the first call. Without this, the second call
                # would run the pair again and write one slot's verdict twice.
                if slot.slot_id(day) in seen:
                    continue
                # It selects by clock, so a slot is reached by standing inside its own
                # window rather than by calling `run_slot` directly - the point is to
                # exercise the scheduler's path, not to step around it.
                begins, _ = slot.window(day)
                just_ran = run_due(
                    schedule, begins,
                    source_for=source_for, pipeline=pipeline, model_key=args.model,
                    connection=connection, evidence_dir=args.evidence_dir, on_slot=_report,
                )
                seen.update(just_ran)
                ran += just_ran
    finally:
        connection.close()

    print(f"\n{len(ran)} slot(s) written to the database")


if __name__ == "__main__":
    main()
