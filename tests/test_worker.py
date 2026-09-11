"""The scheduler's slot loop.

Orchestration bugs are the quiet kind: a dropped camera or a misaligned minute still
produces a plausible verdict. These tests pin the behaviour that keeps a verdict honest
about how much evidence it actually rests on."""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.frame_source import Frame, FrameSource
from pitch_occupancy.worker import run_slot

PLAY, EMPTY = Class3.ACTIVE_PLAY, Class3.EMPTY


class FakeSource(FrameSource):
    """Serves scripted frames; ``None`` entries simulate a camera outage."""

    def __init__(self, script: dict[str, list[object]]) -> None:
        self._script = script

    def cameras(self) -> list[str]:
        return sorted(self._script)

    @property
    def n_minutes(self) -> int:
        return max(len(v) for v in self._script.values())

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        seq = self._script[camera_id]
        if minute_index >= len(seq) or seq[minute_index] is None:
            return None
        return Frame(camera_id, minute_index, np.zeros((4, 4, 3), np.uint8), "x.mp4")


def classifier_from(plan: dict[str, list[Class3]]):
    """Returns a classifier that reads its answer from the call order per camera."""
    state = {k: iter(v) for k, v in plan.items()}
    order: list[str] = []

    def classify(image: np.ndarray) -> tuple[Class3, float]:
        cam = order.pop(0)
        return next(state[cam]), 0.9

    return classify, order


def constant(cls: Class3, conf: float = 0.9):
    return lambda image: (cls, conf)


# --- happy path -------------------------------------------------------------


def test_a_played_slot_is_used() -> None:
    src = FakeSource({"camA": [1] * 60, "camB": [1] * 60})
    run = run_slot("s1", src, constant(PLAY))
    assert run.verdict.status is SlotStatus.USED
    assert run.minutes_captured == 60
    assert run.minutes_missed == 0


def test_an_empty_slot_is_notused() -> None:
    src = FakeSource({"camA": [1] * 60, "camB": [1] * 60})
    assert run_slot("s1", src, constant(EMPTY)).verdict.status is SlotStatus.NOTUSED


def test_one_sample_recorded_per_camera_per_minute() -> None:
    src = FakeSource({"camA": [1] * 10, "camB": [1] * 10})
    run = run_slot("s1", src, constant(PLAY))
    assert len(run.samples) == 20
    assert {(s.camera_id, s.minute_index) for s in run.samples} == {
        (c, m) for c in ("camA", "camB") for m in range(10)
    }


def test_evidence_is_selected_for_the_verdict() -> None:
    src = FakeSource({"camA": [1] * 60})
    run = run_slot("s1", src, constant(PLAY))
    assert len(run.evidence) == 3
    assert all(e.predicted is PLAY for e in run.evidence)


# --- outages ----------------------------------------------------------------


def test_a_minute_with_no_camera_is_missed_not_invented() -> None:
    """A gap must weaken the verdict, never become a fabricated EMPTY."""
    script = [1] * 60
    for m in range(10, 20):
        script[m] = None
    src = FakeSource({"camA": list(script)})
    run = run_slot("s1", src, constant(PLAY))
    assert run.minutes_missed == 10
    assert run.minutes_captured == 50
    assert run.capture_rate == pytest.approx(50 / 60)


def test_a_slot_still_evaluates_when_one_camera_fails_entirely() -> None:
    """Half a pitch is degraded evidence, not no evidence."""
    src = FakeSource({"camA": [1] * 60, "camB": [None] * 60})
    run = run_slot("s1", src, constant(PLAY))
    assert run.verdict.status is SlotStatus.USED
    assert run.minutes_captured == 60
    assert {s.camera_id for s in run.samples} == {"camA"}


def test_a_total_outage_yields_review_not_notused() -> None:
    """No footage is not evidence a pitch was unused."""
    src = FakeSource({"camA": [None] * 60, "camB": [None] * 60})
    run = run_slot("s1", src, constant(PLAY))
    assert run.verdict.status is SlotStatus.REVIEW
    assert run.minutes_captured == 0
    assert run.capture_rate == 0.0


def test_capture_rate_is_one_when_nothing_is_missed() -> None:
    src = FakeSource({"camA": [1] * 5})
    assert run_slot("s1", src, constant(PLAY)).capture_rate == 1.0


# --- fusion through the loop ------------------------------------------------


def test_play_on_either_camera_carries_the_minute() -> None:
    """The goalkeeper-only half: camB sees a match, camA sees an empty half."""
    src = FakeSource({"camA": [1] * 60, "camB": [1] * 60})

    def classify(image: np.ndarray) -> tuple[Class3, float]:
        classify.n += 1  # type: ignore[attr-defined]
        return (EMPTY, 0.95) if classify.n % 2 else (PLAY, 0.80)  # type: ignore[attr-defined]

    classify.n = 0  # type: ignore[attr-defined]
    run = run_slot("s1", src, classify)
    assert run.verdict.status is SlotStatus.USED


def test_on_minute_callback_sees_every_captured_minute() -> None:
    seen: list[int] = []
    src = FakeSource({"camA": [1] * 12})
    run_slot("s1", src, constant(PLAY), on_minute=lambda m, s: seen.append(m))
    assert seen == list(range(12))


# --- degraded mode: what happens when a camera dies mid-slot (WP7-T5) --------


def test_a_slot_that_lost_most_of_its_minutes_is_reviewed_not_decided() -> None:
    """The failure WP7-T5 asks about: a camera dies twenty minutes in and the system
    confidently reports NOTUSED from the twenty minutes before the pitch filled up.

    Degraded mode is REVIEW. The verdict says how much was captured, so an operator can see
    why without opening the slot.
    """
    from pitch_occupancy.data.taxonomy import Class3, SlotStatus
    from pitch_occupancy.slots.aggregate import aggregate_slot

    empty_fragment = [Class3.EMPTY] * 12          # 12 of a 60-minute slot
    decided = aggregate_slot(empty_fragment)
    assert decided.status is SlotStatus.NOTUSED, "without the slot length, the ratios stand"

    degraded = aggregate_slot(empty_fragment, minutes_expected=60)
    assert degraded.status is SlotStatus.REVIEW
    assert "12 of 60" in degraded.reason


def test_a_slot_above_the_capture_floor_is_still_decided() -> None:
    """The gate must not turn every imperfect slot into REVIEW; a few dropped frames are
    ordinary and the verdict should survive them."""
    from pitch_occupancy.data.taxonomy import Class3, SlotStatus
    from pitch_occupancy.slots.aggregate import aggregate_slot

    got = aggregate_slot([Class3.EMPTY] * 55, minutes_expected=60)
    assert got.status is SlotStatus.NOTUSED


def test_the_capture_gate_is_on_by_default_unlike_the_confidence_one() -> None:
    """`review_below_confidence` defaults to 0.0 and is therefore inert - which this project
    found the hard way, three times. A capture floor that shipped at 0.0 would be the same
    defect, so the default is asserted rather than assumed."""
    from pitch_occupancy.slots.aggregate import Thresholds

    assert Thresholds().review_below_capture > 0.0


def test_the_worker_passes_the_slot_length_through() -> None:
    """The gate is only live if the caller supplies the expected length. A worker that
    computed the ratios over whatever arrived would leave it silently vacuous - the camera
    would die, the ratios would be taken over the surviving fragment, and the verdict would
    look exactly as confident as one from a whole hour."""
    # A camera that dies after twelve minutes of an empty pitch.
    script = {"camA": [object()] * 12 + [None] * 48}
    classify, order = classifier_from({"camA": [Class3.EMPTY] * 12})
    source = FakeSource(script)

    original_read = source.read

    def read(camera_id: str, minute_index: int):
        frame = original_read(camera_id, minute_index)
        if frame is not None:
            order.append(camera_id)
        return frame

    source.read = read  # type: ignore[method-assign]
    run = run_slot("s", source, classify)

    assert run.minutes_captured == 12
    assert run.minutes_missed == 48
    assert run.verdict.status is SlotStatus.REVIEW
    assert "12 of 60" in run.verdict.reason


# --- the entry point (WP6-T2) ------------------------------------------------------------
#
# `main` raised NotImplementedError until 2026-09-10, naming the missing classifier. What
# these check is the wiring, not the model: that the run reaches the scheduler rather than
# calling `run_slot` behind its back, that a dry run loads nothing and writes nothing, and
# that evidence images stay off unless asked for. The model itself is `test_classifier.py`.


def _recordings(tmp_path):
    """A directory shaped like the export, with no video in it.

    `schedule_from_recordings` reads the filenames and nothing else, so the schedule can be
    derived without a gigabyte of footage - and every test below stops before a frame is
    read.
    """
    raw = tmp_path / "venue_01"
    raw.mkdir(parents=True)
    for name in ("StatBox_Replay_X_2026-07-11_10-00.mp4",
                 "StatBox_Replay_X_2026-07-11_10-00 (1).mp4"):
        (raw / name).write_bytes(b"")
    return raw


def test_a_dry_run_loads_no_model_and_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    """Deciding is separable from doing, the same shape `pitch retention` and
    `pitch schedule` use. A run that writes verdicts to the database should be something
    that was asked for, and asking twice costs nothing."""
    import sys

    import pitch_occupancy.vision.classifier as classifier
    from pitch_occupancy import worker

    def explode(*a, **k):
        raise AssertionError("a dry run loaded the backbone")

    monkeypatch.setattr(classifier, "load_classifier", explode)
    monkeypatch.setattr(sys, "argv",
                        ["worker", "--raw-dir", str(_recordings(tmp_path)), "--dry-run"])
    worker.main()
    out = capsys.readouterr().out
    assert "venue_01_2026-07-11_1000" in out
    assert "nothing was classified and nothing was written" in out


def test_it_goes_through_the_scheduler_with_the_real_classifier(tmp_path, monkeypatch) -> None:
    """The point of the exercise: `run_due` is the path a deployment takes, so replaying
    recordings has to go through it rather than calling `run_slot` directly. Bypassing it
    would leave the part that decides *when* untested by the thing that runs."""
    import sys

    import pitch_occupancy.scheduler as scheduler
    import pitch_occupancy.vision.classifier as classifier
    from pitch_occupancy import worker
    from pitch_occupancy.config import settings

    seen: dict[str, object] = {}
    sentinel = object()
    monkeypatch.setattr(classifier, "load_classifier",
                        lambda key=None, **k: type("C", (), {
                            "backbone": "dinov2", "n_train": 7, "__call__": lambda *a: None,
                        })())

    def fake_run_due(schedule, now, *, source_for, classify, **kwargs):
        seen["now"] = now
        seen["classify"] = classify
        seen["evidence_dir"] = kwargs.get("evidence_dir", sentinel)
        connection = kwargs.get("connection", sentinel)
        seen["connection"] = connection
        # read the schema here: `main` closes the connection on the way out, which is
        # correct and means it cannot be inspected afterwards
        if connection not in (sentinel, None):
            seen["tables"] = {r[0] for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        return [f"slot_at_{now:%H%M}"]

    monkeypatch.setattr(scheduler, "run_due", fake_run_due)
    # never the developer's real database: `main` opens `settings.db_path`
    monkeypatch.setattr(settings, "db_path", tmp_path / "db" / "test.db")
    monkeypatch.setattr(sys, "argv", ["worker", "--raw-dir", str(_recordings(tmp_path))])
    worker.main()

    assert seen["now"].hour == 10, "the clock must stand inside the slot's own window"
    assert getattr(seen["classify"], "backbone", None) == "dinov2"
    assert seen["evidence_dir"] is None, "evidence images are opt-in: they show real people"

    # `run_due` persists only when it is given a connection; without one it classifies the
    # whole slot, stores nothing, and returns the slot id anyway. The first version of
    # `main` forgot it and then printed "2 slot(s) written to the database" over an empty
    # table - a confident sentence about something that did not happen.
    assert seen["connection"] not in (sentinel, None)
    assert {"slot_evaluations", "frame_samples"} <= seen["tables"]


def test_no_recordings_is_an_error_not_an_empty_success(tmp_path, monkeypatch) -> None:
    """A run that finds nothing and exits 0 reads as "there was nothing to do" when it means
    "the path was wrong", and the two want different responses at 3am."""
    import sys

    from pitch_occupancy import worker

    empty = tmp_path / "nothing"
    empty.mkdir()
    monkeypatch.setattr(sys, "argv", ["worker", "--raw-dir", str(empty), "--dry-run"])
    with pytest.raises(SystemExit):
        worker.main()

def test_a_slot_that_cannot_run_is_reported_and_the_others_continue(
    tmp_path, monkeypatch, capsys
) -> None:
    """`run_due` passes the **exception** to `on_slot` when a slot fails - that is how one
    dead camera avoids stopping the other pitches. The first version of `main` printed
    `run.verdict` unconditionally and so died inside the handler for the failure it was
    reporting, on the very first real run: the schedule derived from the recordings names
    four slot instances and only two were exported.
    """
    import sys

    import pitch_occupancy.scheduler as scheduler
    import pitch_occupancy.vision.classifier as classifier
    from pitch_occupancy import worker
    from pitch_occupancy.config import settings

    monkeypatch.setattr(classifier, "load_classifier",
                        lambda key=None, **k: type("C", (), {
                            "backbone": "dinov2", "n_train": 7, "__call__": lambda *a: None,
                        })())

    def fake_run_due(schedule, now, *, source_for, classify, on_slot=None, **kwargs):
        on_slot("venue_01_2026-07-11_2030", FileNotFoundError("no recording"))
        return []

    monkeypatch.setattr(scheduler, "run_due", fake_run_due)
    monkeypatch.setattr(settings, "db_path", tmp_path / "db" / "test.db")
    monkeypatch.setattr(sys, "argv", ["worker", "--raw-dir", str(_recordings(tmp_path))])
    worker.main()  # must not raise
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "no recording" in out
    assert "0 slot(s) written" in out


# --- the live entry point (WP6-T3, 2026-09-11) ---------------------------------------------
#
# `live_sources` and `RTSPSource` have existed and been tested against an injected capture
# opener since September. What was missing was any way to *invoke* them: `--source` accepted
# only `video` and the camera URLs had nowhere to live, so pointing this at a camera meant
# writing Python. These cover the refusals, because the thing being opened is a real camera
# watching real people and every one of these failures is silent otherwise.


def test_a_missing_camera_config_says_what_to_copy(tmp_path) -> None:
    """Not an empty mapping. A live run that finds no cameras and proceeds reports every slot
    as unobserved, and "no camera was configured" and "no camera responded" are different
    facts that must not arrive as the same verdict."""
    from pitch_occupancy import worker

    with pytest.raises(SystemExit, match="cameras.example.json"):
        worker.load_camera_urls(tmp_path / "absent.json")


def test_an_empty_camera_config_is_refused(tmp_path) -> None:
    config = tmp_path / "cameras.json"
    config.write_text('{"_comment": "only a comment"}', encoding="utf-8")
    from pitch_occupancy import worker

    with pytest.raises(SystemExit, match="no cameras"):
        worker.load_camera_urls(config)


def test_comment_keys_are_dropped_so_the_example_can_explain_itself(tmp_path) -> None:
    """The example file carries its own instructions, which is where someone will look."""
    config = tmp_path / "cameras.json"
    config.write_text(
        '{"_comment": ["read me"], "venue_01": {"_note": "x", "camera_A": "rtsp://h/1"}}',
        encoding="utf-8",
    )
    from pitch_occupancy import worker

    assert worker.load_camera_urls(config) == {"venue_01": {"camera_A": "rtsp://h/1"}}


def test_a_scheduled_venue_with_no_cameras_stops_the_run(tmp_path, monkeypatch, capsys) -> None:
    """Found now rather than an hour later inside `live_sources`. A slot with no reachable
    camera has no state, and defaulting to one invents it."""
    import argparse

    from pitch_occupancy import worker

    config = tmp_path / "cameras.json"
    config.write_text('{"venue_99": {"camera_A": "rtsp://h/1"}}', encoding="utf-8")
    args = argparse.Namespace(cameras=config, dry_run=True, yes=True, model="dinov2",
                              evidence_dir=None, iterations=1)
    with pytest.raises(SystemExit, match="no cameras in the config"):
        worker._run_live(args)


def test_a_slot_camera_with_no_url_stops_the_run(tmp_path) -> None:
    """The venue check is not enough. `live_sources` does catch a missing camera id, but it
    catches it at slot time: a config naming camera_A for a slot that wants A and B starts
    cleanly, runs for an hour, and dies at the first boundary. Half a pitch reported as the
    whole one is the failure this refuses, so it has to happen before anything connects."""
    import argparse

    from pitch_occupancy import worker

    config = tmp_path / "cameras.json"
    # venue_01 is scheduled with camera_A and camera_B; only one of them is configured.
    config.write_text('{"venue_01": {"camera_A": "rtsp://h/1"}}', encoding="utf-8")
    args = argparse.Namespace(cameras=config, dry_run=True, yes=True, model="dinov2",
                              evidence_dir=None, iterations=1)
    with pytest.raises(SystemExit, match="camera_B"):
        worker._run_live(args)


def test_the_confirmation_hides_credentials(capsys, monkeypatch) -> None:
    """A terminal scrollback is a place URLs leak from, and the host is enough to tell you
    whether you are pointed at the right camera."""
    from pitch_occupancy import worker
    from pitch_occupancy.scheduler import Schedule

    monkeypatch.setattr("builtins.input", lambda *_: "no")
    with pytest.raises(SystemExit):
        worker._confirm_live(
            {"venue_01": {"camera_A": "rtsp://user:hunter2@10.0.0.5:554/s1"}},
            Schedule(()), None,
        )
    out = capsys.readouterr().out
    assert "hunter2" not in out and "user" not in out
    assert "10.0.0.5" in out


def test_live_says_whether_evidence_images_will_be_written(capsys, monkeypatch) -> None:
    """They are frames of identifiable people, so the prompt must say which way it is set
    before anything connects."""
    from pathlib import Path

    from pitch_occupancy import worker
    from pitch_occupancy.scheduler import Schedule

    monkeypatch.setattr("builtins.input", lambda *_: "no")
    with pytest.raises(SystemExit):
        worker._confirm_live({"v": {"c": "rtsp://h/1"}}, Schedule(()), Path("/tmp/ev"))
    assert "identifiable people" in capsys.readouterr().out
