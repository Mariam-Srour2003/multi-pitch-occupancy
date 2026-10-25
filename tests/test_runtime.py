"""The shared live step (`runtime`): whole-picture grabbing, one camera counted, a pitch decided.

No camera is contacted and no model loads: the capture and the pipeline are scripted.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from playground_env import RULES, make_env, scripted_verdict
from playground_env import new_playground as _new

from pitch_occupancy import cameras, playgrounds, runtime
from pitch_occupancy.config import settings
from pitch_occupancy.db.schema import connect, initialise
from pitch_occupancy.vision.rules import RuleConfig

RNG = np.random.default_rng(0)
WHOLE = RNG.integers(0, 255, (96, 128, 3), dtype=np.uint8)
HALF_DRAWN = WHOLE.copy()
HALF_DRAWN[-30:] = 250  # the flat pale band a stalled stream leaves


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):  # noqa: N802 - OpenCV's name
        return self.opened

    def get(self, _prop):
        return 2.0

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        self.released = True


# --- grabbing ---------------------------------------------------------------------------------

def test_a_half_drawn_picture_is_skipped_for_the_next_whole_one(monkeypatch) -> None:
    capture = FakeCapture([HALF_DRAWN, WHOLE])
    monkeypatch.setattr(cameras, "open_capture", lambda url, timeout_ms: capture)
    frame, problem, _ = runtime.grab_whole("rtsp://h/1", 2.0)
    assert problem is None and np.array_equal(frame, WHOLE) and capture.released


@pytest.mark.parametrize("capture, problem", [
    (FakeCapture([], opened=False), "did not answer"),
    (FakeCapture([HALF_DRAWN] * 20), "No whole picture"),
])
def test_a_camera_without_a_whole_picture_says_why(monkeypatch, capture, problem) -> None:
    monkeypatch.setattr(cameras, "open_capture", lambda url, timeout_ms: capture)
    frame, said, _ = runtime.grab_whole("rtsp://h/1", 2.0)
    assert frame is None and problem in said and capture.released


# --- one camera, one pitch --------------------------------------------------------------------

def test_counting_one_camera_goes_through_the_pipeline_with_its_venue() -> None:
    seen = {}

    class Pipeline:
        def observe_minute(self, camera, frames, previous=None, *, polygon=None, venue=None):
            seen.update(camera=camera, frames=len(frames), venue=venue, polygon=polygon)
            from types import SimpleNamespace
            return SimpleNamespace(verdict=scripted_verdict(6), instances=())

    counted = runtime.count_camera(Pipeline(), "site", "a", WHOLE)
    assert seen == {"camera": "a", "frames": 1, "venue": "site", "polygon": None}
    assert counted["people"] == 6 and counted["state"] == "C2_ACTIVE_PLAY"


def test_add_decides_the_pitch_with_the_slot_rule() -> None:
    play = runtime.combine_pitch("add", {"a": scripted_verdict(3), "b": scripted_verdict(4)},
                                 RULES)
    assert play == (7, 1), "two halves below play_min each are a game together"
    dark = runtime.combine_pitch("add", {"a": scripted_verdict(0, brightness=5.0),
                                         "b": scripted_verdict(0)}, RULES)
    assert dark == (0, None), "a pitch too dark to see is not sure, not empty"
    standing = runtime.combine_pitch(
        "add", {"a": scripted_verdict(3, matched=3, moved=0),
                "b": scripted_verdict(4, matched=4, moved=0)}, RULES)
    assert standing == (7, 0), "seven people standing still are not playing"
    assert runtime.combine_pitch("add", {}, RULES) == (None, None)


def test_the_other_modes_compare_counts_with_the_rules_play_min() -> None:
    six = RuleConfig(**{**RULES.to_json(), "play_min": 6})
    verdicts = {"a": scripted_verdict(5), "b": scripted_verdict(2)}
    assert runtime.combine_pitch("largest", verdicts, six) == (5, 0)
    assert runtime.combine_pitch("largest", verdicts, RULES) == (5, 1)
    assert runtime.combine_pitch("vote", verdicts, RULES) == (5, 1)


def test_an_unreadable_rule_file_fails_loudly(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "rules_path", tmp_path / "absent.json")
    with pytest.raises(Exception):  # noqa: B017 - whatever RuleConfig.load raises
        runtime.play_min()


# --- a playground check ----------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    return make_env(tmp_path, monkeypatch)


def test_an_add_playground_that_is_too_dark_stores_not_sure(env) -> None:
    env.pipeline.people.update(a=scripted_verdict(0, brightness=5.0), b=0)
    record = playgrounds.check_playground(_new("Dark", ("site/a", "site/b")))
    assert record["people"] == 0 and record["playing"] is None
    conn = connect(settings.db_path)
    row = conn.execute("SELECT people, playing FROM playground_checks").fetchone()
    conn.close()
    assert (row["people"], row["playing"]) == (0, None)


def test_an_add_playground_with_a_standing_crowd_is_not_playing(env) -> None:
    env.pipeline.people.update(a=scripted_verdict(3, matched=3, moved=0),
                               b=scripted_verdict(4, matched=4, moved=0))
    record = playgrounds.check_playground(_new("Talk", ("site/a", "site/b")))
    assert record["people"] == 7 and record["playing"] == 0


def test_a_not_sure_check_counts_neither_way_in_a_session(env) -> None:
    p = _new("Big", ("site/a",), every=60)
    conn = connect(settings.db_path)
    initialise(conn)
    for minute, people, playing in ((40, 8, 1), (41, 0, None), (42, 0, None), (43, None, None)):
        conn.execute(
            """INSERT INTO playground_checks (playground_id, checked_at, people, playing,
                   cameras_read, cameras_total, counting, duration_ms)
               VALUES (?, ?, ?, ?, 1, 1, 'add', 1)""",
            (p["id"], f"2026-10-17T09:{minute:02d}:00", people, playing))
    conn.commit()
    rows = playgrounds.sessions.sessions(conn, datetime(2026, 10, 17, 9, 30),
                                         datetime(2026, 10, 17, 11, 0))
    conn.close()
    (row,) = rows
    assert row["playing_share"] == 1.0 and row["verdict"] == "Playing"
    assert row["checks_not_sure"] == 2 and row["checks_without_picture"] == 1


def test_a_session_where_nothing_could_be_judged_says_not_sure() -> None:
    from pitch_occupancy.playgrounds.sessions import _session_verdict

    assert _session_verdict(None, 3, not_sure=4) == "Not sure"
    assert _session_verdict(None, None) == "No pictures"
