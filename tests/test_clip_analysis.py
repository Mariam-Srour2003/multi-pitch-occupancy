"""The clip reviewer (WP6-T6): sampling, smoothing, segmentation, and what it keeps.

The behaviour worth pinning here is not "it classifies frames" - `classify` is injected and
the model has its own tests. It is the two claims the page makes to the person reading it:
that a correction is *reported* rather than applied silently, and that the uploaded footage
is *gone*. Both are promises about people, and neither fails loudly if it breaks.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api.app import app
from pitch_occupancy.clip_analysis import analyse_clip
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.slots.stan import majority_smooth

E, P, M = Class3.EMPTY, Class3.ACTIVE_PLAY, Class3.MAINTENANCE_NON_SPORTING


def _video(path: Path, seconds: int = 12, fps: int = 10) -> Path:
    """A tiny real video. Small enough that a test decoding it costs nothing."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for i in range(seconds * fps):
        frame = np.full((64, 64, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _states(result) -> str:
    return "".join({E: ".", P: "P", M: "M"}[s.smoothed] for s in result.samples)


def _raw(result) -> str:
    return "".join({E: ".", P: "P", M: "M"}[s.raw] for s in result.samples)


# --- the smoother itself ------------------------------------------------------------------


def test_a_lone_disagreeing_sample_is_overruled_by_its_neighbours() -> None:
    """The case the feature exists for: one EMPTY frame in the middle of play is far more
    likely a misread than four seconds of empty pitch."""
    assert majority_smooth([P, P, E, P, P], 3) == [P, P, P, P, P]


def test_a_sustained_run_survives_smoothing() -> None:
    """The other half, and the one that makes the first safe. If a window of 3 could erase a
    two-sample run, it would be deleting events rather than correcting reads."""
    assert majority_smooth([P, P, E, E, P, P], 3) == [P, P, E, E, P, P]


def test_a_wider_window_erases_more() -> None:
    """Not a bug - the point of the control, and the reason the page exposes the window.

    The same three-sample run of play interrupted by two EMPTY samples survives a window of
    3 and is erased completely by a window of 5. At a ten-second interval that is twenty
    seconds of real event gone, so which window to use is the reviewer's trade, not a
    constant to bake in.
    """
    run = [P, P, P, E, E, P, P, P]
    assert majority_smooth(run, 3) == run
    assert majority_smooth(run, 5) == [P] * 8


# --- what the analysis reports -------------------------------------------------------------


def test_the_corrected_sample_is_flagged_not_hidden(tmp_path) -> None:
    """The promise the page is built on. A reviewer who cannot see what changed cannot judge
    whether the change was right, and only they can: the samples alone cannot separate a
    misread frame from a genuinely brief event."""
    cycle = iter([P, P, E, P, P, P, P, P, P, P, P, P])
    result = analyse_clip(_video(tmp_path / "c.mp4"), lambda img, **_kw: (next(cycle), 0.9),
                          interval_s=1.0, window=3)
    corrected = [s for s in result.samples if s.corrected]
    assert len(corrected) == 1
    assert corrected[0].raw is E and corrected[0].smoothed is P
    assert result.n_corrected == 1
    assert "corrected by its neighbours" in result.summary()


def test_neighbours_never_smooth_a_small_group_into_play(tmp_path) -> None:
    """Found 2026-09-29. The person gate turns a frame with three people inside into C3, and
    majority smoothing - which reads only the states - turned it straight back into
    ACTIVE_PLAY when its neighbours were playing. Fewer than five people is never play, so
    the frame keeps its own verdict; and a frame with people on it is never smoothed EMPTY.
    """
    from pitch_occupancy.vision.people import Counted

    class Scripted:
        play_min = 5

        def __init__(self, script):
            self.script = iter(script)

        def inspect(self, state, frame, polygon=None):
            n, out = next(self.script)
            return out, Counted(people=n, ball=True)

    # frame 2: three people and a ball among a match; frame 6: two people among empties
    script = [(9, P), (9, P), (3, M), (9, P), (9, P), (0, E), (2, M), (0, E)] + [(9, P)] * 4
    result = analyse_clip(_video(tmp_path / "c.mp4"), lambda img, **_kw: (P, 0.9),
                          interval_s=1.0, window=3, person_gate=Scripted(script))
    assert _raw(result).startswith("PPMPP.M.")
    assert _states(result).startswith("PPMPP.M."), _states(result)
    for sample in result.samples:
        if sample.people is not None and sample.people < 5:
            assert sample.smoothed is not P, sample
        if sample.people:
            assert sample.smoothed is not E, sample
        if sample.people == 0:
            assert sample.smoothed is E, sample


def test_the_raw_prediction_is_kept_beside_the_smoothed_one(tmp_path) -> None:
    cycle = iter([P, P, E, P, P, P, P, P, P, P, P, P])
    result = analyse_clip(_video(tmp_path / "c.mp4"), lambda img, **_kw: (next(cycle), 0.9),
                          interval_s=1.0, window=3)
    assert _raw(result).startswith("PP.P")
    assert _states(result).startswith("PPPP")


def test_window_one_disables_smoothing_entirely(tmp_path) -> None:
    """The escape hatch. A reviewer who decides the flickers are real turns it off rather
    than arguing with the tool."""
    cycle = iter([P, P, E, P, P, P, P, P, P, P, P, P])
    result = analyse_clip(_video(tmp_path / "c.mp4"), lambda img, **_kw: (next(cycle), 0.9),
                          interval_s=1.0, window=1)
    assert result.n_corrected == 0
    assert _raw(result) == _states(result)


def test_an_even_window_is_refused(tmp_path) -> None:
    """An even window has no centre, so "the neighbours overruled it" stops being symmetric
    and the correction silently leans earlier."""
    with pytest.raises(ValueError, match="odd"):
        analyse_clip(_video(tmp_path / "c.mp4"), lambda img, **_kw: (P, 0.9), window=4)


# --- segments and their boundaries ---------------------------------------------------------


def test_segments_collapse_the_smoothed_sequence(tmp_path) -> None:
    cycle = iter([E, E, E, P, P, P, P, P, P, E, E, E])
    result = analyse_clip(_video(tmp_path / "c.mp4"), lambda img, **_kw: (next(cycle), 0.9),
                          interval_s=1.0, window=3)
    assert [s.state for s in result.segments] == [E, P, E]


def test_a_boundary_is_reported_as_the_interval_it_falls_in(tmp_path) -> None:
    """Sampling every second locates a change to within a second and no better. Naming a
    timestamp would claim more than was observed, so the segment brackets it."""
    cycle = iter([E, E, E, P, P, P, P, P, P, P, P, P])
    result = analyse_clip(_video(tmp_path / "c.mp4"), lambda img, **_kw: (next(cycle), 0.9),
                          interval_s=1.0, window=3)
    play = next(s for s in result.segments if s.state is P)
    assert play.starts_after == 2.0 and play.starts_by == 3.0
    assert "between" in play.describe()


def test_the_opening_segment_has_no_bracket(tmp_path) -> None:
    """Nothing precedes it, so there is no interval to bracket - the clip began in that
    state and saying "between 0:00 and 0:00" would be noise."""
    result = analyse_clip(_video(tmp_path / "c.mp4"), lambda img, **_kw: (P, 0.9), interval_s=1.0)
    first = result.segments[0]
    assert first.starts_after == first.starts_by == 0.0
    assert "between" not in first.describe()


# --- refusals and degradation ---------------------------------------------------------------


def test_a_file_that_is_not_video_is_refused(tmp_path) -> None:
    bad = tmp_path / "not.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(ValueError, match="could not open|no frame"):
        analyse_clip(bad, lambda img, **_kw: (P, 0.9))


def test_a_long_clip_is_sampled_coarser_rather_than_truncated(tmp_path) -> None:
    """A partial answer that looks complete is the worse failure. Covering the whole clip at
    a wider interval is honest; analysing its first two minutes and stopping is not."""
    result = analyse_clip(_video(tmp_path / "c.mp4", seconds=12), lambda img, **_kw: (P, 0.9),
                          interval_s=0.6, window=1, max_samples=5)
    assert result.interval_widened
    assert len(result.samples) <= 5
    # the last sample still lands near the end of the clip, not near its beginning
    assert result.samples[-1].t_s > result.duration_s * 0.7


# --- the API, and the thing it promises not to do -------------------------------------------


def test_the_clip_route_keeps_nothing(tmp_path, monkeypatch) -> None:
    """Named in `ALLOWED_MUTATING`'s reason, so it is checked rather than asserted in a
    comment. This route is a POST because the body is a video, not because it stores one:
    the upload is deleted on the way out, whatever happened in between.

    Watches the temp directory the route actually writes into, because "we call unlink" is
    a claim about the code and "no file is left" is a claim about the disk.
    """
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    video = _video(tmp_path / "src.mp4").read_bytes()

    monkeypatch.setattr("pitch_occupancy.api.clip_review._classifier",
                        lambda: (lambda img, **_kw: (P, 0.9)))
    client = TestClient(app)
    response = client.post("/api/v1/clip/analyse?interval_s=2&window=1", content=video,
                           headers={"Content-Type": "application/octet-stream"})

    assert response.status_code == 200, response.text
    # Scoped to the suffix this route writes: torch drops an inductor cache directory in the
    # temp root, and asserting on "nothing new at all" would fail on somebody else's file.
    leftover = [p for p in tmp_path.iterdir() if p.suffix == ".upload"]
    assert not leftover, f"the upload was left on disk: {sorted(p.name for p in leftover)}"


def test_an_unreadable_upload_is_also_cleaned_up(tmp_path, monkeypatch) -> None:
    """The `finally` matters most on the path that raised."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client = TestClient(app)
    response = client.post("/api/v1/clip/analyse", content=b"not a video at all",
                           headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 422
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".upload"]


def test_an_empty_body_is_refused() -> None:
    """"No video" and "an empty pitch" must not arrive as the same answer."""
    client = TestClient(app)
    response = client.post("/api/v1/clip/analyse", content=b"",
                           headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 422
    assert "no video" in response.json()["detail"]


def test_an_even_window_is_refused_by_the_route() -> None:
    client = TestClient(app)
    response = client.post("/api/v1/clip/analyse?window=4", content=b"x",
                           headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 422


def test_the_page_is_served_and_shows_which_samples_smoothing_changed() -> None:
    """A smoothed timeline must not be presented as simply the answer.

    This asked for a *sentence* saying so - "it can equally be a real brief event" - and that
    note was removed on request on 2026-09-29, with the rest of the explanatory prose on the
    review pages. The requirement behind it is unchanged and is now carried by the page's
    structure rather than its wording: the corrected samples keep their striped band and its
    legend, the table strikes the raw prediction through beside the smoothed one, and the
    Corrected tile counts them. So this pins those, which is what a reviewer actually reads.

    If the *marks* go, the reviewer genuinely cannot tell a correction from a prediction and
    this should fail. That is a stronger test than the wording one it replaces, and it is not
    the reason it changed - the reason was that the sentence was noise to the person using the
    page.
    """
    client = TestClient(app)
    page = client.get("/clip")
    assert page.status_code == 200
    assert "corrected by neighbours" in page.text, "the timeline legend"
    assert "Show raw predictions only" in page.text, "the toggle back to what the model said"
    assert "<del>" in page.text, "the struck-through raw prediction in the table"
    assert "m-corr" in page.text, "the Corrected tile"


def test_the_route_reports_the_segments_and_the_corrections(tmp_path, monkeypatch) -> None:
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    video = _video(tmp_path / "src.mp4", seconds=12).read_bytes()
    cycle = itertools.chain([P, P, E, P], itertools.repeat(P))
    monkeypatch.setattr("pitch_occupancy.api.clip_review._classifier",
                        lambda: (lambda img, **_kw: (next(cycle), 0.9)))
    # This test is about the neighbour smoothing, so the gates are off. Left on, the person
    # gate turns every frame of a generated video EMPTY - correctly, there is nobody in it -
    # and there is no isolated sample left for the smoothing to correct.
    monkeypatch.setattr("pitch_occupancy.api.clip_review._gates", lambda: (None, None))

    client = TestClient(app)
    body = client.post("/api/v1/clip/analyse?interval_s=1&window=3", content=video,
                       headers={"Content-Type": "application/octet-stream"}).json()

    assert body["n_corrected"] == 1
    flagged = [s for s in body["samples"] if s["corrected"]]
    assert flagged[0]["raw"] == "C1_EMPTY" and flagged[0]["smoothed"] == "C2_ACTIVE_PLAY"
    assert body["segments"] and body["segments"][0]["state"] == "C2_ACTIVE_PLAY"
