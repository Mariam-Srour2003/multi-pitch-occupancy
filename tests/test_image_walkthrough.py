"""Checking still images, one or many.

The image reviewer runs the same detector and rule as the clip walkthrough, so the tests that
matter here are the ones about what is *different*:

* stills have no neighbours, so nothing is smoothed, and the page has to say so rather than
  imply the clip's temporal correction is at work;
* a batch arrives as JSON with base64 rather than as raw bytes, which is a wider door than
  `/clip/analyse` has and needs its bounds asserted;
* a file that is not an image yields nothing rather than a guess, and the count that reaches
  the page has to reconcile with the number of files submitted - otherwise a skipped upload
  is invisible.

The classifier is scripted (`_StubClassifier`) unless a test says otherwise, so nothing here
loads the detector.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api import image_walkthrough as iw
from pitch_occupancy.api.app import app
from pitch_occupancy.api.image_walkthrough import MAX_IMAGES
from pitch_occupancy.api.limits import WORKSPACE_PREFIX
from pitch_occupancy.data.taxonomy import Class3

E, P = Class3.EMPTY, Class3.ACTIVE_PLAY


def _image(path: Path, value: int = 120) -> Path:
    cv2.imwrite(str(path), np.full((64, 64, 3), value, dtype=np.uint8))
    return path


def _b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


class _StubClassifier:
    """A scripted classifier that records the outline each image was scored inside."""

    def __init__(self) -> None:
        self.polygons: list[object] = []

    def __call__(self, image_bgr, *, polygon=None, **_kw):
        self.polygons.append(polygon)
        return P, 0.9


@pytest.fixture(autouse=True)
def stub(monkeypatch) -> _StubClassifier:
    classifier = _StubClassifier()
    monkeypatch.setattr(iw, "_classifier", lambda: classifier)
    return classifier


def _records(paths, **kwargs) -> list[dict]:
    names = kwargs.pop("names", [Path(p).name for p in paths])
    return [json.loads(line) for line in iw.walk_records(
        list(paths), names=names, explain_n=kwargs.pop("explain_n", 0), redact=False,
        **kwargs)]


def _shots(paths, **kwargs) -> list[dict]:
    return [r for r in _records(paths, **kwargs) if r["type"] == "shot"]


# --- walk_records ---------------------------------------------------------------------


def test_shots_carry_their_position_and_name(tmp_path) -> None:
    paths = [_image(tmp_path / f"{i}.jpg") for i in range(3)]
    shots = _shots(paths)
    assert [s["index"] for s in shots] == [0, 1, 2]
    assert [s["name"] for s in shots] == ["0.jpg", "1.jpg", "2.jpg"]


def test_shots_carry_the_size_the_image_actually_was(tmp_path) -> None:
    """The page prints it beside the prediction, and a 64x64 thumbnail and a 4K frame are
    not the same evidence even when they score the same."""
    cv2.imwrite(str(tmp_path / "wide.jpg"), np.zeros((90, 160, 3), dtype=np.uint8))
    (shot,) = _shots([tmp_path / "wide.jpg"])
    assert (shot["width"], shot["height"]) == (160, 90)


def test_a_file_that_is_not_an_image_yields_nothing_rather_than_a_guess(tmp_path) -> None:
    """Yielding a prediction for a file the decoder rejected would be inventing an
    observation, which is the one thing this system is built never to do."""
    (tmp_path / "notes.txt").write_text("this is not an image", encoding="utf-8")
    good = _image(tmp_path / "real.jpg")
    shots = _shots([tmp_path / "notes.txt", good])
    assert len(shots) == 1
    assert shots[0]["name"] == "real.jpg"


def test_explain_n_bounds_how_many_get_the_detectors_drawing(tmp_path) -> None:
    paths = [_image(tmp_path / f"{i}.jpg") for i in range(4)]
    shots = _shots(paths, explain_n=0)
    assert not any(s["explained"] for s in shots)
    assert not any("boxes" in s for s in shots)


def test_max_images_is_enforced_in_the_generator_not_only_the_route(tmp_path) -> None:
    """The route rejects an oversized batch, but `walk_records` is importable on its own and
    a caller that bypassed the route should not be able to start an unbounded run."""
    paths = [_image(tmp_path / f"{i}.jpg") for i in range(6)]
    records = _records(paths, max_images=4)
    assert len([r for r in records if r["type"] == "shot"]) == 4
    assert records[-1]["n_unreadable"] == 0


# --- one outline per image ---------------------------------------------------------------
#
# A folder of stills is routinely a folder of *different cameras* - nothing in an upload says
# otherwise - so one outline across the batch is right only by luck, and wrong in the
# direction that matters: an outline drawn round image 1's pitch, applied to image 2, counts
# image 2's car park instead of its pitch. The stub records the outline each image was
# actually scored inside, so these assert on that rather than on what was passed in.

LEFT = [[0.05, 0.05], [0.45, 0.05], [0.45, 0.95], [0.05, 0.95]]
RIGHT = [[0.55, 0.05], [0.95, 0.05], [0.95, 0.95], [0.55, 0.95]]


def test_each_image_is_scored_inside_its_own_outline(tmp_path, stub) -> None:
    paths = [_image(tmp_path / "a.jpg"), _image(tmp_path / "b.jpg", 80)]
    _shots(paths, polygons=[LEFT, RIGHT])
    assert stub.polygons == [LEFT, RIGHT]


def test_an_image_with_no_outline_of_its_own_falls_back_to_the_batch(tmp_path, stub) -> None:
    """Undecided is not the same as declined, and only the second is a choice."""
    paths = [_image(tmp_path / "a.jpg"), _image(tmp_path / "b.jpg", 80)]
    _shots(paths, polygon=RIGHT, polygons=[LEFT, None])
    assert stub.polygons == [LEFT, RIGHT]


def test_choosing_the_whole_frame_for_one_image_beats_every_outline(tmp_path, stub) -> None:
    """*Use the whole frame* on image 2 has to survive a batch outline and its own.

    The complaint this comes from: pressing the button cleared the outline on screen and the
    image was scored inside a boundary anyway. An absent outline could not carry the choice,
    because that is also what "not measured yet" looks like - and the page answers *that* by
    measuring one. `wholes` says it in a way nothing else can be mistaken for.
    """
    paths = [_image(tmp_path / "a.jpg"), _image(tmp_path / "b.jpg", 80)]
    _shots(paths, polygon=RIGHT, polygons=[LEFT, LEFT], wholes=[False, True])
    assert stub.polygons == [LEFT, None]


def test_the_route_carries_the_whole_frame_choice_per_image(tmp_path) -> None:
    """End to end, and past the stored camera: a saved boundary is not a stronger claim than
    the operator looking at this image."""
    from pitch_occupancy.vision import roi

    roi.save("camera_A", RIGHT)
    body = {"images": [
        {"name": "a.jpg", "data": _b64(_image(tmp_path / "a.jpg")), "polygon": LEFT},
        {"name": "b.jpg", "data": _b64(_image(tmp_path / "b.jpg", 80)), "whole_frame": True},
    ]}
    response = TestClient(app).post(
        "/api/v1/images/walkthrough?explain_n=0&camera=camera_A", json=body)
    assert response.status_code == 200
    meta = json.loads(response.text.strip().splitlines()[0])
    assert meta["n_outlines"] == 1
    assert meta["n_whole_frame"] == 1


# --- the route ------------------------------------------------------------------------


def test_the_stream_is_newline_delimited_json_with_a_meta_and_a_done(tmp_path) -> None:
    client = TestClient(app)
    body = {"images": [{"name": "a.jpg", "data": _b64(_image(tmp_path / "a.jpg"))}]}
    response = client.post("/api/v1/images/walkthrough?explain_n=0", json=body)
    assert response.status_code == 200

    records = [json.loads(line) for line in response.text.strip().splitlines()]
    assert records[0]["type"] == "meta"
    assert records[0]["n_submitted"] == 1
    assert records[-1]["type"] == "done"


def test_a_skipped_file_is_reconcilable_from_the_done_record(tmp_path) -> None:
    """`n` and `n_submitted` can differ, and a page showing only what it managed to read
    would drop the difference silently."""
    junk = base64.b64encode(b"not an image at all").decode()
    body = {"images": [{"name": "junk.txt", "data": junk}]}
    response = TestClient(app).post("/api/v1/images/walkthrough", json=body)

    records = [json.loads(line) for line in response.text.strip().splitlines()]
    assert records[0]["n_submitted"] == 1
    # Exact, not a subset: the done record is what a page reconciles its counts from, and a
    # field appearing in it unannounced is a field nobody accounted for.
    assert records[-1] == {"type": "done", "n": 0, "n_unreadable": 1}


def test_a_data_url_prefix_is_accepted(tmp_path) -> None:
    """`FileReader.readAsDataURL` produces the prefixed form, so the browser's natural
    output has to work without every caller stripping it first."""
    payload = "data:image/jpeg;base64," + _b64(_image(tmp_path / "a.jpg"))
    response = TestClient(app).post(
        "/api/v1/images/walkthrough?explain_n=0", json={"images": [{"name": "a", "data": payload}]}
    )
    records = [json.loads(line) for line in response.text.strip().splitlines()]
    assert records[-1]["n"] == 1


def test_the_batch_size_is_capped(tmp_path) -> None:
    data = _b64(_image(tmp_path / "a.jpg"))
    body = {"images": [{"name": f"{i}.jpg", "data": data} for i in range(MAX_IMAGES + 1)]}
    assert TestClient(app).post("/api/v1/images/walkthrough", json=body).status_code == 413


def test_an_empty_batch_is_refused() -> None:
    assert TestClient(app).post(
        "/api/v1/images/walkthrough", json={"images": []}
    ).status_code == 422


def test_invalid_base64_is_refused_before_anything_streams() -> None:
    assert TestClient(app).post(
        "/api/v1/images/walkthrough", json={"images": [{"name": "x", "data": "not!!b64"}]}
    ).status_code == 422


def test_the_uploads_are_deleted_after_streaming(tmp_path) -> None:
    """Same guarantee as the clip route, and it has the same trap: a StreamingResponse's
    body runs after the handler returns, so deleting in the handler would remove the files
    before a single one had been read."""
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), f"{WORKSPACE_PREFIX}*")))
    body = {"images": [{"name": "a.jpg", "data": _b64(_image(tmp_path / "a.jpg"))}]}
    TestClient(app).post("/api/v1/images/walkthrough?explain_n=0", json=body)
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), f"{WORKSPACE_PREFIX}*")))
    assert after == before


def test_two_uploads_sharing_a_name_do_not_collide(tmp_path) -> None:
    """A browser can send two files called `frame.jpg` from different folders. Writing both
    to one path would analyse the second twice and report the first's name against it."""
    a = _b64(_image(tmp_path / "a.jpg", value=10))
    b = _b64(_image(tmp_path / "b.jpg", value=240))
    body = {"images": [{"name": "frame.jpg", "data": a}, {"name": "frame.jpg", "data": b}]}
    response = TestClient(app).post("/api/v1/images/walkthrough?explain_n=0", json=body)

    shots = [r for r in (json.loads(x) for x in response.text.strip().splitlines())
             if r["type"] == "shot"]
    assert len(shots) == 2
    assert [s["index"] for s in shots] == [0, 1]


# --- the page -------------------------------------------------------------------------


def test_the_page_is_served_and_reaches_the_batch_endpoint() -> None:
    html = TestClient(app).get("/images").text
    assert "/api/v1/images/walkthrough" in html
    assert 'id="file"' in html and "multiple" in html


def test_the_page_says_stills_are_not_smoothed() -> None:
    """The one claim this page must not let a reader assume. The clip reviewer corrects an
    isolated misread from its neighbours; a set of stills has none, so a page that looked
    like the clip page without saying this would imply a correction that never happened.
    """
    html = " ".join(TestClient(app).get("/images").text.split())
    assert "Each image stands on its own" in html
    assert "no neighbours" in html


def test_the_page_is_honest_that_slow_motion_is_not_the_model() -> None:
    """Inherited from the clip page for the same reason: a reader who thinks they sped the
    model up has learnt something false about the system.

    Asserted on whitespace-normalised text, because the claim is the sentence and not where
    the source happens to wrap it - the first version of this test failed on a line break in
    the middle of the very phrase it was protecting.
    """
    html = " ".join(TestClient(app).get("/images").text.split())
    assert "the model runs at the same speed either way" in html


def test_the_two_reviewers_lead_back_to_the_testing_hub() -> None:
    """They reach each other through the shared navigation's Testing hub, which links both."""
    client = TestClient(app)
    assert 'href="/testing"' in client.get("/clip").text
    assert 'href="/testing"' in client.get("/images").text
    hub = client.get("/testing").text
    assert 'href="/clip"' in hub and 'href="/images"' in hub


def test_a_failure_mid_stream_becomes_a_record_rather_than_a_truncated_body(
    tmp_path, monkeypatch
) -> None:
    """The response is committed as 200 before the generator runs.

    So an exception the generator does not catch escapes as an unhandled ASGI error and the
    client gets a truncated NDJSON body with no `error` record in it - a page that spins
    forever while the server console holds the only account of what happened. This was
    observed for real: Windows Application Control blocked torch's `_C` DLL, the model
    failed to load with an `ImportError` on the very first record, and the browser showed
    nothing at all.

    `ImportError` specifically, because the model load is both the likeliest failure here
    and the one furthest from the `ValueError` the handler used to catch.
    """
    def _boom(*args, **kwargs):
        raise ImportError("DLL load failed while importing _C")

    monkeypatch.setattr(iw, "_classifier", _boom)

    body = {"images": [{"name": "a.jpg", "data": _b64(_image(tmp_path / "a.jpg"))}]}
    response = TestClient(app).post("/api/v1/images/walkthrough", json=body)

    assert response.status_code == 200
    records = [json.loads(line) for line in response.text.strip().splitlines()]
    errors = [r for r in records if r["type"] == "error"]
    assert errors, f"the failure never reached the client: {records}"
    assert "ImportError" in errors[0]["detail"]
    # `fatal` tells the page not to suggest re-uploading: the model is missing, and no
    # different image would help.
    assert errors[0]["fatal"] is True


def test_the_uploads_are_still_deleted_when_the_stream_fails(tmp_path, monkeypatch) -> None:
    """The cleanup is in a `finally`, and a broadened `except` must not have moved it."""
    def _boom(*args, **kwargs):
        raise ImportError("no model")

    monkeypatch.setattr(iw, "_classifier", _boom)
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), f"{WORKSPACE_PREFIX}*")))

    body = {"images": [{"name": "a.jpg", "data": _b64(_image(tmp_path / "a.jpg"))}]}
    TestClient(app).post("/api/v1/images/walkthrough", json=body)

    assert set(glob.glob(os.path.join(tempfile.gettempdir(), f"{WORKSPACE_PREFIX}*"))) == before


def _workspaces() -> set[str]:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), f"{WORKSPACE_PREFIX}*")))


def _no_model(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise ImportError("no model in this test")

    monkeypatch.setattr(iw, "_classifier", _boom)


@pytest.mark.parametrize("name", ["a\x00b.jpg", "x" * 255 + ".jpg", "../../escape.jpg"])
def test_the_callers_filename_never_reaches_the_disk(tmp_path, monkeypatch, name) -> None:
    """UP-6. A NUL byte or a 255-character name made the write fail after earlier images had
    been written, the request returned 500, and those images stayed in the temp directory.
    The files on disk are named by index alone now; the name only travels back in records."""
    _no_model(monkeypatch)
    before = _workspaces()
    data = _b64(_image(tmp_path / "a.jpg"))
    body = {"images": [{"name": "first.jpg", "data": data}, {"name": name, "data": data}]}

    response = TestClient(app).post("/api/v1/images/walkthrough?explain_n=0", json=body)

    assert response.status_code == 200
    assert _workspaces() == before


def test_a_batch_that_fails_while_being_written_leaves_nothing(tmp_path, monkeypatch) -> None:
    """UP-6. The workspace is removed on any failure while staging, not only on success."""
    _no_model(monkeypatch)
    real = Path.write_bytes
    calls = []

    def flaky(self, data):
        calls.append(self)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(self, data)

    monkeypatch.setattr(iw.Path, "write_bytes", flaky)
    before = _workspaces()
    data = _b64(_image(tmp_path / "a.jpg"))
    body = {"images": [{"name": "a.jpg", "data": data}, {"name": "b.jpg", "data": data}]}

    with pytest.raises(OSError, match="disk full"):
        TestClient(app).post("/api/v1/images/walkthrough", json=body)
    assert len(calls) == 2
    assert _workspaces() == before


def _png_claiming(width: int, height: int) -> bytes:
    """A tiny, valid-looking PNG whose header declares ``width`` x ``height`` pixels."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(b"\x00" * 64)) + chunk(b"IEND", b""))


@pytest.mark.parametrize("side", [12000, 20000])
def test_an_image_declaring_too_many_pixels_is_refused_before_it_is_decoded(side) -> None:
    """UP-7. A 17 KiB PNG of 12000x12000 passed every byte cap and decoded to a gigabyte.
    The header is read first and anything over `MAX_PIXELS` is a 413; 20000x20000 is past
    PIL's own bomb threshold too, which must also come back as a 413 and not a 500."""
    before = _workspaces()
    body = {"images": [{"name": "bomb.png",
                        "data": base64.b64encode(_png_claiming(side, side)).decode()}]}
    response = TestClient(app).post("/api/v1/images/walkthrough", json=body)
    assert response.status_code == 413
    assert "pixels" in response.text
    assert _workspaces() == before


def test_the_json_body_is_bounded_before_it_is_parsed(monkeypatch) -> None:
    """UP-8. The image caps applied after FastAPI had buffered and parsed the whole body."""
    monkeypatch.setattr(iw, "MAX_BODY_BYTES", 1024)
    body = {"images": [{"name": "a.jpg", "data": "A" * 4096}]}
    assert TestClient(app).post("/api/v1/images/walkthrough", json=body).status_code == 413


def test_a_malformed_outline_is_refused_before_anything_streams(tmp_path) -> None:
    """UP-9/UP-15. Too many points, or a coordinate too large for a float, is a 422 - not an
    error record inside a 200 stream, and not a 500."""
    data = _b64(_image(tmp_path / "a.jpg"))
    ring = [[0.5, 0.1], [0.9, 0.5], [0.5, 0.9], [0.1, 0.5]] * 20
    client = TestClient(app)
    assert client.post("/api/v1/images/walkthrough",
                       json={"images": [{"data": data}], "polygon": ring}).status_code == 422
    huge = [[10 ** 400, 0], [0, 1], [1, 1]]
    assert client.post("/api/v1/images/walkthrough", content=json.dumps(
        {"images": [{"data": data, "polygon": huge}]}),
        headers={"Content-Type": "application/json"}).status_code == 422


# --- the detector's drawing ------------------------------------------------------------------


@pytest.mark.slow
def test_an_explained_image_carries_the_detectors_boxes(tmp_path) -> None:
    """A box per person, a ring round the ball, and the rule row that fired - so "the model
    found six people and a ball" can be checked against the picture."""
    client = TestClient(app)
    body = {"images": [{"name": "a.jpg", "data": _b64(_image(tmp_path / "a.jpg"))}]}
    response = client.post("/api/v1/images/walkthrough?explain_n=1", json=body)
    assert response.status_code == 200

    shots = [json.loads(line) for line in response.text.strip().splitlines()
             if json.loads(line)["type"] == "shot"]
    explained = [s for s in shots if s["explained"]]
    assert explained, "nothing was explained, so there is nothing to check"
    for shot in explained:
        assert shot["boxes"].startswith("data:image/jpeg;base64,"), "a picture, not a promise"
        found = shot["detector"]
        assert found["model"] and found["state"]
        assert found["rule"] >= 1, "which row of the table fired, so the state is checkable"
        assert isinstance(found["people_inside"], int)
        assert found["trace"], "the steps that led to the state, in words"
    for shot in shots:
        if not shot["explained"]:
            assert "boxes" not in shot, "a second detector pass is bounded by explain_n"


def test_the_detector_pane_says_not_checked_rather_than_nobody(monkeypatch) -> None:
    """`None` from a detector means it did not run. It has never meant an empty pitch, and
    this pane is the newest place that distinction could be lost."""
    import numpy as np

    from pitch_occupancy.vision import detector, overlay

    class Unavailable:
        def detect(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(detector.Detector, "load", staticmethod(lambda *a, **k: Unavailable()))
    frame = np.zeros((90, 160, 3), np.uint8)
    canvas, found = overlay.detector_pane(frame)
    assert found["checked"] is False
    assert found["state"] == "UNCERTAIN" and found["rule"] == 1
    assert canvas.shape == frame.shape
    assert np.array_equal(canvas, frame), "nothing found is not something to draw"


# --- the decision table on a still ------------------------------------------


def _meta(tmp_path, monkeypatch) -> dict:
    """The meta record alone, with nothing heavy loaded.

    `walk_records` yields it before the first image is read; the autouse stub stands in for
    a model load the table does not depend on.
    """
    path = tmp_path / "f.jpg"
    cv2.imwrite(str(path), np.full((90, 160, 3), 120, np.uint8))
    record = next(iter(iw.walk_records([path], names=["f.jpg"], explain_n=0, redact=False)))
    return json.loads(record)


def test_the_rule_table_reaches_the_page_with_the_configured_numbers(
    tmp_path, monkeypatch
) -> None:
    """The thresholds are `play_min` and `small_group_max` from `configs/rules.json`, and
    the page must not restate them.

    It used to draw the seven rows as typed HTML with `5` and `4` spelled out in it, which
    is right until the file changes and silently wrong afterwards. The rows now arrive with
    the meta record, built by `rules.rule_table` from the same config `decide` reads.
    """
    from pitch_occupancy.config import settings
    from pitch_occupancy.vision.rules import RuleConfig

    cfg = RuleConfig.load(settings.rules_path)
    meta = _meta(tmp_path, monkeypatch)

    assert meta["type"] == "meta"
    rows = {r["row"]: r for r in meta["rules"]}
    assert set(rows) == {1, 2, 3, 4, 5, 6, 7}
    assert str(cfg.small_group_max) in rows[5]["condition"]
    assert str(cfg.play_min) in rows[6]["condition"]
    assert rows[6]["state"] == "C2_ACTIVE_PLAY"


def test_the_rows_that_cannot_fire_on_a_still_say_so_rather_than_being_drawn_live(
    tmp_path, monkeypatch
) -> None:
    """The point of the whole panel, and the reason it is data rather than a picture.

    Row 4 is "nobody found, but something moved". One image has no predecessor, so movement
    cannot be measured at all - and a reader shown seven live rows would conclude the still
    was checked for movement and found still. That is the "not measured is not zero"
    confusion the rest of the rule module exists to prevent. Row 2 is the same kind of fact
    for a different reason: the interactive path counts the whole frame, so no-boundary
    cannot fire here either.
    """
    meta = _meta(tmp_path, monkeypatch)

    from pitch_occupancy.config import settings
    from pitch_occupancy.vision.rules import RuleConfig

    rows = {r["row"]: r for r in meta["rules"]}
    # Row 4 can still fire on a still through its darkness clause when one is configured;
    # without it, a still has no way to reach the row.
    if getattr(RuleConfig.load(settings.rules_path), "dark_luma", None) is None:
        assert rows[4]["applies"] is False
        assert "predecessor" in rows[4]["why_not"]
    assert rows[2]["applies"] is False
    assert all(rows[n]["applies"] for n in (1, 3, 5, 6, 7))

    motion = [c for c in meta["clauses"] if c["cue"] == "motion"]
    assert motion and motion[0]["status"] == "unavailable"
    assert "skipped rather than passed" in motion[0]["detail"]


def test_the_page_draws_the_rules_from_the_served_table_not_from_typed_rows() -> None:
    """A second copy of the table in the page source is the failure this guards.

    The panel and its renderer have to be there, and the thresholds must not be: if `5` and
    `4` appear as rule text in the HTML, someone has retyped the table.
    """
    html = TestClient(app).get("/images").text
    assert 'id="rules"' in html and "paintRules(" in html
    assert "RULES=d.rules" in html, "the page is not reading the served table"
    for typed in ("5+ people inside", "1 to 4 people inside"):
        assert typed not in html, f"the rule table has been retyped into the page: {typed!r}"


def test_the_gallery_flags_an_empty_verdict_with_people_found() -> None:
    """Both halves of the disagreement, not just one.

    The card's comment always said it flagged "the verdict says nobody is playing and yet
    people were found, or the reverse", and the condition only ever implemented the first.
    A confident EMPTY on a frame where the detector found four people has to be flagged.

    Asserted on the condition rather than on a rendered card because the flag is one line of
    browser JavaScript; what can go stale here is the claim, and the claim is the `empty`
    half being present at all.
    """
    html = " ".join(TestClient(app).get("/images").text.split())
    assert "const empty=s.predicted.indexOf('EMPTY')>=0;" in html
    assert "(play&&n===0)||(empty&&n>0)" in html, "the card flags only one direction again"
    assert "'Empty, people found'" in html, "the batch summary lost its second flag"
