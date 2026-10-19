"""The "never bill" constraint, checked rather than stated.

Reconciliation can implicate a named member of staff, so the design constraint is that the
system produces decision support and takes no automated financial action. That sentence is
worth very little on its own, so these tests walk the actual surface: the
advisory type, the HTTP routes and the database layer.

The point is not that a test stops a determined operator - anyone with the database can do
what they like. It is that "decision support" describes the artefact rather than the
intention, and stays true as the code grows.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from pitch_occupancy.data.taxonomy import SlotStatus
from pitch_occupancy.slots.authority import (
    FORBIDDEN,
    MAX_SEVERITY_WITHOUT_HUMAN,
    PERMITTED,
    Action,
    Advisory,
    advise,
)
from pitch_occupancy.slots.reconcile import Anomaly, Booking, Reconciliation, reconcile

SRC = Path(__file__).resolve().parents[1] / "src" / "pitch_occupancy"


# --- the advisory ----------------------------------------------------------------


def test_an_advisory_always_requires_a_person() -> None:
    """Not a policy setting. There is no argument that turns it off."""
    a = Advisory(field_id="f1", date="2026-07-11", start="10:00", finding="x")
    assert a.requires_human_confirmation is True
    assert "requires_human_confirmation" not in Advisory.__dataclass_fields__


def test_an_advisory_carries_no_monetary_quantity() -> None:
    """The absence is the constraint, so it is asserted rather than assumed."""
    money = re.compile(r"amount|price|charge|invoice|fee|penalt|refund|currenc", re.I)
    named = [f for f in Advisory.__dataclass_fields__ if money.search(f)]
    assert not named, f"Advisory grew a monetary field: {named}"


def test_a_forbidden_action_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="not a permitted action"):
        Advisory(field_id="f1", date="d", start="s", finding="x",
                 actions=("apply_penalty",))  # type: ignore[arg-type]


def test_the_permitted_vocabulary_is_small_and_has_no_money_in_it() -> None:
    """Adding to it should be a design decision. This is what makes it one."""
    assert {a.value for a in PERMITTED} == {
        "flag_for_review", "record_observation", "aggregate_by_field"
    }
    assert not (FORBIDDEN & {a.value for a in PERMITTED})


def test_there_is_no_severity_at_which_a_person_stops_being_required() -> None:
    assert MAX_SEVERITY_WITHOUT_HUMAN is None


def test_an_anomaly_is_flagged_for_a_person_rather_than_acted_on() -> None:
    booking = Booking(field_id="f1", date="2026-07-11", start="10:00",
                      booked=True, staff_recorded_used=True)
    rec = reconcile(booking, SlotStatus.NOTUSED)
    assert rec.is_anomaly
    got = advise(rec)
    assert Action.FLAG_FOR_REVIEW in got.actions
    assert got.requires_human_confirmation


def test_a_consistent_slot_still_produces_an_advisory() -> None:
    """A caller that has to handle a null is a caller that can forget to."""
    booking = Booking(field_id="f1", date="2026-07-11", start="10:00",
                      booked=True, staff_recorded_used=True)
    got = advise(reconcile(booking, SlotStatus.USED))
    assert Action.FLAG_FOR_REVIEW not in got.actions
    assert Action.RECORD_OBSERVATION in got.actions


def test_a_review_verdict_never_becomes_an_anomaly() -> None:
    """`reconcile.py` rule 1, restated at this boundary: the system may not convert its own
    uncertainty into someone else's error."""
    booking = Booking(field_id="f1", date="2026-07-11", start="10:00",
                      booked=True, staff_recorded_used=False)
    rec = reconcile(booking, SlotStatus.REVIEW)
    assert rec.anomaly is Anomaly.NEEDS_REVIEW
    assert not rec.is_anomaly
    assert Action.FLAG_FOR_REVIEW not in advise(rec).actions


# --- the surfaces that could grow one --------------------------------------------


def _http_methods(path: Path) -> list[tuple[str, str]]:
    """(method, route) for every FastAPI decorator in a module.

    **Async handlers count, and for a while they did not.** This walked `ast.FunctionDef`
    alone, which does not match `async def` - so every asynchronous route was invisible to
    the allowlist below, and a route that streams or reads a request body is exactly the
    kind that is written `async`. `/images/walkthrough` was added and never challenged; the
    two clip routes are on the list only because someone put them there by hand, not because
    this check ever saw them.

    A guard that silently skips the routes most likely to need it is the failure this
    project keeps rediscovering: the safeguard existed, passed, and certified nothing.
    """
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
                continue
            if dec.func.attr in {"get", "post", "put", "patch", "delete"} and dec.args:
                first = dec.args[0]
                if isinstance(first, ast.Constant):
                    out.append((dec.func.attr, str(first.value)))
    return out


def test_no_http_route_offers_a_financial_action() -> None:
    """The externally reachable surface, walked rather than remembered."""
    money = re.compile(r"charge|invoice|bill|payment|penalt|fee|refund", re.I)
    offenders = []
    for path in (SRC / "api").glob("*.py"):
        for method, route in _http_methods(path):
            if money.search(route):
                offenders.append(f"{path.name}: {method.upper()} {route}")
    assert not offenders, f"a financial route appeared: {offenders}"


#: Booking routes that use POST only because they carry a body, and write nothing. Each is
#: here deliberately; anything else with "booking" in its path and a mutating verb fails below.
BOOKING_READS_WITH_A_BODY = {
    #: Reads the configured source and reports what it found. POST so the check is an explicit
    #: action rather than something a crawler or a prefetch can trigger against a client's
    #: database.
    "/api/v1/bookings/test",
    #: Runs the quality check over a posted CSV, or over the configured source when the body
    #: is empty. Returns a report; stores nothing.
    "/api/v1/bookings/check",
}


def test_no_http_route_writes_to_a_booking() -> None:
    """Booking data is read-only: the audit compares against the record, it does not amend
    it. A write route would make the system a participant in what it is auditing."""
    offenders = [
        f"{path.name}: {method.upper()} {route}"
        for path in (SRC / "api").glob("*.py")
        for method, route in _http_methods(path)
        if method in {"post", "put", "patch", "delete"} and "booking" in route.lower()
        and not (method == "post" and route in BOOKING_READS_WITH_A_BODY)
    ]
    assert not offenders, f"a booking write route appeared: {offenders}"


def test_the_booking_module_has_no_write_path() -> None:
    """What makes the two POSTs above safe: the source module issues no write. The SQL
    guard there rejects write statements in a configured query; this pins that the module's
    own code never opens a source for writing."""
    text = (SRC / "booking_sources.py").read_text(encoding="utf-8")
    assert "mode=ro" in text
    assert not re.search(r"\.commit\(|open\([^)]*[\"']w", text)


#: Routes allowed to use a mutating HTTP verb, each with the reason it is not the system
#: acting. Adding to this set is the deliberate look the test below asks for - and the reason
#: belongs here, next to the entry, rather than in a commit message nobody re-reads.
ALLOWED_MUTATING = {
    #: A person correcting the system. The model's verdict is retained beside the correction.
    "/slots/{slot_id}/override",
    #: An operator editing this system's *own* capture configuration. Not the same
    #: category as writing to a client system: `bookings.py` has no write method at all,
    #: because a booking sheet is the facility's financial record. This is defensible, and it
    #: is still the most dangerous route here - see `api/schedule_editor.py` for the three
    #: protections it carries.
    "/schedule",
    #: A POST that writes nothing. It carries a proposed schedule in its body, which is why
    #: it cannot be a GET, and returns whether the scheduler would accept it. Pinned by
    #: `test_the_validate_route_really_writes_nothing` below rather than trusted.
    "/schedule/validate",
    #: The clip reviewer. A POST because the body is a video, not because it
    #: persists one: the upload is streamed to a temp file, read, and deleted in a
    #: `finally`, and no database row is written. Same category as `/schedule/validate`,
    #: and pinned the same way by `test_the_clip_route_keeps_nothing` below.
    "/clip/analyse",
    #: The two-camera reviewer: two videos of one pitch in one body, A then B. Same category
    #: and same guarantee as `/clip/analyse` - both temp files are removed in a `finally` and
    #: nothing reaches the database. Pinned by `test_the_pair_route_keeps_neither_file` in
    #: test_clip_pair.
    "/clip/pair/analyse",
    #: The same analysis streamed step by step with the detector's boxes. A POST for
    #: the same reason and with the same guarantee - the upload is removed in the streaming
    #: generator's `finally` rather than the handler's, because the body outlives the
    #: handler. Pinned by the upload tests in test_upload_limits.
    "/clip/walkthrough",
    #: The same analysis over still images. A POST because the body carries the
    #: images, not because it keeps them: they are written to a temp directory, read, and
    #: removed in the streaming generator's `finally`, and nothing reaches the database.
    #: Pinned by the upload tests in test_upload_limits.
    #:
    #: It was added without ever appearing here, because this check could not see `async def`
    #: routes - see `_http_methods`. That is the entry's own small lesson.
    "/images/walkthrough",
    #: A person configuring this system's own capture geometry: which part of the
    #: frame is this camera's pitch. Same category as `/schedule` - it edits the system's
    #: configuration, never a client record - and a narrower one, because a boundary is
    #: geometry rather than money or time. PUT and DELETE share the path.
    #:
    #: The write lands in `configs/roi.json`, not the database, so it is reviewable in a
    #: diff. `roi.validate` refuses a mis-click rather than persisting one, which is the
    #: protection that matters here: a boundary keeping 1% of the frame would not raise, would
    #: not look wrong, and would produce confident answers about a sliver of a pitch.
    "/roi",
    #: Writes nothing at all. It applies a proposed boundary to one frame and returns what
    #: the model makes of it, twice - masked and unmasked - so an operator can check a
    #: boundary before saving it. A POST because the body carries an image. Same category as
    #: `/schedule/validate`, and pinned by `test_the_preview_saves_nothing` below.
    "/roi/preview",
    #: Writes nothing. It measures an outline from posted stills so the images page can open
    #: its drawing step on a suggestion rather than an empty canvas - an operator facing a
    #: blank canvas reasonably picks "whole frame", which is the thing that step exists to
    #: stop. A POST because the body carries images, and the result is returned, never saved:
    #: saving a boundary is `PUT /roi`, which a person does deliberately.
    "/roi/suggest",
    #: Writes nothing either. It takes a posted video and returns its first frame, so a
    #: boundary for a clip can be drawn on the frame the analysis will actually see rather
    #: than on one the browser decoded separately. A POST because the body is a video; the
    #: upload is deleted in a `finally`, as on every other route here that takes footage.
    "/roi/first-frame",
    #: The booking-day test. A POST because the body is one or two videos; they are written to
    #: temp files that the background job deletes in its `finally` (or at once if the request
    #: fails). What is kept is the per-slot result, under `data/test_runs/`, never footage and
    #: never a database row. Pinned by the cleanup tests in test_testing_api.
    "/api/v1/testing/day",
    #: Compares a saved test run with booking records. A POST because the body may carry a
    #: booking CSV; it reads bookings and writes only the comparison onto the run's own result
    #: file. It never writes to a booking source.
    "/api/v1/testing/runs/{run_id}/compare",
    #: Read-only booking checks with a body; see BOOKING_READS_WITH_A_BODY above.
    *BOOKING_READS_WITH_A_BODY,
    #: A person saving or removing a camera connection: this system's own configuration, in
    #: the gitignored `configs/cameras.json`, written atomically with a history copy. Same
    #: category as `/roi`. URLs are never returned unmasked.
    "/api/v1/cameras",
    "/api/v1/cameras/{venue}/{camera_id}",
    #: Writes nothing: opens a stream, grabs one frame, reports what happened.
    "/api/v1/cameras/test",
    #: Writes nothing: a spot check of what a pitch's cameras see now. Nothing is recorded;
    #: the recorded history comes from the live schedule.
    "/api/v1/cameras/predict",
    #: Writes nothing: the slots a start time and slot length would generate.
    "/api/v1/cameras/schedule/preview",
    #: A person saving the capture schedule, through `schedule_editor.write_schedule` and its
    #: protections. Same category as `/schedule`.
    "/api/v1/cameras/schedule",
    #: Writes nothing: tests every saved camera in turn and reports which answered.
    "/api/v1/cameras/test-all",
    #: Re-judges a saved test run with different slot rules. It rewrites only that run's own
    #: result file under `data/test_runs/` - no footage, no database row, no booking source.
    "/api/v1/testing/runs/{run_id}/rejudge",
    #: The booking-day test on a recording already in the configured recordings folder. Reads
    #: the file in place (and never deletes it - it is the client's own), saves the per-slot
    #: result like `/api/v1/testing/day`.
    "/api/v1/testing/day/local",
    #: Sends one test message to one configured alert channel and logs it. Changes no
    #: product data; there is deliberately no route that saves the alert configuration.
    "/api/v1/alerts/test",
    #: Computes what needs attention now (reading the database read-only) and, only with
    #: `send=true`, sends it and updates the alert state and log files under `data/`.
    "/api/v1/alerts/check",
    #: Cancels a queued or running test job. A person stopping their own test; uploads are
    #: deleted as on any finish, local recordings are left untouched.
    "/api/v1/testing/jobs/{job_id}/cancel",
    #: Playgrounds (`playgrounds.py`): a person making, changing, switching or deleting a
    #: group of cameras and how often it is checked. The system's own configuration (the
    #: gitignored configs/playgrounds.json), never a client record; what is written afterwards
    #: are the system's own observations, like the worker's.
    "/api/v1/playgrounds",
    "/api/v1/playgrounds/{playground_id}",
    "/api/v1/playgrounds/{playground_id}/enabled",
    #: "Show pictures now": one check, asked for by a person; the pictures are not kept.
    "/api/v1/playgrounds/{playground_id}/look",
    #: Measures this computer's speed and saves it (configs/computer_speed.json).
    "/api/v1/computer-speed",
}


def test_the_only_mutating_routes_are_a_person_acting() -> None:
    """Every write the API offers should be a person correcting or configuring the system,
    never the system acting. If this fails, a new mutating route needs a deliberate look -
    add it to ALLOWED_MUTATING with the reason, or reconsider the design."""
    mutating = {
        route
        for path in (SRC / "api").glob("*.py")
        for method, route in _http_methods(path)
        if method in {"post", "put", "patch", "delete"}
    }
    assert mutating <= ALLOWED_MUTATING, (
        f"unexpected mutating routes: {sorted(mutating - ALLOWED_MUTATING)}"
    )


def test_the_validate_route_really_writes_nothing() -> None:
    """`/schedule/validate` is on the allowlist as "a POST that writes nothing", and that
    claim is checked rather than believed - an allowlist entry justified by a property is only
    as good as the test for the property."""
    import json

    from pitch_occupancy.api import schedule_editor as se

    entry = {
        "venue_id": "venue_01", "start": "10:00", "duration_minutes": 60,
        "cameras": ["camera_A"],
    }
    if not se.DEFAULT_SCHEDULE.exists():
        pytest.skip("no schedule file")
    def history() -> list[str]:
        folder = se.history_dir()
        return sorted(q.name for q in folder.glob("*.json")) if folder.exists() else []

    before, history_before = se.DEFAULT_SCHEDULE.read_text(encoding="utf-8"), history()

    se.validate([entry])
    se.validate([{**entry, "start": "nonsense"}])

    assert se.DEFAULT_SCHEDULE.read_text(encoding="utf-8") == before
    assert history() == history_before
    json.loads(before)  # still readable


def test_the_database_layer_has_no_financial_table_or_column() -> None:
    money = re.compile(r"\b(price|amount|invoice|charge|billing|payment|penalty)\b", re.I)
    for name in ("schema.py", "store.py"):
        text = (SRC / "db" / name).read_text(encoding="utf-8")
        found = money.findall(text)
        assert not found, f"db/{name} names {sorted(set(found))}"


def test_reconciliation_is_never_attributed_to_a_person() -> None:
    """`reconcile.py` rule 3. Attributing a discrepancy to an individual adds nothing
    scientifically and a great deal of ethical exposure."""
    fields = set(Reconciliation.__dataclass_fields__)
    assert not (fields & {"entered_by", "staff_id", "person", "user"}), fields


def test_the_preview_saves_nothing() -> None:
    """`/roi/preview` is on the allowlist as "writes nothing at all", and that claim is
    checked rather than believed - an allowlist entry justified by a property is only as
    good as the test for the property, which is the rule `/schedule/validate` set.

    The risk here is specific: preview and save take the same polygon and differ by one
    call, so a preview that quietly persisted its argument would be an easy change to make
    and an invisible one to review. It would also be the worst kind of wrong - an operator
    trying boundaries to see which is right would be saving every one they tried.
    """
    import base64
    import io

    import cv2
    import numpy as np
    from fastapi.testclient import TestClient

    from pitch_occupancy.api.app import app
    from pitch_occupancy.vision import roi

    before = roi.load_all()
    ok, buf = cv2.imencode(".jpg", np.full((48, 48, 3), 90, np.uint8))
    assert ok

    TestClient(app).post("/api/v1/roi/preview", json={
        "image": base64.b64encode(io.BytesIO(buf).getvalue()).decode(),
        "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        # `explain=False` keeps this off the detector: the property under test is whether
        # anything is persisted, and loading a model to establish it would make a guard
        # about writes into the slowest test in the file.
        "explain": False,
    })
    assert roi.load_all() == before
