"""The "never bill" constraint, checked rather than stated (WP6-T12).

Reconciliation can implicate a named member of staff, so the design constraint is that the
system produces decision support and takes no automated financial action. That sentence is
worth very little at an examination on its own, so these tests walk the actual surface: the
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
    """(method, route) for every FastAPI decorator in a module."""
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.FunctionDef):
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


def test_no_http_route_writes_to_a_booking() -> None:
    """Booking data is read-only: the audit compares against the record, it does not amend
    it. A write route would make the system a participant in what it is auditing."""
    offenders = [
        f"{path.name}: {method.upper()} {route}"
        for path in (SRC / "api").glob("*.py")
        for method, route in _http_methods(path)
        if method in {"post", "put", "patch", "delete"} and "booking" in route.lower()
    ]
    assert not offenders, f"a booking write route appeared: {offenders}"


#: Routes allowed to use a mutating HTTP verb, each with the reason it is not the system
#: acting. Adding to this set is the deliberate look the test below asks for - and the reason
#: belongs here, next to the entry, rather than in a commit message nobody re-reads.
ALLOWED_MUTATING = {
    #: A person correcting the system. The model's verdict is retained beside the correction.
    "/slots/{slot_id}/override",
    #: The preprocessing search control, which acts on this project's own experiment state.
    "/preprocess",
    #: An operator editing this system's *own* capture configuration (WP6-T6). Not the same
    #: category as writing to a client system: `bookings.py` has no write method at all,
    #: because a booking sheet is the facility's financial record. This is defensible, and it
    #: is still the most dangerous route here - see `api/schedule_editor.py` for the three
    #: protections it carries.
    "/schedule",
    #: A POST that writes nothing. It carries a proposed schedule in its body, which is why
    #: it cannot be a GET, and returns whether the scheduler would accept it. Pinned by
    #: `test_the_validate_route_really_writes_nothing` below rather than trusted.
    "/schedule/validate",
    #: The clip reviewer (WP6-T6). A POST because the body is a video, not because it
    #: persists one: the upload is streamed to a temp file, read, and deleted in a
    #: `finally`, and no database row is written. Same category as `/schedule/validate`,
    #: and pinned the same way by `test_the_clip_route_keeps_nothing` below.
    "/clip/analyse",
    #: The same analysis streamed step by step with its evidence maps (WP4-T5). A POST for
    #: the same reason and with the same guarantee - the upload is removed in the streaming
    #: generator's `finally` rather than the handler's, because the body outlives the
    #: handler. Pinned by `test_the_walkthrough_deletes_the_upload_after_streaming`.
    "/clip/walkthrough",
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
