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

from pitch_occupancy.slots.authority import (
    FORBIDDEN,
    MAX_SEVERITY_WITHOUT_HUMAN,
    PERMITTED,
    Action,
    Advisory,
    advise,
)
from pitch_occupancy.slots.reconcile import Anomaly, Booking, Reconciliation, reconcile
from pitch_occupancy.data.taxonomy import SlotStatus

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


def test_the_only_mutating_route_is_a_human_override() -> None:
    """Every write the API offers should be a person correcting the system, never the
    system acting. If this fails, a new mutating route needs a deliberate look."""
    mutating = {
        route
        for path in (SRC / "api").glob("*.py")
        for method, route in _http_methods(path)
        if method in {"post", "put", "patch", "delete"}
    }
    assert mutating <= {"/slots/{slot_id}/override", "/preprocess"}, (
        f"unexpected mutating routes: {sorted(mutating - {'/slots/{slot_id}/override', '/preprocess'})}"
    )


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
