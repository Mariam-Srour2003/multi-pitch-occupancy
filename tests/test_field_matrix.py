"""The field matrix, and what the evidence endpoint sends (WP6-T6).

The matrix is the operator's overview: every pitch, hour by hour, for one day. The case that
matters in it is the one that is easy to draw wrongly — a scheduled slot with no verdict. An
hour nobody looked at and an hour observed to be empty are different claims, and a grid that
gives them the same chip hides exactly what reconciliation exists to catch.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api import routes
from pitch_occupancy.api.app import app
from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.db.schema import connect, initialise
from pitch_occupancy.db.store import Sample, override_verdict, record_slot
from pitch_occupancy.slots.aggregate import aggregate_slot

DAY = "2026-07-11"


@pytest.fixture()
def client():
    """Two pitches: field_01 has an evaluated slot and an overridden one, field_02 has a
    scheduled slot that was never evaluated."""
    from pitch_occupancy.db.store import ensure_slot

    conn = connect(":memory:", same_thread=False)
    initialise(conn)

    def add(slot_id, field, start, verdict=None, states=None):
        ensure_slot(conn, slot_id=slot_id, venue_id="venue_01", field_id=field,
                    cameras=["camera_A"], slot_date=DAY, start_time=start,
                    end_time="11:00")
        if states is not None:
            record_slot(conn, slot_id, aggregate_slot(states),
                        samples=[Sample("camera_A", 0, states[0].value, 0.9,
                                        f"{DAY}T{start}:00")],
                        model_key="test")

    add("venue_01_2026-07-11_1000", "field_01", "10:00", states=[Class3.ACTIVE_PLAY] * 10)
    add("venue_01_2026-07-11_1100", "field_01", "11:00", states=[Class3.EMPTY] * 10)
    add("venue_01_2026-07-11_1200", "field_02", "12:00")  # scheduled, never evaluated
    override_verdict(conn, "venue_01_2026-07-11_1100", status=SlotStatus.USED,
                     operator="a.demir", note="staff say it ran")
    conn.commit()

    app.dependency_overrides[routes.get_conn] = lambda: conn
    yield TestClient(app)
    app.dependency_overrides.clear()


def matrix(client) -> list[dict]:
    response = client.get(f"/api/v1/fields/day/{DAY}")
    assert response.status_code == 200
    return response.json()


# --- the endpoint ----------------------------------------------------------------------


def test_it_returns_one_entry_per_field(client) -> None:
    assert [f["field_id"] for f in matrix(client)] == ["field_01", "field_02"]


def test_slots_are_grouped_under_their_field_in_time_order(client) -> None:
    """The shape an operator reads is one row per pitch, so the grouping is the response's
    job rather than the browser's."""
    field_01 = matrix(client)[0]
    assert [s["start_time"] for s in field_01["slots"]] == ["10:00", "11:00"]


def test_a_scheduled_but_unevaluated_slot_is_present_with_a_null_status(client) -> None:
    """The case the matrix exists for. Dropping it from the response would make an
    unobserved hour invisible - and an unobserved hour is what reconciliation catches."""
    field_02 = next(f for f in matrix(client) if f["field_id"] == "field_02")
    assert len(field_02["slots"]) == 1
    assert field_02["slots"][0]["status"] is None


def test_an_override_is_reported_alongside_the_model_verdict(client) -> None:
    """Never instead of it. The audit record is that a human disagreed with something."""
    slot = matrix(client)[0]["slots"][1]
    assert slot["status"] == "NOTUSED"
    assert slot["is_overridden"] == 1
    assert slot["override_status"] == "USED"


def test_confidence_comes_through_for_the_chips(client) -> None:
    assert matrix(client)[0]["slots"][0]["mean_confidence"] is not None


def test_a_day_with_nothing_scheduled_is_an_empty_list_not_an_error(client) -> None:
    assert client.get("/api/v1/fields/day/2026-01-01").json() == []


def test_a_malformed_day_is_rejected(client) -> None:
    assert client.get("/api/v1/fields/day/not-a-date").status_code == 422


# --- what the page draws ----------------------------------------------------------------


def test_the_matrix_section_is_rendered(client) -> None:
    page = client.get("/client").text
    assert "Field matrix" in page
    assert "/fields/day/" in page


def test_an_unevaluated_slot_is_drawn_hollow_rather_than_neutral(client) -> None:
    """A filled grey chip would read as "we looked and it was quiet". The dashed outline and
    the words "no verdict" say the opposite, which is the true thing."""
    page = client.get("/client").text
    assert "chip none" in page
    assert "no verdict" in page
    assert "border:1px dashed" in page.replace(" ", " ")


# --- the evidence endpoint must not leak server paths ------------------------------------


def test_the_evidence_endpoint_sends_names_not_absolute_paths(client) -> None:
    """It used to send whatever was stored, which is an absolute path on the server. That
    leaks the deployment's directory layout into a page for no benefit: the images are
    fetched by index, never by path."""
    import json

    conn = app.dependency_overrides[routes.get_conn]()
    conn.execute(
        "UPDATE slot_evaluations SET evidence_paths = ? WHERE slot_id = ?",
        (json.dumps(["/srv/pitch/data/interim/evidence/s/minute_003_camera_A.jpg"]),
         "venue_01_2026-07-11_1000"),
    )
    conn.commit()
    body = client.get("/api/v1/slots/venue_01_2026-07-11_1000/evidence").json()
    assert body["evidence_paths"] == ["minute_003_camera_A.jpg"]
    assert "/srv/" not in str(body)
