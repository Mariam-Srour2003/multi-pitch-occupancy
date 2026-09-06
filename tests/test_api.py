"""Dashboard API.

The endpoints are thin by design, so the tests focus on the contract an operator depends
on: evidence is reachable from a verdict, a correction never destroys the original, and
the anomaly list stays short enough to be read."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api.app import app
from pitch_occupancy.api.routes import get_conn
from pitch_occupancy.data.taxonomy import SlotStatus
from pitch_occupancy.db.schema import connect, initialise
from pitch_occupancy.db.store import Sample, record_slot
from pitch_occupancy.slots.aggregate import SlotVerdict


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    # TestClient runs the app on another thread; the connection is shared deliberately
    conn = connect(tmp_path / "api.db", same_thread=False)
    initialise(conn)
    conn.execute("INSERT INTO venues VALUES ('v1', 'Venue One')")
    conn.execute("INSERT INTO fields VALUES ('f1', 'v1', 'Pitch 1')")
    conn.execute("INSERT INTO cameras VALUES ('camA', 'f1', 'A', NULL, NULL)")
    conn.execute("INSERT INTO rental_slots VALUES ('s1', 'f1', '2026-07-11', '10:00', '11:00')")
    conn.execute("INSERT INTO rental_slots VALUES ('s2', 'f1', '2026-07-11', '11:00', '12:00')")
    conn.commit()

    record_slot(
        conn, "s1", SlotVerdict(SlotStatus.USED, "play in 90%", 0.9, 0.1, 0.0, 60, 0.94),
        [Sample("camA", i, "C2_ACTIVE_PLAY", 0.94, f"2026-07-11T10:0{i}:00") for i in range(3)],
        model_key="dinov2", evidence_paths=["a.jpg", "b.jpg", "c.jpg"],
    )
    record_slot(
        conn, "s2", SlotVerdict(SlotStatus.REVIEW, "intermittent", 0.2, 0.6, 0.2, 60, 0.55),
        [Sample("camA", 0, "C1_EMPTY", 0.55, "2026-07-11T11:00:00")],
        model_key="dinov2",
    )
    conn.execute(
        """INSERT INTO reconciliations
           (slot_id, anomaly, severity, explanation, created_at)
           VALUES ('s1', 'UNBOOKED_USAGE', 'serious', 'no booking on record', '2026-07-11')"""
    )
    conn.execute(
        """INSERT INTO reconciliations
           (slot_id, anomaly, severity, explanation, created_at)
           VALUES ('s2', 'NEEDS_REVIEW', 'info', 'verdict is REVIEW', '2026-07-11')"""
    )
    conn.commit()

    app.dependency_overrides[get_conn] = lambda: conn
    yield TestClient(app)
    app.dependency_overrides.clear()
    conn.close()


# --- meters -----------------------------------------------------------------


def test_meters_counts_verdicts(client) -> None:
    m = client.get("/api/v1/meters").json()
    assert m["slots_evaluated"] == 2
    assert m["used"] == 1
    assert m["review"] == 1


def test_review_rate_is_reported(client) -> None:
    """The number that decides how much manual work the facility is buying."""
    assert client.get("/api/v1/meters").json()["review_rate"] == pytest.approx(0.5)


def test_meters_on_an_empty_database_does_not_divide_by_zero(tmp_path: Path) -> None:
    conn = connect(tmp_path / "empty.db", same_thread=False)
    initialise(conn)
    app.dependency_overrides[get_conn] = lambda: conn
    try:
        m = TestClient(app).get("/api/v1/meters").json()
        assert m["slots_evaluated"] == 0
        assert m["review_rate"] == 0.0
    finally:
        app.dependency_overrides.clear()


# --- slots and evidence -----------------------------------------------------


def test_lists_slots(client) -> None:
    assert len(client.get("/api/v1/slots").json()) == 2


def test_filters_by_status(client) -> None:
    rows = client.get("/api/v1/slots", params={"status": "REVIEW"}).json()
    assert [r["slot_id"] for r in rows] == ["s2"]


def test_evidence_reaches_the_samples_behind_a_verdict(client) -> None:
    ev = client.get("/api/v1/slots/s1/evidence").json()
    assert ev["evidence_paths"] == ["a.jpg", "b.jpg", "c.jpg"]
    assert len(ev["samples"]) == 3
    assert ev["samples"][0]["camera_id"] == "camA"


def test_evidence_for_an_unknown_slot_is_404(client) -> None:
    assert client.get("/api/v1/slots/ghost/evidence").status_code == 404


# --- override ---------------------------------------------------------------


def test_override_records_the_correction_without_losing_the_original(client) -> None:
    r = client.post(
        "/api/v1/slots/s1/override",
        json={"status": "NOTUSED", "operator": "maria", "note": "rain"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "USED"           # what the model said
    assert body["override_status"] == "NOTUSED"  # what the human said
    assert body["is_overridden"]


def test_override_requires_an_operator(client) -> None:
    """Corrections are audit records; an anonymous one is not auditable."""
    r = client.post("/api/v1/slots/s1/override", json={"status": "NOTUSED", "operator": ""})
    assert r.status_code == 422


def test_override_rejects_an_invalid_status(client) -> None:
    r = client.post("/api/v1/slots/s1/override", json={"status": "MAYBE", "operator": "m"})
    assert r.status_code == 422


def test_override_of_an_unknown_slot_is_404(client) -> None:
    r = client.post("/api/v1/slots/ghost/override", json={"status": "USED", "operator": "m"})
    assert r.status_code == 404


# --- anomalies --------------------------------------------------------------


def test_anomaly_list_excludes_agreements_and_reviews(client) -> None:
    """Padding the list with agreements is how an operator learns to ignore it."""
    rows = client.get("/api/v1/anomalies").json()
    assert [r["anomaly"] for r in rows] == ["UNBOOKED_USAGE"]


def test_field_day_matrix_includes_slots_without_a_verdict(client) -> None:
    """A slot with no evaluation is a gap the operator should see, not a missing row."""
    rows = client.get("/api/v1/fields/f1/day/2026-07-11").json()
    assert len(rows) == 2
    assert [r["start_time"] for r in rows] == ["10:00", "11:00"]


def test_health_still_works(client) -> None:
    assert client.get("/health").json()["status"] == "ok"
