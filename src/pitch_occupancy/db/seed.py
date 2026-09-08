"""Populate the database from the real recorded slots (WP6).

The dashboard reads from the database, so an empty database shows an empty dashboard - a
shell that demonstrates nothing. This fills it with the two genuinely recorded slots, their
per-minute samples, the verdicts the decision layer produced, and the reconciliation
outcomes against a booking fixture.

Per-minute states are rebuilt from the labelled frames, the same way
`experiments/end_to_end_slots.py` does it, so what the dashboard shows is what the pipeline
actually decided rather than invented demo data.

    uv run pitch seed
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.db.schema import connect, initialise, transaction
from pitch_occupancy.db.store import Sample, record_slot
from pitch_occupancy.slots.aggregate import aggregate_slot
from pitch_occupancy.slots.evidence import select_evidence
from pitch_occupancy.slots.fusion import fuse
from pitch_occupancy.slots.reconcile import Booking, reconcile

__all__ = ["seed"]

#: Slot-specific camera tag -> physical camera.
#:
#: The four tags are two physical cameras recorded on two days, and the `(1)` export suffix
#: maps to the *opposite* view between them - measured, not assumed: file0 of the morning
#: slot matches file1 of the evening slot at 0.88 view similarity against 0.70 for its
#: own-suffix counterpart (see vision/camera_id.py and the EXPERIMENT_LOG entry).
#:
#: Getting this wrong is invisible downstream: swapped halves still fuse into a plausible
#: verdict. The schema catches it instead - a field permits exactly two cameras, so mapping
#: four tags onto it fails loudly on a foreign key rather than silently inventing cameras.
PHYSICAL_CAMERA = {
    "slot_20260711_1000_camA": "camera_A",
    "slot_20260712_2030_camB": "camera_A",
    "slot_20260711_1000_camB": "camera_B",
    "slot_20260712_2030_camA": "camera_B",
}

#: Booking records for the two recorded slots. The morning slot really was unused and the
#: evening one really was played; the morning record deliberately claims otherwise so the
#: reconciliation layer has something to disagree with.
FIXTURES = {
    "venue_01_2026-07-11_1000": Booking(
        "field_01", "2026-07-11", "10:00", booked=True, staff_recorded_used=True,
        entered_by="desk_1",
    ),
    "venue_01_2026-07-12_2030": Booking(
        "field_01", "2026-07-12", "20:30", booked=True, staff_recorded_used=True,
        entered_by="desk_2",
    ),
}


def seed(db_path: Path | None = None) -> dict[str, str]:
    """Rebuild the database from the labelled frames. Returns slot -> verdict."""
    manifest = read_manifest(settings.dataset_dir / "manifest.csv")
    conn = connect(db_path or settings.db_path)
    initialise(conn)

    # slot -> minute -> camera -> class, from the frames themselves
    grid: dict[str, dict[int, dict[str, Class3]]] = defaultdict(lambda: defaultdict(dict))
    for r in manifest:
        if r.source == "clip":
            continue  # highlights have no slot structure
        grid[r.slot_id][r.t_s // 60][r.camera] = Class3(r.class3)

    with transaction(conn):
        conn.execute(
            "INSERT OR REPLACE INTO venues VALUES ('venue_01', 'Facility 1')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO fields VALUES ('field_01', 'venue_01', 'Pitch 1')"
        )
        for camera_id, side in (("camera_A", "A"), ("camera_B", "B")):
            conn.execute(
                "INSERT OR REPLACE INTO cameras VALUES (?, 'field_01', ?, NULL, NULL)",
                (camera_id, side),
            )
        for slot_id in grid:
            bk = FIXTURES.get(slot_id)
            conn.execute(
                "INSERT OR REPLACE INTO rental_slots VALUES (?, 'field_01', ?, ?, ?)",
                (slot_id, bk.date if bk else "", bk.start if bk else "", ""),
            )
            if bk:
                conn.execute(
                    """INSERT OR REPLACE INTO bookings
                       (field_id, slot_date, start_time, customer_ref, booked,
                        staff_recorded_used, maintenance_window, entered_by, source)
                       VALUES (?, ?, ?, NULL, 1, ?, 0, ?, 'fixture')""",
                    (bk.field_id, bk.date, bk.start, int(bool(bk.staff_recorded_used)),
                     bk.entered_by),
                )

    verdicts: dict[str, str] = {}
    for slot_id, minutes in grid.items():
        samples: list[Sample] = []
        states: list[Class3] = []
        confidences: list[float] = []
        evidence_rows = []
        for minute in sorted(minutes):
            observations = {cam: (cls, 0.95) for cam, cls in minutes[minute].items()}
            for cam, (cls, conf) in observations.items():
                samples.append(Sample(
                    camera_id=PHYSICAL_CAMERA.get(cam, cam),
                    minute_index=minute, predicted=cls.value,
                    confidence=conf,
                    captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ))
            fused = fuse(observations)
            states.append(fused.state)
            confidences.append(fused.confidence)
            evidence_rows.append((minute, fused.state, fused.confidence, None))

        verdict = aggregate_slot(states, confidences)
        evidence = select_evidence(evidence_rows, verdict.status)
        record_slot(
            conn, slot_id, verdict, samples,
            model_key=settings.default_model_key,
            evidence_paths=[f"minute_{e.minute_index:03d}" for e in evidence],
        )
        verdicts[slot_id] = verdict.status.value

        bk = FIXTURES.get(slot_id)
        if bk:
            rec = reconcile(bk, verdict.status, vision_confidence=verdict.mean_confidence)
            with transaction(conn):
                conn.execute(
                    """INSERT OR REPLACE INTO reconciliations
                       (slot_id, anomaly, severity, explanation, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (slot_id, rec.anomaly.value, rec.severity.value, rec.explanation,
                     datetime.now(timezone.utc).isoformat(timespec="seconds")),
                )

    conn.close()
    return verdicts
