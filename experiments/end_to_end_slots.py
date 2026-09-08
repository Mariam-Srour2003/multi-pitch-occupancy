"""End-to-end decision layer on the two real recorded slots (WP6-T2 accept criterion).

Chains the pieces that turn predictions into a billing decision:

    per-frame class -> two-camera fusion -> slot aggregation -> reconciliation

Run on the **ground-truth labels** rather than model output, deliberately. The question
here is whether the decision layer is correct given correct perception - if fusion or
aggregation is wrong, no classifier can rescue it, and mixing model error into this run
would hide that. Model-driven runs come with the scheduler (WP6-T2).

Per-minute states are reconstructed from the manifest: frames carry a camera tag and an
offset in seconds, so grouping by minute and fusing the two cameras rebuilds exactly what
the live sampler would have seen.

Bookings are synthetic fixtures covering the reconciliation matrix, since no booking
export exists yet (WP6-T4). Their purpose is to prove each anomaly type is reachable, not
to measure precision - that needs adjudicated slots (WP6-T11).

    uv run python experiments/end_to_end_slots.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.slots.aggregate import Thresholds, aggregate_slot
from pitch_occupancy.slots.fusion import fuse
from pitch_occupancy.slots.reconcile import Booking, reconcile

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def per_minute_states(rows) -> dict[str, list[Class3]]:
    """Rebuild each slot's fused per-minute sequence from the labelled frames."""
    # slot -> minute -> camera -> class
    grid: dict[str, dict[int, dict[str, Class3]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.source == "clip":
            continue  # highlights have no slot structure
        grid[r.slot_id][r.t_s // 60][r.camera] = Class3(r.class3)

    out: dict[str, list[Class3]] = {}
    for slot, minutes in grid.items():
        seq = []
        for minute in sorted(minutes):
            obs = {cam: (cls, 1.0) for cam, cls in minutes[minute].items()}
            seq.append(fuse(obs).state)
        out[slot] = seq
    return out


def main() -> None:
    rows = read_manifest(DATASET / "manifest.csv")
    slots = per_minute_states(rows)
    print(f"reconstructed {len(slots)} real slots from labelled frames\n")

    # Synthetic bookings exercising the matrix. The morning slot really was unused and the
    # evening one really was played, so these are the records a facility might hold.
    fixtures = {
        "venue_01_2026-07-11_1000": [
            ("booked, staff say used", Booking("field_01", "2026-07-11", "10:00", True, True)),
            ("booked, staff say unused", Booking("field_01", "2026-07-11", "10:00", True, False)),
        ],
        "venue_01_2026-07-12_2030": [
            ("booked, staff say used", Booking("field_01", "2026-07-12", "20:30", True, True)),
            ("not booked at all", Booking("field_01", "2026-07-12", "20:30", False)),
        ],
    }

    records = []
    for slot, states in sorted(slots.items()):
        verdict = aggregate_slot(states, thresholds=Thresholds(review_below_confidence=0.0))
        print(f"=== {slot} ===")
        print(f"  {len(states)} fused minutes")
        print(f"  {verdict}")

        for label, bk in fixtures.get(slot, []):
            rec = reconcile(bk, verdict.status)
            flag = "ANOMALY" if rec.is_anomaly else "ok     "
            print(f"    [{flag}] {label:<26} -> {rec.anomaly.value} ({rec.severity.value})")
            records.append(
                {
                    "slot": slot, "n_minutes": len(states),
                    "play_ratio": round(verdict.play_ratio, 4),
                    "empty_ratio": round(verdict.empty_ratio, 4),
                    "verdict": verdict.status.value,
                    "booking_case": label,
                    "anomaly": rec.anomaly.value,
                    "severity": rec.severity.value,
                }
            )
        print()

    expected = {
        "venue_01_2026-07-11_1000": SlotStatus.NOTUSED,
        "venue_01_2026-07-12_2030": SlotStatus.USED,
    }
    print("=== accept check ===")
    ok = True
    for slot, want in expected.items():
        got = aggregate_slot(slots[slot]).status
        mark = "PASS" if got is want else "FAIL"
        ok &= got is want
        print(f"  [{mark}] {slot}: expected {want.value}, got {got.value}")
    print(f"\n{'all slots correct' if ok else 'MISMATCH - the decision layer is wrong'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "end_to_end_slots.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"wrote {out}")

    record(
        "end-to-end slots",
        "`python experiments/end_to_end_slots.py`",
        f"`{out.name}`",
        f"{len(slots)} real slots, ground-truth labels",
    )


if __name__ == "__main__":
    main()
