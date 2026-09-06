# The database

`data/db/pitch_monitor.db` — SQLite, eight tables, ~144 KB seeded. Stdlib `sqlite3` rather
than an ORM: the schema is small, the queries are straightforward, and the deployment
target is a Mini-PC that should not carry a dependency it does not need.

Schema in `src/pitch_occupancy/db/schema.py`, access in `store.py`, and
`uv run pitch seed` fills it from the real recorded slots.

---

## What it holds, and why

The database exists to make a verdict **auditable**. Anyone disputing a slot's outcome must
be able to see the observations behind it, who corrected it if anyone did, and what the
booking record claimed at the time. Every design choice below follows from that.

```
venues ──< fields ──< cameras          the physical estate
                 │         │
                 └──< rental_slots ──< frame_samples      what was observed
                             ├──── slot_evaluations       what was decided
                             └──< reconciliations         what disagreed
         bookings                                          what the records claim
```

---

## Tables

### `venues` · `fields` · `cameras` — the estate

| table | rows seeded | holds |
|---|---|---|
| `venues` | 1 | facility id and name |
| `fields` | 1 | a pitch, belonging to a venue |
| `cameras` | 2 | **exactly two per field**, side `A` or `B`, with RTSP URL and ROI polygon |

**`cameras` is its own table, and this matters.** The original blueprint put
`primary_camera_rtsp` / `secondary_camera_rtsp` columns on the field. Those columns cannot
carry a per-camera ROI polygon, cannot express a third camera, and make "which half did this
frame come from" unanswerable — which is the question the entire two-camera fusion argument
depends on.

`UNIQUE (field_id, side)` enforces two cameras per pitch. That constraint **caught a real
bug**: the seeder initially tried to insert four camera tags for one field, because the four
source recordings are two physical cameras across two days. It failed loudly on a foreign
key rather than silently creating phantom cameras and fusing the wrong halves together.

### `rental_slots` — the schedule

One row per bookable slot: field, date, start and end. `UNIQUE (field_id, slot_date,
start_time)` so a slot cannot be double-registered.

### `frame_samples` — what was observed

237 rows seeded. **One row per camera per sampled minute**: the predicted class, the
confidence, when it was captured, and the image path.

This is the evidence layer. It is kept alongside the decision rather than discarded after
aggregation, because an operator disputing a verdict needs to see the minutes behind it —
"play in 100% of 60 samples" is only checkable if the 60 samples still exist.

`UNIQUE (slot_id, camera_id, minute_index)` prevents a minute being recorded twice, which
would silently reweight a ratio.

### `slot_evaluations` — what was decided

One row per slot, 16 columns. The verdict (`USED` / `NOTUSED` / `REVIEW`), the reason in
plain language, the three class ratios, sample count, mean confidence, the three evidence
image paths, when it was evaluated and by which model.

Then the override fields: `is_overridden`, `override_status`, `override_by`, `override_at`,
`override_note`.

**A correction never overwrites the model's verdict — both are kept.** The disagreement is
the signal: it is how model quality is monitored in production, and a corrected slot is a
labelled slot. Those corrections are currently the cheapest route to the real slot labels
STAN needs, since only two exist.

### `bookings` — what the records claim

Field, date, start, customer reference, whether it was booked, whether staff recorded it as
used, whether it was a maintenance window, who entered it, and the import source.

`staff_recorded_used` is deliberately **nullable**. `NULL` means *no record exists*, which
is a different thing from *recorded as unused* — and reconciliation treats it differently,
routing to review rather than raising a discrepancy.

**Read-only with respect to client systems.** Nothing in this project writes back to a
facility's booking database.

`entered_by` is stored for the operator's own use and is **never aggregated** by anything
here. Per-staff discrepancy reporting is out of scope — it adds no scientific value and
considerable ethical exposure. See `thesis/ethics.md`.

### `reconciliations` — what disagreed

One row per slot where the records and the observation were compared: the anomaly type, its
severity, an explanation, when it was created, and who resolved it.

Anomaly types: `CONSISTENT`, `NO_SHOW_OR_OVERRECORDED`, `PLAYED_NOT_RECORDED`,
`UNBOOKED_USAGE`, `BLOCKED_SLOT_SOLD`, `NEEDS_REVIEW`.

`NEEDS_REVIEW` is stored like the others but is **not an anomaly**. The system is not
permitted to convert its own uncertainty into someone else's error, so an unsure verdict
routes to an inspector rather than appearing in the anomalies queue.

---

## Guarantees the code enforces

**A verdict is never written without its evidence.** `record_slot()` writes the samples and
the evaluation in one transaction. A test drives a foreign-key failure mid-write and asserts
that *neither* survives — a verdict with half its evidence missing is worse than no verdict,
because it is exactly what an operator would dispute and nobody could reconstruct.

**Foreign keys are on.** `PRAGMA foreign_keys = ON` at every connection, so an orphaned
sample or a phantom camera fails at write time rather than surfacing as a confusing query
result later.

**Statuses and sides are constrained in the schema**, not just in Python: `CHECK (status IN
('USED','NOTUSED','REVIEW'))`, `CHECK (side IN ('A','B'))`. A bug in the application cannot
put an invalid state on disk.

**Connections are thread-bound by default.** `connect(..., same_thread=False)` exists only
for deliberately sharing one connection, such as a test client running the app on another
thread. The API opens one connection per request in the thread that serves it.

---

## Working with it

```bash
uv run pitch seed          # rebuild from the labelled frames
uv run pitch serve         # then http://127.0.0.1:8000/client
```

```bash
sqlite3 data/db/pitch_monitor.db
```

```sql
-- what needs acting on
SELECT slot_id, anomaly, severity FROM reconciliations
WHERE anomaly NOT IN ('CONSISTENT', 'NEEDS_REVIEW') AND resolved_at IS NULL;

-- the minutes behind a verdict
SELECT minute_index, camera_id, predicted, confidence
FROM frame_samples WHERE slot_id = ? ORDER BY minute_index;

-- corrections, i.e. free slot labels
SELECT slot_id, status, override_status, override_by FROM slot_evaluations
WHERE is_overridden = 1;
```

The database is **regenerable** — `pitch seed` rebuilds it from the labelled frames — so it
is gitignored along with the rest of `data/`. The frames are not regenerable; the database
is.
