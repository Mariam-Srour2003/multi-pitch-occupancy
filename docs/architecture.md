# How the code is organised

A CPU-only service. Cameras are opened, YOLO counts the people standing inside each pitch's
drawn boundary, and a small rule turns the count into *empty*, *playing*, *other use* or
*not sure*. Results are stored in SQLite and shown on web pages.

## Layers

```
api/            web pages and JSON endpoints (FastAPI). Thin: reads requests, calls below.
  access.py       sign-in, host check, cross-site refusal, security headers
  templates/      each page's HTML, CSS and JS, filled by templating.render
  static/         style sheet, fonts and scripts every page shares (layout.py)
playgrounds/    pitches checked on a timer inside the web server (the main live path)
  store / capacity / check / snapshots / runner / sessions
worker.py       slot-schedule recorder (separate process) + scheduler.py, timetable.py
assist/         optional free vision-model second opinion on doubtful checks
pipeline.py     builds the one shared detector + rule pipeline (`pipeline.shared()`)
vision/         detection and the rule
  detector.py     YOLO registry and loading (weights never downloaded implicitly)
  counting.py     who stands inside the boundary (by the foot of the box)
  movement.py     how many people moved between burst frames
  rules.py        the decision table; numbers in configs/rules.json
  pitch_classifier.py  a burst of frames from one camera -> one verdict
  roi.py          pitch boundaries (configs/roi.json)
  overlay.py, redact.py  drawing and privacy blur for anything shown or sent
slots/          minutes -> slot verdicts (aggregate), cameras -> pitch (fusion), bookings
db/             SQLite schema (schema.py) and queries (store.py)
config.py       every setting (PITCH_* environment variables or .env)
jsonstore.py    read / atomic write of the operator's JSON files under configs/
cameras.py, alerts.py, bookings.py, booking_sources.py, reports.py, retention.py, backup.py
```

Dependencies point downward. `api` may import anything. Domain modules (`playgrounds`,
`worker`, `slots`, `vision`, `assist`) never import from `api`.

## Where the numbers live

| what | where |
|---|---|
| detector, image size, confidences, play threshold, rule clauses | `configs/rules.json`, measured with `scripts/evaluate_rules.py` |
| server, paths, limits, sign-in | `config.py` (`PITCH_*` variables, `.env`) |
| cameras (with passwords) | `configs/cameras.json` (gitignored) |
| pitch boundaries | `configs/roi.json` |
| playgrounds | `configs/playgrounds.json` (gitignored) |
| second opinion | `configs/assist.json` (gitignored), keys in `.env` |
| alerts | `configs/alerts.json` (gitignored) |

## How to add…

**A rule clause.**
1. Add the threshold to `RuleConfig` in `vision/rules.py`, defaulting to `None` (= off), and
   validate it in `_validate_rule_values`.
2. Read it in `decide` and record in the trace why it fired.
3. If it needs a new measurement, compute it in `pitch_classifier.observe` (one camera) and
   carry it through `slots/fusion.fuse_pitch` (whole pitch). Otherwise the stored verdict
   ignores it.
4. Measure it with `scripts/evaluate_rules.py`, then set it in `configs/rules.json` with the
   numbers in its `_comment`.
5. Test it in `tests/test_rules.py` or `tests/test_movement_and_edges.py`.

**A detector.** Add an entry to `DETECTORS` in `vision/detector.py`. Run
`uv run pitch fetch-weights <key>`, then
`uv run python scripts/evaluate_rules.py detect --detector <key> --imgsz 960` and `score`.
Switch `configs/rules.json` only if the new detector wins on accuracy per CPU second.

**A free vision provider.** Add a `Provider` to `assist/providers.py`. Its endpoint must be
OpenAI-compatible, and the limits you enter must be the provider's published *free* limits.
Then add it to `providers` in `configs/assist.json`.

**A page.**
1. Create a router module in `api/`. Use `layout.page(...)` for the shell and the shared JS
   helpers (escape every value placed in HTML).
   Put the body, CSS and script in `api/templates/` and load them with
   `templating.template(...)` (or `render(...)` when `{{slot}}`s are filled from Python); keep
   markup out of the module. Pages inline these files, so nothing extra is served and the
   sign-in and security headers cover them like any other response.
2. Open the database with `db.schema.open_db()` / `open_readonly()`.
3. Register the router in `api/app.py` and the link in `api/nav.py`.
4. Writes must be POST/PUT/DELETE. `RequestGuard` refuses them when they come from another
   site.

**An operator JSON file.** Use `jsonstore.read_json` / `write_json_atomic`. Keep validation
in the owning module, add the file to `.gitignore` if it holds secrets, and redirect it in
`tests/conftest.py`.

## Conventions

- **Tests never touch the installation.** `tests/conftest.py` points the database and every
  operator file at a temporary folder. A test that needs a specific file sets its own.
- **Docstrings say what the code does and any non-obvious constraint.** Measured facts behind
  a constant may stay as one line beside it. History ("used to", dates, incident stories)
  belongs in commit messages.
- **Fail closed.** A detector that cannot run gives *not sure* (UNCERTAIN), never *empty*.
  Unknown is never zero.
- **Privacy.** Pictures that leave the machine or reach a page go through `vision/redact.py`.
  Camera passwords are masked everywhere and are only sent back to the address they were
  saved for.
- **CI** (`.github/workflows/ci.yml`) runs ruff, the tests and a dependency audit on every
  push and pull request. Run the same locally:
  `uv run ruff check src tests scripts` and `uv run python -m pytest -q`.
