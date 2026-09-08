"""One entry per experiment in `results/EXPERIMENT_LOG.md`, written once.

The log is the index a reader scans to find which run produced which CSV, so it is a
first-class artefact: the Findings tab renders it, and `tests/test_frontends.py` asserts it
renders. Thirteen experiments were each appending to it with their own copy of

    with (RESULTS / "EXPERIMENT_LOG.md").open("a") as fh:
        fh.write(f"\\n- {date} | task | command | file | note\\n")

which has two problems that only show up in use. **Every rerun adds another identical
line** — debugging one script during a single session put three copies of the same entry in
the file, and an index nobody trusts is an index nobody reads. And thirteen copies of a
format string drift: two separator styles were already in the committed log.

So the write lives here. It is idempotent on the exact line, which is the behaviour a log of
*what was run* wants — rerunning an experiment on the same day to reproduce it is not a new
result, and a run that produces a different note (a different fold count, a changed verdict)
writes a new line because its text differs.

Deliberately in the library rather than in `experiments/`: the dependency points one way,
and anything two experiments both need belongs on the library side of it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pitch_occupancy.config import settings

__all__ = ["DEFAULT_LOG", "entry_line", "record"]

DEFAULT_LOG = settings.results_dir / "EXPERIMENT_LOG.md"

#: The separator between fields. One style, because the file had two.
SEP = " | "


def entry_line(*fields: str, date: str | None = None) -> str:
    """The line an entry would write, without touching the file.

    Separated out so a caller — or a test — can ask what would be written. Fields are
    stripped and empties dropped, so an optional field left blank does not leave a dangling
    separator.
    """
    stamp = date or f"{datetime.now(timezone.utc):%Y-%m-%d}"
    parts = [stamp, *(f.strip() for f in fields if f and f.strip())]
    return "- " + SEP.join(parts)


def record(*fields: str, path: Path | None = None, date: str | None = None) -> str:
    """Append one dated entry, unless that exact line is already there. Returns the line.

    Conventional field order, matching what the file already holds:
    ``record(task, command, artefact, note)``.
    """
    log = path or DEFAULT_LOG
    line = entry_line(*fields, date=date)

    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch()
    existing = log.read_text(encoding="utf-8")
    if line in existing:
        return line

    # A blank line before the entry so it renders as its own list item even when the
    # preceding content is a paragraph or a table.
    prefix = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"{prefix}{line}\n")
    return line
