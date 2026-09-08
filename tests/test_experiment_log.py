"""The experiment log's write path.

`EXPERIMENT_LOG.md` is the index a reader scans to find which run produced which CSV, and
the Findings tab renders it verbatim. It had accumulated duplicate entries - three pairs
already committed - because thirteen experiments each appended unconditionally, so every
rerun added another identical line. An index nobody trusts is an index nobody reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pitch_occupancy.evaluation.experiment_log import DEFAULT_LOG, entry_line, record


def test_the_line_leads_with_a_date_and_joins_the_fields() -> None:
    line = entry_line("H3", "`python x.py`", "`out.csv`", date="2026-09-08")
    assert line == "- 2026-09-08 | H3 | `python x.py` | `out.csv`"


def test_blank_fields_do_not_leave_a_dangling_separator() -> None:
    """An optional field left empty is common - a run with no note, say."""
    assert entry_line("H3", "", "  ", "`out.csv`", date="2026-09-08") == (
        "- 2026-09-08 | H3 | `out.csv`"
    )


def test_an_entry_is_written_once_however_often_the_experiment_reruns(tmp_path: Path) -> None:
    """The defect this module exists for. Debugging one script in a single session put
    three copies of its line in the file."""
    log = tmp_path / "LOG.md"
    for _ in range(4):
        record("H3", "`python x.py`", "`out.csv`", path=log, date="2026-09-08")
    assert log.read_text(encoding="utf-8").count("- 2026-09-08 | H3") == 1


def test_a_run_whose_note_changed_writes_a_new_line(tmp_path: Path) -> None:
    """Idempotence is on the exact text, so a *different* result is still recorded. A rerun
    that reports 8 folds where the last one reported 7 is news."""
    log = tmp_path / "LOG.md"
    record("H3", "`python x.py`", "7 folds", path=log, date="2026-09-08")
    record("H3", "`python x.py`", "8 folds", path=log, date="2026-09-08")
    assert log.read_text(encoding="utf-8").count("- 2026-09-08 | H3") == 2


def test_a_later_day_is_a_separate_entry(tmp_path: Path) -> None:
    log = tmp_path / "LOG.md"
    record("H3", "`python x.py`", path=log, date="2026-09-08")
    record("H3", "`python x.py`", path=log, date="2026-09-09")
    assert len([l for l in log.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]) == 2


def test_it_creates_the_file_and_its_directory(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "LOG.md"
    record("H3", "`python x.py`", path=log, date="2026-09-08")
    assert log.exists()


def test_the_entry_is_separated_from_preceding_prose(tmp_path: Path) -> None:
    """The log interleaves narrative sections with index lines; an entry appended straight
    onto a paragraph would render as part of that paragraph."""
    log = tmp_path / "LOG.md"
    log.write_text("Some narrative about a finding.\n", encoding="utf-8")
    record("H3", "`python x.py`", path=log, date="2026-09-08")
    body = log.read_text(encoding="utf-8")
    assert body == "Some narrative about a finding.\n\n- 2026-09-08 | H3 | `python x.py`\n"


def test_it_does_not_stack_blank_lines(tmp_path: Path) -> None:
    log = tmp_path / "LOG.md"
    log.write_text("Narrative.\n\n", encoding="utf-8")
    record("H3", "`python x.py`", path=log, date="2026-09-08")
    assert "\n\n\n" not in log.read_text(encoding="utf-8")


def test_the_committed_log_has_no_duplicate_entries() -> None:
    """A regression guard on the artefact itself, not only on the writer.

    Three duplicated pairs were already in the committed file when this was written. Every
    experiment now records through :func:`record`, so a new one means a script has gone
    back to appending by hand.
    """
    if not DEFAULT_LOG.exists():
        pytest.skip("log not present")
    entries = [
        line for line in DEFAULT_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("- 20")
    ]
    duplicated = {e for e in entries if entries.count(e) > 1}
    assert not duplicated, f"{len(duplicated)} duplicated log entries: {sorted(duplicated)[:3]}"


def test_every_experiment_records_through_this_module() -> None:
    """Thirteen hand-rolled copies of one format string had already drifted into two
    separator styles. A new script that opens the log directly bypasses the deduplication
    and the format, so the import is what is checked."""
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    offenders = [
        p.name for p in experiments.glob("*.py")
        if 'EXPERIMENT_LOG.md").open(' in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"these append to the log directly: {offenders}"
