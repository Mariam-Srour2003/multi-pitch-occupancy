"""Retention (WP6-T8), and the deletion it must refuse to make.

`thesis/ethics.md` commits to purging sampled frames after 7 days and keeping evidence for
365, and says "Code enforces this" — which was, until now, a claim about code that did not
exist. This is the code; these are the reasons to believe it.

**The important tests are the refusals.** A retention worker in this repository sits next to
4.2 GB of irreplaceable client footage and a hand-labelled corpus. The interesting question is
not whether it deletes old files, it is whether it can be persuaded to delete the wrong ones.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pitch_occupancy.config import settings
from pitch_occupancy.retention import (
    PROTECTED_ROOTS,
    RetentionPolicy,
    apply,
    plan,
)

DAY = 86400.0


def aged(directory: Path, name: str, days: float, *, now: float, size: int = 10) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"x" * size)
    stamp = now - days * DAY
    import os

    os.utime(path, (stamp, stamp))
    return path


# --- the refusals ----------------------------------------------------------------


def test_the_source_footage_and_the_corpus_are_protected() -> None:
    """The most expensive bug this project could ship. "Raw frames" in the ethics
    commitment means frames sampled by the running system, not the 4.2 GB that came from a
    client facility once and cannot be re-recorded."""
    assert settings.raw_dir in PROTECTED_ROOTS
    assert settings.dataset_dir in PROTECTED_ROOTS


def test_planning_against_a_protected_root_raises_rather_than_scanning_it() -> None:
    with pytest.raises(ValueError, match="protected root"):
        plan(sampled_dir=settings.raw_dir)
    with pytest.raises(ValueError, match="protected root"):
        plan(evidence_dir=settings.dataset_dir)


def test_apply_refuses_a_plan_containing_a_protected_path(tmp_path, monkeypatch) -> None:
    """A plan with one protected path in it was built wrongly, so the rest of it is not to
    be trusted either - abort before deleting anything rather than skipping the one."""
    from pitch_occupancy import retention

    now = time.time()
    victim = aged(settings.raw_dir, "_retention_test_do_not_delete.tmp", 999, now=now)
    try:
        bad = retention.RetentionPlan(
            delete=(retention.Candidate(victim, 999, 10, "sampled frame"),)
        )
        with pytest.raises(ValueError, match="protected path"):
            apply(bad, confirm=True)
        assert victim.exists(), "apply() deleted a protected file before aborting"
    finally:
        victim.unlink(missing_ok=True)


def test_apply_does_nothing_without_confirmation(tmp_path) -> None:
    """Calling it by accident must be inert. The default is the safe one."""
    now = time.time()
    old = aged(tmp_path / "interim", "old.jpg", 30, now=now)
    got = plan(sampled_dir=tmp_path / "interim", evidence_dir=tmp_path / "evidence", now=now)
    assert got.delete
    assert apply(got) == 0
    assert old.exists()


# --- the policy ------------------------------------------------------------------


def test_the_periods_match_the_ethics_commitment() -> None:
    """The numbers live in one place in code, and the document is the source. If they drift,
    whichever docstring was edited second is wrong."""
    text = (Path(__file__).resolve().parents[1] / "thesis" / "ethics.md").read_text(
        encoding="utf-8"
    )
    policy = RetentionPolicy()
    assert f"{policy.sampled_frame_days} days" in text
    assert f"{policy.evidence_days}" in text


def test_sampled_frames_older_than_a_week_are_swept(tmp_path) -> None:
    now = time.time()
    aged(tmp_path / "interim", "old.jpg", 8, now=now)
    aged(tmp_path / "interim", "fresh.jpg", 6, now=now)
    got = plan(sampled_dir=tmp_path / "interim", evidence_dir=tmp_path / "evidence", now=now)
    assert [c.path.name for c in got.delete] == ["old.jpg"]
    assert got.kept == 1


def test_evidence_is_kept_a_year_not_a_week(tmp_path) -> None:
    """The two periods differ by a factor of fifty; a worker that applied one rule to both
    would silently destroy the audit trail the system exists to provide."""
    now = time.time()
    aged(tmp_path / "evidence", "recent.jpg", 200, now=now)
    aged(tmp_path / "evidence", "ancient.jpg", 400, now=now)
    got = plan(sampled_dir=tmp_path / "interim", evidence_dir=tmp_path / "evidence", now=now)
    assert [c.path.name for c in got.delete] == ["ancient.jpg"]


def test_a_365_day_rule_is_testable_because_the_clock_is_injectable(tmp_path) -> None:
    """Otherwise this rule gets no test at all, which is how a year-long policy stays
    unverified until the year is up."""
    now = time.time()
    aged(tmp_path / "evidence", "e.jpg", 366, now=now)
    assert plan(sampled_dir=tmp_path / "i", evidence_dir=tmp_path / "evidence", now=now).delete


def test_an_unreadable_file_is_kept_and_reported(tmp_path, monkeypatch) -> None:
    """Deleting on a failed stat() is how a retention worker becomes a data-loss incident."""
    now = time.time()
    aged(tmp_path / "interim", "old.jpg", 30, now=now)
    real_stat = Path.stat

    def boom(self, *a, **k):
        if self.name == "old.jpg":
            raise OSError("nope")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", boom)
    got = plan(sampled_dir=tmp_path / "interim", evidence_dir=tmp_path / "e", now=now)
    assert not got.delete
    assert [p.name for p in got.unreadable] == ["old.jpg"]


def test_a_missing_directory_is_not_an_error(tmp_path) -> None:
    """Before the system has run there is nothing to retain, and that is not a failure."""
    got = plan(sampled_dir=tmp_path / "absent", evidence_dir=tmp_path / "also_absent")
    assert not got.delete and not got.scanned_roots


# --- the dry run is the artefact -------------------------------------------------


def test_the_plan_prints_the_purge_set_which_is_the_acceptance_criterion(tmp_path) -> None:
    now = time.time()
    aged(tmp_path / "interim", "old.jpg", 30, now=now, size=2_000_000)
    text = plan(
        sampled_dir=tmp_path / "interim", evidence_dir=tmp_path / "e", now=now
    ).describe()
    assert "1 file(s) to delete" in text
    assert "old.jpg" in text
    assert "sampled frames 7 d, evidence 365 d" in text
    assert "thesis/ethics.md" in text


def test_deletion_actually_removes_the_files_when_confirmed(tmp_path) -> None:
    now = time.time()
    old = aged(tmp_path / "interim", "old.jpg", 30, now=now)
    keep = aged(tmp_path / "interim", "new.jpg", 1, now=now)
    got = plan(sampled_dir=tmp_path / "interim", evidence_dir=tmp_path / "e", now=now)
    assert apply(got, confirm=True) == 1
    assert not old.exists()
    assert keep.exists()


# --- free space (runbook row 7, 2026-09-11) -----------------------------------------------


def test_free_space_walks_up_to_a_directory_that_exists(tmp_path) -> None:
    """A slot's evidence directory does not exist until the slot starts, and the question is
    about the filesystem it will live on, not about the directory."""
    from pitch_occupancy.retention import free_bytes

    assert free_bytes(tmp_path / "not" / "created" / "yet") is not None


def test_an_unmeasurable_disk_does_not_stop_the_write(tmp_path, monkeypatch) -> None:
    """None means "could not measure", which is the absence of an answer rather than an
    answer. Refusing to record a verdict because a stat call failed would turn a diagnostic
    problem into lost observation, and that is the wrong direction for this system."""
    import shutil

    from pitch_occupancy import retention

    def boom(_):
        raise OSError("no statvfs here")

    monkeypatch.setattr(shutil, "disk_usage", boom)
    assert retention.free_bytes(tmp_path) is None
    room, why = retention.has_room(tmp_path)
    assert room is True
    assert "could not be determined" in why


def test_a_full_disk_refuses_before_the_first_write(tmp_path) -> None:
    """The point is the *ordering*. Deciding once, before minute zero, is what stops a slot
    ending with a partial set of evidence images - which is worse than none, because an
    inspector cannot tell it from a slot that never saved any."""
    from pitch_occupancy.retention import has_room

    room, why = has_room(tmp_path, need=10 ** 15)
    assert room is False
    assert "below the" in why


def test_a_full_disk_still_produces_the_verdict(tmp_path, monkeypatch) -> None:
    """A full disk must not cost an hour of observation. The slot runs, the verdict is
    returned, and only the pictures are given up."""
    import numpy as np

    from pitch_occupancy import worker
    from pitch_occupancy.data.taxonomy import Class3
    from pitch_occupancy.frame_source import Frame, FrameSource

    class OneCamera(FrameSource):
        def cameras(self):
            return ["camA"]

        @property
        def n_minutes(self):
            return 3

        def read(self, camera_id, minute_index):
            return Frame(camera_id, minute_index, np.zeros((8, 8, 3), np.uint8), "x.mp4")

    monkeypatch.setattr(worker, "has_room", lambda p: (False, "0 MB free, below the floor"))
    run = worker.run_slot("slot", OneCamera(), lambda img: (Class3.ACTIVE_PLAY, 0.9),
                          evidence_dir=tmp_path)
    assert run.verdict is not None
    assert run.minutes_captured == 3
    assert not any(tmp_path.iterdir()), "wrote evidence despite refusing for space"
    assert all(e.image_path is None for e in run.evidence)
