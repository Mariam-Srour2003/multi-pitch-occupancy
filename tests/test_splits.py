"""Splits decide what every reported number means, and their failures are silent:
a leaked group or a single-class test set still produces a plausible accuracy."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pitch_occupancy.data.manifest import ManifestRow
from pitch_occupancy.data.splits import (
    DEFAULT_FINAL_VENUES,
    DEFAULT_SPLIT_DIR,
    LEGACY_GROUP_KEY,
    check_split,
    development_rows,
    final_test_rows,
    grouped_split,
    leave_one_group_out,
    load_final_venues,
    random_split,
    read_split,
    temporal_split,
    write_split,
)


def row(
    file: str, *, venue: str = "v1", slot: str = "s1", cls: str = "C1_EMPTY", date: str = "2026-07-11"
) -> ManifestRow:
    return ManifestRow(
        file=file, class4="1_empty", class3=cls, venue=venue, camera=f"{slot}_camA",
        slot_date=date, slot_time="10:00", slot_id=slot, t_s=0, source="regular",
        labeled_by="human", lighting="day", quality="unknown", split_role="",
    )


@pytest.fixture
def rows() -> list[ManifestRow]:
    out = []
    for v in ("v1", "v2", "v3", "v4"):
        for s in ("a", "b"):
            for i in range(10):
                cls = "C1_EMPTY" if i < 5 else "C2_ACTIVE_PLAY"
                out.append(row(f"{v}_{s}_{i}.jpg", venue=v, slot=f"{v}{s}", cls=cls))
    return out


@pytest.fixture
def lock(tmp_path: Path) -> frozenset[str]:
    return frozenset({"v4"})


# --- the lock ---------------------------------------------------------------


def test_development_rows_excludes_locked_venues(rows, lock) -> None:
    dev = development_rows(rows, final_venues=lock)
    assert {r.venue for r in dev} == {"v1", "v2", "v3"}


def test_final_test_rows_refuses_without_the_explicit_flag(rows, lock) -> None:
    with pytest.raises(RuntimeError, match="locked"):
        final_test_rows(rows, final_venues=lock)


def test_final_test_rows_returns_only_locked_venues_when_unlocked(rows, lock) -> None:
    final = final_test_rows(rows, i_have_finished_all_development=True, final_venues=lock)
    assert {r.venue for r in final} == {"v4"}


def test_the_lock_path_does_not_depend_on_the_working_directory(monkeypatch, tmp_path) -> None:
    """The lock used to fail *open* from any other directory.

    `DEFAULT_FINAL_VENUES` was `Path("results/splits/...")`, so a run launched from a
    parent directory, a scheduler, or a notebook found no file, got an empty frozenset,
    and `development_rows` quietly returned all 1,692 rows - both locked venues included.
    Nothing raised and nothing warned. The path is resolved from the package now, and
    this pins it: the venues locked before any model was fitted must stay locked wherever
    the process happens to be started.
    """
    assert DEFAULT_FINAL_VENUES.is_absolute()
    monkeypatch.chdir(tmp_path)
    assert load_final_venues() == frozenset(
        {"clipvenue_b_floodlit_track", "clipvenue_c_teal_boards"}
    )


def test_a_missing_lock_file_raises_instead_of_unlocking_everything(tmp_path) -> None:
    """"No lock file" must not evaluate to "nothing is locked"."""
    with pytest.raises(RuntimeError, match="lock file is missing"):
        load_final_venues(tmp_path / "absent.csv")


def test_splits_never_leak_the_locked_venue(rows, monkeypatch) -> None:
    """The guard must hold for every strategy, not just the one that was remembered."""
    monkeypatch.setattr("pitch_occupancy.data.splits.load_final_venues", lambda *a, **k: frozenset({"v4"}))
    for split in (
        grouped_split(rows),
        random_split(rows),
        temporal_split(rows, cutoff_date="2026-07-12"),
        *leave_one_group_out(rows),
    ):
        seen = {r.venue for r in split.train} | {r.venue for r in split.test}
        assert "v4" not in seen, f"{split.name} leaked the locked venue"


# --- strategies -------------------------------------------------------------


def test_grouped_split_keeps_groups_whole(rows) -> None:
    s = grouped_split(rows, group_key="slot_id")
    assert not ({r.slot_id for r in s.train} & {r.slot_id for r in s.test})
    assert check_split(s) == [] or all("single-class" not in p for p in check_split(s))


def test_grouped_split_is_deterministic(rows) -> None:
    a = grouped_split(rows, seed=7)
    b = grouped_split(rows, seed=7)
    assert [r.file for r in a.test] == [r.file for r in b.test]


def test_different_seeds_give_different_splits(rows) -> None:
    a = grouped_split(rows, seed=1)
    b = grouped_split(rows, seed=99)
    assert {r.file for r in a.test} != {r.file for r in b.test}


def test_leave_one_venue_out_covers_every_venue_once(rows) -> None:
    folds = list(leave_one_group_out(rows, group_key="venue"))
    assert [f.name.split("__")[-1] for f in folds] == ["v1", "v2", "v3", "v4"]
    for f in folds:
        assert {r.venue for r in f.test} == {f.name.split("__")[-1]}
        assert f.name.split("__")[-1] not in {r.venue for r in f.train}


def test_temporal_split_respects_the_cutoff(rows) -> None:
    late = [row(f"late_{i}.jpg", date="2026-08-01", cls="C2_ACTIVE_PLAY") for i in range(5)]
    s = temporal_split(rows + late, cutoff_date="2026-08-01")
    assert all(r.slot_date < "2026-08-01" for r in s.train)
    assert all(r.slot_date >= "2026-08-01" for r in s.test)


def test_random_split_is_leaky_by_design(rows) -> None:
    """Documented behaviour: it exists as H1's control arm, so it must NOT be group-safe."""
    s = random_split(rows)
    assert {r.slot_id for r in s.train} & {r.slot_id for r in s.test}


# --- validation -------------------------------------------------------------


def test_check_split_flags_group_overlap(rows) -> None:
    bad = random_split(rows)
    object.__setattr__(bad, "group_key", "slot_id")
    assert any("both sides" in p for p in check_split(bad))


def test_check_split_flags_duplicate_frames(rows) -> None:
    from pitch_occupancy.data.splits import Split

    bad = Split("x", train=tuple(rows[:10]), test=tuple(rows[5:15]), group_key="<none - leaky>")
    assert any("both train and test" in p for p in check_split(bad))


def test_check_split_flags_a_near_single_class_test_set() -> None:
    """The failure mode the current dataset actually produces."""
    from pitch_occupancy.data.splits import Split

    train = [row(f"t{i}.jpg", cls="C1_EMPTY" if i % 2 else "C2_ACTIVE_PLAY") for i in range(20)]
    test = [row(f"e{i}.jpg", cls="C2_ACTIVE_PLAY") for i in range(20)]
    problems = check_split(Split("x", tuple(train), tuple(test), group_key="<none - leaky>"))
    assert any("single-class" in p for p in problems)
    assert any("absent from test" in p for p in problems)


def test_check_split_flags_empty_sides() -> None:
    from pitch_occupancy.data.splits import Split

    assert "test side is empty" in check_split(Split("x", (row("a.jpg"),), (), group_key="v"))


# --- materialisation --------------------------------------------------------


def test_split_round_trips_through_csv(rows, tmp_path: Path) -> None:
    s = grouped_split(rows)
    back = read_split(write_split(s, tmp_path), rows)
    assert {r.file for r in back.train} == {r.file for r in s.train}
    assert {r.file for r in back.test} == {r.file for r in s.test}


def test_read_split_raises_if_the_dataset_changed(rows, tmp_path: Path) -> None:
    """A split that silently shrinks is worse than one that crashes."""
    path = write_split(grouped_split(rows), tmp_path)
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["deleted_frame.jpg", "test"])
    with pytest.raises(KeyError, match="not in the manifest"):
        read_split(path, rows)


def test_the_round_trip_preserves_the_group_key_and_seed(rows, tmp_path: Path) -> None:
    """Without these the round-trip is lossy in the field the leakage check needs.

    A split read back used to carry `group_key="<from file>"`, which is not a manifest
    field - so `check_split`, the validation this module exists for, raised AttributeError
    on every materialised split instead of checking it.
    """
    s = grouped_split(rows, group_key="slot_id", seed=7)
    back = read_split(write_split(s, tmp_path), rows)
    assert back.group_key == "slot_id"
    assert back.seed == 7


def test_check_split_gives_the_same_answer_on_disk_as_in_memory(rows, tmp_path: Path) -> None:
    s = grouped_split(rows)
    back = read_split(write_split(s, tmp_path), rows)
    assert check_split(back) == check_split(s)


def test_check_split_reports_an_unrunnable_group_check_instead_of_raising(
    rows, tmp_path: Path
) -> None:
    """A split file written before group_key was recorded must still be checkable.

    Silently skipping the group-overlap check would be worse than crashing: the whole
    point of this function is to say when a split would make results misleading, so "I
    could not check this" has to appear in the output.
    """
    s = grouped_split(rows)
    path = tmp_path / "legacy.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "role"])  # the old two-column format
        for r in s.train:
            w.writerow([r.file, "train"])
        for r in s.test:
            w.writerow([r.file, "test"])

    back = read_split(path, rows)
    assert back.group_key == LEGACY_GROUP_KEY
    problems = check_split(back)
    assert any("could not be checked" in p and "not fully validated" in p for p in problems)


def test_read_split_re_applies_the_final_venue_lock(rows, tmp_path: Path, monkeypatch) -> None:
    """A materialised split is not a licence to skip the lock.

    A file written before the lock existed - or while it was failing open, which was
    possible until the path became package-relative - can name a locked venue, and nothing
    downstream looks again.
    """
    s = grouped_split(rows)
    path = write_split(s, tmp_path)
    # now declare one of the venues in that file locked, as if the lock post-dates it
    victim = s.test[0].venue
    monkeypatch.setattr(
        "pitch_occupancy.data.splits.load_final_venues", lambda *a, **k: frozenset({victim})
    )
    with pytest.raises(RuntimeError, match="locked final-test venue"):
        read_split(path, rows)


def test_a_split_file_that_disagrees_with_itself_raises(rows, tmp_path: Path) -> None:
    s = grouped_split(rows, group_key="slot_id")
    path = write_split(s, tmp_path)
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([s.test[0].file, "test", "venue", "42"])
    with pytest.raises(ValueError, match="more than one group_key"):
        read_split(path, rows)


def test_the_default_split_directory_does_not_depend_on_the_working_directory() -> None:
    """`write_split(s)` with no out_dir used to write into whatever cwd happened to be -
    scattering materialised splits outside the repo, into the same folder the final-venue
    lock lives in."""
    assert DEFAULT_SPLIT_DIR.is_absolute()
    assert DEFAULT_SPLIT_DIR.name == "splits"

