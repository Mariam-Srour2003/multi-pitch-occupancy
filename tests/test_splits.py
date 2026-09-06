"""Splits decide what every reported number means, and their failures are silent:
a leaked group or a single-class test set still produces a plausible accuracy."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pitch_occupancy.data.manifest import ManifestRow
from pitch_occupancy.data.splits import (
    check_split,
    development_rows,
    final_test_rows,
    grouped_split,
    leave_one_group_out,
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
