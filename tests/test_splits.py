"""Splits decide what every reported number means, and their failures are silent:
a leaked group or a single-class test set still produces a plausible accuracy."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pitch_occupancy.data.manifest import ManifestRow
from pitch_occupancy.data.splits import (
    DEFAULT_FINAL_VENUES,
    DEFAULT_SPLIT_DIR,
    LEGACY_GROUP_KEY,
    check_split,
    SYNTHETIC_SOURCE,
    Split,
    development_rows,
    final_test_rows,
    grouped_split,
    leave_one_group_out,
    load_final_venues,
    random_split,
    read_split,
    split_identity,
    temporal_split,
    write_split,
)


ROOT = Path(__file__).resolve().parents[1]


def row(
    file: str, *, venue: str = "v1", slot: str = "s1", cls: str = "C1_EMPTY",
    date: str = "2026-07-11", source: str = "regular",
) -> ManifestRow:
    return ManifestRow(
        file=file, label="1_empty", class3=cls, venue=venue, camera=f"{slot}_camA",
        slot_date=date, slot_time="10:00", slot_id=slot, t_s=0, source=source,
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
    """Two calls in one process agree.

    Kept, but read the next test before trusting it. This one passed for the entire life
    of the defect it was written to prevent: set iteration order is fixed *within* a
    process, so calling the function twice here could never have detected that the order
    changed between processes.
    """
    a = grouped_split(rows, seed=7)
    b = grouped_split(rows, seed=7)
    assert [r.file for r in a.test] == [r.file for r in b.test]


@pytest.mark.parametrize("group_key", ["slot_id", "venue"])
def test_grouped_split_order_survives_a_different_hash_seed(group_key: str) -> None:
    """The same seed must give the same *sequence* in a fresh process, not just the same set.

    `test_names` is a set, and `PYTHONHASHSEED` is randomised per process, so iterating it
    to build the test side returned identical frames in a different order on every run.
    Point estimates were unaffected - which is why nothing caught it - but
    `bootstrap_metric_ci` resamples positions, so every confidence interval computed on a
    grouped split was a draw from a different ordering and none of them reproduced.

    This has to spawn interpreters. Any assertion made inside one process compares two
    orderings that are equal by construction.
    """
    script = (
        "import sys, hashlib;"
        "sys.path.insert(0, 'src');"
        "from pitch_occupancy.data.manifest import ManifestRow;"
        "from pitch_occupancy.data.splits import grouped_split;"
        "rows=[ManifestRow(file=f'f{i}.jpg', label='1_empty', class3='C1_EMPTY',"
        " venue=f'v{i%4}', camera='c', slot_date='2026-07-11', slot_time='10:00',"
        " slot_id=f's{i%12}', t_s=i, source='regular', labeled_by='human',"
        " lighting='day', quality='unknown', split_role='') for i in range(240)];"
        f"s=grouped_split(rows, group_key='{group_key}', seed=7);"
        "print(hashlib.md5('|'.join(r.file for r in s.test).encode()).hexdigest(),"
        " hashlib.md5('|'.join(r.file for r in s.train).encode()).hexdigest())"
    )
    seen = set()
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1], env=env, check=True,
        )
        seen.add(out.stdout.strip())
    assert len(seen) == 1, (
        f"grouped_split(group_key={group_key!r}) produced {len(seen)} different orderings "
        f"across hash seeds: {seen}"
    )


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


def test_undated_frames_are_excluded_from_a_temporal_split_by_default(rows) -> None:
    """`"" < "2026-08-01"` is true, so undated frames used to land in *train* silently.

    On the real corpus all 282 undated frames are the seven clip venues, none of which
    appear on the test side - so the split named "temporal" was a venue-and-time split and
    nothing in its output said so. A frame with no date has no position on a timeline.
    """
    undated = [row(f"nodate_{i}.jpg", venue="v9", slot="s9", date="") for i in range(6)]
    late = [row(f"late_{i}.jpg", date="2026-08-01", cls="C2_ACTIVE_PLAY") for i in range(5)]
    s = temporal_split(rows + undated + late, cutoff_date="2026-08-01")
    placed = {r.file for r in s.train} | {r.file for r in s.test}
    assert not any(r.file in placed for r in undated)


def test_the_old_behaviour_is_still_available_but_has_to_be_asked_for(rows) -> None:
    undated = [row(f"nodate_{i}.jpg", venue="v9", slot="s9", date="") for i in range(6)]
    late = [row(f"late_{i}.jpg", date="2026-08-01", cls="C2_ACTIVE_PLAY") for i in range(5)]
    s = temporal_split(rows + undated + late, cutoff_date="2026-08-01", undated="train")
    assert {r.file for r in undated} <= {r.file for r in s.train}


def test_undated_error_names_how_many_rows_are_undated(rows) -> None:
    undated = [row(f"nodate_{i}.jpg", venue="v9", slot="s9", date="") for i in range(6)]
    with pytest.raises(ValueError, match="6 row"):
        temporal_split(rows + undated, cutoff_date="2026-08-01", undated="error")


def test_a_fully_dated_corpus_is_unaffected_by_the_policy(rows) -> None:
    late = [row(f"late_{i}.jpg", date="2026-08-01", cls="C2_ACTIVE_PLAY") for i in range(5)]
    data = rows + late
    everything = [
        temporal_split(data, cutoff_date="2026-08-01", undated=policy)
        for policy in ("exclude", "train", "error")
    ]
    assert all(
        [r.file for r in s.train] == [r.file for r in everything[0].train]
        and [r.file for r in s.test] == [r.file for r in everything[0].test]
        for s in everything
    )


def test_an_unknown_undated_policy_raises_rather_than_silently_excluding(rows) -> None:
    with pytest.raises(ValueError, match="exclude/train/error"):
        temporal_split(rows, cutoff_date="2026-08-01", undated="keep")


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



# --- what identifies a split (WP0-T4, decided 2026-09-10) ---------------------------------
#
# The plan claimed splits were "materialised, referenced by name, never re-randomised". They
# were not: `write_split`/`read_split` have no callers outside these tests and every
# experiment calls `grouped_split(seed=42)` directly. That is deterministic given the same
# rows, and the row list depends on which feature caches the calling script filtered to - so
# two experiments that filter differently get different splits from one seed, which is how
# 94 distinct scenes and 95 distinct scenes were both correct at once in September. They
# agree again now, and nothing recorded either fact.
#
# The retrofit was not taken (it would change which rows several published experiments were
# fitted on) and the claim was dropped instead. What replaces it is a fingerprint over
# everything that actually decides the partition, so the disagreement is detectable.


def test_the_same_rows_and_seed_give_the_same_identity(rows) -> None:
    a = grouped_split(rows, group_key="slot_id", seed=42)
    b = grouped_split(rows, group_key="slot_id", seed=42)
    assert split_identity(a) == split_identity(b)


def test_a_different_seed_gives_a_different_identity(rows) -> None:
    a = grouped_split(rows, group_key="slot_id", seed=42)
    b = grouped_split(rows, group_key="slot_id", seed=7)
    if [r.file for r in a.test] == [r.file for r in b.test]:
        pytest.skip("too few groups for the seed to change the partition")
    assert split_identity(a) != split_identity(b)


def test_one_fewer_row_changes_the_identity_even_at_the_same_seed(rows) -> None:
    """The whole point. This is the case a seed cannot distinguish and the one that actually
    happened: a script that filters to one feature cache hands `grouped_split` a shorter row
    list and gets a different partition, reporting it under the same seed."""
    full = grouped_split(rows, group_key="slot_id", seed=42)
    fewer = grouped_split(rows[:-1], group_key="slot_id", seed=42)
    assert split_identity(full) != split_identity(fewer)


def test_the_identity_covers_the_order_and_not_only_the_membership(rows) -> None:
    """Order mattered once and expensively: `grouped_split` built its test side from a set,
    so the same seed returned the same frames in a different order every process and every
    bootstrap interval on a grouped split was a different draw. Membership was right, which
    is why nothing caught it."""
    from dataclasses import replace

    split = grouped_split(rows, group_key="slot_id", seed=42)
    if len(split.test) < 2:
        pytest.skip("test side too small to reorder")
    reordered = replace(split, test=tuple(reversed(split.test)))
    assert split_identity(reordered) != split_identity(split)


def test_the_identity_is_short_enough_to_quote(rows) -> None:
    """It goes next to a number in a table or a log line; a 64-character hash would not be
    written down, and one nobody writes down detects nothing."""
    ident = split_identity(grouped_split(rows, group_key="slot_id", seed=42))
    assert len(ident) == 12 and ident.isalnum()


def test_a_materialised_split_verifies_its_own_partition(tmp_path, rows) -> None:
    """A split file is a CSV and a CSV is editable. Flipping one row from test to train
    produces a file that reads back cleanly and describes a different experiment - so the
    digest is checked on read rather than trusted."""
    split = grouped_split(rows, group_key="slot_id", seed=42)
    path = write_split(split, tmp_path)
    read_split(path, rows)  # round-trips

    text = path.read_text(encoding="utf-8")
    tampered = text.replace(",test,", ",train,", 1)
    assert tampered != text, "no test row to flip"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="reads back as"):
        read_split(path, rows)


def test_copying_a_split_to_a_new_name_stays_legal(tmp_path, rows) -> None:
    """The digest covers the partition and not the name, because renaming a materialised
    split is a thing people do and is not a corruption."""
    split = grouped_split(rows, group_key="slot_id", seed=42)
    path = write_split(split, tmp_path)
    copy = path.with_name("a_different_name.csv")
    copy.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    assert read_split(copy, rows).name == "a_different_name"


def test_a_split_written_before_the_digest_existed_still_reads(tmp_path, rows) -> None:
    """Refusing them would break the only artefacts this path has ever produced. A missing
    digest is a real state, not a failure."""
    split = grouped_split(rows, group_key="slot_id", seed=42)
    path = write_split(split, tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].replace(",partition", "")
    body = [ln.rsplit(",", 1)[0] for ln in lines[1:]]
    path.write_text("\n".join([header, *body]) + "\n", encoding="utf-8")
    assert len(read_split(path, rows).test) == len(split.test)

def test_the_two_scripts_that_once_disagreed_now_build_the_same_split() -> None:
    """The guard the WP0-T4 item actually wanted, and the cheapest form of it.

    `effective_sample_audit.py` filters to the DINOv2 cache and `h4_model_equivalence.py` to
    all three, so they hand `grouped_split` different row lists - and a seed does not
    determine the partition, only the partition *given* the rows. In September they reported
    94 and 95 distinct scenes from `seed=42`, both correctly, and nothing recorded that they
    differed. The reproducibility repair of 2026-09-08 brought them back together, and
    nothing recorded that either.

    Now both write their `split_identity` and this compares them. It is a cross-script check
    that no single script could make, and it fails the moment the two drift apart again -
    which is what the materialisation retrofit would have prevented at twenty times the cost.
    """
    import csv

    audit = ROOT / "results" / "effective_sample_audit.csv"
    equivalence = ROOT / "results" / "h4_model_equivalence.csv"
    if not (audit.exists() and equivalence.exists()):
        pytest.skip("both artefacts are needed to compare them")

    audit_rows = [r for r in csv.DictReader(audit.open(encoding="utf-8"))
                  if r.get("split") == "grouped" and r.get("split_identity")]
    equiv_rows = [r for r in csv.DictReader(equivalence.open(encoding="utf-8"))
                  if r.get("split_identity")]
    if not audit_rows or not equiv_rows:
        pytest.skip("regenerate both: they predate the split_identity column")

    audit_ids = {r["split_identity"] for r in audit_rows}
    equiv_ids = {r["split_identity"] for r in equiv_rows}
    assert len(audit_ids) == 1 and len(equiv_ids) == 1, "one grouped split per script"
    assert audit_ids == equiv_ids, (
        f"the two scripts report different grouped splits from the same seed: "
        f"{audit_ids} vs {equiv_ids}. Their row lists have drifted apart again - see the "
        f"WP0-T4 decision in splits.py."
    )


# --- A13: generated frames are training-side only -------------------------------------


def test_check_split_flags_synthetic_frames_on_the_test_side() -> None:
    """A13 condition 1. The convention is worthless unless the checker enforces it.

    A model scored against its own generator's output measures the generator, and this
    repository's recurring defect is a guard that exists and does not operate.
    """
    train = [row(f"t{i}.jpg", slot="a", cls="C1_EMPTY" if i < 5 else "C2_ACTIVE_PLAY")
             for i in range(10)]
    test = [row(f"e{i}.jpg", slot="b", cls="C1_EMPTY" if i < 5 else "C2_ACTIVE_PLAY")
            for i in range(10)]
    test[0] = row("syn_e0.jpg", slot="b", cls="C1_EMPTY", source=SYNTHETIC_SOURCE)

    problems = check_split(Split(name="a13", train=train, test=test, group_key="slot_id", seed=0))
    assert any("generated frame" in p for p in problems), problems
    assert any("A13" in p or "training-only" in p for p in problems), problems


def test_check_split_allows_synthetic_frames_on_the_train_side() -> None:
    """The other half of the rule: training-side generated frames are the intended use."""
    train = [row(f"t{i}.jpg", slot="a", cls="C1_EMPTY" if i < 5 else "C2_ACTIVE_PLAY")
             for i in range(10)]
    train.append(row("syn_t0.jpg", slot="a", cls="C1_EMPTY", source=SYNTHETIC_SOURCE))
    test = [row(f"e{i}.jpg", slot="b", cls="C1_EMPTY" if i < 5 else "C2_ACTIVE_PLAY")
            for i in range(10)]

    problems = check_split(Split(name="a13", train=train, test=test, group_key="slot_id", seed=0))
    assert not any("generated frame" in p for p in problems), problems


def test_development_rows_excludes_generated_frames_by_default() -> None:
    """A13. Ingesting a batch must not silently redefine every published number."""
    real = [row(f"r{i}.jpg", venue="v1") for i in range(5)]
    syn = [row(f"syn{i}.jpg", venue="v1", source=SYNTHETIC_SOURCE) for i in range(3)]

    default = development_rows(real + syn, final_venues=frozenset())
    assert {r.file for r in default} == {r.file for r in real}

    opted_in = development_rows(real + syn, final_venues=frozenset(),
                                include_synthetic=True)
    assert len(opted_in) == 8


def test_generated_frames_reach_the_train_side_when_opted_in() -> None:
    """A13's opt-in was inert, and silently so.

    `development_rows(include_synthetic=True)` returned the generated rows, and then every
    split function called `development_rows(rows)` again with the default and stripped them
    back out. The flag was documented as the way to use generated frames in an ablation, and
    no split protocol could deliver one to a training set.
    """
    real = [row(f"r{i}.jpg", venue="v1", slot=f"s{i}") for i in range(8)]
    syn = [row(f"g{i}.jpg", venue="v1", slot="gen", source=SYNTHETIC_SOURCE)
           for i in range(4)]

    without = grouped_split(real + syn, seed=42)
    assert not [r for r in without.train if r.source == SYNTHETIC_SOURCE]

    with_gen = grouped_split(real + syn, seed=42, include_synthetic=True)
    assert len({r.file for r in with_gen.train if r.source == SYNTHETIC_SOURCE}) == 4


def test_generated_frames_never_reach_a_test_side_even_when_opted_in() -> None:
    """"Training only" implemented literally.

    Routing them *through* the partition was the first attempt and it is not a near miss:
    on the real manifest 171 of 189 landed on the test side, where `check_split` rejects the
    split outright. They are appended to train instead, never partitioned.
    """
    real = [row(f"r{i}.jpg", venue="v1", slot=f"s{i}") for i in range(8)]
    syn = [row(f"g{i}.jpg", venue="v1", slot="gen", source=SYNTHETIC_SOURCE)
           for i in range(4)]

    split = grouped_split(real + syn, seed=42, include_synthetic=True)
    assert not [r for r in split.test if r.source == SYNTHETIC_SOURCE]
    assert not [p for p in check_split(split) if "generated" in p]


def test_opting_generated_frames_in_does_not_move_the_test_side() -> None:
    """The property that makes the two scores an ablation rather than two numbers.

    Because the generated rows never enter the partition, the test side is identical with
    and without them - so a difference in score is attributable to the training data and to
    nothing else. The first implementation grew the real test set from 907 frames to 961 and
    the comparison quietly stopped meaning anything.
    """
    real = [row(f"r{i}.jpg", venue="v1", slot=f"s{i}") for i in range(12)]
    syn = [row(f"g{i}.jpg", venue="v1", slot="gen", source=SYNTHETIC_SOURCE)
           for i in range(6)]

    without = grouped_split(real + syn, seed=42)
    with_gen = grouped_split(real + syn, seed=42, include_synthetic=True)
    assert [r.file for r in with_gen.test] == [r.file for r in without.test]


# --- scenes: one frame per distinct scene on the training side (A15) --------------------


def test_distinct_rows_keeps_one_frame_per_scene(tmp_path):
    """The corpus is 1,881 frames and 290 scenes; the EMPTY class is 525 and 28."""
    import csv as _csv

    from pitch_occupancy.data.splits import distinct_rows

    rows = [row(f"f{i}.jpg") for i in range(6)]
    p = tmp_path / "scene_ids.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=["file", "scene_id"])
        w.writeheader()
        # three scenes: two frames, three frames, one frame
        for f, s in zip([r.file for r in rows],
                        ["s1", "s1", "s2", "s2", "s2", "s3"]):
            w.writerow({"file": f, "scene_id": s})

    kept = distinct_rows(rows, path=p)
    assert [r.file for r in kept] == ["f0.jpg", "f2.jpg", "f5.jpg"]


def test_distinct_rows_refuses_to_run_without_the_sidecar(tmp_path):
    """Silently returning every frame would mean an experiment reporting itself as
    deduplicated while fitting on 1,881 near-copies - the failure this exists to prevent."""
    import pytest as _pytest

    from pitch_occupancy.data.splits import distinct_rows

    with _pytest.raises(FileNotFoundError):
        distinct_rows([row("f0.jpg")], path=tmp_path / "absent.csv")


def test_a_frame_the_sidecar_does_not_know_is_kept_not_dropped(tmp_path):
    """A new frame ingested before the ids were regenerated must not vanish from training."""
    import csv as _csv

    from pitch_occupancy.data.splits import distinct_rows

    rows = [row("known.jpg"), row("brand_new.jpg")]
    p = tmp_path / "scene_ids.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=["file", "scene_id"])
        w.writeheader()
        w.writerow({"file": "known.jpg", "scene_id": "s1"})

    assert [r.file for r in distinct_rows(rows, path=p)] == ["known.jpg", "brand_new.jpg"]


# --- the per-video cap (2026-09-21) ----------------------------------------------------------


def _scenes(tmp_path, mapping: dict[str, str]):
    import csv as _csv

    p = tmp_path / "scene_ids.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=["file", "scene_id"])
        w.writeheader()
        for f, s in mapping.items():
            w.writerow({"file": f, "scene_id": s})
    return p


def test_the_cap_bounds_what_one_recording_can_be_worth(tmp_path):
    """Four videos are 75% of every recorded frame; the median video contributes six.

    A fit on that is a fit on four afternoons at one facility. `distinct_rows` alone does
    not close it - 103 of the 197 scenes come from those same four recordings.
    """
    from pitch_occupancy.data.splits import balanced_rows

    big = [row(f"big{i}.jpg", slot="big") for i in range(20)]
    small = [row(f"small{i}.jpg", slot="small") for i in range(3)]
    scenes = _scenes(tmp_path, {**{r.file: f"b{i}" for i, r in enumerate(big)},
                                **{r.file: f"s{i}" for i, r in enumerate(small)}})

    kept = balanced_rows(big + small, per_video=5, path=scenes)
    assert sum(1 for r in kept if r.camera == "big_camA") == 5
    assert sum(1 for r in kept if r.camera == "small_camA") == 3, "a small video is untouched"
    assert [r.file for r in kept] == [f"big{i}.jpg" for i in range(5)] + [
        f"small{i}.jpg" for i in range(3)], "input order is preserved, so no seed is needed"


def test_the_cap_spends_its_budget_on_scenes_before_second_frames(tmp_path):
    """Fifteen frames of one moment and fifteen different moments are not the same
    fifteen, and taking the first N in order would often pick the former."""
    from pitch_occupancy.data.splits import balanced_rows

    rows = [row(f"f{i}.jpg", slot="one") for i in range(6)]
    # f0..f3 are all the same moment; f4 and f5 are two others
    scenes = _scenes(tmp_path, dict(zip([r.file for r in rows],
                                        ["s1", "s1", "s1", "s1", "s2", "s3"])))
    kept = [r.file for r in balanced_rows(rows, per_video=3, path=scenes)]
    assert kept == ["f0.jpg", "f4.jpg", "f5.jpg"], "one per scene first, then seconds"


def test_the_cap_falls_back_to_more_of_a_scene_once_every_scene_has_one(tmp_path):
    from pitch_occupancy.data.splits import balanced_rows

    rows = [row(f"f{i}.jpg", slot="one") for i in range(4)]
    scenes = _scenes(tmp_path, dict(zip([r.file for r in rows], ["s1", "s1", "s1", "s2"])))
    kept = [r.file for r in balanced_rows(rows, per_video=3, path=scenes)]
    assert kept == ["f0.jpg", "f1.jpg", "f3.jpg"], "s1's second frame beats nothing"


def test_the_cap_refuses_a_nonsense_budget_and_a_missing_sidecar(tmp_path):
    import pytest as _pytest

    from pitch_occupancy.data.splits import balanced_rows

    rows = [row("f0.jpg")]
    scenes = _scenes(tmp_path, {"f0.jpg": "s1"})
    with _pytest.raises(ValueError, match="at least 1"):
        balanced_rows(rows, per_video=0, path=scenes)
    with _pytest.raises(FileNotFoundError):
        balanced_rows(rows, per_video=5, path=tmp_path / "absent.csv")


def test_no_ingested_frame_ever_lands_in_a_locked_venue() -> None:
    """The locked venues are opened once, at the end, on footage nobody has looked at.

    Seven frames of `playing day.mp4` went into the labelled set on 2026-09-21 - *after* the
    clip had been rendered, gridded and compared frame by frame against
    `clipvenue_b_floodlit_track` to establish that it came from there. The locked set went
    114 -> 121. `development_rows` kept them out of training, which is why nothing failed;
    what it cannot do is un-look at them. A test set you have inspected is not a test set.

    This asserts on the real dataset because that is where the mistake happened. Frames still
    reach locked venues legitimately - the 114 were put there deliberately - so the rule is
    narrower than "nothing new in a locked venue": nothing *this project extracted from a
    clip it was choosing venues for*. The `dv` prefix is that provenance.
    """
    from pitch_occupancy.config import settings
    from pitch_occupancy.data.manifest import read_manifest
    from pitch_occupancy.data.splits import load_final_venues

    manifest = settings.dataset_dir / "manifest.csv"
    if not manifest.exists():
        pytest.skip("manifest not present")
    locked = load_final_venues()
    assert locked, "there are supposed to be locked venues"
    intruders = [r.file for r in read_manifest(manifest)
                 if r.venue in locked and r.camera.startswith("dv")]
    assert not intruders, (
        f"{len(intruders)} DaVinci frame(s) reached a locked final-test venue: "
        f"{intruders[:3]}. scripts/ingest_davinci.py refuses these; if they are on disk, "
        f"they predate the guard and must be removed rather than left."
    )
