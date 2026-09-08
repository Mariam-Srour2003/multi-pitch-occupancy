"""Train/test splits, materialised and leakage-guarded (WP0-T4).

Three rules this module exists to enforce, none of which survive six months of good
intentions on their own:

1. **Splits are materialised, never re-randomised.** A split is written to
   ``results/splits/<name>.csv`` and afterwards referenced by name. Two runs of the same
   experiment must see byte-identical splits.

   **Stated but not yet practised, and the gap has bitten once.** As of 2026-09-07
   :func:`write_split` and :func:`read_split` have no callers outside the tests, and
   ``results/splits/`` holds only the lock file. Every experiment calls
   :func:`grouped_split` directly with ``seed=42`` instead, which is deterministic *given
   the same rows* - and that proviso is the whole problem. The row list depends on which
   feature caches a script filters to, so two experiments that filter differently get
   different splits from the same seed: ``effective_sample_audit.py`` (DINOv2 cache only)
   counts 94 distinct scenes where ``h4_model_equivalence.py`` (all three caches) counts
   95. Harmless there, and exactly the drift materialisation exists to prevent. See
   TODO WP0-T4.
2. **Groups never straddle the boundary.** Frames sampled seconds apart, or the two
   cameras watching one pitch at one moment, are the same scene. Splitting them across
   train and test measures memorisation.
3. **The final test set is unreachable by accident.** Every function here silently drops
   rows belonging to the locked final venues. Getting at them requires calling
   :func:`final_test_rows`, which exists to be greppable: one call site, at the end.

The locked venues live in ``results/splits/FINAL_TESTSET_venues.csv``, fixed before any
model was fitted. See ``thesis/preregistration.md``.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pitch_occupancy.config import RESULTS_DIR
from pitch_occupancy.data.manifest import ManifestRow

__all__ = [
    "Split",
    "load_final_venues",
    "development_rows",
    "final_test_rows",
    "grouped_split",
    "leave_one_group_out",
    "random_split",
    "temporal_split",
    "check_split",
    "LEGACY_GROUP_KEY",
    "DEFAULT_SPLIT_DIR",
    "write_split",
    "read_split",
]

#: Resolved from the installed package, not the working directory. It was
#: ``Path("results/splits/...")``, which made the lock fail **open**: launched from
#: anywhere but the repo root the file was simply not found, :func:`load_final_venues`
#: returned an empty set, and :func:`development_rows` handed both locked venues - 114
#: frames, 29% of the clip data - straight to training with no warning at all. `config.py`
#: states the contract this restores: paths "work identically from a notebook, a test, the
#: CLI, and the scheduler on the Mini-PC".
DEFAULT_FINAL_VENUES = RESULTS_DIR / "splits" / "FINAL_TESTSET_venues.csv"

#: `group_key` of a split read from a file written before it was recorded. Not a manifest
#: field, so the group-overlap check cannot run against it - and `check_split` says so
#: rather than raising, which is what it used to do on every materialised split.
LEGACY_GROUP_KEY = "<from file - not recorded>"


@dataclass(frozen=True, slots=True)
class Split:
    """One materialised train/test partition."""

    name: str
    train: tuple[ManifestRow, ...]
    test: tuple[ManifestRow, ...]
    group_key: str
    seed: int | None = None

    @property
    def n(self) -> int:
        return len(self.train) + len(self.test)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Split({self.name!r}, train={len(self.train)}, test={len(self.test)})"


# ---------------------------------------------------------------------------
# the lock
# ---------------------------------------------------------------------------


def load_final_venues(path: Path | None = None) -> frozenset[str]:
    """Venues reserved for the final evaluation.

    A **missing** lock file raises. "No lock file" and "no locked venues" must never be
    the same value: the second is a deliberate configuration, the first is a broken
    checkout, and returning an empty set for both is precisely how a locked venue reaches
    the training set unnoticed. A file that exists and locks nothing is still allowed -
    that is a choice someone made and can be read in git.
    """
    path = path or DEFAULT_FINAL_VENUES
    if not path.exists():
        raise RuntimeError(
            f"the final-test-set lock file is missing: {path}\n"
            "Every split strategy drops locked-venue rows by consulting this file, so "
            "without it the lock is not enforced and the held-out venues silently enter "
            "development - which invalidates the one guarantee thesis/preregistration.md "
            "rests on. Restore results/splits/FINAL_TESTSET_venues.csv from git rather "
            "than running without it."
        )
    with path.open(newline="", encoding="utf-8") as fh:
        return frozenset(
            r["venue"] for r in csv.DictReader(fh) if r.get("role") == "FINAL_TEST"
        )


def development_rows(
    rows: list[ManifestRow], *, final_venues: frozenset[str] | None = None
) -> list[ManifestRow]:
    """Everything a split is allowed to touch: all rows outside the locked venues."""
    locked = load_final_venues() if final_venues is None else final_venues
    return [r for r in rows if r.venue not in locked]


def final_test_rows(
    rows: list[ManifestRow],
    *,
    i_have_finished_all_development: bool = False,
    final_venues: frozenset[str] | None = None,
) -> list[ManifestRow]:
    """The locked final test set. **Evaluate once, at the very end.**

    The keyword argument is deliberately awkward: it makes every use of the final test set
    a conscious, greppable act rather than something that happens by autocomplete. Each
    evaluation against these rows must be disclosed in the thesis.
    """
    if not i_have_finished_all_development:
        raise RuntimeError(
            "The final test set is locked. Development experiments must use "
            "development_rows(). If development is genuinely complete, pass "
            "i_have_finished_all_development=True and disclose the evaluation in "
            "thesis/preregistration.md."
        )
    locked = load_final_venues() if final_venues is None else final_venues
    return [r for r in rows if r.venue in locked]


# ---------------------------------------------------------------------------
# split strategies
# ---------------------------------------------------------------------------


def _group(rows: list[ManifestRow], key: str) -> dict[str, list[ManifestRow]]:
    out: dict[str, list[ManifestRow]] = defaultdict(list)
    for r in rows:
        out[str(getattr(r, key))].append(r)
    return dict(out)


def grouped_split(
    rows: list[ManifestRow],
    *,
    group_key: str = "slot_id",
    test_frac: float = 0.25,
    seed: int = 42,
    name: str | None = None,
) -> Split:
    """Partition whole groups, so no scene appears on both sides.

    Groups are shuffled and assigned to the test side until ``test_frac`` of rows is
    reached. Because whole groups move together the realised fraction is approximate.
    """
    rows = development_rows(rows)
    groups = _group(rows, group_key)
    names = sorted(groups)
    random.Random(seed).shuffle(names)

    target = len(rows) * test_frac
    test_names: set[str] = set()
    got = 0
    for g in names:
        if got >= target:
            break
        test_names.add(g)
        got += len(groups[g])

    # Both sides are emitted in `groups` insertion order - which follows the manifest -
    # rather than by iterating `test_names`.
    #
    # **A set is a membership test here, never an ordering.** Iterating it put the test rows
    # in an order that depends on `PYTHONHASHSEED`, so `grouped_split(seed=42)` returned the
    # same *frames* in a different *sequence* on every process. Point estimates were
    # unaffected, and that is exactly why it survived: every macro-F1 and accuracy in
    # `h1_h2_baseline_floor.csv` reproduced exactly while the bootstrap intervals beside them
    # did not, because `bootstrap_metric_ci` resamples positions and a reordered vector is a
    # different resample. A seeded function that is not reproducible across processes
    # defeats the claim the reproduction pipeline exists to back.
    train = [r for g, rs in groups.items() if g not in test_names for r in rs]
    test = [r for g, rs in groups.items() if g in test_names for r in rs]
    return Split(
        name=name or f"grouped_{group_key}_seed{seed}",
        train=tuple(train),
        test=tuple(test),
        group_key=group_key,
        seed=seed,
    )


def leave_one_group_out(
    rows: list[ManifestRow], *, group_key: str = "venue"
) -> Iterator[Split]:
    """One fold per group: train on every other group, test on this one.

    With ``group_key="venue"`` this is the leave-one-venue-out protocol. Report the
    distribution across folds, not only the mean — a single venue can dominate it.
    """
    rows = development_rows(rows)
    groups = _group(rows, group_key)
    for held in sorted(groups):
        train = [r for g, rs in groups.items() if g != held for r in rs]
        yield Split(
            name=f"lo_{group_key}_out__{held}",
            train=tuple(train),
            test=tuple(groups[held]),
            group_key=group_key,
        )


def random_split(
    rows: list[ManifestRow], *, test_frac: float = 0.25, seed: int = 42
) -> Split:
    """Stratified-by-nothing random split — **deliberately leaky**.

    Frames sampled seconds apart land on both sides, so this flatters every model. It
    exists only as the control arm of H1 (quantifying how much same-scene evaluation
    inflates accuracy). Never report a headline number from it.
    """
    rows = development_rows(rows)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * test_frac)
    return Split(
        name=f"random_seed{seed}",
        train=tuple(shuffled[cut:]),
        test=tuple(shuffled[:cut]),
        group_key="<none - leaky>",
        seed=seed,
    )


def temporal_split(
    rows: list[ManifestRow], *, cutoff_date: str, undated: str = "exclude"
) -> Split:
    """Train on everything before ``cutoff_date`` (ISO), test on and after it.

    Measures drift: does a model fitted on earlier recordings hold up later? Needs
    recordings spread over time to mean anything.

    **A frame with no ``slot_date`` has no position on the timeline**, so ``undated``
    says what to do with it rather than letting string comparison decide. This is not
    hypothetical: ``"" < "2026-07-12"`` is true, so the original version put every undated
    frame in *train* silently — and on this corpus all 282 undated frames come from the
    seven clip venues, none of which appear in the test side. The split called "temporal"
    was a venue-and-time split, and nothing said so.

    * ``"exclude"`` (default) — leave them out and let the split be about time alone.
    * ``"train"`` — the old behaviour, available deliberately rather than by accident.
    * ``"error"`` — refuse, for a caller that believes its data is fully dated.
    """
    if undated not in {"exclude", "train", "error"}:
        raise ValueError(f"undated must be exclude/train/error, not {undated!r}")
    rows = development_rows(rows)

    missing = [r for r in rows if not r.slot_date]
    if missing and undated == "error":
        raise ValueError(
            f"{len(missing)} row(s) have no slot_date and cannot be placed on a timeline; "
            "pass undated='exclude' or undated='train'"
        )
    dated = rows if undated == "train" else [r for r in rows if r.slot_date]

    train = [r for r in dated if r.slot_date < cutoff_date]
    test = [r for r in dated if r.slot_date >= cutoff_date]
    return Split(
        name=f"temporal_{cutoff_date}",
        train=tuple(train),
        test=tuple(test),
        group_key="slot_date",
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def check_split(split: Split) -> list[str]:
    """Problems that would make results from this split misleading.

    The degenerate-test-set check matters more than it sounds: on the current data a
    grouped split by slot puts the day recording in train and the night one in test,
    leaving a test set that is ~99% a single class. Accuracy on such a split is close to
    meaningless, and it is easy not to notice.
    """
    problems: list[str] = []

    if not split.train:
        problems.append("training side is empty")
    if not split.test:
        problems.append("test side is empty")
    if not split.train or not split.test:
        return problems

    # The group-overlap check is the leakage check this function exists for, so it reports
    # when it *cannot* run instead of skipping quietly. It used to raise AttributeError on
    # any split read from disk, because the round-trip lost the group key and left a
    # placeholder that is not a manifest field - so the validation was unavailable on
    # exactly the splits the plan says are referenced by name and never re-randomised.
    if split.group_key == "<none - leaky>":
        pass
    elif not hasattr(next(iter(split.train)), split.group_key):
        problems.append(
            f"group overlap could not be checked: group_key {split.group_key!r} is not a "
            f"manifest field, so this split is not fully validated"
        )
    else:
        tr = {str(getattr(r, split.group_key)) for r in split.train}
        te = {str(getattr(r, split.group_key)) for r in split.test}
        if overlap := tr & te:
            problems.append(
                f"{len(overlap)} group(s) appear on both sides of a {split.group_key} "
                f"split: {sorted(overlap)[:5]}"
            )

    if shared := {r.file for r in split.train} & {r.file for r in split.test}:
        problems.append(f"{len(shared)} frame(s) appear in both train and test")

    train_classes = {r.class3 for r in split.train}
    test_counts = Counter(r.class3 for r in split.test)
    for missing in sorted(train_classes - set(test_counts)):
        problems.append(f"class {missing} is in train but absent from test - untestable")

    total = sum(test_counts.values())
    if test_counts:
        top, n = test_counts.most_common(1)[0]
        if n / total >= 0.95 and len(train_classes) > 1:
            problems.append(
                f"test set is {n / total:.0%} {top} - close to single-class, so accuracy "
                f"on it is not informative"
            )
    return problems


# ---------------------------------------------------------------------------
# materialisation
# ---------------------------------------------------------------------------


#: Same fix as DEFAULT_FINAL_VENUES, same reason. This was ``Path("results/splits")``, so
#: `write_split(split)` from any other directory scattered materialised splits into
#: whatever the working directory happened to be - and the final-test-set lock lives in
#: this very folder.
DEFAULT_SPLIT_DIR = RESULTS_DIR / "splits"


def write_split(split: Split, out_dir: Path | None = None) -> Path:
    """Persist a split so experiments reference it by name instead of regenerating it.

    ``group_key`` and ``seed`` are written alongside every row. They are constant per file
    and so technically redundant, but without them the round-trip is lossy: a split read
    back had ``group_key="<from file>"``, which is not a manifest field, so
    :func:`check_split` - the leakage validation this module exists for - raised
    ``AttributeError`` on every materialised split rather than checking it. Redundant
    columns also make the file self-describing to a human, which matters because these
    files are cited artefacts.
    """
    out_dir = out_dir or DEFAULT_SPLIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split.name}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "role", "group_key", "seed"])
        seed = "" if split.seed is None else split.seed
        for role, part in (("train", split.train), ("test", split.test)):
            for r in part:
                w.writerow([r.file, role, split.group_key, seed])
    return path


def read_split(path: Path, rows: list[ManifestRow]) -> Split:
    """Rehydrate a materialised split against a manifest.

    Raises if the manifest no longer contains a file the split names - a split that
    silently shrinks because frames were relabelled or deleted is worse than a crash.

    **The final-venue lock is re-applied on read.** A split is a file, and a file written
    before the lock existed - or written while the lock was failing open, which was possible
    until the path became package-relative - can name a locked venue. Reconstructing it
    without checking would walk the lock straight past the guard in
    :func:`development_rows`, since nothing downstream looks again.
    """
    by_file = {r.file: r for r in rows}
    train, test = [], []
    group_keys: set[str] = set()
    seeds: set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            row = by_file.get(rec["file"])
            if row is None:
                raise KeyError(
                    f"{path.name} references {rec['file']!r}, which is not in the "
                    f"manifest - the dataset changed since this split was written"
                )
            (train if rec["role"] == "train" else test).append(row)
            if rec.get("group_key"):
                group_keys.add(rec["group_key"])
            if rec.get("seed"):
                seeds.add(rec["seed"])

    locked = load_final_venues()
    if trespassers := sorted({r.venue for r in (*train, *test)} & locked):
        raise RuntimeError(
            f"{path.name} contains locked final-test venue(s) {trespassers}. It was "
            f"written before the lock, or while it was not being enforced. Regenerate it "
            f"from the manifest rather than using it - a materialised split is not a "
            f"licence to skip the lock."
        )

    if len(group_keys) > 1:
        raise ValueError(
            f"{path.name} names more than one group_key ({sorted(group_keys)}); it is one "
            f"value per split and this file disagrees with itself"
        )
    # A file written before group_key was recorded keeps the old sentinel, and
    # `check_split` reports the group check as unrunnable rather than crashing on it.
    group_key = next(iter(group_keys), LEGACY_GROUP_KEY)
    seed = int(next(iter(seeds))) if len(seeds) == 1 else None
    return Split(
        name=path.stem, train=tuple(train), test=tuple(test),
        group_key=group_key, seed=seed,
    )
