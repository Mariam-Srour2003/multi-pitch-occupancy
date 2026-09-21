"""Train/test splits, materialised and leakage-guarded (WP0-T4).

Three rules this module exists to enforce, none of which survive six months of good
intentions on their own:

1. **A split is identified by its rows, not by its seed** - and the machinery to
   materialise one exists for when that is not enough. :func:`write_split` writes to
   ``results/splits/<name>.csv`` and :func:`read_split` reads it back, re-applying the
   final-venue lock.

   **Decision, 2026-09-10 (WP0-T4).** Those two have no callers outside the tests, and
   ``results/splits/`` holds only the lock file: every experiment calls :func:`grouped_split`
   with ``seed=42`` directly. That is deterministic *given the same rows*, and the proviso is
   the whole problem - the row list depends on which feature caches a script filtered to, so
   two experiments that filter differently get different splits from one seed. In September
   ``effective_sample_audit.py`` (DINOv2 cache only) counted 94 distinct scenes where
   ``h4_model_equivalence.py`` (all three) counted 95 — both correct, from ``seed=42``.
   Harmless there, and exactly the drift materialisation exists to prevent.

   **Run today the two agree** (identity ``f73f5a29b425``), because the reproducibility
   repair of 2026-09-08 brought their row lists back together. Nothing recorded that they
   had diverged, and nothing recorded that they had converged either, which is the actual
   gap: not that a number was wrong, but that no artefact could have told you.

   The plan's intent was to retrofit ``read_split`` across some twenty scripts. That was
   **not** taken, and the reason is worth stating rather than leaving as an omission:
   materialising the canonical splits now would change which rows several published
   experiments were fitted on, so it would invalidate results in order to protect them. The
   claim was dropped instead. A split's identity is the strategy, the group key, the seed
   **and the ordered row list**, and :func:`split_identity` returns a fingerprint of exactly
   that, so a disagreement between two runs is detectable instead of surfacing as an
   unexplained difference in a count. It detects; it does not prevent. Preventing is what
   the retrofit would have done, and it stays available for anyone who takes it.
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
import hashlib
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
    "check_split", "split_identity",
    "LEGACY_GROUP_KEY",
    "SYNTHETIC_SOURCE",
    "SCENE_IDS",
    "scene_ids",
    "distinct_rows",
    "balanced_rows",
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

#: `source` value marking a frame that was generated rather than recorded (A13). Generated
#: frames may be used for training only. They must never reach a test side: a model scored
#: against its own generator's output measures the generator, and the whole point of this
#: project is that an evaluation which looks fine can be measuring the wrong thing.
#: :func:`check_split` enforces this, because a convention that is only written down is the
#: defect class this repository keeps finding - a guard that exists and does not operate.
SYNTHETIC_SOURCE = "synthetic"

#: Written by `scripts/assign_scene_ids.py`. One row per frame: which scene it belongs to.
SCENE_IDS = RESULTS_DIR.parent / "data" / "processed" / "scene_ids.csv"


def scene_ids(path: Path | None = None) -> dict[str, str]:
    """``{file: scene_id}``, or empty if the sidecar has not been written.

    Empty rather than raising, because every caller treats a missing mapping as "no scenes
    known" and falls back to using every frame - which is the behaviour the project had
    before scenes existed. A caller that needs it to exist should say so; `distinct_rows`
    does.
    """
    path = path or SCENE_IDS
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["file"]: r["scene_id"] for r in csv.DictReader(fh)}


def distinct_rows(rows: list[ManifestRow], *, path: Path | None = None) -> list[ManifestRow]:
    """One frame per scene, in the order given.

    **This is a training-side tool and belongs nowhere near a test set.** Pruning a test set
    to distinct scenes would change what a reported number means - `effective_sample_audit`
    already reports the distinct count *beside* the full one for that reason, rather than
    replacing it.

    The corpus is 1,881 frames and 290 scenes. On the EMPTY class it is 525 and 28, and the
    five recorded ones are one venue over two days. Fitting on all of them tells the probe
    that those backgrounds are what an empty pitch looks like with a confidence the evidence
    does not carry: refitting on one frame per scene took false-play on unseen footage from
    0.31 to 0.00, against a cost of 0.10 macro-F1 on a test set drawn from the same repeated
    scenes.

    Raises if the sidecar is missing, because silently returning every frame would mean an
    experiment reporting itself as deduplicated while training on 1,881 near-copies.
    """
    mapping = scene_ids(path)
    if not mapping:
        raise FileNotFoundError(
            f"no scene ids at {path or SCENE_IDS}; run scripts/assign_scene_ids.py"
        )
    seen: set[str] = set()
    out: list[ManifestRow] = []
    for r in rows:
        scene = mapping.get(r.file)
        if scene is None:  # a frame the sidecar does not know; keep it rather than drop it
            out.append(r)
            continue
        if scene not in seen:
            seen.add(scene)
            out.append(r)
    return out


def balanced_rows(rows: list[ManifestRow], *, per_video: int = 12,
                  path: Path | None = None) -> list[ManifestRow]:
    """At most ``per_video`` frames from any one source recording, the most distinct first.

    **Four videos are 75% of every recorded frame.** `slot_20260712_2030_camB` alone is 516
    of 1,720, and the four venue_01 slot recordings together are 1,296; the median source
    video contributes six. A fit on that is a fit on four afternoons at one facility, and it
    will learn what those four afternoons look like - which is A20's finding arriving from a
    different direction.

    `distinct_rows` is not enough on its own here. It takes the corpus to 197 scenes, but 103
    of those still come from the same four recordings, so venue_01 keeps half the weight. The
    cap is what bounds a single camera's vote.

    **Scenes first, then frames.** Within a video the first frame of each scene is taken in
    order, and only once every scene has one are second frames taken - so a video with
    fifteen scenes contributes fifteen different moments rather than fifteen frames of its
    first. Order is preserved within a video, so the result is reproducible without a seed.

    **Training-side only**, for the reason `distinct_rows` gives: a capped test set would
    change what its number means.

    **The cap is a knob and there is no free value**, which is why it is a knob. The four
    videos that carry the bias are also the only EMPTY footage there is, so capping trades
    one problem for the other - measured on the 2026-09-21 corpus:

    ===============  ======  ======  =======  =======
    arm              frames  top 4   C1_EMPTY  C2
    ===============  ======  ======  =======  =======
    all recorded      1,720     75%      494    1,210
    distinct scenes     197     53%        5      182
    cap 12              472     10%       21      438
    cap 20              504     16%       41      450
    cap 40              584     27%       81      490
    ===============  ======  ======  =======  =======

    Twelve nearly removes the single-camera vote and leaves 21 empty frames; forty keeps 81
    and lets four videos back to a quarter of the weight. `experiments/dataset_redundancy.py`
    re-derives this table, so the choice is made against current numbers rather than these.
    """
    if per_video < 1:
        raise ValueError(f"per_video must be at least 1, got {per_video}")
    mapping = scene_ids(path)
    if not mapping:
        raise FileNotFoundError(
            f"no scene ids at {path or SCENE_IDS}; run scripts/assign_scene_ids.py"
        )
    by_video: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_video.setdefault(row.camera, []).append(row)

    keep: set[str] = set()
    for group in by_video.values():
        seen: set[str] = set()
        first_of_scene: list[ManifestRow] = []
        rest: list[ManifestRow] = []
        for row in group:
            scene = mapping.get(row.file, row.file)
            (rest if scene in seen else first_of_scene).append(row)
            seen.add(scene)
        keep.update(r.file for r in (first_of_scene + rest)[:per_video])
    return [r for r in rows if r.file in keep]


def _training_only(rows: list[ManifestRow], include_synthetic: bool) -> list[ManifestRow]:
    """The generated rows to append to a train side, or none.

    A13 says generated frames are "for training only", and the only way to implement that
    literally is to keep them out of the partition entirely and add them to train afterwards.
    Passing them *through* a split does not work and is not a near miss: handed to
    `grouped_split`, 171 of 189 landed on the **test** side, where `check_split` rejects
    them outright and every number from that split is unreportable.

    Doing it this way has a second property worth more than the first. The partition runs on
    exactly the rows it ran on before, so the test side of a with-generated split is
    *identical* to the test side without them - which is what makes the two scores an
    ablation rather than two numbers measured on different data. The first attempt grew the
    test set from 907 frames to 961 and the comparison silently stopped meaning anything.
    """
    if not include_synthetic:
        return []
    return [
        r for r in development_rows(rows, include_synthetic=True)
        if getattr(r, "source", "") == SYNTHETIC_SOURCE
    ]


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
    rows: list[ManifestRow],
    *,
    final_venues: frozenset[str] | None = None,
    include_synthetic: bool = False,
) -> list[ManifestRow]:
    """Everything a split is allowed to touch: all rows outside the locked venues.

    **Generated frames are excluded by default** (A13). They are an augmentation to be
    reported as a with/without ablation, so the default development set has to be the one
    every existing table describes - otherwise ingesting a batch silently redefines every
    published number, and nobody re-reads the tables. Pass ``include_synthetic=True`` in the
    ablation itself, and only there.
    """
    locked = load_final_venues() if final_venues is None else final_venues
    out = [r for r in rows if r.venue not in locked]
    if not include_synthetic:
        out = [r for r in out if getattr(r, "source", "") != SYNTHETIC_SOURCE]
    return out


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
    include_synthetic: bool = False,
) -> Split:
    """Partition whole groups, so no scene appears on both sides.

    Groups are shuffled and assigned to the test side until ``test_frac`` of rows is
    reached. Because whole groups move together the realised fraction is approximate.

    ``include_synthetic`` is forwarded to :func:`development_rows`, and it has to be: this
    function re-filters its input, so a caller that opted generated frames in and handed
    the result here got them **silently stripped again**. A13 admitted generated frames
    "for training only" and documented `include_synthetic=True` as the way to use them -
    and no split protocol could deliver one to a training set, so the opt-in did nothing
    at all. `check_split` still refuses to let one reach a test side.
    """
    generated = _training_only(rows, include_synthetic)
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
    train = [r for g, rs in groups.items() if g not in test_names for r in rs] + generated
    test = [r for g, rs in groups.items() if g in test_names for r in rs]
    return Split(
        name=name or f"grouped_{group_key}_seed{seed}",
        train=tuple(train),
        test=tuple(test),
        group_key=group_key,
        seed=seed,
    )


def leave_one_group_out(
    rows: list[ManifestRow], *, group_key: str = "venue", include_synthetic: bool = False
) -> Iterator[Split]:
    """One fold per group: train on every other group, test on this one.

    With ``group_key="venue"`` this is the leave-one-venue-out protocol. Report the
    distribution across folds, not only the mean — a single venue can dominate it.
    """
    generated = _training_only(rows, include_synthetic)
    rows = development_rows(rows)
    groups = _group(rows, group_key)
    for held in sorted(groups):
        train = [r for g, rs in groups.items() if g != held for r in rs] + generated
        yield Split(
            name=f"lo_{group_key}_out__{held}",
            train=tuple(train),
            test=tuple(groups[held]),
            group_key=group_key,
        )


def random_split(
    rows: list[ManifestRow], *, test_frac: float = 0.25, seed: int = 42,
    include_synthetic: bool = False,
) -> Split:
    """Stratified-by-nothing random split — **deliberately leaky**.

    Frames sampled seconds apart land on both sides, so this flatters every model. It
    exists only as the control arm of H1 (quantifying how much same-scene evaluation
    inflates accuracy). Never report a headline number from it.
    """
    generated = _training_only(rows, include_synthetic)
    rows = development_rows(rows)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * test_frac)
    return Split(
        name=f"random_seed{seed}",
        train=tuple(list(shuffled[cut:]) + generated),
        test=tuple(shuffled[:cut]),
        group_key="<none - leaky>",
        seed=seed,
    )


def temporal_split(
    rows: list[ManifestRow], *, cutoff_date: str, undated: str = "exclude",
    include_synthetic: bool = False,
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
    generated = _training_only(rows, include_synthetic)
    rows = development_rows(rows)

    missing = [r for r in rows if not r.slot_date]
    if missing and undated == "error":
        raise ValueError(
            f"{len(missing)} row(s) have no slot_date and cannot be placed on a timeline; "
            "pass undated='exclude' or undated='train'"
        )
    dated = rows if undated == "train" else [r for r in rows if r.slot_date]

    train = [r for r in dated if r.slot_date < cutoff_date] + generated
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


def _partition_digest(train, test, group_key: str, seed: int | None) -> str:
    """The hash of everything except the split's *name*.

    Separate from :func:`split_identity` because the two answer different questions. The
    identity says "is this the same split", and a split is referenced by name, so the name
    belongs in it. This says "is this the same partition", which must survive a file being
    copied to a new name - a legitimate thing to do with a materialised split, and not a
    reason to refuse to read it.
    """
    payload = "|".join([
        group_key, str(seed),
        *(r.file for r in train), "--test--", *(r.file for r in test),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def split_identity(split: Split) -> str:
    """A short fingerprint of everything that decides which rows land on which side.

    **A seed does not identify a split here.** `grouped_split(rows, seed=42)` is
    deterministic given the same rows, and that proviso is the whole problem: the row list
    depends on which feature caches the calling script filtered to.
    In September `effective_sample_audit.py` read the DINOv2 cache and counted 94 distinct
    scenes while `h4_model_equivalence.py` read all three and counted 95 - from the same
    seed, the same function and the same manifest. Harmless that time. **Not detectable**,
    which is the part worth fixing: the two agree again today, and nothing recorded either
    the divergence or the return.

    So the identity is the strategy, the group key, the seed **and the ordered row list**.
    Two runs that agree on all four have the same split; two that differ in any of them do
    not, whatever the seed says. Quoting this next to a split-derived number makes the
    difference visible instead of leaving it to be discovered by a discrepancy in a count.

    Deliberately not a guarantee of reproducibility, which is a stronger claim than a hash
    can support: it detects disagreement, it does not prevent it. Preventing it is
    `write_split`/`read_split`, and WP0-T4 records why that retrofit was not taken.
    """
    partition = _partition_digest(split.train, split.test, split.group_key, split.seed)
    return hashlib.sha256(f"{split.name}|{partition}".encode()).hexdigest()[:12]


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

    # A13. Generated frames are a training-side-only augmentation; on the test side they
    # would make every headline number unreportable, so this is checked rather than trusted.
    if synthetic := [
        r for r in split.test if getattr(r, "source", "") == SYNTHETIC_SOURCE
    ]:
        problems.append(
            f"{len(synthetic)} generated frame(s) are on the test side - synthetic data is "
            f"training-only under A13, and results from this split cannot be reported"
        )

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
        w.writerow(["file", "role", "group_key", "seed", "partition"])
        seed = "" if split.seed is None else split.seed
        digest = _partition_digest(split.train, split.test, split.group_key, split.seed)
        for role, part in (("train", split.train), ("test", split.test)):
            for r in part:
                w.writerow([r.file, role, split.group_key, seed, digest])
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
    digests: set[str] = set()
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
            if rec.get("partition"):
                digests.add(rec["partition"])

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
    # The partition is verified rather than trusted. A materialised split is a CSV, and a
    # CSV is editable: deleting a row, or flipping one from test to train, produces a file
    # that reads back cleanly and silently describes a different experiment. The digest
    # covers the rows, their roles, their order, the group key and the seed - everything
    # except the name, so copying a split to a new filename stays legal. Files written
    # before the column existed carry no digest and are read as before; that is a real
    # state, not a failure, and refusing them would break the only artefacts this path has.
    if digests:
        if len(digests) > 1:
            raise ValueError(
                f"{path.name} carries more than one partition digest ({sorted(digests)}); "
                f"it is one value per split and this file disagrees with itself"
            )
        recorded = next(iter(digests))
        actual = _partition_digest(tuple(train), tuple(test), group_key, seed)
        if actual != recorded:
            raise ValueError(
                f"{path.name} was written as partition {recorded} and reads back as "
                f"{actual}. The file has been edited, or the manifest rows it names have "
                f"changed order or role since. Regenerate it rather than using it: the "
                f"numbers it produced were computed on the partition it no longer is."
            )

    return Split(
        name=path.stem, train=tuple(train), test=tuple(test),
        group_key=group_key, seed=seed,
    )
