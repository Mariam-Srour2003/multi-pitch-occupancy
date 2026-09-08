"""Retention: delete what the commitment says must be deleted, and nothing else (WP6-T8).

`thesis/ethics.md` commits to *"raw frames purged after 7 days in production; evidence images
retained 365 days for audit"*, and adds **"Code enforces this (WP6-T8)"** — which was a claim
about code that did not exist. This is that code, and the numbers here are read from that
commitment rather than chosen, so the two cannot drift apart.

The dangerous half
------------------

**A retention worker in this repository can delete the one irreplaceable thing.** `data/raw/`
is the source footage — 4.2 GB from a client facility, one copy, not regenerable — and
`data/processed/` is the labelled corpus that represents weeks of hand labelling.
`docs/backup.md` exists because of exactly that.

So the protected roots are named, the planner refuses to consider anything under them, and a
test asserts the refusal. "Raw frames" in the ethics commitment means *frames sampled by the
running system*, which live in `data/interim/`; it does not mean the research corpus, and a
worker that conflated the two would be the most expensive bug this project could ship.

The cautious half
-----------------

* **Dry run is the default and the only thing `plan()` does.** Deleting requires
  :func:`apply`, an explicit call with an explicit flag.
* **Nothing is deleted without an age.** A file whose modification time cannot be read is
  kept and reported, never swept up in a wildcard.
* **The plan is printable and the acceptance criterion is that it prints the right set** — so
  the plan is the artefact, and deletion is an afterthought that operates on it.

    uv run pitch retention              # what would be deleted, and how much space
    uv run pitch retention --apply      # actually delete it
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from pitch_occupancy.config import settings

__all__ = [
    "RetentionPolicy", "Candidate", "RetentionPlan", "PROTECTED_ROOTS", "plan", "apply",
]

#: Never considered for deletion, whatever the policy says. The source footage came from a
#: client facility on one occasion and the labelled corpus is weeks of hand labelling;
#: neither is regenerable and neither is what "raw frames" means in the ethics commitment.
#: This is a list rather than a convention because a convention is a thing to be forgotten
#: at four in the afternoon.
PROTECTED_ROOTS: tuple[Path, ...] = (
    settings.raw_dir,
    settings.dataset_dir,
    settings.reference_dir,
    settings.feature_cache_dir,
    settings.results_dir,
)


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Periods from `thesis/ethics.md`, not from preference.

    If the ethics document is revised, these change with it and the docstring of whichever
    is edited second is wrong - so the numbers appear in one place in code and the test
    checks them against the document.
    """

    #: Frames sampled by the running system. Purged after a week.
    sampled_frame_days: int = 7
    #: Images bound to a verdict for audit. Kept a year.
    evidence_days: int = 365


@dataclass(frozen=True, slots=True)
class Candidate:
    path: Path
    age_days: float
    bytes: int
    reason: str


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    """What would be deleted, what is kept, and what could not be read.

    `unreadable` is not an error state to be swallowed: a file whose age cannot be
    determined is kept, listed, and left for a person. Deleting on a failed `stat()` is how
    a retention worker becomes a data-loss incident.
    """

    delete: tuple[Candidate, ...] = ()
    kept: int = 0
    unreadable: tuple[Path, ...] = ()
    scanned_roots: tuple[Path, ...] = ()
    policy: RetentionPolicy = field(default_factory=RetentionPolicy)

    @property
    def bytes_freed(self) -> int:
        return sum(c.bytes for c in self.delete)

    def describe(self) -> str:
        mb = self.bytes_freed / 1e6
        lines = [
            f"retention: {len(self.delete)} file(s) to delete, {mb:.1f} MB, "
            f"{self.kept} kept",
            f"  policy: sampled frames {self.policy.sampled_frame_days} d, "
            f"evidence {self.policy.evidence_days} d  (thesis/ethics.md)",
            "  scanned: " + (", ".join(str(r) for r in self.scanned_roots) or "nothing"),
        ]
        for c in sorted(self.delete, key=lambda c: -c.age_days)[:10]:
            lines.append(f"    {c.age_days:6.1f} d  {c.bytes / 1e6:7.2f} MB  {c.reason}  {c.path}")
        if len(self.delete) > 10:
            lines.append(f"    ... and {len(self.delete) - 10} more")
        if self.unreadable:
            lines.append(f"  {len(self.unreadable)} file(s) could not be aged and are KEPT:")
            for p in self.unreadable[:5]:
                lines.append(f"    {p}")
        return "\n".join(lines)


def _is_protected(path: Path) -> bool:
    resolved = path.resolve()
    for root in PROTECTED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def plan(
    policy: RetentionPolicy | None = None,
    *,
    now: float | None = None,
    sampled_dir: Path | None = None,
    evidence_dir: Path | None = None,
) -> RetentionPlan:
    """What retention *would* delete. This function never deletes anything.

    ``now`` is injectable so the behaviour is testable without waiting a year, which is the
    only way a 365-day rule ever gets a test.
    """
    pol = policy or RetentionPolicy()
    clock = time.time() if now is None else now
    roots = [
        (sampled_dir or settings.interim_dir, pol.sampled_frame_days, "sampled frame"),
        (evidence_dir or settings.evidence_dir, pol.evidence_days, "evidence"),
    ]

    delete: list[Candidate] = []
    unreadable: list[Path] = []
    kept = 0
    scanned: list[Path] = []

    for root, days, reason in roots:
        if _is_protected(root):
            raise ValueError(
                f"refusing to apply retention to {root}: it is under a protected root. "
                f"'Raw frames' in the ethics commitment means frames sampled by the running "
                f"system, not the source footage or the labelled corpus."
            )
        if not root.exists():
            continue
        scanned.append(root)
        for path in sorted(root.rglob("*")):
            # `is_file()` stats too, so it belongs inside the guard. It was outside, and a
            # test that made stat() fail escaped through it - which is the same shape as
            # the bug the guard exists to prevent, one call earlier.
            try:
                if not path.is_file():
                    continue
                stat = path.stat()
                age = (clock - stat.st_mtime) / 86400.0
                size = stat.st_size
            except OSError:
                unreadable.append(path)
                continue
            if age > days:
                delete.append(Candidate(path, age, size, reason))
            else:
                kept += 1

    return RetentionPlan(
        delete=tuple(delete), kept=kept, unreadable=tuple(unreadable),
        scanned_roots=tuple(scanned), policy=pol,
    )


def apply(retention_plan: RetentionPlan, *, confirm: bool = False) -> int:
    """Delete what the plan names. Returns the number removed.

    ``confirm`` is required and defaults to False, so calling this by accident does nothing.
    A protected path in the plan aborts before anything is removed rather than being skipped
    quietly - a plan that contains one was built wrongly, and the rest of it is not to be
    trusted either.
    """
    if not confirm:
        return 0
    protected = [c.path for c in retention_plan.delete if _is_protected(c.path)]
    if protected:
        raise ValueError(
            f"plan contains {len(protected)} protected path(s), starting with "
            f"{protected[0]} - refusing to delete anything from it"
        )
    removed = 0
    for candidate in retention_plan.delete:
        try:
            candidate.path.unlink()
            removed += 1
        except OSError:
            continue
    return removed
