"""Regenerate the branch summary in `docs/CODEBASE.md` from git itself.

The summary had already drifted once, and silently: it named `feat/prompt-search` as the
tip long after sixteen further commits had landed, so anyone following the checkout
instruction - a supervisor, an examiner, a future reader - would have got a copy missing the
entire front end, the database and both searches, with nothing to indicate anything was
absent. Hand-maintained counts in a document that changes every session cannot stay true.

So the numbers come from git. This rewrites the block between the two markers in
`docs/CODEBASE.md` and leaves the rest of the file alone.

    uv run python scripts/branch_report.py           # show what it would write
    uv run python scripts/branch_report.py --write   # rewrite the block
    uv run python scripts/branch_report.py --check    # exit 1 if the block is stale

**Run it after merging to main.** Everything here is computed from the branches already
merged into `main`, so it is safe to run - and to check - from anywhere; it simply will not
mention work that has not landed yet. Merging is what adds a branch to the history, so
merging is what changes the document.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "CODEBASE.md"
START = "<!-- branch-report:start -->"
END = "<!-- branch-report:end -->"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def branches() -> list[str]:
    """Every local branch."""
    return git("for-each-ref", "--format=%(refname:short)", "refs/heads/").splitlines()


def documented_branches() -> list[str]:
    """The branches this reference describes: those already merged into `main`.

    **Not simply every branch, and the difference is what makes the checks usable.** The
    document says where the code is and how it was built, so it describes the history that
    is *in* `main`. A branch created five minutes ago and not yet merged is not part of that
    history, and treating it as missing documentation made the tests fail on every feature
    branch from the moment it was created - which taught the only person running them to
    pass --deselect, and a test that is routinely skipped protects nothing.

    Merging is what adds a branch to the story, so merging is what updates the document.
    """
    merged = set(git("branch", "--merged", "main").replace("*", "").split())
    return [b for b in branches() if b in merged]


def contains_everything(branch: str) -> bool:
    """Does this branch contain every other branch?

    `pilot/model-selection` is excluded by construction rather than by name: it branches
    from the first commit and is an ancestor of nothing, so it is never contained by
    anything and would make this false for every branch.
    """
    merged = set(git("branch", "--merged", branch).replace("*", "").split())
    return all(b in merged for b in documented_branches() if b != "pilot/model-selection")


def tip_branch() -> str:
    """Where a reader should go to get the whole project.

    `main` wins whenever it qualifies. It is what a repository link opens on, so if it
    carries the work then sending anyone anywhere else is a needless instruction to
    follow - and one more thing that can go stale.
    """
    if contains_everything("main"):
        return "main"
    best, best_count = "", -1
    for branch in branches():
        contained = [
            b for b in git("branch", "--merged", branch).replace("*", "").split()
            if b != branch
        ]
        if len(contained) > best_count:
            best, best_count = branch, len(contained)
    return best


def counts(ref: str) -> dict[str, int]:
    def n(*pattern: str) -> int:
        out = git("ls-tree", "-r", "--name-only", ref)
        return sum(
            1 for line in out.splitlines()
            if line.endswith(".py") and any(line.startswith(p) for p in pattern)
        )

    return {
        "commits": int(git("rev-list", "--count", ref)),
        "branches": len(documented_branches()),
        "src": n("src/"),
        "experiments": n("experiments/"),
        "tests": n("tests/"),
    }


def block() -> str:
    tip = tip_branch()
    # Counts come from the documented tip, never from HEAD: run on a feature branch this
    # would otherwise describe that branch and disagree with itself once merged.
    c = counts(tip)
    others = len(documented_branches()) - 2  # the tip and the pilot are named separately

    if tip == "main":
        # No commit count here, deliberately. It changes with every commit - including the
        # commit that regenerates this block - so the block could never be current and its
        # own test failed permanently. The file counts move only when real work lands.
        where = f"""`main` carries the whole project — {c['src']} source modules,
{c['experiments']} experiment scripts, {c['tests']} test files. Clone it and everything is
there; no branch to check out first.

```bash
git clone <repo> && cd multi-pitch-occupancy
uv sync
```

The work was built on a chain of {others} stacked feature branches, each on top of the last, and
**they are all still there**. That history is the point: each branch is one reviewable step
with its own diff and its own commit message explaining what was found. To read the project as
it was built, walk them in the order of the table below.

```
main  <- everything is here
  └── history: setup/project-scaffold → feat/dataset-manifest → ... → {others} branches, each stacked on the last
```"""
    else:
        where = f"""`main` holds **only the planning documents**. Every line of code sits on a chain of
feature branches, each built on the one before, with the newest work at the tip.

```
main                     planning documents only
  └── setup/project-scaffold
        └── feat/dataset-manifest
              └── ... {others} more branches, each stacked on the last ...
                    └── {tip}      <- the tip: everything is here
```

**The tip branch (`{tip}`) contains the whole project** — {c['commits']} commits,
{c['src']} source modules, {c['experiments']} experiment scripts, {c['tests']} test files.

```bash
git checkout {tip}
```

Because the branches are stacked rather than parallel, they cannot be merged independently —
each contains all its ancestors. To bring everything into `main` at once:

```bash
git checkout main && git merge {tip}
```"""

    return f"""{START}
{where}

> This block is generated by `scripts/branch_report.py` from git. It once named a tip that
> was sixteen commits stale, handing readers a copy with no front end and neither search;
> regenerate it rather than editing the numbers by hand.
{END}"""


TABLE_START = "<!-- branch-table:start -->"
TABLE_END = "<!-- branch-table:end -->"


def chain() -> list[str]:
    """The stacked branches in the order they were built.

    Ordered by how many commits each contains: because every branch sits on top of the last,
    that count only ever grows, so it recovers the build order exactly - and unlike commit
    dates it cannot be disturbed by a rebase or a clock.
    """
    ordered = sorted(
        (b for b in documented_branches() if b != "pilot/model-selection"),
        key=lambda b: int(git("rev-list", "--count", b)),
    )
    return ordered


def branch_table() -> str:
    """One row per branch, described by its own commit subjects.

    Generated rather than written, because the hand-maintained version went stale the moment
    a branch was added and the test that noticed it became a chore rather than a signal. The
    commit messages are the description: they were written when the work was fresh, and a
    branch whose subjects do not explain it has a commit-message problem, not a table problem.
    """
    rows = [
        "| # | Branch | Commits | What it contains |",
        "|---|---|---|---|",
        "| — | `pilot/model-selection` | 2 | The **original pilot**, preserved. The bake-off "
        "that chose the four models, with its own run guide. Branches from the first commit "
        "and is deliberately not an ancestor of the others. |",
    ]
    previous: str | None = None
    step = 0
    for branch in chain():
        span = branch if previous is None else f"{previous}..{branch}"
        subjects = [s for s in git("log", "--format=%s", span).splitlines() if s]
        if branch == "main":
            # main is the integration point, not a step in the chain. It carries no commits
            # of its own - it is fast-forwarded to whatever the newest branch reached - so a
            # row saying "0 commits" would read as though nothing were there.
            rows.append(
                "| — | `main` | all | **The integration branch.** Fast-forwarded to the tip "
                "of the chain, so it contains every commit above. |"
            )
        else:
            step += 1
            rows.append(
                f"| {step} | `{branch}` | {len(subjects)} | {' · '.join(subjects)} |"
            )
        previous = branch
    return f"{TABLE_START}\n" + "\n".join(rows) + f"\n{TABLE_END}"


def rewrite_table(text: str) -> str:
    before, _, rest = text.partition(TABLE_START)
    _, _, after = rest.partition(TABLE_END)
    if not rest:
        return text  # markers absent; the table is still hand-maintained
    return before + branch_table() + after


def rewrite(text: str) -> str:
    before, _, rest = text.partition(START)
    _, _, after = rest.partition(END)
    if not rest or not after and END not in rest:
        raise SystemExit(f"markers {START} / {END} not found in {DOC.name}")
    return before + block() + after


def main() -> None:
    # The block contains the box-drawing characters of the branch tree, and a Windows
    # console defaults to cp1252, which cannot encode them - so printing the preview
    # crashed while --write (which writes UTF-8 explicitly) worked.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="rewrite the block in place")
    ap.add_argument("--check", action="store_true", help="exit 1 if the block is stale")
    args = ap.parse_args()

    current = DOC.read_text(encoding="utf-8")
    updated = rewrite_table(rewrite(current))
    if args.write:
        DOC.write_text(updated, encoding="utf-8")
        print(f"rewrote the branch block in {DOC.relative_to(ROOT)}")
    elif args.check:
        if updated != current:
            print("the branch block in docs/CODEBASE.md is stale; run with --write")
            sys.exit(1)
        print("branch block is current")
    else:
        print(block())


if __name__ == "__main__":
    main()
