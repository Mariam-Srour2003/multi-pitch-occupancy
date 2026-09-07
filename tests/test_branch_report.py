"""The branch reference in docs/CODEBASE.md.

It drifted once and silently: it named `feat/prompt-search` as the tip long after sixteen
further commits had landed, so the checkout instruction handed a reader a copy missing the
entire front end, the database and both searches - with nothing on the page to suggest
anything was absent. A stale pointer is worse than no pointer, because it is followed.

These check the property a reader depends on - wherever the document sends them, they get
all the work - rather than the exact counts, which change every session and would fail the
suite for reasons that are not defects.
"""

from __future__ import annotations

import re

from scripts.branch_report import (
    DOC, block, branches, contains_everything, git, tip_branch,
)

# Roots a document may legitimately name besides the tip. `pilot/model-selection` branches
# from the first commit and is deliberately not an ancestor of anything: it preserves the
# original bake-off, so pointing a reader at it is correct rather than stale.
DELIBERATE = {"pilot/model-selection", "main"}


def documented_tip() -> str:
    """Where docs/CODEBASE.md tells a reader to go for the whole project."""
    text = DOC.read_text(encoding="utf-8")
    if "`main` carries the whole project" in text:
        return "main"
    m = re.search(r"The tip branch \(`([^`]+)`\)", text)
    assert m, "docs/CODEBASE.md names neither main nor a tip branch as the place to look"
    return m.group(1)


def test_the_documented_tip_is_a_real_branch() -> None:
    assert documented_tip() in branches()


def test_the_documented_tip_actually_contains_every_other_branch() -> None:
    """The one claim a reader relies on: go here and you have all the work."""
    tip = documented_tip()
    merged = set(git("branch", "--merged", tip).replace("*", "").split())
    missing = [b for b in branches() if b not in merged and b != "pilot/model-selection"]
    assert not missing, f"{tip} does not contain {missing}; the instruction is wrong"


def test_the_document_agrees_with_git_about_where_the_code_is() -> None:
    """If main carries everything the document must say so, and stop sending readers to a
    branch - a needless instruction is one more thing that can go stale."""
    assert documented_tip() == tip_branch()


def test_main_is_preferred_once_it_carries_everything() -> None:
    if contains_everything("main"):
        assert tip_branch() == "main"


def test_the_generated_block_is_current() -> None:
    """Regenerate with `uv run python scripts/branch_report.py --write` when this fails."""
    assert block() in DOC.read_text(encoding="utf-8"), (
        "the branch block is stale; run scripts/branch_report.py --write"
    )


def test_every_branch_appears_in_the_branch_table() -> None:
    """A branch missing from the table is work nobody can find."""
    text = DOC.read_text(encoding="utf-8")
    missing = [b for b in branches() if f"`{b}`" not in text]
    assert not missing, f"branches absent from docs/CODEBASE.md: {missing}"


def test_no_document_tells_a_reader_to_check_out_a_stale_branch() -> None:
    """The README's checkout line was stale too, and it is the first thing anyone reads."""
    tip = documented_tip()
    root = DOC.parent.parent
    for name in ("README.md", "docs/CODEBASE.md"):
        text = (root / name).read_text(encoding="utf-8")
        named = set(re.findall(r"git checkout ([\w/.-]+)", text))
        stale = {b for b in named if b in branches() and b not in DELIBERATE | {tip}}
        assert not stale, f"{name} tells the reader to check out {stale}, not {tip}"
