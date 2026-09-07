"""The branch reference in docs/CODEBASE.md.

It drifted once and silently: it named `feat/prompt-search` as the tip long after sixteen
further commits had landed, so the checkout instruction handed a reader a copy missing the
entire front end, the database and both searches - with nothing on the page to suggest
anything was absent. A stale pointer is worse than no pointer, because it is followed.

These tests check the property that actually matters - the documented tip is a real branch
that contains everything - rather than the exact numbers, which change every session and
would make the suite fail for reasons that are not defects.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from scripts.branch_report import DOC, block, branches, git, tip_branch


def documented_tip() -> str:
    m = re.search(r"The tip branch \(`([^`]+)`\)", DOC.read_text(encoding="utf-8"))
    assert m, "docs/CODEBASE.md no longer names a tip branch"
    return m.group(1)


def test_the_documented_tip_is_a_real_branch() -> None:
    assert documented_tip() in branches()


def test_the_documented_tip_actually_contains_every_other_branch() -> None:
    """The one claim a reader relies on: check this out and you have all the work.

    `pilot/model-selection` is excluded because it branches from the first commit and is
    deliberately not an ancestor - it preserves the original bake-off.
    """
    tip = documented_tip()
    merged = set(git("branch", "--merged", tip).replace("*", "").split())
    missing = [b for b in branches() if b not in merged and b != "pilot/model-selection"]
    assert not missing, f"{tip} does not contain {missing}; the checkout instruction is wrong"


def test_the_checkout_command_names_the_same_branch() -> None:
    """The prose and the command have to agree; a reader copies the command."""
    text = DOC.read_text(encoding="utf-8")
    assert f"git checkout {documented_tip()}" in text


def test_the_generated_block_is_current() -> None:
    """Regenerate with `uv run python scripts/branch_report.py --write` when this fails."""
    text = DOC.read_text(encoding="utf-8")
    assert block() in text, "the branch block is stale; run scripts/branch_report.py --write"


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
        # `pilot/model-selection` is a deliberate separate root, preserved on purpose, so
        # pointing a reader at it is correct rather than stale
        expected = {tip, "pilot/model-selection", "main"}
        stale = {b for b in named if b in branches() and b not in expected}
        assert not stale, f"{name} tells the reader to check out {stale}, not {tip}"
