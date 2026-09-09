"""The threats-to-validity chapter (WP8-T3).

A validity chapter is where a thesis states its own weaknesses, so a *wrong number* here is
worse than a wrong number anywhere else: it is a self-criticism the author cannot defend, in
the section an examiner reads most carefully.

The document cites claim ids rather than only printing figures. These tests check that every
id it cites exists and is one the ledger re-derives, so a claim that is renamed, removed or
starts failing takes this chapter down with it instead of leaving it quietly stale.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "thesis" / "threats_to_validity.md"
LEDGER = ROOT / "thesis" / "claims.toml"

#: Backticked tokens that look like claim ids: lowercase words joined by hyphens. Deliberately
#: narrow - it must not match file paths, function names or `1 - false_play`.
CITATION = re.compile(r"`([a-z][a-z0-9]*(?:-[a-z0-9]+){2,})`")


def cited_ids() -> set[str]:
    return set(CITATION.findall(DOC.read_text(encoding="utf-8")))


def ledger_ids() -> set[str]:
    return {c["id"] for c in tomllib.loads(LEDGER.read_text(encoding="utf-8"))["claim"]}


def test_the_chapter_exists() -> None:
    assert DOC.exists()


def test_it_cites_the_ledger_at_all() -> None:
    """Guards the regex as much as the document: if the pattern stopped matching, every test
    below would pass on an empty set and prove nothing."""
    assert len(cited_ids()) >= 10, cited_ids()


def test_every_cited_claim_exists_in_the_ledger() -> None:
    """A renamed or deleted claim must break this chapter rather than leave a dangling
    reference that reads as evidence."""
    unknown = cited_ids() - ledger_ids()
    assert not unknown, f"cited but not in claims.toml: {sorted(unknown)}"


def test_every_cited_claim_is_one_the_ledger_re_derives() -> None:
    """`verify_claims` re-derives each value from its artefact. A claim carrying no source is
    an assertion, and this chapter must not rest on one."""
    claims = {c["id"]: c for c in tomllib.loads(LEDGER.read_text(encoding="utf-8"))["claim"]}
    for claim_id in sorted(cited_ids()):
        assert claims[claim_id].get("source"), f"{claim_id} has no source artefact"


def test_the_three_confidence_labels_are_all_used() -> None:
    """The taxonomy is the point of the chapter. If every threat were "mitigated" it would be
    a marketing document, and if none were the project would have no defence."""
    text = DOC.read_text(encoding="utf-8")
    for label in ("*(mitigated", "*(quantified", "*(unquantifiable"):
        assert label in text.lower(), label


def test_it_covers_all_four_kinds_of_validity() -> None:
    text = flattened()
    for kind in ("construct validity", "internal validity", "external validity",
                 "conclusion validity"):
        assert kind in text, kind


def test_the_false_play_correction_is_stated() -> None:
    """The chapter's sharpest claim, and the one most likely to be softened in an edit: the
    published rate has been read as something it does not measure."""
    text = flattened()
    assert "camera-transfer test" in text
    assert "false-play-is-not-accuracy" in text


def flattened() -> str:
    """The document with its line wrapping removed.

    Prose is wrapped at 100 columns, so a phrase test that matches the raw text fails the
    moment a paragraph is re-wrapped - which says nothing about the content. The first version
    of the test below did exactly that on "has never run this\\ncode".

    Markdown line prefixes go too, before this file needs it: a sentence spanning two lines of
    a blockquote flattens to "how > to evaluate" if the ``>`` survives the join, which is how
    the defence-deck test came to fail on a document that said exactly the right thing.
    """
    lines = (
        re.sub(r"^[>\-*#\s]+", "", line)
        for line in DOC.read_text(encoding="utf-8").lower().splitlines()
    )
    return " ".join(" ".join(lines).split())


@pytest.mark.parametrize("unfixable", [
    "no empty pitch outside",
    "the split does not exist",
    "never run this code",
])
def test_the_unquantifiable_limits_are_named_rather_than_softened(unfixable) -> None:
    """These are the ones an examiner will find anyway. Naming them is the strong move."""
    assert unfixable in flattened(), unfixable
