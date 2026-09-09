"""The defence deck (WP8-T4).

Same discipline as the threats chapter: every figure it quotes cites a claim id, and a test
asserts each id exists and re-derives. A slide plan is the last place a stale number should
survive, because it is the one document read aloud with an examiner holding the results.

The rest of these check structural promises the deck makes about itself - that the demo
section exists and is marked un-cuttable, that the reserve slides cover the questions the
red-team document says will be asked, and that the sharpest findings have not been softened
into something safer during an edit.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "thesis" / "defence_deck.md"
LEDGER = ROOT / "thesis" / "claims.toml"
REDTEAM = ROOT / "thesis" / "defence_redteam.md"

CITATION = re.compile(r"`([a-z][a-z0-9]*(?:-[a-z0-9]+){2,})`")


def text() -> str:
    return DECK.read_text(encoding="utf-8")


def flattened() -> str:
    """Wrapping *and* markdown line prefixes removed, so a phrase test matches the prose.

    Stripping the wrapping alone is not enough and the first version of this file proved it:
    a sentence spanning two lines of a blockquote flattens to "how > to evaluate", because the
    ``>`` marker survives the join. Leading ``>``, ``-``, ``*`` and ``#`` all belong to the
    formatting rather than the sentence.
    """
    lines = (re.sub(r"^[>\-*#\s]+", "", line) for line in text().lower().splitlines())
    return " ".join(" ".join(lines).split())


def cited_ids() -> set[str]:
    return set(CITATION.findall(text()))


def ledger() -> dict[str, dict]:
    return {c["id"]: c for c in tomllib.loads(LEDGER.read_text(encoding="utf-8"))["claim"]}


# --- the numbers -----------------------------------------------------------------------


def test_the_deck_exists() -> None:
    assert DECK.exists()


def test_it_cites_the_ledger_at_all() -> None:
    """Guards the regex as much as the deck: if the pattern stopped matching, the two tests
    below would pass on an empty set."""
    assert len(cited_ids()) >= 8, cited_ids()


def test_every_cited_claim_exists() -> None:
    unknown = cited_ids() - set(ledger())
    assert not unknown, f"cited but not in claims.toml: {sorted(unknown)}"


def test_every_cited_claim_re_derives_from_an_artefact() -> None:
    claims = ledger()
    for claim_id in sorted(cited_ids()):
        assert claims[claim_id].get("source"), f"{claim_id} has no source artefact"


def test_it_does_not_hardcode_a_claim_count() -> None:
    """A count in prose goes stale the moment a claim is added - which is the drift the ledger
    exists to prevent, so the deck must not reintroduce it. It did, once, saying 28."""
    assert not re.search(r"\b\d+ claims re-derived", text()), (
        "the deck states a claim count; say 'every claim' instead"
    )


# --- the figures it names --------------------------------------------------------------


@pytest.mark.parametrize("figure", [
    "leakage_decomposition", "ranking_inversion", "onboarding_cost",
])
def test_every_named_figure_has_been_generated(figure: str) -> None:
    """The deck names figures by file so they are regenerated rather than redrawn. A named
    figure that does not exist is a blank slide discovered on the day."""
    assert figure in text(), f"{figure} is no longer referenced"
    if not (ROOT / "results" / "figs").exists():
        pytest.skip("figures not generated")
    assert (ROOT / "results" / "figs" / f"{figure}.png").exists(), figure


# --- structure -------------------------------------------------------------------------


def test_the_demo_section_exists_and_is_marked_un_cuttable() -> None:
    """A defence that runs long cuts the demo, and the demo is the part a panel remembers."""
    body = flattened()
    assert "demo" in body
    assert "never cut" in body


def test_the_demo_has_a_fallback() -> None:
    """The most common defence failure is a demo that worked yesterday. Debugging in front of
    a panel is worse than the demo not running."""
    assert "if the demo fails" in flattened()


def test_the_reserve_slides_cover_the_red_team_questions() -> None:
    """The deck and the red-team document must not drift apart: a question rehearsed in one
    and absent from the other is a slide nobody prepared."""
    if not REDTEAM.exists():
        pytest.skip("red-team document not present")
    body = flattened()
    for topic in ("colour histogram", "temporal smoothing", "human ceiling", "reproduce"):
        assert topic in body, topic


def test_the_framing_decision_is_stated_up_front() -> None:
    """The deck's own argument: an apologetic reading and a strong reading of the same results
    need different opening sentences."""
    assert "how to evaluate this problem" in flattened()


@pytest.mark.parametrize("sharp", [
    "it is correct zero times",
    "camera-transfer test",
    "the real test set is two slots",
    "no inter-annotator figure exists",
])
def test_the_uncomfortable_lines_have_not_been_softened(sharp: str) -> None:
    """These are the four sentences most likely to be edited into something safer, and each is
    stronger stated plainly than discovered by an examiner."""
    assert sharp in flattened(), sharp
