"""The claims ledger (WP8-T5).

Every quantitative claim the write-up makes, checked against the artefact that produced it.
This is the generalisation of the guards added one at a time elsewhere: a figure whose ranks
were hardcoded, a README describing a project that had not started, an export that stopped
covering the thesis. Each was one claim with nothing checking it; the ledger is the place
where that stops being a per-claim accident.

The suite therefore has to check the ledger itself as well as the claims, because a ledger
that quietly loses entries would be the same failure one level up.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
vc = importlib.import_module("experiments.verify_claims")


@pytest.fixture(scope="module")
def claims() -> list[dict]:
    return vc.load()


# --- the claims themselves ------------------------------------------------------


def test_every_claim_verifies(claims) -> None:
    """The whole point. A failure here is either a result that moved without the prose
    following, or prose that moved without the ledger following - the message says which."""
    failed = [
        f"{r['id']}: {r['state']} - {r['detail']}"
        for r in (vc.check(c) for c in claims)
        if r["state"] not in ("ok", "unsupported")
    ]
    assert not failed, "claims that no longer hold:\n  " + "\n  ".join(failed)


def test_unsupported_claims_say_why(claims) -> None:
    """An unsupported claim is not a failure; an unsupported claim nobody knows about is."""
    for claim in claims:
        if not claim.get("source"):
            assert claim.get("note"), f"{claim['id']} has no source and no reason given"


def test_claim_ids_are_unique(claims) -> None:
    ids = [c["id"] for c in claims]
    assert len(ids) == len(set(ids))


def test_every_claim_has_a_statement_and_a_value(claims) -> None:
    for claim in claims:
        assert claim.get("statement"), claim["id"]
        assert isinstance(claim.get("value"), (int, float)), claim["id"]


def test_every_named_document_exists(claims) -> None:
    """A `where` pointing at a renamed file would silently stop checking anything."""
    for claim in claims:
        for where in claim.get("where", []):
            assert (ROOT / where).exists(), f"{claim['id']} names a missing {where}"


def test_the_log_is_not_used_as_a_live_document(claims) -> None:
    """`EXPERIMENT_LOG.md` is append-only history: entries record what was true when
    written, and superseded ones stay with a retraction beside them. Requiring it to match
    current values would forbid keeping that history."""
    for claim in claims:
        assert "EXPERIMENT_LOG" not in " ".join(claim.get("where", [])), claim["id"]


# --- the verifier's own machinery -----------------------------------------------


def test_a_moved_result_is_caught(tmp_path, monkeypatch) -> None:
    """The first failure mode, exercised end to end rather than assumed."""
    source = tmp_path / "r.csv"
    source.write_text("model,score\ndinov2,0.9297\n", encoding="utf-8")
    monkeypatch.setattr(vc, "ROOT", tmp_path)
    claim = {"id": "x", "value": 0.9297, "source": "r.csv",
             "select": {"model": "dinov2"}, "column": "score", "where": []}
    assert vc.check(claim)["state"] == "ok"

    source.write_text("model,score\ndinov2,0.8000\n", encoding="utf-8")
    got = vc.check(claim)
    assert got["state"] == "stale result"
    assert "0.8000" in got["detail"]


def test_prose_that_lost_the_number_is_caught(tmp_path, monkeypatch) -> None:
    """The second failure mode: the ledger and the artefact agree, the document does not."""
    (tmp_path / "r.csv").write_text("model,score\ndinov2,0.9297\n", encoding="utf-8")
    doc = tmp_path / "doc.md"
    doc.write_text("DINOv2 holds 0.9297 recall.\n", encoding="utf-8")
    monkeypatch.setattr(vc, "ROOT", tmp_path)
    claim = {"id": "x", "value": 0.9297, "source": "r.csv",
             "select": {"model": "dinov2"}, "column": "score", "where": ["doc.md"]}
    assert vc.check(claim)["state"] == "ok"

    doc.write_text("DINOv2 holds high recall.\n", encoding="utf-8")
    assert vc.check(claim)["state"] == "stale prose"


def test_an_ambiguous_select_is_an_error_not_a_first_match(tmp_path, monkeypatch) -> None:
    """"The ConvNeXtV2 row" quietly becoming "the first of five" is how a claim starts
    describing something other than what it says."""
    (tmp_path / "r.csv").write_text(
        "model,score\ndinov2,0.9\ndinov2,0.5\n", encoding="utf-8")
    monkeypatch.setattr(vc, "ROOT", tmp_path)
    claim = {"id": "x", "value": 0.9, "source": "r.csv",
             "select": {"model": "dinov2"}, "column": "score", "where": []}
    got = vc.check(claim)
    assert got["state"] == "cannot derive"
    assert "aggregate" in got["detail"]

    claim["aggregate"] = "mean"
    claim["value"] = 0.7
    assert vc.check(claim)["state"] == "ok"


def test_a_missing_source_is_reported_not_skipped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(vc, "ROOT", tmp_path)
    claim = {"id": "x", "value": 1.0, "source": "gone.csv",
             "select": {}, "column": "c", "where": []}
    assert vc.check(claim)["state"] == "missing source"


def test_renderings_accept_how_a_number_is_actually_written() -> None:
    """A claim of 0.9918 may appear as 0.9918, 0.992 or 99.2%, and a negative delta is as
    often written without its sign. Accepting those keeps the prose check from being a
    formatting rule."""
    got = vc.renderings(0.9918)
    assert "0.9918" in got and "0.992" in got and "99.2%" in got
    assert "0.2712" in vc.renderings(-0.2712), "a signed delta must match its bare form"


def test_the_generated_ledger_is_current(claims) -> None:
    """`thesis/claims.md` is generated; a stale copy is a document that disagrees with the
    ledger it was made from - the exact failure this whole file exists for."""
    if not vc.OUT.exists():
        pytest.skip("claims.md not generated")
    results = [vc.check(c) for c in claims]
    assert vc.markdown(claims, results) == vc.OUT.read_text(encoding="utf-8"), (
        "thesis/claims.md is stale; run `python -m experiments.verify_claims`"
    )


def test_a_rendering_must_be_a_number_not_a_prefix_of_one() -> None:
    """The loose renderings are prefixes of other numbers: a claim of 0.9297 offers "0.93",
    and plain `in` accepts that inside "0.9302". The check would then pass on a document
    that states a different number and never states this one."""
    assert "0.93" in "the value was 0.9302"          # what plain `in` would accept
    assert not vc.states("the value was 0.9302", 0.9297)


def test_prose_is_allowed_to_round() -> None:
    """A rounded statement is still a statement of the value; requiring four decimals in
    prose would make the ledger a formatting rule rather than a consistency check."""
    assert vc.states("DINOv2 holds 0.930 recall across unseen venues.", 0.9297)
    assert vc.states("It calls 99.2% of empty pitches a match.", 0.9918)
