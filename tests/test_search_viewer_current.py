"""The published search viewer must agree with the search it claims to show.

`results/preprocess_search.html` is a committed, publishable artefact: it is what a
supervisor or examiner opens. For a day it embedded a snapshot of the run this project had
itself renamed `preprocess_search_500frame_UNTRUSTWORTHY.json` - 16 evaluations from
2026-09-06 - while the JSON beside it had moved on. Nothing on the page said so, and nothing
caught it, because the page states its provenance honestly and simply had old provenance.

This is the same guard `tests/test_branch_report.py` puts on the generated branch table, for
the same reason: a generated artefact that can silently fall behind its source will.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "results" / "preprocess_search.html"
JSON = ROOT / "results" / "preprocess_search.json"

pytestmark = pytest.mark.skipif(
    not (HTML.exists() and JSON.exists()),
    reason="the search viewer or its source has not been generated here",
)


def _payload() -> dict:
    return json.loads(JSON.read_text(encoding="utf-8"))


def _html() -> str:
    return HTML.read_text(encoding="utf-8", errors="replace")


def test_the_viewer_shows_the_run_that_is_on_disk() -> None:
    stamp = _payload()["generated"]
    assert stamp in _html(), (
        f"preprocess_search.html does not embed the current run ({stamp}); "
        "regenerate it with `uv run python experiments/make_search_viewer.py`"
    )


def test_the_viewer_covers_every_evaluation_in_the_source() -> None:
    """A page built from a truncated run is worse than no page: it reads as complete.

    Asserted as a *subset* rather than an exact count, because the payload also carries a
    `best` block whose entries repeat their winning config's hash - counting every `"hash"`
    in the file gives 90 for an 88-evaluation run. Truncation is the failure mode that
    matters, and a missing hash catches it without depending on how many times the
    generator happens to mention a winner.
    """
    payload = _payload()
    html = _html()
    assert re.search(r'"evaluations":\s*\[', html), "no evaluations array in the page"

    in_page = set(re.findall(r'"hash":\s*"([0-9a-f]{6,})"', html))
    in_source = {e["hash"] for e in payload["evaluations"] if "hash" in e}
    assert in_source, "the source run records no config hashes to check against"

    if missing := sorted(in_source - in_page):
        raise AssertionError(
            f"{len(missing)} of {len(in_source)} evaluations are absent from the page "
            f"(e.g. {missing[:3]}); regenerate it with "
            f"`uv run python experiments/make_search_viewer.py`"
        )


def test_the_source_run_is_not_the_withdrawn_one() -> None:
    """Pin the specific mistake, so re-publishing that run needs a deliberate edit."""
    assert not _payload()["generated"].startswith("2026-09-06T16:5"), (
        "preprocess_search.json is the 500-frame run this project withdrew as "
        "UNTRUSTWORTHY; do not publish a viewer from it"
    )
