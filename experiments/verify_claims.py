"""WP8-T5 - re-derive every quantitative claim from the artefact that produced it.

The ledger lives in `thesis/claims.toml`. This reads it, recomputes each claim's value from
its source, and checks that the same number appears in every document the claim says it
appears in. Then it writes `thesis/claims.md`, which is generated and must not be edited.

**Why a ledger has to be executable.** A hand-maintained one is a document like any other,
and every hand-maintained claim in this project has drifted at least once: a figure with its
ranks hardcoded contradicted its own CSV while the reproduction stage reported success; the
README described a project that had not started; the standalone export silently stopped
covering the thesis; a diagnostic was quoted for a day before a second measurement retracted
it. None of those were caught by a test. A ledger nobody runs would join them.

A claim can fail in three distinct ways and the output separates them, because they call for
different work:

* **stale result** - the source no longer produces the recorded value, so a number changed
  and the prose did not;
* **stale prose** - the value is absent from a document that is supposed to state it, so the
  prose changed and the ledger did not;
* **unsupported** - the claim names no source, so nothing checks it at all. These are listed
  rather than hidden, and the reason each one cannot be checked is recorded with it. An
  unsupported claim is not a failure; an unsupported claim that nobody knows about is.

    uv run python -m experiments.verify_claims          # check, and rewrite claims.md
    uv run python -m experiments.verify_claims --check  # check only; exit 1 on any failure
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tomllib
from pathlib import Path
from typing import Any

from pitch_occupancy.config import settings
from pitch_occupancy.evaluation.experiment_log import record

ROOT = settings.results_dir.parent
LEDGER = ROOT / "thesis" / "claims.toml"
OUT = ROOT / "thesis" / "claims.md"
DEFAULT_TOLERANCE = 5e-4


def load() -> list[dict]:
    return tomllib.loads(LEDGER.read_text(encoding="utf-8"))["claim"]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _matching(rows: list[dict[str, str]], select: dict[str, Any]) -> list[dict[str, str]]:
    return [r for r in rows if all(str(r.get(k, "")) == str(v) for k, v in select.items())]


def _value(source: Path, claim: dict, select_key: str, column_key: str) -> float:
    """One number from one artefact, or a ValueError naming what was wrong.

    Deliberately strict about ambiguity: a select that matches several rows without an
    explicit ``aggregate`` is an error rather than a silent first-match, because "the
    ConvNeXtV2 row" quietly becoming "the first of five ConvNeXtV2 rows" is how a claim
    starts describing something other than what it says.
    """
    if source.suffix == ".json":
        blob = json.loads(source.read_text(encoding="utf-8"))
        for part in claim["path"].split("."):
            blob = blob[part]
        return float(blob)

    matched = _matching(_rows(source), claim.get(select_key, {}))
    if not matched:
        raise ValueError(f"no row matches {claim.get(select_key)}")
    column = claim[column_key]
    values = [float(r[column]) for r in matched if r.get(column) not in (None, "")]
    if not values:
        raise ValueError(f"column {column!r} is empty on every matching row")

    aggregate = claim.get("aggregate")
    if aggregate is None:
        if len(values) > 1:
            raise ValueError(
                f"{len(values)} rows match {claim.get(select_key)}; add aggregate = "
                f'"mean" or "max" if that is intended'
            )
        return values[0]
    if aggregate == "mean":
        return sum(values) / len(values)
    if aggregate == "max":
        return max(values)
    if aggregate == "min":
        return min(values)
    if aggregate == "sum":
        return sum(values)
    raise ValueError(f"unknown aggregate {aggregate!r}")


def renderings(value: float) -> list[str]:
    """How a number might legitimately be written in prose.

    A claim of 0.9918 may appear as 0.9918, 0.992, or 99.2%, and a negative delta is as
    often written without its sign ("costs 0.2712") as with one. Accepting all of those is
    what keeps the prose check from being a formatting rule.
    """
    out: set[str] = set()
    for v in {value, abs(value)}:
        for places in (2, 3, 4):
            out.add(f"{v:.{places}f}")
            out.add(f"{v * 100:.{places - 2}f}%")
        out.add(f"{v:g}")
        out.add(f"{v * 100:.1f}%")
    return sorted(x for x in out if x)


def states(text: str, value: float) -> bool:
    """Does ``text`` actually state ``value``, as a number rather than as a substring?

    Plain `in` is not enough, because the loose renderings are prefixes of other numbers: a
    claim of 0.9297 offers "0.93", and `"0.93" in "the value was 0.9302"` is true. The check
    would then pass on a document that states a different number and never states this one.

    So each rendering has to sit on a numeric boundary - no digit or decimal point directly
    either side. `"0.93"` still matches a document that writes the claim as `0.930`, which is
    correct: that is the same value rounded, and prose is allowed to round.
    """
    return any(
        re.search(rf"(?<![\d.]){re.escape(r)}(?![\d])", text)
        for r in renderings(value)
    )


def check(claim: dict) -> dict:
    """Verify one claim. Returns its state and what went wrong, never raises."""
    result = {"id": claim["id"], "state": "ok", "detail": "", "derived": None}

    if not claim.get("source"):
        result["state"] = "unsupported"
        result["detail"] = claim.get("note", "no source recorded")
        return result

    source = ROOT / claim["source"]
    if not source.exists():
        result["state"] = "missing source"
        result["detail"] = f"{claim['source']} does not exist"
        return result

    try:
        derived = _value(source, claim, "select", "column")
        if "minus_select" in claim or "minus_column" in claim:
            derived -= _value(source, claim, "minus_select", "minus_column")
    except (KeyError, ValueError) as exc:
        result["state"] = "cannot derive"
        result["detail"] = str(exc)
        return result

    result["derived"] = derived
    tolerance = float(claim.get("tolerance", DEFAULT_TOLERANCE))
    if abs(derived - float(claim["value"])) > tolerance:
        result["state"] = "stale result"
        result["detail"] = f"recorded {claim['value']}, source now gives {derived:.4f}"
        return result

    missing = []
    for where in claim.get("where", []):
        path = ROOT / where
        if not path.exists():
            missing.append(f"{where} (absent)")
            continue
        if not states(path.read_text(encoding="utf-8"), float(claim["value"])):
            missing.append(where)
    if missing:
        result["state"] = "stale prose"
        result["detail"] = "value not found in " + ", ".join(missing)
    return result


def markdown(claims: list[dict], results: list[dict]) -> str:
    by_id = {r["id"]: r for r in results}
    supported = [c for c in claims if c.get("source")]
    unsupported = [c for c in claims if not c.get("source")]

    lines = [
        "# Claims ledger",
        "",
        "**Generated by `experiments/verify_claims.py` from `thesis/claims.toml`. Do not edit.**",
        "",
        "Every quantitative claim the write-up makes, with the artefact that produced it and",
        "enough of a locator to re-derive it. The verifier recomputes each value from its",
        "source and checks that the same number appears in each document listed. Run it before",
        "any draft goes out:",
        "",
        "```bash",
        "uv run python -m experiments.verify_claims --check",
        "```",
        "",
        f"**{len(supported)} claims are checked against an artefact. "
        f"{len(unsupported)} are not, and say why.**",
        "",
        "## Checked",
        "",
        "| Claim | Value | Source | Stated in | State |",
        "|---|---|---|---|---|",
    ]
    for claim in supported:
        r = by_id[claim["id"]]
        where = ", ".join(f"`{w}`" for w in claim.get("where", [])) or "—"
        state = "✔" if r["state"] == "ok" else f"**{r['state']}** — {r['detail']}"
        lines.append(
            f"| {claim['statement']} | `{claim['value']}` | `{claim['source']}` | {where} | {state} |"
        )

    if unsupported:
        lines += [
            "",
            "## Not checked",
            "",
            "These are quoted in the write-up but cannot be re-derived from a committed",
            "artefact. Listed rather than hidden: an unsupported claim is not a failure, but an",
            "unsupported claim nobody knows about is.",
            "",
            "| Claim | Value | Why it cannot be checked |",
            "|---|---|---|",
        ]
        for claim in unsupported:
            lines.append(
                f"| {claim['statement']} | `{claim['value']}` | {claim.get('note', '—')} |"
            )

    notes = [c for c in supported if c.get("note")]
    if notes:
        lines += ["", "## Notes on individual claims", ""]
        for claim in notes:
            lines.append(f"- **{claim['id']}** — {claim['note']}")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify only; do not rewrite claims.md")
    args = ap.parse_args()

    claims = load()
    ids = [c["id"] for c in claims]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate claim ids in the ledger")

    results = [check(c) for c in claims]
    by_state: dict[str, list[dict]] = {}
    for r in results:
        by_state.setdefault(r["state"], []).append(r)

    print(f"{len(claims)} claims in the ledger\n")
    width = max(len(c["id"]) for c in claims)
    for claim, r in zip(claims, results, strict=True):
        mark = {"ok": "ok ", "unsupported": "-- "}.get(r["state"], "!! ")
        derived = f"{r['derived']:>10.4f}" if r["derived"] is not None else " " * 10
        print(f"  {mark}{claim['id']:<{width}} {derived}"
              + (f"   {r['state']}: {r['detail']}" if r["state"] != "ok" else ""))

    failed = [r for r in results if r["state"] not in ("ok", "unsupported")]
    unsupported = by_state.get("unsupported", [])
    print(f"\n{len(by_state.get('ok', []))} verified, {len(unsupported)} unsupported, "
          f"{len(failed)} failing")

    if not args.check:
        OUT.write_text(markdown(claims, results), encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")
        record(
            "WP8-T5 claims ledger",
            "`python -m experiments.verify_claims`",
            "`thesis/claims.md`",
            f"{len(by_state.get('ok', []))} claims verified against their artefacts, "
            f"{len(unsupported)} recorded as unsupported",
        )

    if failed:
        print("\nA failing claim is one of two problems, and they need different work:")
        print("  stale result - the source changed and the prose did not;")
        print("  stale prose  - the prose changed and the ledger did not.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
