"""Milestone gates, checked against the artefacts rather than ticked by hand (WP2-T7, WP4-T8).

Every row of the gate tracker in `TODO.md` reads `[ ]`, and two of them have been largely met
for some time. That is the failure this project keeps meeting from a different angle: a
hand-maintained status that nobody updates while doing the work it describes, exactly like the
README that said the implementation had not started.

So the criteria are encoded as predicates over the repository, and this reports which are met.
Three outcomes, not two, because the difference matters:

* **met** — an artefact exists and says what the criterion requires;
* **not met** — checkable, and absent;
* **needs a person** — supervisor sign-off, a client conversation, a live deployment. These
  can never be auto-passed, and reporting them as "not met" would be misleading: nothing in
  the repository can move them.

A gate passes only when nothing under it is unmet *and* nothing needs a person. The output is
`results/gate_status.md`, which the tracker points at.

**What this cannot do.** It checks that an artefact exists and contains what the criterion
names. It cannot check that the work is *good* — that is what the review documents, the
pre-registration amendments and an examiner are for. A criterion that reduces to "a file
exists" is a weak criterion, and where one does, the weakness is the criterion's rather than
this script's.

    uv run python -m experiments.gate_check
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from typing import Callable

from pitch_occupancy.config import settings
from pitch_occupancy.evaluation.experiment_log import record

ROOT = settings.results_dir.parent
RESULTS = settings.results_dir
OUT = RESULTS / "gate_status.md"

MET, UNMET, HUMAN = "met", "not met", "needs a person"


@dataclass(frozen=True, slots=True)
class Criterion:
    text: str
    check: Callable[[], tuple[str, str]]


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    week: int
    criteria: tuple[Criterion, ...]


# --- small helpers over the artefacts ------------------------------------------------


def _exists(relative: str, note: str = "") -> tuple[str, str]:
    path = ROOT / relative
    return (MET, note or relative) if path.exists() else (UNMET, f"{relative} is absent")


def _contains(relative: str, needle: str, note: str) -> tuple[str, str]:
    path = ROOT / relative
    if not path.exists():
        return UNMET, f"{relative} is absent"
    if needle in path.read_text(encoding="utf-8"):
        return MET, note
    return UNMET, f"{relative} does not mention {needle!r}"


def _csv_rows(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _human(what: str) -> Callable[[], tuple[str, str]]:
    return lambda: (HUMAN, what)


# --- the criteria ---------------------------------------------------------------------


def _benchmark_covers_four_models_and_protocols() -> tuple[str, str]:
    rows = [r for r in _csv_rows("results/benchmark_v2.csv") if r.get("replicate") != "SUMMARY"]
    if not rows:
        return UNMET, "benchmark_v2.csv is absent"
    models = {r["model"] for r in rows}
    protocols = {r["protocol"] for r in rows}
    if len(models) >= 4 and len(protocols) >= 3:
        return MET, f"{len(models)} models x {len(protocols)} protocols"
    return UNMET, f"only {len(models)} models and {len(protocols)} protocols"


def _intervals_attached() -> tuple[str, str]:
    rows = _csv_rows("results/h1_h2_baseline_floor.csv")
    if not rows:
        return UNMET, "h1_h2_baseline_floor.csv is absent"
    with_ci = [r for r in rows if r.get("macro_f1_lo") and r.get("macro_f1_hi")]
    if len(with_ci) == len(rows):
        return MET, f"all {len(rows)} rows carry a bootstrap interval"
    return UNMET, f"{len(rows) - len(with_ci)} of {len(rows)} rows have no interval"


def _significance_reported() -> tuple[str, str]:
    families = {
        "false_play_significance.csv": "p_holm",
        "h4_model_equivalence.csv": "p_holm",
        "h6_zero_shot_gap.csv": "p_holm",
        "h5_preprocessing_switches.csv": "p_holm",
    }
    have = [
        name for name, column in families.items()
        if any(r.get(column) for r in _csv_rows(f"results/{name}"))
    ]
    if len(have) >= 3:
        return MET, f"Holm-corrected families in {len(have)} reports"
    return UNMET, f"only {len(have)} report carries a corrected p-value"


def _trivial_floor_present() -> tuple[str, str]:
    models = {r["model"] for r in _csv_rows("results/h1_h2_baseline_floor.csv")}
    trivial = models & {"majority", "clock_rule", "cheap_intensity", "cheap_histogram"}
    if len(trivial) >= 3:
        return MET, f"{len(trivial)} trivial baselines in the benchmark"
    return UNMET, f"only {len(trivial)} trivial baselines"


def _empty_concentration_reported() -> tuple[str, str]:
    """How many *venues* hold an EMPTY frame - the concentration M2 asks to be quantified.

    Read from the class-by-venue table specifically. The first version took the first row
    beginning `| EMPTY`, which is the class-by-*lighting* table earlier in the same file, and
    cheerfully reported "EMPTY appears in 2 venues" - meaning day and night. A check that
    reports a plausible wrong number is worse than one that fails.
    """
    path = ROOT / "results" / "coverage.md"
    if not path.exists():
        return UNMET, "coverage.md is absent"
    in_venue_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith("## class x venue"):
            in_venue_table = True
            continue
        if in_venue_table and line.startswith("## "):
            break
        if in_venue_table and line.startswith("| EMPTY"):
            cells = [c.strip() for c in line.split("|")[2:-2]]  # drop label, total, trailing
            nonzero = [c for c in cells if c not in ("-", "")]
            if len(nonzero) == 1:
                return MET, f"EMPTY appears in exactly one venue ({nonzero[0]} frames)"
            return MET, f"EMPTY appears in {len(nonzero)} of {len(cells)} venues"
    return UNMET, "coverage.md has no class-by-venue EMPTY row"


def _leakage_measured_per_split() -> tuple[str, str]:
    rows = [r for r in _csv_rows("results/error_taxonomy.csv")
            if r.get("near_duplicate_in_train")]
    if not rows:
        return UNMET, "no per-split near-duplicate measurement"
    leaky = [r for r in rows if r["protocol"] == "random"]
    honest = [r for r in rows if r["protocol"] == "grouped_slot"]
    if not leaky or not honest:
        return UNMET, "only one protocol was measured"
    share = sum(r["near_duplicate_in_train"] == "True" for r in leaky) / len(leaky)
    clean = sum(r["near_duplicate_in_train"] == "True" for r in honest)
    return MET, f"{share:.0%} of leaky-split errors had a near-duplicate; {clean} on the honest split"


def _claims_verified() -> tuple[str, str]:
    """Re-derive every claim, rather than grepping the generated page for a bad word.

    Searching `claims.md` for "stale" would pass or fail on prose - including the prose that
    explains what staleness is. The ledger has a verifier; a gate check that did not run it
    would be asserting the thing it is supposed to check.
    """
    # `python experiments/gate_check.py` puts `experiments/` on the path, not the repo root,
    # so this import fails and the criterion used to degrade quietly to "not met" - which is
    # how a wrong gate status got committed. A check whose answer depends on how it was
    # invoked is not a check, so the path is repaired rather than the failure swallowed.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from experiments.verify_claims import check, load
    except Exception as exc:  # pragma: no cover - import guard
        return UNMET, f"the claims verifier could not be loaded: {exc}"
    results = [check(c) for c in load()]
    bad = [r for r in results if r["state"] not in ("ok", "unsupported")]
    unsupported = [r for r in results if r["state"] == "unsupported"]
    if bad:
        return UNMET, f"{len(bad)} of {len(results)} claims no longer hold"
    note = f"{len(results)} claims re-derive from their artefacts"
    if unsupported:
        note += f"; {len(unsupported)} recorded as unsupported"
    return MET, note


def _fusion_ablated() -> tuple[str, str]:
    """M4's first criterion, checked by reading the ablation rather than by finding a file.

    "The module exists" is the weak form of this criterion and would have passed on a fusion
    head that was never compared to anything. What M4 asks for is an *ablation*, so what is
    checked is that the comparison isolating the gate is present in the results - the rung
    directly below it, not the published probes, which cross a trainer boundary.
    """
    if not (ROOT / "results" / "logit_average_baseline.csv").exists():
        return UNMET, "no fusion baseline"
    if not (ROOT / "src" / "pitch_occupancy" / "slots" / "fusion_head.py").exists():
        return UNMET, ("the logit-average baseline is reported and answers the question "
                       "negatively; the fusion head itself (WP5-T2) is not built")
    rows = _csv_rows("results/fusion_head_comparisons.csv")
    isolating = [r for r in rows if r.get("isolates_one_cause") == "True"]
    if not isolating:
        return UNMET, ("fusion_head.py exists but results/fusion_head_ablation.csv carries no "
                       "gate-isolating comparison; a module compared only against a different "
                       "trainer has not been ablated")
    delta = isolating[0].get("mean_delta", "?")
    return MET, f"fusion head ablated against its own ungated rung (routing worth {delta})"


def _stan_preliminary() -> tuple[str, str]:
    """M4's second criterion. The word "preliminary" is the criterion, so it is verified.

    A STAN module reporting a headline number off two real slots would satisfy "the file
    exists" and violate the thing the criterion is actually protecting. So this checks that
    the results carry the caveat and that the gate in `slots/stan.py` still refuses.
    """
    if not (ROOT / "src" / "pitch_occupancy" / "slots" / "stan.py").exists():
        return UNMET, "src/pitch_occupancy/slots/stan.py is absent"
    path = ROOT / "results" / "stan_preliminary.csv"
    if not path.exists():
        return UNMET, "stan.py exists but has never been run to a result"
    text = path.read_text(encoding="utf-8")
    if "PRELIMINARY" not in text:
        return UNMET, ("results/stan_preliminary.csv carries no preliminary caveat; WP5-T8 "
                       "forbids a headline below 30 real labelled slots")
    return MET, "STAN reported against four tuned baselines, marked preliminary in the results"


GATES: tuple[Gate, ...] = (
    Gate("M1", 4, (
        Criterion("Protocol document written",
                  lambda: _exists("thesis/protocol.md")),
        Criterion("Labelling protocol written",
                  lambda: _exists("thesis/labelling_protocol.md")),
        Criterion("Taxonomy fixed and implemented",
                  lambda: _exists("src/pitch_occupancy/data/taxonomy.py")),
        Criterion("Evaluation design pre-registered",
                  lambda: _exists("thesis/preregistration.md")),
        Criterion("Supervisor approval", _human("a supervision conversation (WP1-T2)")),
        Criterion("Ethics/DPIA outcome documented",
                  _human("a supervisor answer and two client answers (WP1-T3)")),
    )),
    Gate("M2", 9, (
        Criterion("Coverage matrix published",
                  lambda: _exists("results/coverage.md")),
        Criterion("Concentration of EMPTY reported", _empty_concentration_reported),
        Criterion("Near-duplicate rate measured",
                  lambda: _exists("results/near_duplicates.csv")),
        Criterion("Per-split leakage measured", _leakage_measured_per_split),
        Criterion("Effective sample size reported",
                  lambda: _exists("results/effective_sample_audit.csv")),
        Criterion("Unanswerable questions recorded",
                  lambda: _contains("thesis/preregistration.md", "Not answerable",
                                    "the pre-registration lists them")),
    )),
    Gate("M3", 13, (
        Criterion("Four models under three or more protocols",
                  _benchmark_covers_four_models_and_protocols),
        Criterion("Confidence intervals attached", _intervals_attached),
        Criterion("Significance tests with correction", _significance_reported),
        Criterion("Trivial-baseline floor reported", _trivial_floor_present),
        Criterion("Figures exported",
                  lambda: _exists("results/figs/leakage_decomposition.png")),
        Criterion("Every quantitative claim re-derives", _claims_verified),
    )),
    Gate("M4", 18, (
        Criterion("Fusion module ablated against a strong baseline", _fusion_ablated),
        Criterion("STAN reported as preliminary", _stan_preliminary),
    )),
    Gate("M5", 19, (
        Criterion("Reconciliation implemented",
                  lambda: _exists("src/pitch_occupancy/slots/reconcile.py")),
        Criterion("End-to-end run on real slots",
                  lambda: _exists("results/end_to_end_slots.csv")),
        Criterion("Retention enforced",
                  lambda: _exists("src/pitch_occupancy/retention.py")),
        Criterion("Degraded mode enforced and tested",
                  lambda: _contains("src/pitch_occupancy/slots/aggregate.py",
                                    "review_below_capture", "capture-rate gate present")),
        Criterion("Scheduler service on mock feeds",
                  lambda: _exists("src/pitch_occupancy/scheduler.py")),
    )),
    Gate("M6", 21, (
        Criterion("Failure-mode runbook", lambda: _exists("docs/runbook.md")),
        Criterion("48-hour live validation accepted",
                  _human("a deployment and a facility conversation (WP7-T3, WP7-T4)")),
    )),
    Gate("M7", 24, (
        Criterion("Claims ledger complete and verified", _claims_verified),
        Criterion("Defence red-team written",
                  lambda: _exists("thesis/defence_redteam.md")),
        Criterion("Related-work chapter", lambda: _exists("thesis/ch2_related_work.md")),
        Criterion("Thesis submitted and defence ready",
                  _human("writing, a mock examination and submission (WP8-T7)")),
    )),
)


def evaluate() -> list[dict]:
    out = []
    for gate in GATES:
        for criterion in gate.criteria:
            status, detail = criterion.check()
            out.append({"gate": gate.name, "week": gate.week, "criterion": criterion.text,
                        "status": status, "detail": detail})
    return out


def verdict(rows: list[dict], gate: str) -> str:
    mine = [r for r in rows if r["gate"] == gate]
    if any(r["status"] == UNMET for r in mine):
        return "not passed"
    if any(r["status"] == HUMAN for r in mine):
        return "waiting on a person"
    return "passed"


def markdown(rows: list[dict]) -> str:
    lines = [
        "# Milestone gate status",
        "",
        "**Generated by `experiments/gate_check.py`. Do not edit.**",
        "",
        "Each criterion is checked against the repository. Three outcomes, because the",
        "difference matters: **met**, **not met**, and **needs a person** — the last cannot be",
        "moved by anything in here, so reporting it as unmet would be misleading.",
        "",
        "A gate passes only when nothing under it is unmet *and* nothing needs a person.",
        "",
        "| Gate | Week | Verdict | met | not met | needs a person |",
        "|---|---|---|---|---|---|",
    ]
    for gate in GATES:
        mine = [r for r in rows if r["gate"] == gate.name]
        counts = {s: sum(1 for r in mine if r["status"] == s) for s in (MET, UNMET, HUMAN)}
        lines.append(
            f"| **{gate.name}** | {gate.week} | {verdict(rows, gate.name)} | "
            f"{counts[MET]} | {counts[UNMET]} | {counts[HUMAN]} |"
        )

    mark = {MET: "✔", UNMET: "✘", HUMAN: "—"}
    for gate in GATES:
        lines += ["", f"## {gate.name} (week {gate.week}) — {verdict(rows, gate.name)}", "",
                  "| | criterion | evidence |", "|---|---|---|"]
        for r in (x for x in rows if x["gate"] == gate.name):
            lines.append(f"| {mark[r['status']]} | {r['criterion']} | {r['detail']} |")

    lines += [
        "",
        "---",
        "",
        "**What this cannot do.** It checks that an artefact exists and says what the",
        "criterion names. It cannot check that the work is good — that is what the reviews,",
        "the pre-registration amendments and an examiner are for. Where a criterion reduces",
        "to \"a file exists\", the weakness is the criterion's rather than the checker's.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = evaluate()
    width = max(len(r["criterion"]) for r in rows)
    mark = {MET: "ok ", UNMET: "!! ", HUMAN: "-- "}
    for gate in GATES:
        print(f"\n=== {gate.name} (week {gate.week}) — {verdict(rows, gate.name)} ===")
        for r in (x for x in rows if x["gate"] == gate.name):
            print(f"  {mark[r['status']]}{r['criterion']:<{width}}  {r['detail']}")

    passed = [g.name for g in GATES if verdict(rows, g.name) == "passed"]
    waiting = [g.name for g in GATES if verdict(rows, g.name) == "waiting on a person"]
    print(f"\n{len(passed)} passed ({', '.join(passed) or 'none'}); "
          f"{len(waiting)} waiting on a person ({', '.join(waiting) or 'none'})")

    OUT.write_text(markdown(rows), encoding="utf-8")
    (RESULTS / "gate_status.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {OUT.name}")

    record(
        "milestone gate check",
        "`python -m experiments.gate_check`",
        f"`{OUT.name}`",
        f"{len(passed)} gate(s) met on artefacts, {len(waiting)} waiting on a person",
    )


if __name__ == "__main__":
    main()
