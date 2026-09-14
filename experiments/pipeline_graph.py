"""The reproduction pipeline as a graph, so the Reproduce page can be drawn (WP8-T6).

`reproduce_all.py` already knows everything worth showing about a run: every stage declares
what it produces, what it requires, how long it takes, and whether its outputs are present,
stale or blocked. None of that reached a reader. The Reproduce page was four code blocks and
a paragraph, which tells someone *how to type the command* and nothing about what happens
when they do - how long it takes, what depends on what, or where a fresh checkout would stop.

So this reads `STAGES` and writes it out as a graph.

**The dependency edges are derived, not declared.** A stage depends on another when one of
its `requires` paths is in the other's `produces` - which means the edges come from the same
declarations the runner obeys, and cannot disagree with the order stages actually run in. A
hand-drawn diagram of this pipeline would be a second description of it, and the one that
goes stale the next time a stage is added.

**Levels are computed, not labelled.** Each stage sits one level below the deepest stage it
depends on, so the layers on the page are the real critical path rather than a tidy grouping
someone chose. A stage with no dependencies on other stages' outputs is at level 0 - and
seeing how many of those there are is itself the answer to "what can I run first?".

Status comes from the same three methods `--check` uses, so the page and the command agree:
`satisfied()` for outputs present, `stale_inputs()` for an output older than its input,
`blocked_by()` for a missing input that no stage here produces.

    uv run python experiments/pipeline_graph.py

Writes `results/pipeline_graph.json`. Cheap - it runs no stage, it only inspects them - but
it does shell out to git per file for the staleness check, so it takes a few seconds.
"""

from __future__ import annotations

import json

from pitch_occupancy.config import settings

from experiments.reproduce_all import ROOT, STAGES

OUT = settings.results_dir / "pipeline_graph.json"


def _rel(path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def edges() -> dict[str, list[str]]:
    """``stage -> the stages whose outputs it consumes``, derived from the declarations.

    Built by matching `requires` against `produces` across every stage. A required path
    that no stage produces is an *external input* - the footage, the manifest, a feature
    cache - and is deliberately not an edge: it is the boundary of what this repository can
    regenerate, which the page reports separately.
    """
    producer: dict[str, str] = {}
    for stage in STAGES:
        for path in stage.produces:
            producer[_rel(path)] = stage.name
    out: dict[str, list[str]] = {}
    for stage in STAGES:
        deps = {
            producer[_rel(p)] for p in stage.requires
            if _rel(p) in producer and producer[_rel(p)] != stage.name
        }
        out[stage.name] = sorted(deps)
    return out


def levels(dep: dict[str, list[str]]) -> dict[str, int]:
    """Depth of each stage in the dependency graph, by repeated relaxation.

    Iterative rather than recursive, and capped: a cycle in the declarations would hang a
    recursive walk, and a pipeline that has grown a cycle should still produce a readable
    page saying so rather than no page at all. The cap is the stage count, which is the
    longest a genuine chain can be.
    """
    level = {name: 0 for name in dep}
    for _ in range(len(dep)):
        changed = False
        for name, parents in dep.items():
            want = max((level[p] + 1 for p in parents), default=0)
            if want > level[name]:
                level[name] = want
                changed = True
        if not changed:
            break
    return level


def main() -> None:
    dep = edges()
    level = levels(dep)
    produced = {_rel(p) for s in STAGES for p in s.produces}

    stages = []
    for stage in STAGES:
        blocked = [_rel(p) for p in stage.blocked_by()]
        stale = [_rel(p) for p in stage.stale_inputs()]
        satisfied = stage.satisfied()
        stages.append({
            "name": stage.name,
            "level": level[stage.name],
            "minutes": stage.minutes,
            "note": stage.note,
            "machine_dependent": stage.machine_dependent,
            "produces": [_rel(p) for p in stage.produces],
            "requires": [_rel(p) for p in stage.requires],
            "depends_on": dep[stage.name],
            # External inputs are the ones no stage here produces - the boundary of what
            # this repository can regenerate on its own.
            "external_inputs": [
                _rel(p) for p in stage.requires if _rel(p) not in produced
            ],
            "blocked_by": blocked,
            "stale_inputs": stale,
            "status": (
                "blocked" if blocked else
                "stale" if stale else
                "done" if satisfied else "missing"
            ),
        })

    by_status: dict[str, int] = {}
    for s in stages:
        by_status[s["status"]] = by_status.get(s["status"], 0) + 1

    # The critical path in wall-clock terms: the slowest chain through the graph, which is
    # the floor on a full rerun no matter how much is parallelised. Reported beside the
    # simple total, because the two answer different questions - "how much compute" and
    # "how long until I have the thesis back".
    minutes = {s["name"]: s["minutes"] for s in stages}
    longest: dict[str, int] = {}
    for s in sorted(stages, key=lambda s: s["level"]):
        longest[s["name"]] = s["minutes"] + max(
            (longest[p] for p in s["depends_on"] if p in longest), default=0
        )

    graph = {
        "stages": stages,
        "n_stages": len(stages),
        "total_minutes": sum(minutes.values()),
        "critical_path_minutes": max(longest.values(), default=0),
        "max_level": max(level.values(), default=0),
        "by_status": by_status,
        "external_inputs": sorted(
            {p for s in stages for p in s["external_inputs"]}
        ),
    }

    print(f"{graph['n_stages']} stages across {graph['max_level'] + 1} dependency levels")
    print(f"total {graph['total_minutes']} min; "
          f"critical path {graph['critical_path_minutes']} min")
    print("status: " + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())))
    print(f"external inputs this repo cannot regenerate: {len(graph['external_inputs'])}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
