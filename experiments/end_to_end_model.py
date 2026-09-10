"""The recorded slots run through the *model*, beside the same slots run through the labels.

`end_to_end_slots.py` demonstrates the pipeline on real slots by rebuilding each slot's
per-minute sequence from the **label column** of the manifest. That was the only thing
available until `vision/classifier.py` existed, and it is what M5's "end-to-end run on real
slots" has been satisfied by ever since — a sentence that reads as though a model were
involved, and one that was true of nothing in the repository.

This runs the same two slots through the deployed classifier and writes the two answers side
by side, so the gap between "the pipeline works" and "the pipeline works with a model in it"
is a committed artefact rather than a paragraph.

**In-sample, and it has to be said in the same breath.** Both slots' frames are in the
probe's training set. Nothing here is an accuracy claim — WP4 measures that honestly, on
held-out venues, and gets very different numbers. What this checks is that the assembled
system produces the verdicts the labels imply when handed the actual video, which is a
different question and one nothing else answers.

**The minutes are not frame-aligned, so the agreement is scene-level.** `VideoSlotSource`
reads the frame at exactly minute x 60 s; the labelled frames of the same minute sit at
irregular instants (0, 4, 5, 14, 21, 27, 29 ... seconds). So a minute's model prediction is
compared against **the labels of that minute**, and minutes whose own labels disagree are
counted separately rather than scored — a minute containing both EMPTY and ACTIVE_PLAY
cannot make a single prediction right or wrong, and folding it in either direction would be
a choice dressed as a measurement.

    uv run python -m experiments.end_to_end_model
"""

from __future__ import annotations

import csv
from collections import defaultdict

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.frame_source import VideoSlotSource, discover_slots
from pitch_occupancy.slots.aggregate import aggregate_slot
from pitch_occupancy.slots.fusion import fuse
from pitch_occupancy.worker import run_slot

OUT = settings.results_dir / "end_to_end_model_slots.csv"

FIELDS = (
    "slot", "verdict_model", "verdict_labels", "verdicts_agree",
    "play_ratio_model", "play_ratio_labels", "empty_ratio_model", "empty_ratio_labels",
    "minutes_model", "minutes_labels",
    "minutes_compared", "minutes_agreeing", "minute_agreement", "minutes_ambiguous",
)


def manifest_slot_id(recording_key: str) -> str:
    """`slot_20260711_1000` -> `venue_01_2026-07-11_1000`.

    The recordings are grouped by the export's filenames and the manifest keys a slot by
    venue and date; `scheduler._recording_key` does the same translation in the other
    direction. Two conventions for one thing, and both directions now live in a named
    function rather than being done from memory.
    """
    _, _, tail = recording_key.partition("slot_")
    stamp, _, clock = tail.partition("_")
    return f"venue_01_{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}_{clock}"


def labels_by_minute(rows) -> dict[int, set[str]]:
    """minute -> the set of classes labelled anywhere in it, across both cameras."""
    out: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        out[row.t_s // 60].add(row.class3)
    return dict(out)


def label_sequence(rows) -> list[Class3]:
    """The fused per-minute sequence the labels imply - the same construction
    `end_to_end_slots.py` uses, so the two artefacts are comparable."""
    grid: dict[int, dict[str, Class3]] = defaultdict(dict)
    for row in rows:
        grid[row.t_s // 60][row.camera] = Class3(row.class3)
    return [
        fuse({cam: (cls, 1.0) for cam, cls in grid[minute].items()}).state
        for minute in sorted(grid)
    ]


def compare(fused: dict[int, Class3], by_minute: dict[int, set[str]]) -> tuple[int, int, int]:
    """(compared, agreeing, ambiguous) for one slot.

    A minute whose own labels disagree — both cameras were labelled, and differently, or two
    frames seconds apart caught a change — cannot make a single prediction right or wrong.
    Counting it either way would be a choice dressed as a measurement, so it is reported
    separately and scored not at all.
    """
    compared = agreeing = ambiguous = 0
    for minute, state in fused.items():
        truth = by_minute.get(minute)
        if not truth:
            continue  # the model saw a minute the labels do not cover
        if len(truth) > 1:
            ambiguous += 1
            continue
        compared += 1
        agreeing += int(state.value in truth)
    return compared, agreeing, ambiguous


def main() -> None:
    from pitch_occupancy.vision.classifier import load_classifier

    rows = read_manifest(settings.dataset_dir / "manifest.csv")
    by_slot: dict[str, list] = defaultdict(list)
    for row in rows:
        if row.source != "clip":
            by_slot[row.slot_id].append(row)

    found = discover_slots(settings.raw_dir / "venue_01")
    if not found:
        raise SystemExit(f"no recordings under {settings.raw_dir / 'venue_01'}")

    classify = load_classifier()
    print(f"classifier: {classify.backbone} probe on {classify.n_train} development frames")
    print("(both slots below are in that training set - this is a wiring check, not accuracy)\n")

    results: list[dict] = []
    for key in sorted(found):
        slot_id = manifest_slot_id(key)
        labelled = by_slot.get(slot_id, [])
        if not labelled:
            print(f"{slot_id}: no labelled frames in the manifest; skipped")
            continue

        print(f"  {slot_id}: classifying...", end="", flush=True)
        run = run_slot(slot_id, VideoSlotSource(found[key]), classify)
        by_minute = labels_by_minute(labelled)

        predicted: dict[int, Class3] = {}
        for sample in run.samples:
            predicted.setdefault(sample.minute_index, Class3(sample.predicted))
        # the fused state is what the verdict rests on, so it is what gets compared
        fused: dict[int, Class3] = {}
        for minute in sorted(predicted):
            observed = {
                s.camera_id: (Class3(s.predicted), s.confidence)
                for s in run.samples if s.minute_index == minute
            }
            fused[minute] = fuse(observed).state

        compared, agreeing, ambiguous = compare(fused, by_minute)

        labels_verdict = aggregate_slot(label_sequence(labelled))
        results.append({
            "slot": slot_id,
            "verdict_model": run.verdict.status.value,
            "verdict_labels": labels_verdict.status.value,
            "verdicts_agree": int(run.verdict.status == labels_verdict.status),
            "play_ratio_model": run.verdict.play_ratio,
            "play_ratio_labels": labels_verdict.play_ratio,
            "empty_ratio_model": run.verdict.empty_ratio,
            "empty_ratio_labels": labels_verdict.empty_ratio,
            "minutes_model": run.minutes_captured,
            "minutes_labels": labels_verdict.n_samples,
            "minutes_compared": compared,
            "minutes_agreeing": agreeing,
            "minute_agreement": agreeing / compared if compared else float("nan"),
            "minutes_ambiguous": ambiguous,
        })
        save(results)
        r = results[-1]
        print(f"\r  {slot_id}: model {r['verdict_model']} / labels {r['verdict_labels']}"
              f"   minutes {r['minutes_agreeing']}/{r['minutes_compared']} agree"
              f"   ({r['minutes_ambiguous']} ambiguous)")

    save(results)
    agree = sum(r["verdicts_agree"] for r in results)
    compared = sum(r["minutes_compared"] for r in results)
    agreeing = sum(r["minutes_agreeing"] for r in results)
    print(f"\nverdicts: {agree}/{len(results)} agree with the labels")
    if compared:
        ambiguous = sum(r["minutes_ambiguous"] for r in results)
        print(f"minutes : {agreeing}/{compared} agree ({agreeing / compared:.1%}), "
              f"{ambiguous} minutes ambiguous (their own labels disagree)")
    else:
        print("minutes : none comparable")
    print(f"\nwrote {OUT.name}")

    record(
        "WP6-T2 end-to-end with the model",
        "`python -m experiments.end_to_end_model`",
        f"`{OUT.name}`",
        f"{agree}/{len(results)} slot verdicts agree with the label-derived ones and "
        f"{agreeing}/{compared} comparable minutes agree; in-sample, so a wiring check "
        f"rather than an accuracy result",
    )


def save(results: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in results:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})


if __name__ == "__main__":
    main()
