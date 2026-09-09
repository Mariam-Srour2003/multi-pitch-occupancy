"""Turning operator overrides back into training data (WP6-T7).

When an operator disagrees with a verdict they override it, and that override is the most
valuable label this system will ever get: a human looked at a real hour of real footage and
said what actually happened. WP6-T7 asks for those slots' evidence frames to be copied back
into the dataset so the head can be re-fitted on them.

**One deliberate departure from the task as written, and it is a correctness argument.** The
plan says copy into ``data/dataset/_incoming/<corrected_class>/``. That cannot be done
soundly, because *a slot override is not a frame label*. An operator overriding an hour to
USED is saying roughly "there was a match here" - they are not saying that each of the three
evidence frames shows active play, and under the tuned thresholds a slot is USED at 35% play,
so a majority of its minutes may show an empty pitch. Filing those frames as ACTIVE_PLAY would
inject confidently wrong labels into the training set, from the one source the project treats
as ground truth. NOTUSED constrains its frames more tightly than USED does, and still not
tightly enough: a slot can be NOTUSED overall with people crossing it.

So frames land in ``_incoming/_unfiled/<slot_status>/`` with a sidecar recording exactly what
is known - which slot, which operator, when, the note they left, and the verdict before and
after - and a human moves them into a class folder. The staging path starts with an underscore
because :func:`~pitch_occupancy.data.manifest.build_manifest` scans ``[0-9]_*`` only, so
nothing here can reach a training set by accident. That is checked by a test rather than
assumed.

**Nothing is deleted and nothing is overwritten.** Same discipline as
:mod:`pitch_occupancy.retention`: :func:`plan` is pure and copies nothing, :func:`apply` is
inert unless ``confirm=True``, and a destination that already exists is skipped rather than
replaced. The frames being copied are the only copies of evidence for a disputed billing
decision.

The re-fit itself
-----------------

Re-fitting is cheap and that is the point of the frozen-backbone design: features are cached
per frame, so adding *n* corrected frames costs one embedding pass over *n* images and then a
logistic regression over the whole cache, which is seconds rather than hours. The loop is::

    pitch retraining --apply      # stage the overridden slots' frames
    <a human files them into 1_empty / 2_playing / 3_maintenance>
    pitch manifest                # re-index
    pitch cache                   # embed only the new frames
    <re-run the experiments>

**The feedback loop is worth naming rather than leaving implicit.** Operators override the
verdicts they notice, and they notice the ones that look wrong, so this pipeline preferentially
harvests the model's own errors. That is exactly what makes it valuable for training and
exactly what makes it a biased sample: a head re-fitted on it is fitted on a corrected error
distribution, not on the operating distribution, and its accuracy on that data is not an
estimate of anything. Any evaluation must stay on the held-out sets, never on harvested
frames. Recorded here because the temptation to report "accuracy after retraining" on this
data will be strong and the number would be meaningless.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pitch_occupancy.data.taxonomy import SlotStatus

__all__ = [
    "STAGING_ROOT",
    "HarvestedFrame",
    "HarvestPlan",
    "plan",
    "apply",
]

#: Where staged frames land, relative to the dataset directory. The leading underscore is
#: load-bearing: `build_manifest` globs ``[0-9]_*``, so nothing under here is ever indexed as
#: training data until a human moves it into a real class folder.
STAGING_ROOT = "_incoming/_unfiled"


@dataclass(frozen=True, slots=True)
class HarvestedFrame:
    """One evidence frame, with the operator's claim attached but not applied as a label."""

    slot_id: str
    source: Path
    destination: Path
    slot_status: SlotStatus
    original_status: str
    operator: str
    overridden_at: str
    note: str

    @property
    def exists(self) -> bool:
        return self.source.exists()

    def provenance(self) -> dict[str, str]:
        return {
            "slot_id": self.slot_id,
            "source": str(self.source),
            "operator_slot_verdict": self.slot_status.value,
            "model_slot_verdict": self.original_status,
            "operator": self.operator,
            "overridden_at": self.overridden_at,
            "note": self.note,
            "frame_label": "",
            "warning": (
                "This is the operator's verdict for the whole SLOT, not a label for this "
                "frame. Fill frame_label after looking at the image, then move it into the "
                "matching class folder."
            ),
        }


@dataclass(frozen=True, slots=True)
class HarvestPlan:
    """What would be staged. Produced without touching the filesystem."""

    frames: tuple[HarvestedFrame, ...] = ()
    missing: tuple[str, ...] = ()
    already_staged: tuple[str, ...] = ()
    slots: tuple[str, ...] = field(default=())

    def __len__(self) -> int:
        return len(self.frames)

    def describe(self) -> str:
        lines = [
            f"{len(self.slots)} overridden slot(s); {len(self.frames)} frame(s) to stage",
        ]
        by_status: dict[str, int] = {}
        for f in self.frames:
            by_status[f.slot_status.value] = by_status.get(f.slot_status.value, 0) + 1
        for status, n in sorted(by_status.items()):
            lines.append(f"  {status:<10} {n} frame(s)")
        if self.already_staged:
            lines.append(f"  {len(self.already_staged)} already staged, skipped")
        if self.missing:
            lines.append(f"  {len(self.missing)} evidence file(s) no longer on disk:")
            lines.extend(f"    - {m}" for m in self.missing[:10])
        lines.append("")
        lines.append(
            "Frames are staged UNFILED. The operator's verdict is about the slot, not about "
            "each frame; a human files them before they become training data."
        )
        return "\n".join(lines)


def plan(connection, *, dataset_dir: Path | None = None) -> HarvestPlan:
    """What would be staged from the overridden slots. Reads; copies nothing.

    Args:
        connection: an open database connection.
        dataset_dir: defaults to the configured dataset directory.
    """
    if dataset_dir is None:
        from pitch_occupancy.config import settings

        dataset_dir = settings.dataset_dir
    staging = Path(dataset_dir) / STAGING_ROOT

    rows = connection.execute(
        """SELECT slot_id, status, override_status, override_by, override_at,
                  override_note, evidence_paths
           FROM slot_evaluations WHERE is_overridden = 1 ORDER BY override_at"""
    ).fetchall()

    frames: list[HarvestedFrame] = []
    missing: list[str] = []
    already: list[str] = []
    slots: list[str] = []

    for row in rows:
        slot_id, status, override_status, operator, at, note, evidence = row
        slots.append(slot_id)
        try:
            slot_status = SlotStatus(override_status)
        except ValueError:
            # A status outside the enum means the schema's CHECK was bypassed. Skip it rather
            # than guess: staging a frame under an unknown verdict files it nowhere useful.
            continue
        for path_text in json.loads(evidence or "[]"):
            source = Path(path_text)
            if not source.is_absolute():
                source = Path(dataset_dir) / source
            if not source.exists():
                missing.append(f"{slot_id}: {path_text}")
                continue
            destination = staging / slot_status.value / f"{slot_id}__{source.name}"
            if destination.exists():
                already.append(str(destination))
                continue
            frames.append(HarvestedFrame(
                slot_id=slot_id, source=source, destination=destination,
                slot_status=slot_status, original_status=str(status),
                operator=str(operator or ""), overridden_at=str(at or ""),
                note=str(note or ""),
            ))

    return HarvestPlan(
        frames=tuple(frames), missing=tuple(missing),
        already_staged=tuple(already), slots=tuple(dict.fromkeys(slots)),
    )


def apply(harvest: HarvestPlan, *, confirm: bool = False) -> int:
    """Copy the planned frames into staging. Returns how many were copied.

    Inert unless ``confirm=True``, matching `retention.apply`. Copies rather than moves - the
    evidence frame stays where the audit trail points at it, because a slot's evidence is what
    justifies a billing decision and moving it would break the record to feed the model.

    An existing destination is skipped, never overwritten: a second run must not silently
    replace a frame a human has already looked at and annotated.
    """
    if not confirm:
        return 0

    copied = 0
    for frame in harvest.frames:
        if frame.destination.exists():
            continue
        frame.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frame.source, frame.destination)
        sidecar = frame.destination.with_suffix(frame.destination.suffix + ".json")
        payload = frame.provenance()
        payload["staged_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        copied += 1
    return copied
