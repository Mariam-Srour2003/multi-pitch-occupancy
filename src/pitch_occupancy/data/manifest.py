"""Build and read the dataset manifest (WP0-T2).

The manifest is the single source of truth about the dataset: one row per labelled frame,
carrying the provenance every downstream step needs. Splits, coverage reports, feature
caches and experiments all read it, and none of them re-derive metadata from filenames.

Deliberately stdlib-only: the manifest must be buildable before any heavy dependency is
installed, and on the Mini-PC.

Frame filenames follow::

    slot_<YYYYMMDD>_<HHMM>_<camA|camB>_t<SSSSSS>[_m].jpg
                                             |       |
                             seconds into the video  motion-mined rather than
                                                     sampled on the fixed interval
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import date, time
from pathlib import Path

from pitch_occupancy.data.taxonomy import Class3, Class4, to_class3

__all__ = ["ManifestRow", "build_manifest", "read_manifest", "write_manifest", "cross_tab"]

FRAME_RE = re.compile(
    r"^slot_(?P<d>\d{8})_(?P<t>\d{4})_(?P<cam>cam[A-Z])_t(?P<sec>\d{6})(?P<motion>_m)?\.jpg$"
)

#: Hours (local) from which a slot counts as floodlit rather than daylight. A crude but
#: explicit rule - override per venue in configs/ once seasons and latitudes vary.
NIGHT_FROM_HOUR = 19
NIGHT_UNTIL_HOUR = 7


@dataclass(frozen=True, slots=True)
class ManifestRow:
    file: str  # path relative to the dataset directory
    class4: str
    class3: str
    venue: str
    camera: str  # camera tag, unique within a venue
    slot_date: str  # ISO date
    slot_time: str  # HH:MM
    slot_id: str  # venue x date x time - THE grouping key for splits
    t_s: int  # seconds into the source video
    source: str  # regular | motion
    labeled_by: str  # human | bulk
    lighting: str  # day | night
    quality: str  # unknown until WP3-T5 runs
    split_role: str  # assigned once by WP0-T4; never re-randomised


def _lighting_for(t: time) -> str:
    return "night" if (t.hour >= NIGHT_FROM_HOUR or t.hour < NIGHT_UNTIL_HOUR) else "day"


def _parse_frame_name(name: str) -> dict[str, object] | None:
    m = FRAME_RE.match(name)
    if m is None:
        return None
    d = date(int(m["d"][:4]), int(m["d"][4:6]), int(m["d"][6:8]))
    t = time(int(m["t"][:2]), int(m["t"][2:4]))
    return {
        "camera": m["cam"],
        "slot_date": d.isoformat(),
        "slot_time": t.strftime("%H:%M"),
        "t_s": int(m["sec"]),
        "source": "motion" if m["motion"] else "regular",
        "lighting": _lighting_for(t),
    }


def _read_label_provenance(dataset_dir: Path) -> dict[str, str]:
    """Map relative path -> label, for frames that went through the labelling tool.

    Frames present on disk but absent here were filed some other way (a bulk move of a
    whole camera-slot, say). That is a legitimate action but a different provenance, and
    the manifest records the difference rather than hiding it.
    """
    labels_csv = dataset_dir / "labels.csv"
    if not labels_csv.exists():
        return {}
    with labels_csv.open(newline="", encoding="utf-8") as fh:
        return {row["file"]: row["label"] for row in csv.DictReader(fh)}


def build_manifest(
    dataset_dir: Path | None = None,
    *,
    venue: str = "venue_01",
) -> tuple[list[ManifestRow], list[str]]:
    """Scan the class folders and build manifest rows.

    The folder a frame sits in is the authoritative label; ``labels.csv`` supplies
    provenance only. Returns the rows plus a list of problems worth a human's attention.
    """
    if dataset_dir is None:
        # Imported lazily so this module stays runnable with the standard library alone.
        from pitch_occupancy.config import settings

        dataset_dir = settings.dataset_dir
    labelled = _read_label_provenance(dataset_dir)

    rows: list[ManifestRow] = []
    problems: list[str] = []

    for class_dir in sorted(dataset_dir.glob("[0-9]_*")):
        if not class_dir.is_dir():
            continue
        try:
            class4 = Class4(class_dir.name)
        except ValueError:
            problems.append(f"unexpected class folder: {class_dir.name}")
            continue

        for frame in sorted(class_dir.glob("*.jpg")):
            rel = f"{class_dir.name}/{frame.name}"
            parsed = _parse_frame_name(frame.name)
            if parsed is None:
                problems.append(f"unparseable filename: {rel}")
                continue

            recorded = labelled.get(rel)
            if recorded is not None and recorded != class4.value:
                problems.append(
                    f"label mismatch for {rel}: folder={class4.value} labels.csv={recorded}"
                )

            # The camera tag matches how configs/cameras.json is keyed: per slot, not
            # globally, because the A/B <-> physical view mapping is not stable across
            # days in the source exports.
            hhmm = str(parsed["slot_time"]).replace(":", "")
            camera_tag = f"slot_{str(parsed['slot_date']).replace('-', '')}_{hhmm}_{parsed['camera']}"
            # Both cameras of one pitch see the SAME scene at the same moment, so they
            # must never straddle a split boundary - the slot id deliberately omits the
            # camera.
            slot_id = f"{venue}_{parsed['slot_date']}_{hhmm}"

            rows.append(
                ManifestRow(
                    file=rel,
                    class4=class4.value,
                    class3=to_class3(class4).value,
                    venue=venue,
                    camera=camera_tag,
                    slot_date=str(parsed["slot_date"]),
                    slot_time=str(parsed["slot_time"]),
                    slot_id=slot_id,
                    t_s=int(parsed["t_s"]),  # type: ignore[arg-type]
                    source=str(parsed["source"]),
                    labeled_by="human" if recorded is not None else "bulk",
                    lighting=str(parsed["lighting"]),
                    quality="unknown",
                    split_role="",
                )
            )

    return rows, problems


def write_manifest(rows: list[ManifestRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [f.name for f in fields(ManifestRow)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            ManifestRow(**{**row, "t_s": int(row["t_s"])})  # type: ignore[arg-type]
            for row in csv.DictReader(fh)
        ]


def cross_tab(rows: list[ManifestRow], row_key: str, col_key: str) -> str:
    """Render a counts table of one field against another.

    Used to make confounding visible: if a class only ever appears in one slot or one
    lighting condition, a classifier can score well by recognising the *scene* instead
    of the state, and no split strategy can separate the two after the fact.
    """
    counts: Counter[tuple[str, str]] = Counter(
        (str(getattr(r, row_key)), str(getattr(r, col_key))) for r in rows
    )
    r_vals = sorted({r for r, _ in counts})
    c_vals = sorted({c for _, c in counts})
    width = max((len(c) for c in c_vals), default=8) + 2
    label_w = max((len(r) for r in r_vals), default=8) + 2

    lines = [" " * label_w + "".join(f"{c:>{width}}" for c in c_vals) + f"{'total':>{width}}"]
    for rv in r_vals:
        total = sum(counts[(rv, cv)] for cv in c_vals)
        lines.append(
            f"{rv:<{label_w}}"
            + "".join(f"{counts.get((rv, cv), 0):>{width}}" for cv in c_vals)
            + f"{total:>{width}}"
        )
    totals = [sum(counts[(rv, cv)] for rv in r_vals) for cv in c_vals]
    lines.append(
        f"{'total':<{label_w}}"
        + "".join(f"{t:>{width}}" for t in totals)
        + f"{sum(totals):>{width}}"
    )
    return "\n".join(lines)


def confound_warnings(rows: list[ManifestRow]) -> list[str]:
    """Flag classes that are concentrated in a single slot or lighting condition.

    This is the check that would have caught the pilot's inflated accuracy before it was
    reported: when class and scene coincide, high accuracy measures scene recognition.
    """
    warnings: list[str] = []
    by_class: dict[str, list[ManifestRow]] = {}
    for r in rows:
        by_class.setdefault(r.class3, []).append(r)

    for class3, members in sorted(by_class.items()):
        for field_name, label in (("slot_id", "slot"), ("lighting", "lighting condition")):
            spread = Counter(str(getattr(m, field_name)) for m in members)
            top, top_n = spread.most_common(1)[0]
            share = top_n / len(members)
            if share >= 0.95 and len(members) > 0:
                name = Class3(class3).name
                warnings.append(
                    f"{name}: {share:.0%} of {len(members)} frames come from a single "
                    f"{label} ({top}) - accuracy on this class cannot be separated "
                    f"from recognising that {label}"
                )
    return warnings
