"""Which labelled frames does the detector disagree with? (2026-09-21)

The folder a frame sits in is its label (`docs/data_layout.md` rule 3), and nothing has ever
checked those folders against what is in the pictures. This does, with the only instrument
available that is independent of the label: **count the people and look for a ball**, then
ask whether the folder is consistent with the count under the facility's own rule.

**It cannot say a label is wrong.** A detector that misses far-side players will call a real
match C3, which is a detector failure and not a labelling one - A16 measured 88 of 278 real
play frames with four or fewer people inside a single camera's boundary. So this ranks frames
by **how hard they are to reconcile**, worst first, and a person looks at the top of the
list. The output is a queue for a human, not a verdict, and `results/label_audit.csv` is
named so that nothing mistakes it for one.

Three disagreements are worth different amounts:

- **EMPTY with people found** is the strongest signal, because the detector's false-person
  rate on a pitch is the one error this project has measured directly - 11% of venue_01
  camera B's recorded empty frames hold at least one detection (A16). One person is within
  that; three is not.
- **ACTIVE_PLAY with nobody found at all** is next. A missed player is ordinary; a frame
  where every player is missed is not.
- **C3 with a crowd and a ball** is weakest and is scored low deliberately, because
  `3_maintenance_non_sporting` legitimately holds team talks and groups standing about -
  that is the class A40 widened it to be.

    uv run python experiments/label_audit.py
    uv run python experiments/label_audit.py --limit 200 --contact-sheet
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter

import cv2

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.vision import roi
from pitch_occupancy.vision.counting import count_inside
from pitch_occupancy.vision.detector import DEFAULT_DETECTOR, Detector
from pitch_occupancy.vision.rules import RuleConfig

RESULTS = settings.results_dir

#: At or above this, a frame goes in the queue for a person to look at.
FLAG = 1.0


def suspicion(cls: str, people: int, ball: bool, cfg: RuleConfig) -> tuple[float, str]:
    """How hard this frame is to reconcile with its folder, and why. 0 means consistent."""
    if cls == Class3.EMPTY.value:
        return (float(people), f"{people} person(s) found in a frame labelled EMPTY")
    if cls == Class3.ACTIVE_PLAY.value:
        if people == 0:
            return (3.0, "no people found at all in a frame labelled ACTIVE_PLAY")
        if people <= cfg.small_group_max:
            # Expected on half-pitch cameras. Worth counting in bulk, not per frame.
            return (0.5, f"only {people} inside the boundary, below the play threshold")
        return (0.0, "")
    if people == 0:
        return (2.0, "no people found at all in a frame labelled C3")
    if people > cfg.small_group_max and ball:
        return (0.25, f"{people} people and a ball in a frame labelled C3")
    return (0.0, "")


def contact_sheet(worst: list[dict], path) -> None:
    import numpy as np

    tiles = []
    for r in worst:
        image = cv2.imread(str(settings.dataset_dir / r["file"]))
        if image is None:
            continue
        tile = cv2.resize(image, (320, 184))
        cv2.rectangle(tile, (0, 0), (320, 16), (0, 0, 0), -1)
        cv2.putText(tile, f"{r['class3'][:2]} n={r['people_inside']} {r['file'][-24:]}",
                    (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    blank = np.zeros((184, 320, 3), np.uint8)
    grid = [np.hstack(tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(grid))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_DETECTOR)
    ap.add_argument("--limit", type=int, default=0, help="audit only the first N frames")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="render the worst offenders for a human to look at")
    args = ap.parse_args()

    cfg = RuleConfig.load(settings.rules_path)
    det = Detector.load(args.model)
    rows = [r for r in read_manifest(settings.dataset_dir / "manifest.csv")
            if r.source != "synthetic"]
    if args.limit:
        rows = rows[:args.limit]
    print(f"auditing {len(rows)} recorded frames with {args.model}")
    print("a queue for a person, not a verdict - see the module docstring")

    out: list[dict] = []
    for n, row in enumerate(rows, 1):
        frame = cv2.imread(str(settings.dataset_dir / row.file))
        if frame is None:
            continue
        polygon = roi.resolve(row.camera)
        found = det.detect(frame, confidence=min(cfg.person_conf, cfg.ball_conf),
                           imgsz=cfg.imgsz, tiles=cfg.tiles)
        if found is None:
            continue
        count = count_inside(found, polygon, frame.shape[:2], person_conf=cfg.person_conf,
                             ball_conf=cfg.ball_conf)
        score, why = suspicion(row.class3, count.people_inside, count.ball_seen, cfg)
        out.append({"file": row.file, "label": row.label, "class3": row.class3,
                    "venue": row.venue, "lighting": row.lighting, "source": row.source,
                    "boundary": bool(polygon), "people_inside": count.people_inside,
                    "people_total": count.people_total, "ball": count.ball_seen,
                    "suspicion": round(score, 2), "why": why})
        if n % 100 == 0:
            print(f"    {n}/{len(rows)}")
    print(f"    {len(out)}/{len(rows)} audited")

    out.sort(key=lambda r: -r["suspicion"])
    flagged = [r for r in out if r["suspicion"] >= FLAG]
    share = len(flagged) / max(len(out), 1)
    print(f"\n{len(flagged)} frame(s) hard to reconcile with their folder ({share:.1%})")
    print(f"\n{'class':<32}{'audited':>9}{'flagged':>9}{'no boundary':>13}")
    for cls in sorted({r["class3"] for r in out}):
        group = [r for r in out if r["class3"] == cls]
        hard = [r for r in group if r["suspicion"] >= FLAG]
        unbounded = sum(1 for r in group if not r["boundary"])
        print(f"{cls:<32}{len(group):>9}{len(hard):>9}{unbounded:>13}")

    print(f"\nby venue, flagged only:")
    per_venue = Counter(r["venue"] for r in flagged)
    for venue, n in per_venue.most_common():
        total = sum(1 for r in out if r["venue"] == venue)
        print(f"  {venue:<32}{n:>5} of {total}")

    print(f"\nworst {min(25, len(flagged))}, for a human to look at:")
    print(f"{'file':<58}{'venue':<26}{'n':>4}  why")
    for r in flagged[:25]:
        print(f"{r['file'][-56:]:<58}{r['venue'][:24]:<26}{r['people_inside']:>4}  {r['why']}")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "label_audit.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"\nwrote {path}")

    if args.contact_sheet and flagged:
        sheet = RESULTS / "figs" / "label_audit_worst.png"
        contact_sheet(flagged[:24], sheet)
        print(f"wrote {sheet}")

    record(
        "label audit: folders against what the detector finds",
        "uv run python experiments/label_audit.py",
        "label_audit.csv",
        f"{len(flagged)} of {len(out)} recorded frames hard to reconcile with their folder; "
        + "; ".join(f"{c} {sum(1 for r in out if r['class3'] == c and r['suspicion'] >= FLAG)}"
                    f"/{sum(1 for r in out if r['class3'] == c)}"
                    for c in sorted({r["class3"] for r in out}))
        + "; a queue for a person, not a verdict - a detector that misses far-side players "
          "calls a real match C3",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
