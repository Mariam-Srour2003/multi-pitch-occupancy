"""Choose the frames for the hand-count audit, and render them for counting (WP9-T0b).

The detector-first path (A36) stands on a detector's count, so the detector needs a truth
that is not another detector: a person looking at each frame and counting the people whose
feet are inside the boundary. This script picks the frames and prepares them; it counts
nothing.

**A hundred frames, stratified, spread over scenes.** Every development venue, both cameras
at venue_01, every recorded class, both lighting conditions where they exist - and within a
stratum, one frame per perceptual-hash scene before a second from any scene
(`data/processed/scene_ids.csv`), because A28 measured the corpus at 179 distinct scenes in
1,692 frames and twenty frames of one passage of play would be one measurement wearing
twenty rows. The locked venues are excluded: a hand count is development work.

**Rendered with the boundary drawn**, at 1280 wide, into a folder the counter opens. The
boundary is the same one the detector is scored inside (`roi.resolve`), so "inside" means
the same thing to the person and to the model. The rendered copies are working files, never
published, and carry identifiable people - they stay under the scratch folder given.

The output CSV has the frame list with the truth columns **empty**, for a person to fill:

    file, venue, label, lighting, camera, people_inside, ball_visible, counter, date, note

    uv run python scripts/sample_hand_counts.py --out results/hand_counts.csv --render <folder>
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision import roi

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
SEED = 42

#: How many frames each stratum gets. Venue_01's EMPTY is five scenes in 494 frames, so it
#: does not need more than this to cover them; the clip venues have one class each and are
#: where far-side recall is decided, so they get the most. The 6 recorded C3 frames are all
#: taken. Anything left over from a stratum that runs short is given to venue_01 play.
QUOTA = {
    ("venue_01", "1_empty"): 20,
    ("venue_01", "2_playing"): 24,
    ("venue_01", "3_maintenance_non_sporting"): 6,
    ("clip", "2_playing"): 50,
}


def _scenes() -> dict[str, str]:
    path = DATASET / "scene_ids.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["file"]: row["scene_id"] for row in csv.DictReader(fh)}


def _spread(rows, n: int, scenes: dict[str, str], rng: random.Random, *, by=None) -> list:
    """``n`` rows, one per scene before any second one, balanced over ``by`` if given."""
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[by(r) if by else ""].append(r)
    picked: list = []
    # round-robin over the sub-groups (cameras, venues), and within each, scene-first
    queues = {}
    for key, members in sorted(groups.items()):
        by_scene: dict[str, list] = defaultdict(list)
        for r in sorted(members, key=lambda r: r.file):
            by_scene[scenes.get(r.file, r.file)].append(r)
        order = []
        scene_lists = [rng.sample(v, len(v)) for _, v in sorted(by_scene.items())]
        rng.shuffle(scene_lists)
        while any(scene_lists):
            for lst in scene_lists:
                if lst:
                    order.append(lst.pop())
        queues[key] = order
    keys = sorted(queues)
    while len(picked) < n and any(queues.values()):
        for key in keys:
            if queues[key] and len(picked) < n:
                picked.append(queues[key].pop(0))
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "hand_counts.csv")
    ap.add_argument("--render", type=Path, default=None,
                    help="folder for boundary-drawn copies at 1280 wide (working files)")
    ap.add_argument("--total", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; it may hold counts. Pass --force to overwrite.")

    rng = random.Random(SEED)
    scenes = _scenes()
    rows = [r for r in development_rows(read_manifest(DATASET / "manifest.csv"))
            if r.source != "synthetic"]

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    v01 = [r for r in rows if r.venue == "venue_01"]
    clips = [r for r in rows if r.venue != "venue_01"]
    chosen: list = []
    chosen += _spread([r for r in v01 if r.label == "1_empty"],
                      QUOTA[("venue_01", "1_empty")], scenes, rng, by=cam)
    chosen += _spread([r for r in v01 if r.label == "3_maintenance_non_sporting"],
                      QUOTA[("venue_01", "3_maintenance_non_sporting")], scenes, rng)
    chosen += _spread([r for r in clips if r.label == "2_playing"],
                      QUOTA[("clip", "2_playing")], scenes, rng, by=lambda r: r.venue)
    remaining = args.total - len(chosen)
    chosen += _spread([r for r in v01 if r.label == "2_playing"], remaining, scenes, rng,
                      by=cam)

    if args.render is not None:
        args.render.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "venue", "label", "lighting", "camera",
                    "people_inside", "ball_visible", "counter", "date", "note"])
        for i, r in enumerate(chosen, 1):
            w.writerow([r.file, r.venue, r.label, r.lighting, cam(r), "", "", "", "", ""])
            if args.render is not None:
                frame = cv2.imread(str(DATASET / r.file))
                if frame is None:
                    continue
                polygon = roi.resolve(r.camera)
                drawn = roi.outline(frame, polygon, colour=(0, 255, 255), thickness=3)
                h, wide = drawn.shape[:2]
                if wide > 1280:
                    drawn = cv2.resize(drawn, (1280, int(h * 1280 / wide)),
                                       interpolation=cv2.INTER_AREA)
                cv2.putText(drawn, f"#{i:03d} {Path(r.file).name}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(args.render / f"{i:03d}.jpg"), drawn,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    counts = defaultdict(int)
    for r in chosen:
        counts[(r.venue, r.label, r.lighting)] += 1
    print(f"{len(chosen)} frames -> {args.out}")
    for (venue, cls, light), n in sorted(counts.items()):
        print(f"  {venue:<28} {cls:<22} {light:<8} {n:>3}")
    if args.render is not None:
        print(f"rendered with boundaries to {args.render}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
