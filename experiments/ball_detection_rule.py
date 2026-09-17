"""Does finding a football help, and where? (A17)

The requested rule was three parts: nobody means empty, four or fewer means present-but-not-
playing, and "detect if there is a ball, it may help to know". The first is wired in as
`PersonGate`. The second is refused by the data - `person_count_rule.py` found a third of
genuine ACTIVE_PLAY frames show four or fewer people inside the boundary, so the count alone
cannot carry C3. The third has not been measured, and it is exactly the evidence the second
part is missing: **four people with a ball is a kickabout; four people without one is not.**

What makes this worth a run rather than an assumption: a football on a CCTV frame is a few
dozen pixels, and COCO's `sports ball` class was trained on photographs where it fills a
useful fraction of the image. The honest prior is that recall will be poor. That is a
reportable answer - it says the rule cannot be built from this detector - and it is not
knowable without looking.

Measured the same way the person count was, so the two are comparable:

- **inside the derived boundary**, by the box centre this time rather than the foot point, a
  ball being an object in the air rather than a person standing somewhere
- **on the same labelled split**, venue_01 camera B, plus every clip venue for cross-venue
  recall
- **cross-tabulated against the person count**, because the question is not "does a ball
  predict play" on its own - it is whether a ball resolves the frames the count cannot

    uv run python experiments/ball_detection_rule.py --clip-venues
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision import roi

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
EMPTY, PLAY, C3 = "C1_EMPTY", "C2_ACTIVE_PLAY", "C3_MAINTENANCE_NON_SPORTING"

#: COCO class ids. 0 is person, 32 is `sports ball`.
PERSON, BALL = 0, 32


def _mask_for(polygon, h: int, w: int):
    if polygon is None:
        return None
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array([[int(x * w), int(y * h)] for x, y in polygon], np.int32)], 1)
    return m


def detect(model, frame, polygon, *, imgsz: int, person_conf: float, ball_conf: float):
    """(people inside, balls inside, best ball confidence, largest ball area in px).

    The ball is run at its own lower confidence. A football is small and the detector is
    correspondingly unsure of it; using the person threshold for both would be measuring the
    threshold rather than the object. The confidence is written to the CSV so the cost of that
    choice stays visible rather than buried in a rate.
    """
    res = model.predict(frame, verbose=False, conf=min(person_conf, ball_conf),
                        classes=[PERSON, BALL], imgsz=imgsz)[0]
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    h, w = frame.shape[:2]
    mask = _mask_for(polygon, h, w)

    def inside(box, foot: bool) -> bool:
        if mask is None:
            return True
        x = int((box[0] + box[2]) / 2)
        y = int(box[3]) if foot else int((box[1] + box[3]) / 2)
        return 0 <= y < h and 0 <= x < w and bool(mask[y, x])

    people = sum(1 for b, c, p in zip(xyxy, cls, conf, strict=True)
                 if c == PERSON and p >= person_conf and inside(b, True))
    balls = [(b, p) for b, c, p in zip(xyxy, cls, conf, strict=True)
             if c == BALL and p >= ball_conf and inside(b, False)]
    best = max((p for _b, p in balls), default=0.0)
    area = max((float((b[2] - b[0]) * (b[3] - b[1])) for b, _p in balls), default=0.0)
    return people, len(balls), float(best), area


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--person-conf", type=float, default=0.25)
    ap.add_argument("--ball-conf", type=float, default=0.10)
    ap.add_argument("--clip-venues", action="store_true",
                    help="also run every recorded clip-venue frame (all ACTIVE_PLAY)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    rows = [r for r in read_manifest(DATASET / "manifest.csv") if r.source != "synthetic"]

    def cam(r):
        return PHYSICAL_CAMERA.get(r.camera, r.camera)

    frames = [r for r in rows if r.venue == "venue_01" and cam(r) == "camera_B"]
    if args.clip_venues:
        frames += [r for r in rows if r.venue.startswith("clipvenue_")]

    print(f"{len(frames)} recorded frames | yolov8n imgsz={args.imgsz}, "
          f"person conf {args.person_conf}, ball conf {args.ball_conf}, "
          f"counted inside each camera's boundary\n")

    records = []
    for n, r in enumerate(frames, 1):
        frame = cv2.imread(str(DATASET / r.file))
        if frame is None:
            continue
        people, balls, best, area = detect(model, frame, roi.get(r.camera), imgsz=args.imgsz,
                                           person_conf=args.person_conf,
                                           ball_conf=args.ball_conf)
        records.append({"file": r.file, "venue": r.venue, "class3": r.class3,
                        "people": people, "balls": balls,
                        "ball_conf": round(best, 3), "ball_area_px": round(area, 1)})
        if n % 50 == 0:
            print(f"  {n}/{len(frames)}", end="\r", flush=True)
    print(" " * 30, end="\r")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "ball_detection_rule.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    def rate(sub):
        return sum(1 for d in sub if d["balls"] > 0) / len(sub) if sub else float("nan")

    v01 = [d for d in records if d["venue"] == "venue_01"]
    print(f"{'class':<32}{'n':>6}{'ball found':>12}{'median conf':>13}")
    for cls in (EMPTY, PLAY, C3):
        sub = [d for d in v01 if d["class3"] == cls]
        if not sub:
            continue
        hits = [d["ball_conf"] for d in sub if d["balls"] > 0]
        med = f"{np.median(hits):.3f}" if hits else "-"
        print(f"{cls:<32}{len(sub):>6}{rate(sub):>11.0%}{med:>13}")

    # The question the count could not answer, asked only of the frames it could not answer.
    print("\nthe frames the person count cannot separate (1-4 people inside):")
    print(f"{'class':<32}{'n':>6}{'ball found':>12}")
    for cls in (EMPTY, PLAY, C3):
        sub = [d for d in v01 if d["class3"] == cls and 1 <= d["people"] <= 4]
        if sub:
            print(f"{cls:<32}{len(sub):>6}{rate(sub):>11.0%}")

    e = [d for d in v01 if d["class3"] == EMPTY]
    p = [d for d in v01 if d["class3"] == PLAY]
    if e and p:
        print("\nthe rule 'a ball inside the boundary means play':")
        print(f"  recall on ACTIVE_PLAY   {rate(p):.3f}")
        print(f"  false-play on EMPTY     {rate(e):.3f}")
        print(f"  balanced                {rate(p) - rate(e):+.3f}")

    clips = [d for d in records if d["venue"].startswith("clipvenue_")]
    if clips:
        print("\ncross-venue, every frame ACTIVE_PLAY:")
        print(f"{'venue':<30}{'n':>6}{'ball found':>12}")
        for v in sorted({d["venue"] for d in clips}):
            sub = [d for d in clips if d["venue"] == v]
            print(f"{v:<30}{len(sub):>6}{rate(sub):>11.0%}")
        print(f"{'all clip venues':<30}{len(clips):>6}{rate(clips):>11.0%}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
