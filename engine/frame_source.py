"""Frame sources.

VIDEO_SIM: replays a recorded slot from video files — the two videos of one slot
act as camera A and camera B of one field, sampled once per simulated minute.
(RTSP_LIVE and API_SIM sources plug in here later with the same interface.)

Yields SampleSet(t_s, frames={cam_tag: BGR ndarray}).
"""
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@dataclass
class SampleSet:
    t_s: int                      # seconds since slot start
    frames: dict                  # cam_tag -> BGR ndarray


def camera_tag(video_path: Path) -> str:
    name = video_path.stem
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})", name)
    slot = f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}{m.group(5)}" if m else "unknown"
    cam = "camB" if "(1)" in name else "camA"
    return f"slot_{slot}_{cam}"


def discover_slots(data_dir: Path = DATA) -> dict:
    """Group videos by slot key -> {cam_tag: path}. Expects <=2 videos per slot."""
    slots: dict[str, dict[str, Path]] = {}
    for v in sorted(data_dir.glob("*.mp4")):
        tag = camera_tag(v)
        slot_key = tag.rsplit("_cam", 1)[0]
        slots.setdefault(slot_key, {})[tag] = v
    return slots


class VideoSlotSource:
    """Iterate a recorded slot: one frame per `interval_s` from each camera."""

    def __init__(self, videos: dict, interval_s: int = 60):
        self.videos = videos          # cam_tag -> Path
        self.interval_s = interval_s
        self.caps = {}
        self.fps = {}
        self.dur = {}
        for tag, path in videos.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {path}")
            self.caps[tag] = cap
            self.fps[tag] = cap.get(cv2.CAP_PROP_FPS) or 20.0
            self.dur[tag] = cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps[tag]

    def __iter__(self):
        t = 0
        max_dur = max(self.dur.values())
        while t < max_dur:
            frames = {}
            for tag, cap in self.caps.items():
                if t >= self.dur[tag]:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * self.fps[tag]))
                ok, frame = cap.read()
                if ok:
                    frames[tag] = frame
            if frames:
                yield SampleSet(t_s=t, frames=frames)
            t += self.interval_s

    def close(self):
        for cap in self.caps.values():
            cap.release()
