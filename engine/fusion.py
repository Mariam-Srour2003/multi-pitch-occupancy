"""Dual-camera fusion: two half-pitch views -> one field-level state per sample.

Rules (each camera sees one half of the pitch):
  - PLAYING on either half        -> field PLAYING
  - MAINTENANCE on either half    -> field MAINTENANCE (unless the other half plays)
  - PEOPLE on either half         -> field PEOPLE_NOT_PLAYING
  - EMPTY requires BOTH halves empty

Confidence of the fused state = confidence of the camera that determined it
(max over the cameras reporting that state).
"""

PRIORITY = ["2_playing", "4_maintenance", "3_people_not_playing", "1_empty"]


def fuse(per_camera: dict) -> tuple[str, float]:
    """per_camera: {cam_tag: (label, confidence)} -> (field_label, confidence)."""
    if not per_camera:
        return "1_empty", 0.0
    for state in PRIORITY:
        holders = [conf for (label, conf) in per_camera.values() if label == state]
        if holders:
            if state == "1_empty" and len(holders) < len(per_camera):
                continue  # empty needs unanimity; keep looking (won't happen given priority)
            return state, max(holders)
    return "1_empty", 0.0
