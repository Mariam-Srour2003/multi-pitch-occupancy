"""Slot aggregation: fused per-minute samples -> USED / NOTUSED / REVIEW + evidence.

Thresholds from config/system_config.json AGGREGATION (blueprint §6.2):
  ratio_playing >= 0.35                          -> USED
  ratio_playing < 0.10 and ratio_empty >= 0.75   -> NOTUSED
  otherwise                                      -> REVIEW

Evidence (blueprint §6.3): highest-confidence playing frame from each third of the
slot; backfilled with highest-confidence empty frames when fewer than 3 exist.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGG = json.loads((ROOT / "config" / "system_config.json").read_text())["AGGREGATION"]


@dataclass
class SlotResult:
    status: str
    confidence: float
    reason: str
    ratios: dict
    n_samples: int
    evidence: list = field(default_factory=list)   # [(t_s, label, conf)]


def aggregate(samples: list) -> SlotResult:
    """samples: [(t_s, fused_label, confidence)] in time order."""
    n = len(samples)
    if n == 0:
        return SlotResult("REVIEW", 0.0, "no samples collected", {}, 0)

    ratios = {}
    for cls in ["1_empty", "2_playing", "3_people_not_playing", "4_maintenance"]:
        ratios[cls] = sum(1 for _, lbl, _ in samples if lbl == cls) / n
    r_play, r_empty = ratios["2_playing"], ratios["1_empty"]

    if r_play >= AGG["USED_MIN_PLAYING_RATIO"]:
        status = "USED"
        reason = (f"football activity in {r_play:.0%} of {n} samples "
                  f"(threshold {AGG['USED_MIN_PLAYING_RATIO']:.0%})")
    elif r_play < AGG["NOTUSED_MAX_PLAYING_RATIO"] and r_empty >= AGG["NOTUSED_MIN_EMPTY_RATIO"]:
        status = "NOTUSED"
        reason = (f"pitch empty in {r_empty:.0%} of {n} samples, "
                  f"activity in only {r_play:.0%}")
    else:
        status = "REVIEW"
        reason = (f"inconclusive: playing {r_play:.0%}, empty {r_empty:.0%}, "
                  f"people {ratios['3_people_not_playing']:.0%}, "
                  f"maintenance {ratios['4_maintenance']:.0%}")

    mean_conf = sum(c for _, _, c in samples) / n
    return SlotResult(status, round(mean_conf, 4), reason, ratios, n,
                      evidence=select_evidence(samples))


def select_evidence(samples: list, k: int = 3) -> list:
    """Top playing frame per slot-third; backfill with top empty frames."""
    if not samples:
        return []
    t_max = max(t for t, _, _ in samples) or 1
    picks = []
    for third in range(3):
        lo, hi = third * t_max / 3, (third + 1) * t_max / 3
        cands = [s for s in samples
                 if s[1] == "2_playing" and lo <= s[0] <= hi and s not in picks]
        if cands:
            picks.append(max(cands, key=lambda s: s[2]))
    if len(picks) < k:
        fill = sorted((s for s in samples if s not in picks),
                      key=lambda s: s[2], reverse=True)
        picks += fill[:k - len(picks)]
    return sorted(picks[:k], key=lambda s: s[0])
