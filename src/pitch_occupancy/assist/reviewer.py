"""When to ask, what to send, and how the answer is combined with the detector's count.

Called by the playground runner after each check is stored (`playgrounds.check_playground`).
It decides from the database, not from memory, how many questions this playground has had in
the current window, so a restart does not reset the pace.

**When.** A check is *doubtful* when the detector found people but fewer than a game needs -
the case the detector gets wrong most. Doubtful checks are asked about first; with ``audit``
on, one ordinary check per window is asked about too, so the stored opinions measure how often
the two disagree. Never more than ``per_window`` per window, never two within
``min_gap_minutes``.

**What changes.** Only a doubtful check, only to playing, only when the model says playing
with at least ``upgrade_confidence``. The check's ``playing_source`` then reads ``assist`` and
its ``detector_playing`` keeps what the detector said, so the upgrade is visible and can be
undone.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from pitch_occupancy.assist import budget
from pitch_occupancy.assist.config import AssistConfig
from pitch_occupancy.assist.providers import Opinion, ProviderError, RateLimited, ask

__all__ = ["CameraPicture", "Decision", "should_ask", "prepare", "review"]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CameraPicture:
    """One camera's picture of a check, and the person boxes the detector saw in it."""

    frame: np.ndarray
    people_boxes: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class Decision:
    ask: bool
    doubtful: bool
    why: str


def _utc(when: datetime) -> str:
    """Stored and compared in UTC, so a change of clocks cannot shift the windows."""
    return when.astimezone(UTC).isoformat(timespec="seconds")


def _asked_since(conn: sqlite3.Connection, playground_id: str,
                 since: datetime) -> list[tuple[datetime, bool]]:
    return [(datetime.fromisoformat(r[0]), bool(r[1])) for r in conn.execute(
        """SELECT asked_at, doubtful FROM assist_opinions WHERE playground_id = ?
           AND asked_at >= ? ORDER BY asked_at""", (playground_id, _utc(since)))]


def should_ask(cfg: AssistConfig, conn: sqlite3.Connection, record: dict, now: datetime,
               play_min: int) -> Decision:
    """Whether this stored check should go to the model.

    Doubtful checks come first: the comparison sample (``audit``) is taken only while the
    window is still unused, takes one question at most, and never holds a doubtful check back
    through the minimum gap.
    """
    people, playing = record.get("people"), record.get("playing")
    doubtful = people is not None and people >= 1 and not playing and people < play_min
    asked = _asked_since(conn, record["playground_id"],
                         now - timedelta(minutes=cfg.window_minutes))
    if len(asked) >= cfg.per_window:
        return Decision(False, doubtful, "this window's questions are used")
    if asked:
        last_at, last_doubtful = asked[-1]
        too_soon = last_at > now - timedelta(minutes=cfg.min_gap_minutes)
        if too_soon and (last_doubtful or not doubtful):
            return Decision(False, doubtful, "asked too recently")
    if doubtful:
        return Decision(True, True, f"{people} people, fewer than {play_min}")
    if people is None:
        return Decision(False, False, "no picture")
    if cfg.audit and not asked:
        return Decision(True, False, "one ordinary check per window, for comparison")
    return Decision(False, False, "nothing doubtful")


#: Blur applied to every picture sent, after shrinking: at 1024 pixels across a face is a few
#: pixels, and this makes it unrecognisable while bodies, kit and the ball stay readable.
FLOOR_BLUR = 5


def prepare(pictures: Sequence[CameraPicture], cfg: AssistConfig) -> list[bytes]:
    """Shrink, redact and encode the pictures to send.

    **Two layers, always** (the rule in `vision/redact.py`): the heads of the people the
    detector found are pixelated, and the whole picture is blurred lightly - because the far
    players the detector missed, the very reason for asking, have heads too.
    """
    import cv2

    from pitch_occupancy.vision.redact import pixelate_boxes

    out = []
    for pic in pictures:
        frame = pic.frame
        h, w = frame.shape[:2]
        scale = min(1.0, cfg.max_side / max(h, w))
        if scale < 1:
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        if cfg.blur_heads:
            heads = [(int(x1 * scale), int(y1 * scale), int(x2 * scale),
                      int((y1 + max(1, (y2 - y1) // 3)) * scale))
                     for x1, y1, x2, y2 in pic.people_boxes]
            heads = [(x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)) for x1, y1, x2, y2 in heads]
            if heads:
                frame = pixelate_boxes(frame, heads, blocks=3)
            frame = cv2.GaussianBlur(frame, (FLOOR_BLUR, FLOOR_BLUR), 0)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out.append(buf.tobytes())
    return out


def _consult(cfg: AssistConfig, conn: sqlite3.Connection, images: list[bytes],
             people: int | None, now: datetime, opener=None) -> tuple[Opinion | None, str]:
    """The first provider in the chain that is configured, within budget and answers."""
    reasons = []
    for provider in cfg.chain():
        if not provider.configured:
            reasons.append(f"{provider.name}: no key")
            continue
        ok, why = budget.reserve(conn, provider, now)
        if not ok:
            reasons.append(why)
            continue
        try:
            opinion = ask(provider, images, people=people, timeout_s=cfg.timeout_s,
                          opener=opener)
        except RateLimited as exc:
            budget.settle(conn, provider, now, tokens=0, failed=True)
            budget.rest(conn, provider, now, exc.retry_after_s)
            reasons.append(str(exc))
            continue
        except ProviderError as exc:
            budget.settle(conn, provider, now, tokens=0, failed=True)
            reasons.append(str(exc))
            continue
        budget.settle(conn, provider, now, tokens=opinion.tokens)
        return opinion, ""
    return None, "; ".join(reasons) or "no provider listed"


def review(cfg: AssistConfig, conn: sqlite3.Connection, record: dict,
           pictures: Sequence[CameraPicture], *, now: datetime, play_min: int,
           opener=None) -> dict | None:
    """Ask about one stored check if it is due; store the opinion; upgrade when warranted.

    Returns the stored opinion row, or None when nothing was asked. Never raises for a
    provider's sake: the count stands whatever happens here.
    """
    if not cfg.enabled or not pictures or not any(p.configured for p in cfg.chain()):
        return None
    decision = should_ask(cfg, conn, record, now, play_min)
    if not decision.ask:
        return None
    images = prepare(pictures, cfg)
    opinion, problem = _consult(cfg, conn, images, record.get("people"), now, opener=opener)
    upgraded = bool(
        opinion is not None and decision.doubtful and opinion.state == "playing"
        and opinion.confidence >= cfg.upgrade_confidence)
    row = {
        "check_id": record.get("check_id"), "playground_id": record["playground_id"],
        "asked_at": _utc(now), "doubtful": int(decision.doubtful),
        "provider": opinion.provider if opinion else None,
        "model": opinion.model if opinion else None,
        "state": opinion.state if opinion else None,
        "people": opinion.people if opinion else None,
        "confidence": opinion.confidence if opinion else None,
        "reason": opinion.reason if opinion else None,
        "tokens": opinion.tokens if opinion else None,
        "latency_ms": opinion.latency_ms if opinion else None,
        "upgraded": int(upgraded), "problem": problem or None,
    }
    conn.execute(
        """INSERT INTO assist_opinions (check_id, playground_id, asked_at, doubtful, provider,
               model, state, people, confidence, reason, tokens, latency_ms, upgraded, problem)
           VALUES (:check_id, :playground_id, :asked_at, :doubtful, :provider, :model, :state,
                   :people, :confidence, :reason, :tokens, :latency_ms, :upgraded, :problem)""",
        row)
    if upgraded and record.get("check_id") is not None:
        conn.execute(
            """UPDATE playground_checks SET detector_playing = playing, playing = 1,
                   playing_source = 'assist' WHERE check_id = ?""", (record["check_id"],))
        record["playing"] = 1
    conn.commit()
    if problem:
        log.info("assist: no opinion for %s (%s)", record["playground_id"], problem)
    return row
