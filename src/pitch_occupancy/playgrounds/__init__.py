"""Playgrounds: groups of cameras that watch one pitch, checked together on their own timer.

A playground is what a person means by "the pitch": one camera on a small one, two or three on
a big one. Each has its cameras (a camera belongs to one playground at most), how often it is
checked (every 5 seconds up to every hour), its hours (all day, or from-to), and how its
cameras' counts are combined - the part that decides whether one person is counted once:

* ``add``: the cameras watch *different parts*. Each counts only the people inside its own
  drawn area (`vision/roi.py`) and the counts are added. Nobody is counted twice **provided
  the drawn areas do not overlap**; drawing them edge to edge is the fix.
  Playing is decided by the slot worker's pitch rule (`runtime.combine_pitch`): a pitch too
  dark to see is not sure (stored as NULL), a standing crowd is not playing.
* ``largest``: the cameras watch *the same area*; the largest single count is used.
* ``average``: the same area, one camera sometimes missing people; the counts are averaged.
* ``vote``: each camera decides; playing when at least half say so. The count is the largest.

Matching the same person across cameras without areas (re-identification) would need the
cameras calibrated to each other, so it is not attempted. In the other modes a playground
with ``play_min`` (`configs/rules.json`) or more people is **playing**.

Modules: `store` (the saved settings), `capacity` (what this computer can do), `check` (one
check, and the second opinion), `snapshots` (saved pictures), `runner` (the background
thread), `sessions` (the stored checks, grouped for the pages).

The functions most callers need are re-exported here. Settings a test redirects - ``CONFIG_PATH``
(store), ``SPEED_PATH`` (capacity), ``SNAPSHOT_DIR``/``SNAPSHOT_CAP_BYTES`` (snapshots) - and
seams it swaps - ``runtime.grab_whole`` and ``runtime.shared_pipeline`` (on
`pitch_occupancy.runtime`), ``check.play_min``, ``check.check_playground`` - are deliberately
*not* re-exported: each is read from its own module at call time, so patch it there
(``playgrounds.store.CONFIG_PATH``) and one patch reaches every caller. ``capacity()`` and
``sessions()`` share their module's name; call them as ``playgrounds.capacity.capacity()`` and
``playgrounds.sessions.sessions()``.
"""

from __future__ import annotations

from pitch_occupancy.playgrounds import capacity, check, runner, sessions, snapshots, store
from pitch_occupancy.playgrounds.capacity import HEADROOM, load_speed, measure_speed
from pitch_occupancy.playgrounds.check import check_playground, combine, wait_for_second_opinions
from pitch_occupancy.playgrounds.runner import run_now, start, status, stop
from pitch_occupancy.playgrounds.sessions import (
    PERIOD_CHOICES,
    SESSION_ANCHOR,
    SESSION_MINUTES,
    SESSION_PLAYED,
    SESSION_SOME,
    every_text,
    latest,
    records,
    session_bounds,
)
from pitch_occupancy.playgrounds.snapshots import snapshot_usage
from pitch_occupancy.playgrounds.store import (
    COUNTING,
    COUNTING_SHORT,
    EVERY_CHOICES,
    delete_playground,
    in_hours,
    load,
    save_playground,
    set_enabled,
)

__all__ = [
    "capacity", "check", "runner", "sessions", "snapshots", "store",
    "HEADROOM", "load_speed", "measure_speed",
    "check_playground", "combine", "wait_for_second_opinions",
    "run_now", "start", "status", "stop",
    "PERIOD_CHOICES", "SESSION_ANCHOR", "SESSION_MINUTES", "SESSION_PLAYED", "SESSION_SOME",
    "every_text", "latest", "records", "session_bounds",
    "snapshot_usage",
    "COUNTING", "COUNTING_SHORT", "EVERY_CHOICES", "delete_playground", "in_hours", "load",
    "save_playground", "set_enabled",
]
