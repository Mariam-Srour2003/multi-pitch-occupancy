"""The sampling scheduler (WP6-T2).

Runs as its own process, independent of the API. During an active slot it pulls one frame
per camera per minute from whichever source is configured, classifies it, fuses the two
halves of each pitch, and writes to the database; when the slot closes it aggregates a
verdict, selects evidence, and hands off to reconciliation.

Keeping this separate from ``api/`` is deliberate: a dashboard restart must never drop a
sample, and inference must never happen inside a request handler.

Run with::

    uv run python -m pitch_occupancy.worker

Not implemented yet - it waits on the preprocessing pipeline (WP3) and the frame sources
(WP6-T1/T3). See TODO.md.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "The scheduler arrives with WP6-T2; it depends on WP3 (preprocessing) "
        "and WP6-T1/T3 (frame sources). See TODO.md."
    )


if __name__ == "__main__":
    main()
