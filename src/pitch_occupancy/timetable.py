"""Cut a day of footage into booking slots and say, per slot, whether the pitch was used.

The clip reviewer answers *"what happened in this video, and when?"* on the video's own clock
(0:00, 1:30, ...). A facility does not think in that clock. It thinks in rental slots -
08:00-09:30, 09:30-11:00 - and the question it asks of a day's recording is *"which of these
slots were played in?"*. This module is the translation, and only the translation: it takes
per-moment verdicts that something else produced and adds them up per slot. No footage, no
model and no web framework, so every rule here can be checked by hand in a test.

Four decisions shape it.

**A slot is judged on the time it covers, not on the samples it happens to hold.** Each
sample stands for ``interval_s`` seconds, and those seconds are split across slot edges
rather than credited whole to whichever slot the sample starts in. Otherwise a slot
boundary that is not a multiple of the interval quietly moves minutes between slots.

**Not seeing is not the same as seeing nothing.** A slot only half covered by the video, or
one where the system abstained for most of it, is REVIEW - never NOT USED. An empty answer
built from missing evidence is the failure that accuses a customer of a no-show they did not
commit, so ``seen_share`` is computed against the whole slot, not against what was recorded.

**A close call is said out loud.** When playing is between 40% and 60% of what was seen, the
slot is REVIEW rather than whichever side of 50% the noise put it on. The numbers live in
:class:`Rules` so a facility that wants a different line can say so in one place.

**Comparing with bookings never accuses on a REVIEW slot.** :func:`compare` reports "review"
for any slot the footage could not settle, whatever the booking says.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from enum import StrEnum

from pitch_occupancy.bookings import BookingRecord

__all__ = [
    "SlotWindow", "SlotStatus", "Rules", "DEFAULT_RULES", "SlotResult", "SlotCheck",
    "OFF_SALE", "ACTIVITY_LABELS", "slot_windows", "summarise", "compare", "to_csv",
]

_DAY_S = 24 * 3600

#: Booking statuses that mean the facility took the slot off sale (`bookings.SOLD` is the
#: other side: the slot was sold).
OFF_SALE: frozenset[str] = frozenset({"maintenance", "blocked"})

_PLAY = "C2_ACTIVE_PLAY"
_EMPTY = "C1_EMPTY"
_MAINT = "C3_MAINTENANCE_NON_SPORTING"

#: How each activity is named to a client. No class codes on the page.
ACTIVITY_LABELS: dict[str, str] = {
    _PLAY: "playing",
    _EMPTY: "empty",
    _MAINT: "maintenance / not playing",
    "mixed": "mixed",
    "": "nothing seen",
}


class SlotStatus(StrEnum):
    USED = "USED"
    NOT_USED = "NOT_USED"
    REVIEW = "REVIEW"


@dataclass(frozen=True, slots=True)
class Rules:
    """The three numbers that turn a slot's minutes into a result."""

    #: Below this share of the slot decided, the slot is REVIEW: too little was seen.
    min_seen_share: float = 0.5
    #: Playing between these shares of the *decided* time is a close call: REVIEW.
    close_low: float = 0.40
    close_high: float = 0.60


DEFAULT_RULES = Rules()


@dataclass(frozen=True, slots=True)
class SlotWindow:
    """One rental slot, placed on the video's clock.

    ``start_s``/``end_s`` are seconds from the first frame of the video and may run past
    either end of it; ``covered_start_s``/``covered_end_s`` are the part the video holds.
    ``day_offset`` is how many days after the video's first day the slot starts, so a video
    started at 23:00 labels its 00:30 slot as the next day rather than the same one.
    """

    index: int
    start: time
    end: time
    day_offset: int
    start_s: float
    end_s: float
    covered_start_s: float
    covered_end_s: float

    @property
    def length_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def covered_s(self) -> float:
        return max(0.0, self.covered_end_s - self.covered_start_s)

    @property
    def coverage(self) -> float:
        return self.covered_s / self.length_s if self.length_s > 0 else 0.0

    @property
    def label(self) -> str:
        suffix = f" (+{self.day_offset}d)" if self.day_offset else ""
        return f"{self.start:%H:%M}-{self.end:%H:%M}{suffix}"

    def starts_on(self, day: date) -> datetime:
        """The slot's start as a datetime, given the date the video started on."""
        return datetime.combine(day, self.start) + timedelta(days=self.day_offset)


def _seconds(t: time) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


def _clock(seconds_of_day: float) -> tuple[time, int]:
    whole = int(round(seconds_of_day))
    days, rest = divmod(whole, _DAY_S)
    return time(rest // 3600, rest % 3600 // 60, rest % 60), days


def slot_windows(video_start: time, duration_s: float, slot_minutes: int,
                 first_slot: time | None = None) -> list[SlotWindow]:
    """The consecutive slots the video touches, from ``first_slot`` (default: video start).

    A slot the video only partly covers is kept, with its coverage. Video before
    ``first_slot`` belongs to no slot. ``first_slot`` is read as the nearest such clock time
    to the video start - within twelve hours either way - so 00:00 after a 23:00 start is
    one hour in, and 07:30 before an 08:00 start is half an hour before it (its first slot is
    then only partly covered).
    """
    if slot_minutes <= 0:
        raise ValueError(f"slot_minutes must be positive, not {slot_minutes}")
    if duration_s <= 0:
        return []
    slot_s = slot_minutes * 60
    base = _seconds(video_start)
    delta = 0.0
    if first_slot is not None:
        delta = (_seconds(first_slot) - base) % _DAY_S
        if delta > _DAY_S / 2:
            delta -= _DAY_S
    out: list[SlotWindow] = []
    k = 0
    while True:
        start_s = delta + k * slot_s
        end_s = start_s + slot_s
        k += 1
        if start_s >= duration_s:
            break
        if end_s <= 0:
            continue
        start, day = _clock(base + start_s)
        end, _ = _clock(base + end_s)
        out.append(SlotWindow(
            index=len(out), start=start, end=end, day_offset=day,
            start_s=start_s, end_s=end_s,
            covered_start_s=max(0.0, start_s), covered_end_s=min(duration_s, end_s),
        ))
    return out


@dataclass(frozen=True, slots=True)
class SlotResult:
    """One slot's minutes, shares, main activity and result, with a one-line reason."""

    window: SlotWindow
    playing_s: float
    empty_s: float
    maintenance_s: float
    uncertain_s: float
    #: Decided time over the whole slot length (not over what the video covered).
    seen_share: float
    #: Playing over decided time: the share the close-call rule looks at.
    play_of_seen: float
    #: A Class3 value, ``"mixed"`` on a tie, or ``""`` when nothing was decided.
    main: str
    status: SlotStatus
    reason: str

    @property
    def decided_s(self) -> float:
        return self.playing_s + self.empty_s + self.maintenance_s

    @property
    def unseen_s(self) -> float:
        return max(0.0, self.window.length_s - self.decided_s - self.uncertain_s)

    def share(self, seconds: float) -> float:
        return seconds / self.window.length_s if self.window.length_s > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        w = self.window
        return {
            "index": w.index, "label": w.label, "start": f"{w.start:%H:%M}",
            "end": f"{w.end:%H:%M}", "day_offset": w.day_offset,
            "start_s": w.start_s, "end_s": w.end_s, "coverage": round(w.coverage, 4),
            "playing_min": round(self.playing_s / 60, 1),
            "empty_min": round(self.empty_s / 60, 1),
            "maintenance_min": round(self.maintenance_s / 60, 1),
            "uncertain_min": round(self.uncertain_s / 60, 1),
            "unseen_min": round(self.unseen_s / 60, 1),
            "playing_share": round(self.share(self.playing_s), 4),
            "empty_share": round(self.share(self.empty_s), 4),
            "maintenance_share": round(self.share(self.maintenance_s), 4),
            "uncertain_share": round(self.share(self.uncertain_s), 4),
            "seen_share": round(self.seen_share, 4),
            "play_of_seen": round(self.play_of_seen, 4),
            "main": self.main, "main_label": ACTIVITY_LABELS[self.main],
            "status": self.status.value, "reason": self.reason,
            # Exact values, so a stored run can be rebuilt by `from_dict` for a comparison.
            "playing_s": self.playing_s, "empty_s": self.empty_s,
            "maintenance_s": self.maintenance_s, "uncertain_s": self.uncertain_s,
            "covered_start_s": w.covered_start_s, "covered_end_s": w.covered_end_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SlotResult:
        """The inverse of :meth:`to_dict`, for a run read back from disk."""
        window = SlotWindow(
            index=int(d["index"]), start=time.fromisoformat(d["start"]),
            end=time.fromisoformat(d["end"]), day_offset=int(d["day_offset"]),
            start_s=float(d["start_s"]), end_s=float(d["end_s"]),
            covered_start_s=float(d["covered_start_s"]),
            covered_end_s=float(d["covered_end_s"]))
        playing, empty = float(d["playing_s"]), float(d["empty_s"])
        maint = float(d["maintenance_s"])
        decided = playing + empty + maint
        # The shares are recomputed from the exact seconds; the stored ones are rounded.
        return cls(window=window, playing_s=playing, empty_s=empty, maintenance_s=maint,
                   uncertain_s=float(d["uncertain_s"]),
                   seen_share=decided / window.length_s if window.length_s > 0 else 0.0,
                   play_of_seen=playing / decided if decided > 0 else 0.0,
                   main=str(d["main"]), status=SlotStatus(d["status"]),
                   reason=str(d["reason"]))


def _state_key(state: object) -> str | None:
    """A sample's state as a Class3 value, or None when nothing was decided."""
    if state is None:
        return None
    value = getattr(state, "value", state)
    text = str(value)
    return text if text in (_PLAY, _EMPTY, _MAINT) else None


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _judge(playing: float, empty: float, maint: float, uncertain: float,
           window: SlotWindow, rules: Rules) -> SlotResult:
    decided = playing + empty + maint
    length = window.length_s
    seen = decided / length if length > 0 else 0.0
    play_of_seen = playing / decided if decided > 0 else 0.0
    totals = {_PLAY: playing, _EMPTY: empty, _MAINT: maint}
    if decided <= 0:
        main = ""
    else:
        top = max(totals.values())
        leaders = [k for k, v in totals.items() if abs(v - top) < 1e-6]
        main = leaders[0] if len(leaders) == 1 else "mixed"

    if seen < rules.min_seen_share:
        status = SlotStatus.REVIEW
        why = ("most of the slot is outside the video"
               if window.coverage < rules.min_seen_share
               else "the system could not tell what was happening for much of it")
        reason = f"Less than half of the slot was seen ({_pct(seen)}); {why}."
    elif rules.close_low <= play_of_seen <= rules.close_high:
        status = SlotStatus.REVIEW
        reason = (f"Close call: playing for {_pct(play_of_seen)} of the time seen - "
                  f"please check the footage.")
    elif main == _PLAY:
        status = SlotStatus.USED
        reason = f"Mostly playing ({_pct(play_of_seen)} of the time seen)."
    else:
        status = SlotStatus.NOT_USED
        what = ACTIVITY_LABELS[main] if main != "mixed" else "a mix of empty and maintenance"
        reason = (f"Mostly {what}; playing only {_pct(play_of_seen)} "
                  f"of the time seen.")
    return SlotResult(window=window, playing_s=playing, empty_s=empty, maintenance_s=maint,
                      uncertain_s=uncertain, seen_share=seen, play_of_seen=play_of_seen,
                      main=main, status=status, reason=reason)


def summarise(samples: Iterable[tuple[float, object]], windows: Sequence[SlotWindow],
              interval_s: float, *, rules: Rules = DEFAULT_RULES) -> list[SlotResult]:
    """Add each sample's ``interval_s`` seconds to the slot(s) it overlaps, then judge.

    ``samples`` are ``(t_s, state)`` on the video's clock. ``state`` may be a Class3, a
    MinuteState or its string value; None and ``"UNCERTAIN"`` count as not decided. A sample's
    time is clipped to the stretch the video covers, so the last sample of a file cannot claim
    seconds after the file ended.
    """
    if interval_s <= 0:
        raise ValueError(f"interval_s must be positive, not {interval_s}")
    acc = [{_PLAY: 0.0, _EMPTY: 0.0, _MAINT: 0.0, None: 0.0} for _ in windows]
    for t_s, state in samples:
        key = _state_key(state)
        lo, hi = float(t_s), float(t_s) + interval_s
        for i, w in enumerate(windows):
            overlap = min(hi, w.covered_end_s) - max(lo, w.covered_start_s)
            if overlap > 0:
                acc[i][key] += overlap
    return [_judge(a[_PLAY], a[_EMPTY], a[_MAINT], a[None], w, rules)
            for a, w in zip(acc, windows, strict=True)]


# --- comparison with bookings ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SlotCheck:
    """One slot set against the booking records."""

    index: int
    label: str
    status: SlotStatus
    #: booked_and_used, booked_not_used, used_not_booked, not_booked_not_used,
    #: off_sale_but_used or review.
    outcome: str
    ok: bool
    customer_refs: list[str] = field(default_factory=list)
    booking_statuses: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["status"] = self.status.value
        return out


def _booking_span(r: BookingRecord) -> tuple[datetime, datetime]:
    start = datetime.combine(r.date, r.start)
    end = datetime.combine(r.date, r.end)
    if end <= start:
        end += timedelta(days=1)
    return start, end


def compare(results: Sequence[SlotResult], records: Iterable[BookingRecord], *,
            field_id: str, day: date) -> list[SlotCheck]:
    """Set each slot's result against the bookings for ``field_id`` on the video's day.

    A booking counts for a slot when it overlaps at least half of the slot. Sold bookings
    (`BookingRecord.is_sold`) win over off-sale ones (maintenance/blocked) when both match.
    A REVIEW slot is always "review": the footage did not settle it, so neither does this.
    """
    wanted = field_id.strip()
    mine = [r for r in records if r.field_id.strip() == wanted]
    out: list[SlotCheck] = []
    for res in results:
        w = res.window
        s0 = w.starts_on(day)
        s1 = s0 + timedelta(seconds=w.length_s)
        matched = []
        for r in mine:
            b0, b1 = _booking_span(r)
            overlap = (min(s1, b1) - max(s0, b0)).total_seconds()
            if overlap >= w.length_s / 2:
                matched.append(r)
        sold = [r for r in matched if r.is_sold]
        off = [r for r in matched if r.status in OFF_SALE]
        refs = [r.customer_ref for r in (sold or off) if r.customer_ref]
        statuses = [r.status for r in (sold or off)]
        used = res.status is SlotStatus.USED
        who = f" ({', '.join(refs)})" if refs else ""
        if res.status is SlotStatus.REVIEW:
            outcome, ok = "review", True
            msg = "The footage did not settle this slot, so it is not compared. Check it."
            if sold:
                msg = f"Booked{who}, but the footage did not settle this slot. Check it."
        elif sold and used:
            outcome, ok, msg = "booked_and_used", True, f"Booked{who} and played in."
        elif sold:
            outcome, ok = "booked_not_used", False
            msg = f"Booked{who} but no play was seen - possible no-show."
        elif off and used:
            outcome, ok = "off_sale_but_used", False
            msg = f"Taken off sale ({', '.join(sorted(set(statuses)))}) but play was seen."
        elif used:
            outcome, ok, msg = "used_not_booked", False, "Play was seen but there is no booking."
        else:
            outcome, ok = "not_booked_not_used", True
            msg = ("Taken off sale and not played in." if off
                   else "No booking and no play.")
        out.append(SlotCheck(index=w.index, label=w.label, status=res.status, outcome=outcome,
                             ok=ok, customer_refs=refs, booking_statuses=statuses,
                             message=msg))
    return out


_CSV_COLUMNS = (
    "slot", "start", "end", "day_offset", "coverage_pct", "playing_min", "empty_min",
    "maintenance_min", "uncertain_min", "unseen_min", "playing_pct", "empty_pct",
    "maintenance_pct", "seen_pct", "main_activity", "result", "reason",
)


def to_csv(results: Sequence[SlotResult] | Sequence[dict],
           checks: Sequence[SlotCheck] | Sequence[dict] | None = None) -> str:
    """The slot table as CSV, with the booking columns when ``checks`` are given.

    Accepts the objects or their ``to_dict()`` form, so a stored run can be exported without
    being rebuilt.
    """
    rows = [r.to_dict() if isinstance(r, SlotResult) else r for r in results]
    by_index = {}
    for c in checks or ():
        d = c.to_dict() if isinstance(c, SlotCheck) else c
        by_index[d["index"]] = d
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    header = list(_CSV_COLUMNS)
    if checks is not None:
        header += ["booking_outcome", "customer_refs", "booking_message"]
    writer.writerow(header)
    for d in rows:
        line = [
            d["label"], d["start"], d["end"], d["day_offset"], round(100 * d["coverage"], 1),
            d["playing_min"], d["empty_min"], d["maintenance_min"], d["uncertain_min"],
            d["unseen_min"], round(100 * d["playing_share"], 1),
            round(100 * d["empty_share"], 1), round(100 * d["maintenance_share"], 1),
            round(100 * d["seen_share"], 1), d["main_label"], d["status"], d["reason"],
        ]
        if checks is not None:
            c = by_index.get(d["index"], {})
            line += [c.get("outcome", ""), " ".join(c.get("customer_refs", [])),
                     c.get("message", "")]
        writer.writerow(line)
    return buf.getvalue()
