"""The per-minute vocabulary of the detector-first path (A36, WP9).

Two things live here and a third is coming.

**`MinuteState`** is what one camera, or one pitch, is in for one minute: the four folder
classes of `data/taxonomy.Class4`, plus **UNCERTAIN** - an abstention. The four values are
the folder names, deliberately, so a stored `Sample.predicted` reads the same whether it came
from this path or from a label; a test pins that the two enums agree. UNCERTAIN is not a
fifth class. It is the state a minute is in when the system declines to say - no detector, no
boundary, movement without anyone found - and `slots/aggregate.py` treats it like a minute
that was never captured: it counts against the capture floor and can push a slot toward
REVIEW and nowhere else.

**`FrameVerdict`** is what a classification *records*, as opposed to what it returns.
`worker.Classifier` returns two values, a class and a confidence, and every surface that
wanted to know *why* - which boundary, how many people, whether a gate spoke - rebuilt the
answer from pieces (A35). A verdict carries the pieces, and a `trace`: the steps that led to
the state, in order, in words a reviewer can read.

**`decide`** - the decision table itself - lands with WP9-T3. Until then the probe path fills
a verdict from its class and its gates (`pipeline.Pipeline.classify_frame`), which is enough
for every surface to speak this vocabulary before the detector arrives to speak it better.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pitch_occupancy.data.taxonomy import Class3, Class4

__all__ = ["MinuteState", "FrameVerdict", "from_class3", "from_class4", "to_class3"]


class MinuteState(StrEnum):
    """One minute's state for one camera or one pitch. Values are the folder names."""

    EMPTY = "1_empty"
    PLAYING = "2_playing"
    PEOPLE_NOT_PLAYING = "3_people_not_playing"
    MAINTENANCE = "4_maintenance"
    #: The system declined to say. See the module docstring for what that is not.
    UNCERTAIN = "UNCERTAIN"

    @property
    def decided(self) -> bool:
        return self is not MinuteState.UNCERTAIN


#: The three-class collapse the whole thesis reports in. UNCERTAIN has no Class3: it is the
#: absence of one, and a caller asking for it gets None rather than a guess.
_TO_CLASS3 = {
    MinuteState.EMPTY: Class3.EMPTY,
    MinuteState.PLAYING: Class3.ACTIVE_PLAY,
    MinuteState.PEOPLE_NOT_PLAYING: Class3.MAINTENANCE_NON_SPORTING,
    MinuteState.MAINTENANCE: Class3.MAINTENANCE_NON_SPORTING,
}

#: The probe cannot tell `3_people_not_playing` from `4_maintenance` - it was trained on
#: three classes - so its C3 lands on the folder that has real frames. Six of them.
_FROM_CLASS3 = {
    Class3.EMPTY: MinuteState.EMPTY,
    Class3.ACTIVE_PLAY: MinuteState.PLAYING,
    Class3.MAINTENANCE_NON_SPORTING: MinuteState.PEOPLE_NOT_PLAYING,
}


def to_class3(state: MinuteState | str) -> Class3 | None:
    """The reporting class of a state, or None for UNCERTAIN."""
    return _TO_CLASS3.get(MinuteState(state))


def from_class3(state: Class3 | str) -> MinuteState:
    """The state a three-class verdict maps to. See `_FROM_CLASS3` for where C3 lands."""
    return _FROM_CLASS3[Class3(state)]


def from_class4(label: Class4 | str) -> MinuteState:
    """A folder label as a state; the values coincide, and this says so in one place."""
    return MinuteState(Class4(label).value)


@dataclass(frozen=True, slots=True)
class FrameVerdict:
    """What one classification found, and how it got there.

    ``trace`` is the rule path in order - "boundary slot_..._camA keeps 49% of the frame",
    "probe dinov2: ACTIVE_PLAY 0.93", "person gate -> EMPTY". It is the first thing an
    operator disputing a verdict asks for and was, until A36, reconstructed by hand from four
    fields on three pages.

    ``people``, ``ball`` and ``motion`` are None when the corresponding step did not run,
    which is a different fact from zero and is kept different: a detector that was not
    consulted has not established that nobody was there (`vision/explain.py`).
    """

    state: MinuteState
    confidence: float
    trace: tuple[str, ...] = ()
    polygon: list[list[float]] | None = None
    #: The store key the boundary was found under, ``"given"`` when the caller supplied one,
    #: None when there was none.
    boundary_key: str | None = None
    people: int | None = None
    ball: bool | None = None
    ball_confidence: float = 0.0
    motion: float | None = None
    #: What the model said before any gate spoke, when a model spoke at all.
    probed: Class3 | None = None
    gated: bool = False
    model_key: str = ""

    @property
    def class3(self) -> Class3 | None:
        return to_class3(self.state)

    @property
    def decided(self) -> bool:
        return self.state.decided

    def as_pair(self) -> tuple[Class3, float]:
        """The `worker.Classifier` shape, for callers that still want it.

        Raises on UNCERTAIN rather than inventing a class: the two-value contract has no way
        to say "declined", and a caller that needs one has to look at the verdict.
        """
        cls = self.class3
        if cls is None:
            raise ValueError("an UNCERTAIN verdict has no (class, confidence) form")
        return cls, self.confidence
