"""Class taxonomy and the mapping between labelling and reporting.

Frames are *labelled* into four folders because the finer distinction costs nothing at
labelling time and keeps a 4-class ablation available later. Every thesis metric is
*reported* over three classes, because that is what the slot decision actually needs.

    1_empty                -> C1 EMPTY                     -> NOTUSED
    2_playing              -> C2 ACTIVE_PLAY               -> USED
    3_people_not_playing   -> C3 MAINTENANCE_NON_SPORTING  -> NOTUSED / REVIEW
    4_maintenance          -> C3 MAINTENANCE_NON_SPORTING  -> NOTUSED / REVIEW

Never report a macro-F1 improvement on the 4-class view while C3 support is near zero;
fix the data first. See TODO.md WP2-T11 for the C3 contingency plan.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "Class4",
    "Class3",
    "SlotStatus",
    "to_class3",
    "CLASS4_ORDER",
    "CLASS3_ORDER",
    "DISPLAY_NAMES",
]


class Class4(StrEnum):
    """The four folders frames are labelled into on disk."""

    EMPTY = "1_empty"
    PLAYING = "2_playing"
    PEOPLE_NOT_PLAYING = "3_people_not_playing"
    MAINTENANCE = "4_maintenance"


class Class3(StrEnum):
    """The three operational classes every thesis metric reports."""

    EMPTY = "C1_EMPTY"
    ACTIVE_PLAY = "C2_ACTIVE_PLAY"
    MAINTENANCE_NON_SPORTING = "C3_MAINTENANCE_NON_SPORTING"


class SlotStatus(StrEnum):
    """The verdict a whole rental slot resolves to."""

    USED = "USED"
    NOTUSED = "NOTUSED"
    REVIEW = "REVIEW"


CLASS4_ORDER: tuple[Class4, ...] = (
    Class4.EMPTY,
    Class4.PLAYING,
    Class4.PEOPLE_NOT_PLAYING,
    Class4.MAINTENANCE,
)

CLASS3_ORDER: tuple[Class3, ...] = (
    Class3.EMPTY,
    Class3.ACTIVE_PLAY,
    Class3.MAINTENANCE_NON_SPORTING,
)

_CLASS4_TO_CLASS3: dict[Class4, Class3] = {
    Class4.EMPTY: Class3.EMPTY,
    Class4.PLAYING: Class3.ACTIVE_PLAY,
    Class4.PEOPLE_NOT_PLAYING: Class3.MAINTENANCE_NON_SPORTING,
    Class4.MAINTENANCE: Class3.MAINTENANCE_NON_SPORTING,
}

DISPLAY_NAMES: dict[Class3, str] = {
    Class3.EMPTY: "Empty",
    Class3.ACTIVE_PLAY: "Active play",
    Class3.MAINTENANCE_NON_SPORTING: "Maintenance / non-sporting",
}


def to_class3(label: Class4 | str) -> Class3:
    """Map a 4-class label (enum or folder name) onto its reporting class.

    Raises:
        ValueError: if the label is not one of the four known folder names.
    """
    try:
        return _CLASS4_TO_CLASS3[Class4(label)]
    except ValueError as exc:
        known = ", ".join(c.value for c in CLASS4_ORDER)
        raise ValueError(f"unknown label {label!r}; expected one of: {known}") from exc
