"""Class taxonomy and the mapping between labelling and reporting.

Frames are labelled into **three folders** and every metric reports **three classes**, and
since 2026-09-21 those are the same three:

    1_empty                      -> C1 EMPTY                     -> NOTUSED
    2_playing                    -> C2 ACTIVE_PLAY               -> USED
    3_maintenance_non_sporting   -> C3 MAINTENANCE_NON_SPORTING  -> NOTUSED / REVIEW

**It was four folders until then**, splitting `3_people_not_playing` from `4_maintenance` on
the reasoning that the finer distinction cost nothing at labelling time and kept a four-class
ablation available later. It cost something in the end. The corpus reached 1,692 recorded
frames holding **6 real `3_people_not_playing` frames and 0 real `4_maintenance` frames** -
the ablation was never runnable, the prediction path dropped the split in A40 because nothing
could measure it, and two folders that no measurement could tell apart were still being
maintained, documented and reported on. `LEGACY_FOLDERS` keeps the old names readable so
every CSV written before the collapse still parses; nothing writes them any more.

The old warning stands in its new form: **never report a macro-F1 gain driven by C3 while its
support is near zero.** The operator's 2026-09-21 footage is the first real C3 data this
project has had; before it the class was almost entirely generated.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "Label",
    "Class3",
    "SlotStatus",
    "to_class3",
    "LABEL_ORDER",
    "CLASS3_ORDER",
    "DISPLAY_NAMES",
    "LEGACY_FOLDERS",
]


class Label(StrEnum):
    """The three folders frames are labelled into on disk."""

    EMPTY = "1_empty"
    PLAYING = "2_playing"
    MAINTENANCE_NON_SPORTING = "3_maintenance_non_sporting"

    @classmethod
    def parse(cls, value: Label | str) -> Label:
        """A folder name as a label, accepting the two pre-2026-09-21 names.

        Old manifests, old `labels.csv` backups and every results CSV written before the
        collapse carry `3_people_not_playing` or `4_maintenance`. They are read, not
        rewritten - editing an artefact to match today's vocabulary is how a record stops
        being a record.
        """
        if isinstance(value, cls):
            return value
        text = str(value)
        if text in LEGACY_FOLDERS:
            return LEGACY_FOLDERS[text]
        try:
            return cls(text)
        except ValueError as exc:
            known = ", ".join(c.value for c in LABEL_ORDER)
            raise ValueError(f"unknown label {value!r}; expected one of: {known}") from exc


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


#: The folder names this project wrote before 2026-09-21, and where they land now. Both
#: mapped to C3 even then (`to_class3` has always collapsed them), so nothing is lost by
#: reading them here - only the name of the folder they were filed in.
LEGACY_FOLDERS: dict[str, Label] = {
    "3_people_not_playing": Label.MAINTENANCE_NON_SPORTING,
    "4_maintenance": Label.MAINTENANCE_NON_SPORTING,
}

LABEL_ORDER: tuple[Label, ...] = (
    Label.EMPTY,
    Label.PLAYING,
    Label.MAINTENANCE_NON_SPORTING,
)

CLASS3_ORDER: tuple[Class3, ...] = (
    Class3.EMPTY,
    Class3.ACTIVE_PLAY,
    Class3.MAINTENANCE_NON_SPORTING,
)

_LABEL_TO_CLASS3: dict[Label, Class3] = {
    Label.EMPTY: Class3.EMPTY,
    Label.PLAYING: Class3.ACTIVE_PLAY,
    Label.MAINTENANCE_NON_SPORTING: Class3.MAINTENANCE_NON_SPORTING,
}

DISPLAY_NAMES: dict[Class3, str] = {
    Class3.EMPTY: "Empty",
    Class3.ACTIVE_PLAY: "Active play",
    Class3.MAINTENANCE_NON_SPORTING: "Maintenance / non-sporting",
}


def to_class3(label: Label | str) -> Class3:
    """Map a labelling folder (enum or folder name, current or legacy) onto its class.

    One-to-one since the collapse, and kept as a function because the call sites read the
    same as they did when it was a four-to-three mapping - and because the legacy names
    still have to come through it.

    Raises:
        ValueError: if the label is not a known folder name.
    """
    return _LABEL_TO_CLASS3[Label.parse(label)]
