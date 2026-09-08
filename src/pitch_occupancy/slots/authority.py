"""What this system is permitted to do with a discrepancy (WP6-T12, RQ4).

Reconciliation compares what the cameras saw against what the records claim, and a
disagreement can implicate a named member of staff. The design constraint is one sentence -
**the system never takes an automated financial action** - and a sentence in a chapter is
worth very little at an examination, so it lives here as code with a test behind it.

The constraint has three parts, and they are separable:

1. **Every output is an advisory.** :class:`Advisory` is the only thing reconciliation can
   produce for a discrepancy, it carries no monetary quantity, and
   ``requires_human_confirmation`` is not a parameter - there is no way to construct one
   that does not need a person.
2. **The permitted actions are enumerated, and money is not among them.** :data:`PERMITTED`
   is the whole vocabulary; :data:`FORBIDDEN` names what was deliberately left out, so the
   omission is legible rather than an oversight waiting to be filled in.
3. **The boundary is checked, not asserted.** ``tests/test_authority.py`` walks the HTTP
   surface and the database layer and fails if either grows a way to charge, invoice,
   penalise or alter a booking.

**Why enumerate the forbidden actions at all.** A list of permitted ones would be enough for
the code; the second list is for the reader. "This system does not bill" is a claim an
examiner should be able to check in thirty seconds, and a named absence is checkable where a
silent one is not.

**This is not a safety mechanism against a malicious operator** - anyone with the database
can do what they like with it. It is a statement about what the software offers, kept true
by a test, so that "decision support" describes the artefact rather than the intention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "Action", "PERMITTED", "FORBIDDEN", "Advisory", "advise", "MAX_SEVERITY_WITHOUT_HUMAN",
]


class Action(StrEnum):
    """Everything the system may emit in response to a discrepancy."""

    #: Put the slot in front of a person, with its evidence.
    FLAG_FOR_REVIEW = "flag_for_review"
    #: Write the observation to the audit log. A record, not a judgement.
    RECORD_OBSERVATION = "record_observation"
    #: Count it in a per-field summary. Never per person - see `reconcile.py` rule 3.
    AGGREGATE_BY_FIELD = "aggregate_by_field"


#: The whole vocabulary. Adding to it is a design decision, not a refactor.
PERMITTED: frozenset[Action] = frozenset(Action)

#: Deliberately absent, named so the absence is legible. These are not enum members: there
#: is no code path that could emit one, and that is the point of writing them down here
#: rather than as a comment.
FORBIDDEN: frozenset[str] = frozenset({
    "charge_customer",
    "issue_invoice",
    "adjust_invoice",
    "withhold_payment",
    "apply_penalty",
    "flag_staff_member",
    "amend_booking",
    "cancel_booking",
})

#: There is no severity at which a person stops being required. The constant exists so the
#: claim is greppable and testable rather than implied by the absence of an else-branch.
MAX_SEVERITY_WITHOUT_HUMAN: None = None


@dataclass(frozen=True, slots=True)
class Advisory:
    """The only thing a discrepancy can produce.

    No monetary field, and ``requires_human_confirmation`` is a property rather than an
    argument: an advisory that did not need a person would be a different kind of object,
    and there is no way to construct one here.
    """

    field_id: str
    date: str
    start: str
    finding: str
    actions: tuple[Action, ...] = field(default_factory=tuple)

    @property
    def requires_human_confirmation(self) -> bool:
        """Always true. Not a policy setting; a property of what this system produces."""
        return True

    def __post_init__(self) -> None:
        unknown = [a for a in self.actions if a not in PERMITTED]
        if unknown:
            raise ValueError(
                f"not a permitted action: {unknown}. The permitted vocabulary is "
                f"{sorted(a.value for a in PERMITTED)}, and money is deliberately not in it "
                f"- see FORBIDDEN and thesis/ethics.md."
            )


def advise(reconciliation, *, log: bool = True) -> Advisory:
    """Turn a reconciliation into the advisory it is allowed to become.

    A consistent slot still produces an advisory - one with nothing to confirm - rather than
    ``None``, because a caller that has to handle a null is a caller that can forget to.
    """
    actions: list[Action] = []
    if log:
        actions.append(Action.RECORD_OBSERVATION)
    if reconciliation.is_anomaly:
        actions.append(Action.FLAG_FOR_REVIEW)
        actions.append(Action.AGGREGATE_BY_FIELD)
    return Advisory(
        field_id=reconciliation.field_id,
        date=reconciliation.date,
        start=reconciliation.start,
        finding=reconciliation.explanation,
        actions=tuple(actions),
    )
