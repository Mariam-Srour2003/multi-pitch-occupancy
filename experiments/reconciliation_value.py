"""WP6-T10 - what is reconciliation worth, and how good does it have to be? (RQ4, RQ6)

The task asks for **expected € recovered per 1,000 slots at the chosen operating point, with
the assumptions stated**. Two of the three things that figure needs are not measured, and one
of them cannot be measured on this data at all, so the honest form of the answer is not a
number - it is a **break-even**.

What is known, what is assumed, what is unknown
-----------------------------------------------

* **Known.** What the system flags. `reconcile.py` maps (booking, staff record, vision
  verdict) onto typed anomalies deterministically, so the *flag rate* per 1,000 slots follows
  from the rule table and an assumed mix of booking cases. No model uncertainty enters it.
* **Assumed.** Prices and costs. A slot price, the staff time to investigate a flag, and the
  cost of a wrong accusation. Every one is a parameter here with a stated default and a
  sensitivity sweep - none is presented as a measurement.
* **Unknown, and not knowable here.** The **precision** of a flag: of the slots the system
  calls anomalous, how many really are. WP6-T11 says why - anomaly precision needs
  adjudicated slots, and this corpus has two.

So a single euro figure would be a number invented out of a parameter nobody has measured.
What can be computed without it is the question a facility actually has to answer:

> **How often does a flag have to be right before running this is worth it?**

That is a break-even precision, it follows from the cost ratios alone, and it is decision-
relevant: if the answer is 5% the system is worth deploying almost regardless of how good it
is, and if it is 80% it is not worth deploying until precision is measured.

**This is an evaluation, not an action.** It attaches money to the confusion matrix to size
the feature; the system itself takes no financial action of any kind and cannot be made to -
see `slots/authority.py` and WP6-T12. The two are easy to conflate and are opposites: one
asks *what is this worth*, the other would be *charge someone*.

    uv run python -m experiments.reconciliation_value
    uv run python -m experiments.reconciliation_value --slot-price 45 --investigate-min 20
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.taxonomy import SlotStatus
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.slots.reconcile import Booking, reconcile

OUT = settings.results_dir / "reconciliation_value.csv"


@dataclass(frozen=True, slots=True)
class Assumptions:
    """Every number a facility would have to supply. None of them is measured here.

    Defaults are plausible for a UK/EU five-a-side operator and are **illustrative**: the
    output is reported as a function of them, and the sensitivity table exists so a reader
    substitutes their own rather than inheriting these.
    """

    slot_price_eur: float = 40.0
    #: Staff minutes to look at three evidence frames and reach a decision.
    investigate_minutes: float = 10.0
    staff_cost_per_hour_eur: float = 18.0
    #: What a wrong accusation costs: the investigation, plus dispute handling, plus the
    #: part nobody can price. Deliberately set well above the investigation cost, because
    #: the asymmetry is the whole ethical argument for a human in the loop.
    false_accusation_eur: float = 120.0
    #: Share of a facility's slots in each booking situation. Sums to 1.
    mix_booked_and_used: float = 0.80
    mix_booked_no_show: float = 0.12
    mix_unbooked_usage: float = 0.05
    mix_booked_unrecorded: float = 0.03

    @property
    def investigation_cost_eur(self) -> float:
        return self.investigate_minutes / 60.0 * self.staff_cost_per_hour_eur

    def check(self) -> None:
        total = (self.mix_booked_and_used + self.mix_booked_no_show
                 + self.mix_unbooked_usage + self.mix_booked_unrecorded)
        if abs(total - 1.0) > 1e-9:
            raise SystemExit(f"the booking-case mix sums to {total:.3f}, not 1.0")


#: The four situations, as (booked, staff_recorded_used, true vision verdict, what it is
#: worth to catch). Recovery is what the *facility* recovers by learning the truth, and it
#: is deliberately not the same for every anomaly type: an unbooked slot is revenue that
#: was never invoiced, while a no-show is a correction to a record rather than new money.
CASES: dict[str, tuple[bool, bool | None, SlotStatus, float]] = {
    "booked_and_used": (True, True, SlotStatus.USED, 0.0),
    "booked_no_show": (True, True, SlotStatus.NOTUSED, 0.0),
    "unbooked_usage": (False, None, SlotStatus.USED, 1.0),
    "booked_unrecorded": (True, False, SlotStatus.USED, 0.0),
}


def flag_rate(assumptions: Assumptions) -> dict[str, tuple[float, float]]:
    """``{anomaly: (flags per 1,000 slots, euros recoverable per flag)}``.

    From the rule table rather than from a model: `reconcile.py` is deterministic given
    (booking, record, verdict), so this part carries no estimation error at all - only the
    assumed mix does, and that is a parameter.

    The recovery figure is per *type*, not per flag, and most types recover nothing. An
    unbooked slot is revenue that was never invoiced; a no-show is a correction to a record.
    Flattening them to one number was the first version's mistake.
    """
    mix = {
        "booked_and_used": assumptions.mix_booked_and_used,
        "booked_no_show": assumptions.mix_booked_no_show,
        "unbooked_usage": assumptions.mix_unbooked_usage,
        "booked_unrecorded": assumptions.mix_booked_unrecorded,
    }
    out: dict[str, tuple[float, float]] = {}
    for name, share in mix.items():
        booked, recorded, verdict, recovery_share = CASES[name]
        result = reconcile(
            Booking(field_id="f", date="d", start="s", booked=booked,
                    staff_recorded_used=recorded),
            verdict,
        )
        if not result.is_anomaly:
            continue
        key = result.anomaly.value
        count, _ = out.get(key, (0.0, 0.0))
        out[key] = (count + share * 1000, recovery_share * assumptions.slot_price_eur)
    return out


def value_per_1000(assumptions: Assumptions, precision: float) -> dict[str, float]:
    """Expected euros per 1,000 slots at a given flag precision.

    A flag costs an investigation whether or not it is right. A *correct* flag recovers
    whatever that anomaly type is worth; a *wrong* one costs the false-accusation figure on
    top of the investigation. Missed anomalies are not counted as a cost here - they are the
    status quo, and counting them would flatter the system by charging it for a problem it
    did not create.
    """
    by_type = flag_rate(assumptions)
    flags = sum(count for count, _ in by_type.values())

    # Linear in precision, and per type. The first version multiplied by precision twice -
    # once to count the true flags and again against the total recoverable - which halved
    # the recovery at p=0.5 and moved the break-even by several points.
    recovered = sum(count * precision * per_flag for count, per_flag in by_type.values())
    investigation = flags * assumptions.investigation_cost_eur
    accusation = flags * (1.0 - precision) * assumptions.false_accusation_eur
    return {
        "flags_per_1000": flags,
        "recovered_eur": recovered,
        "investigation_eur": investigation,
        "false_accusation_eur": accusation,
        "net_eur": recovered - investigation - accusation,
    }


def break_even_precision(assumptions: Assumptions, *, grid: int = 2001) -> float | None:
    """The precision at which the net becomes positive, or None if it never does.

    The quantity the facility actually needs, and the one that does not require the
    measurement this corpus cannot supply.
    """
    for p in np.linspace(0.0, 1.0, grid):
        if value_per_1000(assumptions, float(p))["net_eur"] > 0:
            return float(p)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slot-price", type=float, default=Assumptions().slot_price_eur)
    ap.add_argument("--investigate-min", type=float, default=Assumptions().investigate_minutes)
    ap.add_argument("--false-accusation", type=float,
                    default=Assumptions().false_accusation_eur)
    args = ap.parse_args()

    base = Assumptions(
        slot_price_eur=args.slot_price,
        investigate_minutes=args.investigate_min,
        false_accusation_eur=args.false_accusation,
    )
    base.check()

    print("=== assumptions (none of these is measured) ===")
    for k, v in sorted(vars(base).items() if hasattr(base, "__dict__") else
                       ((f, getattr(base, f)) for f in base.__slots__)):
        print(f"  {k:28} {v}")
    print(f"  {'investigation_cost_eur':28} {base.investigation_cost_eur:.2f}  (derived)")

    flags = flag_rate(base)
    print(f"\n=== what the system flags, per 1,000 slots ===")
    print("  From the rule table, not from a model: `reconcile` is deterministic given")
    print("  (booking, staff record, verdict), so only the assumed case mix enters here.")
    for name, (n, per_flag) in sorted(flags.items()):
        worth = f"{per_flag:6.2f} EUR each" if per_flag else "recovers nothing"
        print(f"    {name:26} {n:7.1f}   {worth}")
    print(f"    {'total':26} {sum(n for n, _ in flags.values()):7.1f}")

    print("\n=== euros per 1,000 slots, against flag precision ===")
    print("  Precision is UNKNOWN and unmeasurable here - WP6-T11: it needs adjudicated")
    print("  slots and this corpus has two. So it is swept, not assumed.\n")
    print(f"  {'precision':>10}{'recovered':>12}{'investigate':>13}{'wrong flags':>13}{'net':>11}")
    rows: list[dict] = []
    for p in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        v = value_per_1000(base, p)
        print(f"  {p:>10.0%}{v['recovered_eur']:>12.0f}{-v['investigation_eur']:>13.0f}"
              f"{-v['false_accusation_eur']:>13.0f}{v['net_eur']:>11.0f}")
        rows.append({"kind": "sweep", "precision": round(p, 4),
                     **{k: round(x, 2) for k, x in v.items()}})

    be = break_even_precision(base)
    print("\n=== the answer that does not need the missing measurement ===")
    if be is None:
        print("  Net is negative at every precision: on these assumptions the feature")
        print("  cannot pay for itself, and the assumptions are where to look first.")
    else:
        print(f"  **Break-even precision: {be:.1%}.** Below this, flagging costs more than")
        print("  it recovers; above it, the feature pays for itself.")
        print("  This follows from the cost ratios alone and needs no model measurement.")
    rows.append({"kind": "break_even", "precision": round(be, 4) if be else "",
                 "flags_per_1000": round(sum(n for n, _ in flags.values()), 2),
                 "recovered_eur": "",
                 "investigation_eur": "", "false_accusation_eur": "", "net_eur": ""})

    # --- sensitivity: which assumption moves the answer -----------------------------
    print("\n=== which assumption the break-even is most sensitive to ===")
    print("  Halving and doubling each, one at a time. A break-even that barely moves is")
    print("  one a facility does not have to argue about.\n")
    print(f"  {'assumption':26}{'half':>12}{'default':>12}{'double':>12}")
    for name in ("slot_price_eur", "investigate_minutes", "false_accusation_eur"):
        current = getattr(base, name)
        line = []
        for scale in (0.5, 1.0, 2.0):
            got = break_even_precision(replace(base, **{name: current * scale}))
            line.append("never" if got is None else f"{got:.1%}")
            rows.append({"kind": "sensitivity", "precision": got if got else "",
                         "flags_per_1000": "", "recovered_eur": "",
                         "investigation_eur": f"{name}x{scale}",
                         "false_accusation_eur": "", "net_eur": ""})
        print(f"  {name:26}{line[0]:>12}{line[1]:>12}{line[2]:>12}")

    print("\n  This is an evaluation, not an action. The system takes no financial action of")
    print("  any kind and cannot be made to - see slots/authority.py and WP6-T12.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT.name}")

    record(
        "WP6-T10 reconciliation value",
        "`python -m experiments.reconciliation_value`",
        f"`{OUT.name}`",
        f"break-even flag precision {be:.1%} on stated assumptions; precision itself is "
        f"unmeasurable here (WP6-T11)" if be else "net negative at every precision",
    )


if __name__ == "__main__":
    main()
