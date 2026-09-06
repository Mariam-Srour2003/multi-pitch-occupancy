"""From per-frame predictions to a slot verdict.

Modules that belong here:
  fusion.py       combine the two camera halves of one pitch
  aggregate.py    threshold baseline: ratios -> USED/NOTUSED/REVIEW
  stan.py         learned slot-temporal aggregator               (WP5-T1)
  evidence.py     pick three representative evidence frames
  reconcile.py    verdicts vs bookings -> typed anomalies        (WP6-T5)

Fusion note: the baseline takes the strongest activity across the two halves, but
disagreement between halves is itself signal (occlusion, dirty lens, play confined to
one side). STAN consumes both sequences rather than the pre-fused one - see WP5-T6.
"""
