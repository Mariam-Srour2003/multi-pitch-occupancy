"""From per-frame predictions to a slot verdict.

Modules:
  fusion.py       combine the two camera halves of one pitch
  aggregate.py    threshold baseline: ratios -> USED/NOTUSED/REVIEW
  smoothing.py    majority smoothing over a sequence of per-sample states
  conditions.py   the conditions a verdict was formed under
  evidence.py     pick three representative evidence frames
  reconcile.py    verdicts vs bookings -> typed anomalies

Fusion note: the baseline takes the strongest activity across the two halves, but
disagreement between halves is itself signal (occlusion, dirty lens, play confined to
one side).
"""
