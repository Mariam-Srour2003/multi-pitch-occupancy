"""Metrics, statistics and calibration.

Modules that belong here:
  metrics.py      accuracy, per-class P/R/F1, macro-F1, confusion
  stats.py        bootstrap CIs, McNemar, paired bootstrap,
                  Holm-Bonferroni correction, effect sizes       (WP0-T6)
  calibration.py  reliability diagrams, ECE, temperature scaling (WP4-T5)
  latency.py      median/p95 timing, concurrent-camera throughput (WP0-T10)

Every reported comparison carries a confidence interval and a corrected p-value; a
significant result with a negligible effect size is reported as exactly that.
"""
