"""Low-bandwidth, CPU-only occupancy auditing for multi-pitch sports facilities.

One frame per camera per minute is classified as empty, active play, or maintenance /
non-sporting; a slot's worth of predictions aggregates into a USED / NOTUSED / REVIEW
verdict bound to three evidence images, which is then reconciled against the facility's
booking records.

Layout::

    data/        manifest, splits, taxonomy, feature cache   (WP0, WP2)
    vision/      preprocessing, ROI, backbones, heads,
                 and the deployed classifier                 (WP3, WP4, WP6-T2)
    slots/       two-camera fusion, aggregation, STAN        (WP5)
    evaluation/  metrics, statistics, calibration            (WP0-T6, WP4)
    db/          SQLite schema and access                    (WP6)
    api/         FastAPI app, served by uvicorn              (WP6-T1, WP6-T6)
    worker.py    the sampling scheduler                      (WP6-T2)

CPU-only is a design constraint of the thesis, not an implementation detail: no code
path in this package may require a GPU.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
