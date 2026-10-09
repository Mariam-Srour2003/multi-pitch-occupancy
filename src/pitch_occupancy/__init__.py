"""Low-bandwidth, CPU-only occupancy auditing for multi-pitch sports facilities.

One frame per camera per minute is classified as empty, active play, or maintenance /
non-sporting; a slot's worth of predictions aggregates into a USED / NOTUSED / REVIEW
verdict bound to three evidence images, which is then reconciled against the facility's
booking records.

Layout::

    data/        manifest, splits, taxonomy, feature cache
    vision/      ROI, backbones, heads, detector, gates,
                 and the deployed classifier
    slots/       two-camera fusion, aggregation, reconciliation
    db/          SQLite schema and access
    api/         FastAPI app, served by uvicorn
    worker.py    the sampling scheduler

CPU-only is a design constraint of the project, not an implementation detail: no code
path in this package may require a GPU.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
