"""Frame preprocessing and classification.

One preprocessing code path is shared by offline runs and the live pipeline - an early
prototype showed what happens when they drift apart.

Main modules:
  roi.py          per-camera polygon masking
  backbones.py    frozen feature extractors; MEAN pooling only
  heads.py        logistic-regression heads + calibration
  detector.py     the person/ball detector registry

Feature extraction uses mean pooling, never HF ``pooler_output`` - on a plain ViT that
is randomly initialised, and it cost an early prototype a false 38% score plus a silent
production failure. Every saved head carries a ``pooling`` stamp that is asserted on load.
"""
