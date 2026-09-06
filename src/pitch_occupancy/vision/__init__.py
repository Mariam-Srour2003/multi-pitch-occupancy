"""Frame preprocessing and classification.

One preprocessing code path is shared by experiments and the live pipeline - the pilot
proved what happens when they drift apart.

Modules that belong here:
  preprocess.py   single entry: roi, letterbox, clahe, quality   (WP3)
  roi.py          per-camera polygon masking                     (WP3-T1)
  backbones.py    frozen feature extractors; MEAN pooling only   (WP4)
  heads.py        logistic-regression heads + calibration        (WP4-T5)
  detector.py     tier-2 ambiguity resolver                      (WP4)

Feature extraction uses mean pooling, never HF ``pooler_output`` - on a plain ViT that
is randomly initialised, and it cost the pilot a false 38% score plus a silent
production failure. Every saved head carries a ``pooling`` stamp that is asserted on load.
"""
