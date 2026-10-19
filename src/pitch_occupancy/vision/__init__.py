"""Frame classification: one YOLO detector, a count, a motion cue and a decision rule.

Main modules:
  roi.py               per-camera pitch boundaries
  detector.py          the person/ball detector registry
  counting.py          who stands inside the boundary
  motion.py            the movement cue between frames
  rules.py             the decision table (`configs/rules.json`)
  pitch_classifier.py  the deployed detector-first classifier
  overlay.py           what the detector saw, drawn on the frame
  redact.py            pixelation and blur for frames that are shown or saved
"""
