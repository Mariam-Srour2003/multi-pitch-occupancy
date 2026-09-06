# Notes from the review session — 2026-09-06

*Written by a second Claude session (the one that ran the pilot) after auditing the repo
while you were working. Everything here is either already done (labeled DONE — fold into
your commit flow) or a suggestion for your queue (labeled SUGGEST). Nothing was committed;
all changes sit in the working tree so you keep control of history.*

---

## DONE by this session (uncommitted, ready to fold in)

1. **Venue-grouping audit (your TODO 0.4 sub-item, the `[H]` one).**
   - `clipvenue_g_netting` → confirmed ONE facility: the pitch-4 camera sees the
     neighbouring pitch's "5" sign in-frame, and `cg_...055948` is that pitch 5. Raised
     `low → high` in `configs/clip_venues.csv`.
   - `clipvenue_h_teal_pitch` → night pair + day clip share the distinctive blue-tarp-with-
     teal-cap boards, fine teal mesh, and shanty-hillside backdrop. Kept grouped, raised
     `low → medium`.
   - Cross-check: cg ≠ ch (different fencing systems, skylines, floodlights) and neither
     matches any other group — so no two of the nine venues are secretly one facility.
   - Evidence sheets: `results/figs/venue_check/` (5 jpgs). Log entry appended. TODO 0.4
     sub-box ticked with a pointer.

2. **H3 sensitivity run (the insurance recommended by the audit).**
   New standalone script `experiments/h3_sensitivity_merged_venues.py` (mirrors your H3
   exactly; only relabels cg+ch as one merged venue before leave-one-group-out).
   **Result: H3 survives the pessimistic merge** — DINOv2 0.967 [0.932, 0.993] and
   ConvNeXtV2 0.927 [0.843, 0.989] both meet the 0.90 target with the pair fully held out;
   clock rule still collapses (0.243). Full findings + two write-up caveats appended to
   `EXPERIMENT_LOG.md`; CSV at `results/h3_sensitivity_merged_venues.csv`.
   → You can now quote the worst-fold caveat and the venue grouping without exposure.

## SUGGEST — small queue items (in rough priority order)

1. **TODO 0.2 (data backup) is still open and marked blocking.** 4.2 GB to two places.
   Nothing this session or yours produces is safe until that's done — it protects the one
   thing that cannot be regenerated.
2. **When quoting H3:** main run = primary numbers, sensitivity run = robustness check.
   The sensitivity means are higher partly from fold arithmetic (fewer, larger folds in an
   unweighted mean) — the log entry spells this out so it doesn't get quoted backwards.
3. **ViT in the sensitivity run "meets" the target only nominally** (mean 0.903, CI
   [0.764, 1.000] spans 0.90, worst fold 0.583). Consistent with your ranking-inversion
   finding; suggest never listing ViT as "meets target" without the CI.
4. **Duplicate log line:** the label-efficiency entry's one-line footer appears twice at
   the end of that section in `EXPERIMENT_LOG.md` — cosmetic, worth deleting one on your
   next edit pass.
5. **`leaderboard.csv` + `leaderboard_new.csv` on the pilot branch** still need merging
   when the user closes Excel (pilot-branch-only concern, low priority).
6. **Working-tree hygiene:** this session's uncommitted changes are
   `configs/clip_venues.csv`, `results/EXPERIMENT_LOG.md`, `TODO.md` (one checkbox),
   `docs/REVIEW_SESSION_NOTES.md`, `experiments/h3_sensitivity_merged_venues.py`,
   `results/h3_sensitivity_merged_venues.csv`, `results/figs/venue_check/*`. They're
   logically one unit ("venue audit + sensitivity check") if you want them as one commit,
   and they don't touch any file your RQ6/calibration work has open.

## Context you might want (from the pilot session, not in any doc)

- The pilot's four StatBox slot exports pair as `X.mp4` + `X (1).mp4` = two cameras of one
  slot, but the (1)-suffix ↔ physical-camera mapping flipped between the two days. If you
  ever build venue_01 fusion experiments from those videos, identify cameras by view
  content, not filename suffix.
- The user's machine sleeps overnight and kills long background runs (a stuck HF download
  froze a full benchmark once). Chunk anything > ~30 min and write results incrementally.
