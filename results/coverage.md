# Dataset coverage

Generated 2026-09-13 | `uv run pitch coverage` | 1692 recorded frames | 10 venues | +107 generated (excluded)

**107 generated frame(s) are excluded from every table above the Generated section** (A13). They are a training-side augmentation and counting them here would make a gap look filled that is not.

## Class x lighting

| class | day | night | total |
|---|---|---|---|
| EMPTY | 485 | 9 | 494 |
| ACTIVE_PLAY | 222 | 970 | 1192 |
| MAINTENANCE_NON_SPORTING | 6 | - | 6 |
| **total** | 713 | 979 | 1692 |

## Class x venue

| class | clipvenue_a_blue_barrier | clipvenue_b_floodlit_track | clipvenue_c_teal_boards | clipvenue_d_indoor_dome | clipvenue_e_pink_boards | clipvenue_f_outdoor_bldg | clipvenue_g_netting | clipvenue_h_teal_pitch | clipvenue_i_outdoor_trees | venue_01 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EMPTY | - | - | - | - | - | - | - | - | - | 494 | 494 |
| ACTIVE_PLAY | 168 | 78 | 36 | 18 | 18 | 12 | 30 | 18 | 18 | 796 | 1192 |
| MAINTENANCE_NON_SPORTING | - | - | - | - | - | - | - | - | - | 6 | 6 |
| **total** | 168 | 78 | 36 | 18 | 18 | 12 | 30 | 18 | 18 | 1296 | 1692 |

## Provenance

| class | bulk | human | total |
|---|---|---|---|
| EMPTY | 238 | 256 | 494 |
| ACTIVE_PLAY | 396 | 796 | 1192 |
| MAINTENANCE_NON_SPORTING | - | 6 | 6 |
| **total** | 634 | 1058 | 1692 |

## Gaps

Combinations with **no frames at all** - claims this dataset cannot support:

- `MAINTENANCE_NON_SPORTING` x `night`

Classes below the 100-frame working target **in total**: `MAINTENANCE_NON_SPORTING` (6)

13 of 19 populated cells are below target.

## Concentration

A class drawn overwhelmingly from one venue or one lighting condition cannot be separated from that condition by any split, so accuracy on it measures scene recognition. Share of each class held by its single largest source:

| class | top venue | share | top lighting | share |
|---|---|---|---|---|
| EMPTY | `venue_01` | 99% | `day` | 97% |
| ACTIVE_PLAY | `venue_01` | 67% | `night` | 81% |
| MAINTENANCE_NON_SPORTING | `venue_01` | 99% | `unknown` | 94% |

## Generated frames (A13) - not counted above

Training-side augmentation, listed apart from the recorded corpus because a gap they appear to fill is still a gap: the test set stays real, so a venue whose only empty pitch is a generated one still has no empty pitch.

| class | clipvenue_a_blue_barrier | venue_01 | total |
|---|---|---|---|
| EMPTY | 4 | - | 4 |
| ACTIVE_PLAY | - | - | 0 |
| MAINTENANCE_NON_SPORTING | 1 | 102 | 103 |
| **total** | 5 | 102 | 107 |

| recorded defect | frames |
|---|---|
| `synthetic:cinematic` | 6 |
| `synthetic:cinematic;sign_prop` | 1 |
| `synthetic:ok` | 82 |
| `synthetic:timestamp_overlay` | 2 |
| `synthetic:timestamp_overlay;sign_prop` | 1 |
| `synthetic:timestamp_overlay;wrong_venue` | 6 |
| `synthetic:timestamp_overlay;wrong_venue;sign_prop` | 3 |
| `synthetic:wrong_venue` | 6 |
