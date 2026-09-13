# Dataset coverage

Generated 2026-09-13 | `uv run pitch coverage` | 1794 frames | 10 venues

## Class x lighting

| class | day | night | unknown | total |
|---|---|---|---|---|
| EMPTY | 485 | 9 | - | 494 |
| ACTIVE_PLAY | 222 | 970 | - | 1192 |
| MAINTENANCE_NON_SPORTING | 6 | - | 102 | 108 |
| **total** | 713 | 979 | 102 | 1794 |

## Class x venue

| class | clipvenue_a_blue_barrier | clipvenue_b_floodlit_track | clipvenue_c_teal_boards | clipvenue_d_indoor_dome | clipvenue_e_pink_boards | clipvenue_f_outdoor_bldg | clipvenue_g_netting | clipvenue_h_teal_pitch | clipvenue_i_outdoor_trees | venue_01 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EMPTY | - | - | - | - | - | - | - | - | - | 494 | 494 |
| ACTIVE_PLAY | 168 | 78 | 36 | 18 | 18 | 12 | 30 | 18 | 18 | 796 | 1192 |
| MAINTENANCE_NON_SPORTING | - | - | - | - | - | - | - | - | - | 108 | 108 |
| **total** | 168 | 78 | 36 | 18 | 18 | 12 | 30 | 18 | 18 | 1398 | 1794 |

## Provenance

| class | bulk | human | synthetic | total |
|---|---|---|---|---|
| EMPTY | 238 | 256 | - | 494 |
| ACTIVE_PLAY | 396 | 796 | - | 1192 |
| MAINTENANCE_NON_SPORTING | - | 6 | 102 | 108 |
| **total** | 634 | 1058 | 102 | 1794 |

## Gaps

Combinations with **no frames at all** - claims this dataset cannot support:

- `EMPTY` x `unknown`
- `ACTIVE_PLAY` x `unknown`
- `MAINTENANCE_NON_SPORTING` x `night`

13 of 17 populated cells are below target.

## Concentration

A class drawn overwhelmingly from one venue or one lighting condition cannot be separated from that condition by any split, so accuracy on it measures scene recognition. Share of each class held by its single largest source:

| class | top venue | share | top lighting | share |
|---|---|---|---|---|
| EMPTY | `venue_01` | 100% | `day` | 98% |
| ACTIVE_PLAY | `venue_01` | 67% | `night` | 81% |
| MAINTENANCE_NON_SPORTING | `venue_01` | 100% | `unknown` | 94% |
