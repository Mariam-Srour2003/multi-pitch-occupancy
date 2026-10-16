"""A derived boundary reaches the frame's edge when the turf does (audit VI-4).

`roi_derive.polygon_from_mask` divided pixel *indices* by the width and height, so the last
row of a 180-row mask came out at 0.9944 rather than 1.0. Every derived boundary therefore
stopped a pixel short of the bottom of the frame, and a player whose feet the frame cuts off -
the one `counting.inside` clamps onto the last row - fell outside it.
"""

from __future__ import annotations

import numpy as np

from pitch_occupancy.vision import counting, roi_derive
from pitch_occupancy.vision.detector import PERSON, Detection


def test_an_all_turf_mask_derives_the_whole_frame() -> None:
    mask = np.full((180, 320), 255, np.uint8)
    polygon = roi_derive.polygon_from_mask(mask)
    assert polygon is not None
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    assert min(xs) == 0.0 and max(xs) == 1.0
    assert min(ys) == 0.0 and max(ys) == 1.0


def test_a_bottom_truncated_person_is_inside_a_derived_boundary() -> None:
    mask = np.full((180, 320), 255, np.uint8)
    polygon = roi_derive.polygon_from_mask(mask)
    truncated = Detection(PERSON, (150, 120, 170, 180), 0.9)  # feet cut off by the frame
    count = counting.count_inside([truncated], polygon, (180, 320))
    assert count.people_inside == 1


def test_a_region_short_of_the_edge_still_stops_short_of_it() -> None:
    """The fix maps the last index to 1.0; it must not stretch a region that ends earlier."""
    mask = np.zeros((180, 320), np.uint8)
    mask[:90] = 255
    polygon = roi_derive.polygon_from_mask(mask)
    assert max(y for _, y in polygon) < 0.6
