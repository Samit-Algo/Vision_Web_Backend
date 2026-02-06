"""
Wall Climb Detection - Zone helpers
-----------------------------------

We treat the user-drawn polygon as the wall. Its TOP edge is the "wall line".
- Climbing: part of the person is above the wall (box TOP is above wall line).
- Fully above: whole person is above the wall (box BOTTOM is above wall line).
We also require the box to overlap the wall horizontally (so we don't alert for someone far to the side).
"""

from typing import List, Tuple, Optional


def _wall_top_and_x_range(
    zone_coordinates: List[List[float]],
    frame_width: float,
    frame_height: float,
) -> Tuple[float, float, float]:
    """
    Get wall top (smallest y = top of image) and horizontal range in PIXEL coords.
    Zone is normalized 0-1; we convert to pixel using frame size.
    Returns: (wall_top_y_px, wall_x_min_px, wall_x_max_px)
    """
    if not zone_coordinates or frame_width <= 0 or frame_height <= 0:
        return 0.0, 0.0, float(frame_width)

    max_val = max(max(p[0], p[1]) for p in zone_coordinates)
    is_normalized = max_val <= 1.0

    if is_normalized:
        # Convert normalized 0-1 to pixel
        ys = [p[1] * frame_height for p in zone_coordinates]
        xs = [p[0] * frame_width for p in zone_coordinates]
    else:
        ys = [p[1] for p in zone_coordinates]
        xs = [p[0] for p in zone_coordinates]

    wall_top_y = min(ys)
    wall_x_min = min(xs)
    wall_x_max = max(xs)
    return wall_top_y, wall_x_min, wall_x_max


def _box_to_pixel(
    box: List[float],
    frame_width: float,
    frame_height: float,
) -> Tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2) in pixel coords. Box may be pixel or normalized."""
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    max_val = max(x1, y1, x2, y2)
    if max_val <= 1.0 and frame_width > 1 and frame_height > 1:
        x1, x2 = x1 * frame_width, x2 * frame_width
        y1, y2 = y1 * frame_height, y2 * frame_height
    return x1, y1, x2, y2


def _box_overlaps_wall_x(
    x1: float, x2: float,
    wall_x_min: float, wall_x_max: float,
) -> bool:
    """True if box horizontal range [x1,x2] overlaps wall range [wall_x_min, wall_x_max]."""
    return not (x2 < wall_x_min or x1 > wall_x_max)


def is_box_climbing(
    box: List[float],
    zone_coordinates: List[List[float]],
    frame_width: float,
    frame_height: float,
) -> bool:
    """
    True if the person is attempting to climb: part of the box is above the wall.
    We use the TOP of the box (y1 = head side). If box top is above wall line and overlaps wall horizontally -> climbing.
    """
    if not box or len(box) < 4 or not zone_coordinates or len(zone_coordinates) < 3:
        return False

    wall_top_y, wall_x_min, wall_x_max = _wall_top_and_x_range(
        zone_coordinates, frame_width, frame_height
    )
    x1, y1, x2, y2 = _box_to_pixel(box, frame_width, frame_height)

    if not _box_overlaps_wall_x(x1, x2, wall_x_min, wall_x_max):
        return False

    # In image coords, smaller y = higher on screen. So "above wall" = y < wall_top_y.
    # Climbing = top of person (y1) is above the wall line.
    return y1 < wall_top_y


def is_box_fully_above_wall(
    box: List[float],
    zone_coordinates: List[List[float]],
    frame_width: float,
    frame_height: float,
) -> bool:
    """
    True if the whole person is above the wall: BOTTOM of box is above wall line.
    Once this is true we keep the box RED (no going back to orange).
    """
    if not box or len(box) < 4 or not zone_coordinates or len(zone_coordinates) < 3:
        return False

    wall_top_y, wall_x_min, wall_x_max = _wall_top_and_x_range(
        zone_coordinates, frame_width, frame_height
    )
    x1, y1, x2, y2 = _box_to_pixel(box, frame_width, frame_height)

    if not _box_overlaps_wall_x(x1, x2, wall_x_min, wall_x_max):
        return False

    # Fully above = bottom of person (y2) is above the wall line.
    return y2 < wall_top_y
