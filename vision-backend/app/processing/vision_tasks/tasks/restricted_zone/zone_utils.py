"""
Restricted Zone Utilities
-------------------------

Utilities for checking if detections are within restricted zones.
"""

from typing import List, Tuple


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            else:
                xinters = p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def get_box_center(box: List[float]) -> Tuple[float, float]:
    """Get center point of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def is_box_in_zone(
    box: List[float],
    zone_coordinates: List[List[float]],
    frame_width: float = None,
    frame_height: float = None,
) -> bool:
    """
    Check if a detection box is inside a polygon zone.
    Uses the center point of the box. Handles both normalized (0-1) and pixel coordinates.
    """
    if not box or len(box) < 4 or not zone_coordinates or len(zone_coordinates) < 3:
        return False
    center_x, center_y = get_box_center(box)
    max_zone_val = max(max(p[0], p[1]) for p in zone_coordinates)
    is_zone_normalized = max_zone_val <= 1.0
    if is_zone_normalized and (frame_width is None or frame_height is None):
        frame_width = max(box[0], box[2]) * 2
        frame_height = max(box[1], box[3]) * 2
    if is_zone_normalized and frame_width and frame_height and frame_width > 1 and frame_height > 1:
        center_x = center_x / frame_width
        center_y = center_y / frame_height
    return point_in_polygon((center_x, center_y), zone_coordinates)
