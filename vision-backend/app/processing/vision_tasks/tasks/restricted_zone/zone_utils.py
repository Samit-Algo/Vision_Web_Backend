"""
Restricted zone utilities
--------------------------

Point-in-polygon and box-in-zone checks. Used by restricted_zone and fire_detection scenarios.
Handles both normalized (0–1) and pixel coordinates.

Industrial-grade implementation:
- Box-polygon intersection (not just center point)
- Touch detection (any corner inside or intersection)
- Area-based inside detection (majority of box inside)
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
import numpy as np


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """True if point (x, y) is inside polygon (ray-casting). Polygon needs at least 3 vertices."""
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
    """Center (x, y) of box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_box_corners(box: List[float]) -> List[Tuple[float, float]]:
    """Get all four corners of bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return [
        (x1, y1),  # Top-left
        (x2, y1),  # Top-right
        (x2, y2),  # Bottom-right
        (x1, y2),  # Bottom-left
    ]


def normalize_coordinates(
    coords: List[Tuple[float, float]],
    frame_width: float,
    frame_height: float,
    is_normalized: bool,
) -> List[Tuple[float, float]]:
    """Normalize coordinates if needed."""
    if is_normalized:
        return coords
    return [(x / frame_width, y / frame_height) for x, y in coords]


def calculate_box_area(box: List[float]) -> float:
    """Calculate area of bounding box."""
    x1, y1, x2, y2 = box
    return abs((x2 - x1) * (y2 - y1))


def box_polygon_intersection_area(
    box: List[float],
    polygon: List[List[float]],
    frame_width: float = None,
    frame_height: float = None,
) -> float:
    """
    Calculate intersection area between box and polygon.
    Returns area in normalized coordinates (0.0-1.0).
    
    Uses sampling method: sample points inside box and count how many are inside polygon.
    """
    if not box or len(box) < 4 or not polygon or len(polygon) < 3:
        return 0.0
    
    x1, y1, x2, y2 = box
    box_width = abs(x2 - x1)
    box_height = abs(y2 - y1)
    
    # Determine if coordinates are normalized
    max_zone_val = max(max(p[0], p[1]) for p in polygon)
    is_zone_normalized = max_zone_val <= 1.0
    
    # Normalize box coordinates if needed
    if not is_zone_normalized and frame_width and frame_height:
        x1, y1 = x1 / frame_width, y1 / frame_height
        x2, y2 = x2 / frame_width, y2 / frame_height
        box_width = abs(x2 - x1)
        box_height = abs(y2 - y1)
    
    # Sample points inside box (grid sampling)
    # Use reasonable sample density: ~10x10 grid
    sample_density = 10
    samples_inside = 0
    total_samples = 0
    
    for i in range(sample_density):
        for j in range(sample_density):
            # Sample point inside box
            sample_x = x1 + (x2 - x1) * (i + 0.5) / sample_density
            sample_y = y1 + (y2 - y1) * (j + 0.5) / sample_density
            
            if point_in_polygon((sample_x, sample_y), polygon):
                samples_inside += 1
            total_samples += 1
    
    # Intersection area ratio
    intersection_ratio = samples_inside / total_samples if total_samples > 0 else 0.0
    box_area = box_width * box_height
    intersection_area = box_area * intersection_ratio
    
    return intersection_area


def check_box_zone_intersection(
    box: List[float],
    zone_coordinates: List[List[float]],
    frame_width: float = None,
    frame_height: float = None,
) -> dict:
    """
    Industrial-grade box-zone intersection check.
    
    Returns dict with:
    - 'touches': bool - Any corner inside OR polygon intersects box
    - 'inside': bool - Majority of box area inside polygon
    - 'intersection_ratio': float - Ratio of box area inside polygon (0.0-1.0)
    - 'any_corner_inside': bool - At least one corner is inside
    """
    if not box or len(box) < 4 or not zone_coordinates or len(zone_coordinates) < 3:
        return {
            'touches': False,
            'inside': False,
            'intersection_ratio': 0.0,
            'any_corner_inside': False,
        }
    
    # Determine if coordinates are normalized
    max_zone_val = max(max(p[0], p[1]) for p in zone_coordinates)
    is_zone_normalized = max_zone_val <= 1.0
    
    # Normalize box coordinates if needed
    # If zone is normalized but box is in pixels, normalize box
    # If zone is in pixels but box is normalized, denormalize box (but we assume zone is normalized)
    normalized_box = list(box)
    if frame_width and frame_height and frame_width > 1 and frame_height > 1:
        # Check if box appears to be in pixels (values > 1) or normalized (values <= 1)
        max_box_val = max(abs(box[0]), abs(box[1]), abs(box[2]), abs(box[3]))
        is_box_normalized = max_box_val <= 1.0
        
        # If box is in pixels but zone is normalized, normalize the box
        if not is_box_normalized and is_zone_normalized:
            normalized_box = [
                box[0] / frame_width,
                box[1] / frame_height,
                box[2] / frame_width,
                box[3] / frame_height,
            ]
        # If box is normalized but zone is in pixels, denormalize the box
        elif is_box_normalized and not is_zone_normalized:
            normalized_box = [
                box[0] * frame_width,
                box[1] * frame_height,
                box[2] * frame_width,
                box[3] * frame_height,
            ]
    
    # Get box corners
    corners = get_box_corners(normalized_box)
    
    # Check if any corner is inside polygon
    any_corner_inside = any(point_in_polygon(corner, zone_coordinates) for corner in corners)
    
    # Calculate intersection ratio
    intersection_area = box_polygon_intersection_area(
        normalized_box, zone_coordinates, None, None
    )
    box_area = calculate_box_area(normalized_box)
    intersection_ratio = intersection_area / box_area if box_area > 0 else 0.0
    
    # Touch: any corner inside OR significant intersection (>5% of box)
    touches = any_corner_inside or intersection_ratio > 0.05
    
    # Inside: majority of box area inside (>50%)
    inside = intersection_ratio > 0.5
    
    return {
        'touches': touches,
        'inside': inside,
        'intersection_ratio': intersection_ratio,
        'any_corner_inside': any_corner_inside,
    }


def is_box_in_zone(
    box: List[float],
    zone_coordinates: List[List[float]],
    frame_width: float = None,
    frame_height: float = None,
) -> bool:
    """
    Legacy function: True if box center is inside the polygon zone.
    Kept for backward compatibility.
    Zone can be normalized (0–1) or pixel; if normalized, pass frame_width and frame_height.
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
