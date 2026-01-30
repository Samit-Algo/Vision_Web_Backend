"""
Restricted Zone Utilities
-------------------------

Utilities for checking if detections are within restricted zones.
"""

from typing import List, Tuple


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: (x, y) coordinate of the point
        polygon: List of [x, y] coordinates defining the polygon vertices
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    
    if n < 3:
        return False
    
    inside = False
    p1x, p1y = polygon[0]
    
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside


def get_box_center(box: List[float]) -> Tuple[float, float]:
    """
    Get the center point of a bounding box.
    
    Args:
        box: [x1, y1, x2, y2] bounding box coordinates
        
    Returns:
        (center_x, center_y) tuple
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def is_box_in_zone(box: List[float], zone_coordinates: List[List[float]], frame_width: float = None, frame_height: float = None) -> bool:
    """
    Check if a detection box is inside a polygon zone.
    
    Uses the center point of the box to determine if it's in the zone.
    IMPORTANT: Handles both normalized (0-1) and pixel coordinates automatically.
    
    Args:
        box: Bounding box [x1, y1, x2, y2] in pixel coordinates
        zone_coordinates: Polygon coordinates [[x1, y1], [x2, y2], ...] in normalized (0-1) format
        frame_width: Optional frame width for normalization (if None, assumes zone is in pixel coords)
        frame_height: Optional frame height for normalization (if None, assumes zone is in pixel coords)
    
    Returns:
        True if box center is inside zone, False otherwise
    """
    if not box or len(box) < 4:
        return False
    
    if not zone_coordinates or len(zone_coordinates) < 3:
        return False
    
    # Get box center in pixel coordinates
    center_x, center_y = get_box_center(box)
    
    # Detect if zone is normalized (0-1 range) or pixel coordinates
    # If zone values are all between 0 and 1, it's normalized
    max_zone_val = max(max(point[0], point[1]) for point in zone_coordinates)
    is_zone_normalized = max_zone_val <= 1.0
    
    # If zone is normalized but we don't have frame dimensions, try to detect from box
    if is_zone_normalized and (frame_width is None or frame_height is None):
        # Estimate frame size from box coordinates
        frame_width = max(box[0], box[2]) * 2  # Rough estimate
        frame_height = max(box[1], box[3]) * 2  # Rough estimate
        
    # Normalize box center to match zone coordinates if needed
    if is_zone_normalized:
        if frame_width and frame_height and frame_width > 1 and frame_height > 1:
            center_x = center_x / frame_width
            center_y = center_y / frame_height
    
    center = (center_x, center_y)
    
    # Check if center is in polygon
    return point_in_polygon(center, zone_coordinates)
