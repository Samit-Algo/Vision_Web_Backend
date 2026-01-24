"""
Zone Utilities
--------------

Utilities for checking if detections are within defined zones.
"""

from typing import List


def is_detection_in_zone(box: List[float], zone_coordinates: List[List[float]]) -> bool:
    """
    Check if a detection bounding box is inside a polygon zone.
    
    Uses point-in-polygon algorithm to check if the box center
    or any corner is inside the zone polygon.
    
    Args:
        box: Bounding box [x1, y1, x2, y2] in pixel coordinates
        zone_coordinates: Polygon coordinates [[x1, y1], [x2, y2], ...]
    
    Returns:
        True if box is inside zone, False otherwise
    """
    if not box or len(box) < 4:
        return False
    
    if not zone_coordinates or len(zone_coordinates) < 3:
        return False
    
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    
    # Calculate box center
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    # Check if center is in polygon (primary check)
    if _point_in_polygon(center_x, center_y, zone_coordinates):
        return True
    
    # Also check corners (if center is outside, but box overlaps zone)
    corners = [
        (x1, y1),  # Top-left
        (x2, y1),  # Top-right
        (x2, y2),  # Bottom-right
        (x1, y2),  # Bottom-left
    ]
    
    for corner_x, corner_y in corners:
        if _point_in_polygon(corner_x, corner_y, zone_coordinates):
            return True
    
    return False


def _point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        x: Point x coordinate
        y: Point y coordinate
        polygon: List of polygon vertices [[x1, y1], [x2, y2], ...]
    
    Returns:
        True if point is inside polygon, False otherwise
    """
    if len(polygon) < 3:
        return False
    
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        
        # Check if point is on the edge
        if y == p1y == p2y:
            if min(p1x, p2x) <= x <= max(p1x, p2x):
                return True
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside
