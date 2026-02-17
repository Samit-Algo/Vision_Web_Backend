"""
Wall Climb Detection - Zone helpers
-----------------------------------

User-drawn polygon = wall. Its TOP edge is the "wall line".
- Keypoint-based: person "above" when head/shoulders (COCO 0, 5, 6) are above wall line (primary).
- Box-based fallback: when keypoints missing, use box top above wall line (is_box_climbing).
"""

from typing import List, Tuple, Optional

# COCO pose indices (YOLO Pose same): 
# 0=nose (head), 5=left_shoulder, 6=right_shoulder, 9=left_wrist, 10=right_wrist
UPPER_BODY_KEYPOINT_INDICES = (0, 5, 6)  # Legacy: nose and shoulders
# Keypoints to monitor for climbing detection: head, shoulders, wrists
CLIMBING_KEYPOINT_INDICES = (0, 5, 6, 9, 10)  # nose, left_shoulder, right_shoulder, left_wrist, right_wrist


def _get_keypoint_xy(
    person_keypoints: List[List[float]],
    idx: int,
    frame_width: float,
    frame_height: float,
    confidence_threshold: float = 0.25,
) -> Optional[Tuple[float, float]]:
    """Return (x, y) in pixel coords for keypoint at idx, or None if missing/low conf."""
    if not person_keypoints or idx >= len(person_keypoints):
        return None
    kp = person_keypoints[idx]
    if not kp or len(kp) < 2:
        return None
    if len(kp) >= 3 and float(kp[2]) < confidence_threshold:
        return None
    x, y = float(kp[0]), float(kp[1])
    if frame_width > 1 and frame_height > 1 and max(x, y) <= 1.0:
        x, y = x * frame_width, y * frame_height
    return (x, y)


def wall_top_and_x_range(
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


def is_person_above_wall_by_keypoints(
    person_keypoints: List[List[float]],
    zone_coordinates: List[List[float]],
    frame_width: float,
    frame_height: float,
    kp_confidence_threshold: float = 0.25,
) -> bool:
    """
    True if person is "above" the wall line using upper-body keypoints (nose, shoulders).
    Catches 10-20% body crossing (head/shoulders over wall). Requires at least one
    keypoint above the line and within wall horizontal range.
    """
    if not person_keypoints or not zone_coordinates or len(zone_coordinates) < 3:
        return False
    wall_top_y, wall_x_min, wall_x_max = wall_top_and_x_range(
        zone_coordinates, frame_width, frame_height
    )
    for idx in UPPER_BODY_KEYPOINT_INDICES:
        pt = _get_keypoint_xy(
            person_keypoints, idx, frame_width, frame_height, kp_confidence_threshold
        )
        if pt is None:
            continue
        x, y = pt
        if y < wall_top_y and wall_x_min <= x <= wall_x_max:
            return True
    return False


def check_climbing_keypoints_above_zone(
    person_keypoints: List[List[float]],
    zone_coordinates: List[List[float]],
    frame_width: float,
    frame_height: float,
    kp_confidence_threshold: float = 0.25,
) -> Tuple[bool, List[int]]:
    """
    Check if any climbing-related keypoints (head, shoulders, wrists) are above the wall zone.
    
    Returns:
        (is_above, detected_keypoint_indices): 
        - is_above: True if any keypoint is above zone
        - detected_keypoint_indices: List of keypoint indices that are above zone
    """
    if not person_keypoints or not zone_coordinates or len(zone_coordinates) < 3:
        return False, []
    
    wall_top_y, wall_x_min, wall_x_max = wall_top_and_x_range(
        zone_coordinates, frame_width, frame_height
    )
    
    detected_indices = []
    for idx in CLIMBING_KEYPOINT_INDICES:
        pt = _get_keypoint_xy(
            person_keypoints, idx, frame_width, frame_height, kp_confidence_threshold
        )
        if pt is None:
            continue
        x, y = pt
        # Check if keypoint is above wall line and within horizontal range
        if y < wall_top_y and wall_x_min <= x <= wall_x_max:
            detected_indices.append(idx)
    
    return len(detected_indices) > 0, detected_indices


def box_to_pixel(
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


def box_overlaps_wall_x(
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

    wall_top_y, wall_x_min, wall_x_max = wall_top_and_x_range(
        zone_coordinates, frame_width, frame_height
    )
    x1, y1, x2, y2 = box_to_pixel(box, frame_width, frame_height)

    if not box_overlaps_wall_x(x1, x2, wall_x_min, wall_x_max):
        return False

    # In image coords, smaller y = higher on screen. So "above wall" = y < wall_top_y.
    # Climbing = top of person (y1) is above the wall line.
    return y1 < wall_top_y
