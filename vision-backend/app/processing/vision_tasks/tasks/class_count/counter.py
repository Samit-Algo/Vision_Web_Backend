"""
Class counter (shared helpers)
------------------------------

- count_class_detections: simple per-frame count of target class.
- filter_detections_by_class: (bbox, score) list for tracker.
- generate_count_label: label string for events.
- calculate_iou: Intersection over Union of two boxes (shared by class_count and box_count).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from app.processing.processing_output.data_models import DetectionPacket

# -----------------------------------------------------------------------------
# Shared: IoU (used by class_count and box_count for track–detection matching)
# -----------------------------------------------------------------------------


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Intersection over Union of two boxes [x1, y1, x2, y2]. Returns 0 if no overlap."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    overlap_left = max(x1_1, x1_2)
    overlap_top = max(y1_1, y1_2)
    overlap_right = min(x2_1, x2_2)
    overlap_bottom = min(y2_1, y2_2)
    if overlap_right < overlap_left or overlap_bottom < overlap_top:
        return 0.0
    intersection = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    return (intersection / union) if union else 0.0

# -----------------------------------------------------------------------------
# Counting helpers
# -----------------------------------------------------------------------------


def count_class_detections(
    detections: DetectionPacket,
    target_class: str
) -> Tuple[int, List[int]]:
    """
    Count detections of target class (simple per-frame counting).

    Note: This is for simple counting only. Line-based counting is handled
    by tracking in the scenario's _process_line_counting method.

    Args:
        detections: Detection packet
        target_class: Target class name (normalized)

    Returns:
        Tuple of (count, matched_indices)
    """
    detected_classes = detections.classes
    matched_count = 0
    matched_indices = []

    for idx, detected_class in enumerate(detected_classes):
        if isinstance(detected_class, str) and detected_class.lower() == target_class:
            matched_count += 1
            matched_indices.append(idx)

    return matched_count, matched_indices


def filter_detections_by_class(
    detections: DetectionPacket,
    target_class: str
) -> List[Tuple[List[float], float]]:
    """
    Filter detections to only include specified class (for tracking).

    Args:
        detections: Detection packet
        target_class: Target class name

    Returns:
        List of (bbox, score) tuples for matching detections
    """
    filtered, _ = filter_detections_by_class_with_indices(detections, target_class)
    return filtered


def filter_detections_by_class_with_indices(
    detections: DetectionPacket,
    target_class: str,
) -> Tuple[List[Tuple[List[float], float]], List[int]]:
    """
    Filter detections by class while preserving original detection indices.
    Shared by fall_detection and wall_climb_detection for tracker-based scenarios.

    Returns:
        (filtered_detections, original_indices) where
        filtered_detections = list of (bbox, score), original_indices = list of indices into detections
    """
    boxes = detections.boxes
    classes = detections.classes
    scores = detections.scores
    filtered: List[Tuple[List[float], float]] = []
    original_indices: List[int] = []
    class_name_lower = target_class.lower()
    for i, detected_class in enumerate(classes):
        if isinstance(detected_class, str) and detected_class.lower() == class_name_lower:
            if i < len(boxes) and i < len(scores):
                filtered.append((list(boxes[i]), float(scores[i])))
                original_indices.append(i)
    return filtered, original_indices


def generate_count_label(
    count: int,
    target_class: str,
    custom_label: str = None
) -> str:
    """
    Generate label for count event.

    Args:
        count: Current count
        target_class: Target class name
        custom_label: Optional custom label from config

    Returns:
        Generated label string
    """
    if custom_label and isinstance(custom_label, str) and custom_label.strip():
        return f"{custom_label.strip()}: {count}"

    if count == 0:
        return f"No {target_class} detected"
    elif count == 1:
        return f"1 {target_class} detected"
    else:
        return f"{count} {target_class}s detected"
