"""
Class Counter
-------------

Counts detections of a specified class (simple per-frame counting).
Line-based counting is handled by tracking (see scenario.py).
"""

from typing import List, Tuple

from app.processing.scenarios.contracts import DetectionPacket


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
    boxes = detections.boxes
    classes = detections.classes
    scores = detections.scores
    
    filtered = []
    class_name_lower = target_class.lower()
    
    for i, detected_class in enumerate(classes):
        if isinstance(detected_class, str) and detected_class.lower() == class_name_lower:
            if i < len(boxes) and i < len(scores):
                filtered.append((boxes[i], scores[i]))
    
    return filtered


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
