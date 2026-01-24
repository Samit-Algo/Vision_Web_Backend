"""
Class Counter
-------------

Counts detections of a specified class, optionally within a zone.
"""

from typing import List, Tuple

from app.processing.scenarios.contracts import DetectionPacket
from app.processing.scenarios.class_count.zone_utils import is_detection_in_zone


def count_class_detections(
    detections: DetectionPacket,
    target_class: str,
    zone_coordinates: List[List[float]] = None
) -> Tuple[int, List[int]]:
    """
    Count detections of target class, optionally filtered by zone.
    
    Args:
        detections: Detection packet
        target_class: Target class name (normalized)
        zone_coordinates: Optional zone polygon coordinates
    
    Returns:
        Tuple of (count, matched_indices)
    """
    detected_classes = detections.classes
    boxes = detections.boxes
    matched_count = 0
    matched_indices = []
    
    for idx, detected_class in enumerate(detected_classes):
        if isinstance(detected_class, str) and detected_class.lower() == target_class:
            # Check zone if configured
            if zone_coordinates:
                if idx < len(boxes):
                    box = boxes[idx]
                    if is_detection_in_zone(box, zone_coordinates):
                        matched_count += 1
                        matched_indices.append(idx)
            else:
                matched_count += 1
                matched_indices.append(idx)
    
    return matched_count, matched_indices


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
