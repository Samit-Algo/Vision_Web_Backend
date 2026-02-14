"""
Class presence detector
------------------------

Helpers: normalize class names, match by mode (any/all), find indices, generate label.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import List, Tuple


def normalize_classes(classes: List[str]) -> List[str]:
    """Normalize class names to lowercase."""
    return [str(class_name).lower() for class_name in classes if isinstance(class_name, str) and class_name]


def find_matched_classes(
    detected_classes: List[str],
    required_classes: List[str],
    match_mode: str
) -> Tuple[List[str], bool]:
    """
    Find matched classes based on match mode.

    Args:
        detected_classes: List of detected class names (normalized)
        required_classes: List of required class names (normalized)
        match_mode: "any" or "all"

    Returns:
        Tuple of (matched_classes, matched_now)
    """
    if match_mode == "all":
        matched_now = all(
            req_class in detected_classes
            for req_class in required_classes
        )
        matched_classes = required_classes if matched_now else []
    else:
        matched_classes = [
            req_class for req_class in required_classes
            if req_class in detected_classes
        ]
        matched_now = len(matched_classes) > 0

    return matched_classes, matched_now


def find_matched_indices(
    detection_classes: List[str],
    matched_classes: List[str]
) -> List[int]:
    """Find detection indices that match the required classes."""
    matched_indices = []
    for idx, detected_class in enumerate(detection_classes):
        if isinstance(detected_class, str) and detected_class.lower() in matched_classes:
            matched_indices.append(idx)
    return matched_indices


def generate_label(
    matched_classes: List[str],
    required_classes: List[str],
    match_mode: str,
    custom_label: str = None
) -> str:
    """Generate event label."""
    if custom_label:
        return custom_label

    if match_mode == "all" and len(required_classes) > 1:
        return f"Classes detected: {', '.join(sorted(set(required_classes)))}"
    elif len(matched_classes) == 1:
        return f"{matched_classes[0]} detected"
    else:
        return f"Classes detected: {', '.join(sorted(set(matched_classes)))}"
