"""
Rule: class_count
-----------------

Schema (report mode - always returns count):
{
  "type": "class_count",
  "class": "person",
  "label": "Person count"  # optional, if provided prints "<label>: N"
}
"""
from datetime import datetime
from typing import Any, Dict, Optional

from app.processing.rule_engine.registry import register_rule


@register_rule("class_count")
def evaluate_class_count(
    rule: Dict[str, Any],
    detections: Dict[str, Any],
    task: Dict[str, Any],
    rule_state: Dict[str, Any],
    now: datetime,
) -> Optional[Dict[str, Any]]:
    """
    Always return the current count for the specified class.
    Intended for continuous reporting per frame (including zero).
    """
    detected_classes = detections.get("classes") or []
    target_class_name = str(rule.get("class") or "").strip().lower()
    if not target_class_name:
        return None

    matched_count = sum(
        1
        for detected_class_name in detected_classes
        if isinstance(detected_class_name, str) and detected_class_name.lower() == target_class_name
    )

    custom_label = rule.get("label")
    if isinstance(custom_label, str) and custom_label.strip():
        label = f"{custom_label.strip()}: {matched_count}"
    else:
        if matched_count == 0:
            label = f"No {target_class_name} detected"
        elif matched_count == 1:
            label = f"1 {target_class_name} detected"
        else:
            label = f"{matched_count} {target_class_name}s detected"

    # Compute detection indices that matched (for overlay filtering)
    detected_classes_list = detections.get("classes") or []
    matched_indices: List[int] = []
    for idx, detected_class_name in enumerate(detected_classes_list):
        if isinstance(detected_class_name, str) and detected_class_name.lower() == target_class_name:
            matched_indices.append(idx)

    return {
        "label": label,
        "count": matched_count,
        "matched_classes": [target_class_name] * matched_count,
        "matched_detection_indices": matched_indices,
    }



