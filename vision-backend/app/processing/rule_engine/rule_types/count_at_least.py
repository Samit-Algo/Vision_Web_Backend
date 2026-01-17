"""
Rule: count_at_least
--------------------

Schema:
{
  "type": "count_at_least",
  "class": "person",
  "min_count": 2,
  "duration_seconds": <optional>,
  "label": <optional custom label>
}
"""
from datetime import datetime
from typing import Any, Dict, Optional

from app.processing.rule_engine.registry import register_rule


@register_rule("count_at_least")
def evaluate_count_at_least(
    rule: Dict[str, Any],
    detections: Dict[str, Any],
    task: Dict[str, Any],
    rule_state: Dict[str, Any],
    now: datetime,
) -> Optional[Dict[str, Any]]:
    """
    Trigger when the count of a specific class in detections['classes'] is
    greater than or equal to the provided threshold. Supports optional
    duration_seconds for sustained conditions.
    """
    detected_classes = detections.get("classes") or []
    target_class_name = str(rule.get("class") or "").strip().lower()
    if not target_class_name:
        rule_state["last_matched_since"] = None
        return None

    try:
        min_count = int(rule.get("min_count", 1))
    except Exception:  # noqa: BLE001
        min_count = 1

    try:
        duration_seconds = int(rule.get("duration_seconds", 0) or 0)
    except Exception:  # noqa: BLE001
        duration_seconds = 0

    matched_count = sum(
        1
        for detected_class_name in detected_classes
        if isinstance(detected_class_name, str) and detected_class_name.lower() == target_class_name
    )
    matched_now = matched_count >= min_count

    if not matched_now:
        rule_state["last_matched_since"] = None
        return None

    last_matched_since: Optional[datetime] = rule_state.get("last_matched_since")
    if duration_seconds <= 0:
        rule_state["last_matched_since"] = now
    else:
        if last_matched_since is None:
            rule_state["last_matched_since"] = now
            return None
        if (now - last_matched_since).total_seconds() < duration_seconds:
            return None

    label = rule.get("label")
    if not label:
        plural_suffix = "s" if matched_count != 1 else ""
        label = f"{matched_count} {target_class_name}{plural_suffix} detected"

    # Compute detection indices that matched (for overlay filtering)
    detected_classes_list = detections.get("classes") or []
    matched_indices: List[int] = []
    for idx, detected_class_name in enumerate(detected_classes_list):
        if isinstance(detected_class_name, str) and detected_class_name.lower() == target_class_name:
            matched_indices.append(idx)

    return {
        "label": label,
        "matched_classes": [target_class_name] * matched_count,
        "count": matched_count,
        "matched_detection_indices": matched_indices,
    }



