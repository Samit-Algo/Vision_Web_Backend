"""
Rule: class_presence
--------------------

Schema:
{
  "type": "class_presence",
  "match": "any" | "all",
  "classes": ["person", "car"],
  "duration_seconds": <optional>,
  "label": <optional custom label>
}
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.processing.rule_engine.registry import register_rule


def normalize_classes(classes: List[str]) -> List[str]:
    return [str(class_name).lower() for class_name in classes if isinstance(class_name, str) and class_name]


@register_rule("class_presence")
def evaluate_class_presence(rule: Dict[str, Any], detections: Dict[str, Any], task: Dict[str, Any], rule_state: Dict[str, Any], now: datetime) -> Optional[Dict[str, Any]]:
    """
    Evaluate class presence rule against current detections with optional duration.
    Keeps per-rule state in rule_state["last_matched_since"].
    
    Supports both formats:
    - "class": "person" (singular string) - legacy format from database
    - "classes": ["person", "car"] (plural list) - new format
    """
    detected_classes = normalize_classes(detections.get("classes") or [])
    if not detected_classes:
        rule_state["last_matched_since"] = None
        return None

    # Support both "class" (singular) and "classes" (plural) formats
    rule_classes = rule.get("classes") or []
    rule_class = rule.get("class") or rule.get("target_class")  # Support alias too
    
    # If "class" is provided, convert to list format
    if rule_class and not rule_classes:
        rule_classes = [rule_class]
    
    required_classes = [str(class_name).lower() for class_name in rule_classes if isinstance(class_name, str) and class_name]
    if not required_classes:
        rule_state["last_matched_since"] = None
        return None
    
    # Debug: log what we're looking for vs what we detected
    # Uncomment for debugging:
    # print(f"[class_presence] Required: {required_classes}, Detected: {detected_classes}")

    match_mode = str(rule.get("match", "any")).strip().lower()
    if match_mode == "all":
        matched_now = all(required_class_name in detected_classes for required_class_name in required_classes)
        matched_classes = required_classes if matched_now else []
    else:
        found_classes = sorted(set(required_class_name for required_class_name in required_classes if required_class_name in detected_classes))
        matched_now = len(found_classes) > 0
        matched_classes = found_classes

    # duration handling
    duration_seconds = int(rule.get("duration_seconds", 0) or 0)
    if not matched_now:
        rule_state["last_matched_since"] = None
        return None

    last_matched_since: Optional[datetime] = rule_state.get("last_matched_since")
    if duration_seconds <= 0:
        # immediate trigger, lock since for continuity info but not required
        rule_state["last_matched_since"] = now
    else:
        if last_matched_since is None:
            rule_state["last_matched_since"] = now
            return None
        if (now - last_matched_since).total_seconds() < duration_seconds:
            return None
        # already sustained enough time; keep since to allow continued matches

    # Label
    label = rule.get("label")
    if not label:
        if match_mode == "all" and len(required_classes) > 1:
            label = f"Classes detected: {', '.join(sorted(set(required_classes)))}"
        elif len(matched_classes) == 1:
            label = f"{matched_classes[0]} detected"
        else:
            label = f"Classes detected: {', '.join(sorted(set(matched_classes)))}"

    # Compute detection indices that matched (for overlay filtering)
    detected_classes_list = detections.get("classes") or []
    matched_indices: List[int] = []
    for idx, detected_class in enumerate(detected_classes_list):
        if isinstance(detected_class, str) and detected_class.lower() in matched_classes:
            matched_indices.append(idx)

    return {
        "label": label,
        "matched_classes": matched_classes,
        "matched_detection_indices": matched_indices,
    }


