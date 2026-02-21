"""
Backend state machine for vision rules.
Loads knowledge base, computes missing fields. Tools use this; the LLM does not see it.
"""

import json
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent.parent.parent.parent
KNOWLEDGE_BASE_PATH = APP_ROOT / "knowledge_base" / "vision_rule_knowledge_base.json"

rules_cache: dict[str, dict[str, Any]] | None = None


def load_knowledge_base() -> dict[str, dict[str, Any]]:
    """Load and cache rules from the knowledge base JSON. Returns rule_id -> rule dict."""
    global rules_cache
    if rules_cache is None:
        with open(KNOWLEDGE_BASE_PATH, encoding="utf-8") as file:
            data = json.load(file)
        rules_cache = {rule["rule_id"]: rule for rule in data["rules"]}
    return rules_cache


def get_rule(rule_id: str) -> dict[str, Any] | None:
    """Return the rule template for the given rule_id, or None if not found."""
    return load_knowledge_base().get(rule_id)


def list_rule_ids() -> list[str]:
    """Return all known rule_ids from the knowledge base."""
    return list(load_knowledge_base().keys())


def build_initial_config(rule: dict[str, Any]) -> dict[str, Any]:
    """Config from rule defaults (deep copy)."""
    defaults = rule.get("defaults") or {}
    return json.loads(json.dumps(defaults))


def compute_missing_fields(rule: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """
    Return required fields still missing from config.
    Uses required_fields_from_user, time_window.required, execution_modes, and source (camera_id or video_path).
    """
    missing: list[str] = []

    for field_name in rule.get("required_fields_from_user") or []:
        if config.get(field_name) is None or field_name not in config:
            missing.append(field_name)

    source_type = (config.get("source_type") or "").strip().lower()
    if source_type == "video_file":
        video_path = config.get("video_path")
        if video_path is None or (isinstance(video_path, str) and not video_path.strip()):
            missing.append("video_path")
    else:
        camera_id = config.get("camera_id")
        if camera_id is None or (isinstance(camera_id, str) and not camera_id.strip()):
            missing.append("camera_id")

    time_window = rule.get("time_window") or {}
    if time_window.get("required"):
        if config.get("start_time") is None or "start_time" not in config:
            missing.append("start_time")
        if config.get("end_time") is None or "end_time" not in config:
            missing.append("end_time")

    run_mode = config.get("run_mode") or "continuous"
    execution_modes = rule.get("execution_modes") or {}
    mode_config = execution_modes.get(run_mode)
    if mode_config:
        for field_name in mode_config.get("required_fields") or []:
            if config.get(field_name) is None or field_name not in config:
                missing.append(field_name)

    zone_support = rule.get("zone_support") or {}
    requires_zone = bool(zone_support.get("required"))
    exec_zone_required = bool(mode_config.get("zone_required", False)) if mode_config else False
    zone_value = config.get("zone")
    zone_empty = not zone_value or (isinstance(zone_value, (dict, list)) and len(zone_value) == 0)
    if requires_zone or exec_zone_required:
        if zone_empty:
            missing.append("zone")
    else:
        current_rule_id = rule.get("rule_id", "")
        if current_rule_id in ("class_count", "box_count") and zone_empty:
            missing.append("zone")

    seen_fields: set[str] = set()
    return [field for field in missing if not (field in seen_fields or seen_fields.add(field))]


def get_next_required_field(missing_fields: list[str]) -> str | None:
    """The one field to ask for next. Backend decides; LLM does not choose."""
    return missing_fields[0] if missing_fields else None


def compute_requires_zone(rule: dict[str, Any] | None, run_mode: str) -> bool:
    """
    Whether the rule requires a zone (from zone_support or execution_modes[run_mode].zone_required).
    Used when building the save payload; not stored in config.
    """
    if not rule:
        return False
    zone_support = rule.get("zone_support") or {}
    if zone_support.get("required"):
        return True
    modes = rule.get("execution_modes") or {}
    mode_config = modes.get(run_mode, {})
    return bool(mode_config.get("zone_required", False))


def get_editable_fields(rule: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """
    Return field names that can be edited when reopening after a reject.
    Includes required_fields_from_user, source, time_window, execution mode fields, and zone when applicable.
    """
    editable: list[str] = []
    for field_name in rule.get("required_fields_from_user") or []:
        editable.append(field_name)
    source_type = (config.get("source_type") or "").strip().lower()
    if source_type == "video_file":
        editable.append("video_path")
    else:
        editable.append("camera_id")
    time_window = rule.get("time_window") or {}
    if time_window.get("required"):
        editable.append("start_time")
        editable.append("end_time")
    run_mode = config.get("run_mode") or "continuous"
    execution_modes = rule.get("execution_modes") or {}
    mode_config = execution_modes.get(run_mode)
    if mode_config:
        for field_name in mode_config.get("required_fields") or []:
            editable.append(field_name)
    zone_support = rule.get("zone_support") or {}
    if zone_support.get("required") or (mode_config and mode_config.get("zone_required")):
        editable.append("zone")
    current_rule_id = rule.get("rule_id", "")
    if current_rule_id in ("class_count", "box_count") and "zone" not in editable:
        editable.append("zone")
    seen_fields: set[str] = set()
    return [field for field in editable if not (field in seen_fields or seen_fields.add(field))]
