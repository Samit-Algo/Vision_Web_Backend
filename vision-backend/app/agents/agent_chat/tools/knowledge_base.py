from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from ...exceptions import RuleNotFoundError
from ..state.agent_state import (
    STEP_CAMERA,
    STEP_CONFIRMATION,
    STEP_TIME_WINDOW,
    STEP_ZONE,
    AgentState,
)


def can_enter_confirmation(agent: AgentState, rule: Dict) -> bool:
    """
    Return True only when all required fields for confirmation are present.
    Used to block transition to CONFIRMATION until state is valid.
    No user-facing messages; structure only.
    """
    source_type = (agent.fields.get("source_type") or "").strip().lower()
    if source_type == "video_file":
        video_path = (agent.fields.get("video_path") or "").strip()
        if not video_path:
            return False
    else:
        if not agent.fields.get("camera_id"):
            return False

    run_mode = agent.fields.get("run_mode") or "continuous"
    if compute_requires_zone(rule, run_mode):
        zone = agent.fields.get("zone")
        if not zone or (isinstance(zone, (dict, list)) and not zone):
            return False
        # Rules that use zone_type motion_rois require zone with type and at least one loom
        zone_type = (rule.get("zone_support") or {}).get("zone_type")
        if zone_type == "motion_rois":
            if not isinstance(zone, dict) or zone.get("type") != "motion_rois":
                return False
            looms = zone.get("looms")
            if not isinstance(looms, list) or len(looms) < 1:
                return False
        # Rules that use zone_type polygon require zone with type polygon and at least 3 coordinates
        elif zone_type == "polygon":
            if not isinstance(zone, dict) or zone.get("type") != "polygon":
                return False
            coords = zone.get("coordinates")
            if not isinstance(coords, list) or len(coords) < 3:
                return False

    time_window = rule.get("time_window", {})
    if time_window.get("required", False) and source_type != "video_file":
        if agent.fields.get("start_time") is None or agent.fields.get("end_time") is None:
            return False

    for f in rule.get("required_fields_from_user", []) or []:
        if agent.fields.get(f) is None:
            return False

    execution_modes = rule.get("execution_modes", {})
    execution_config = execution_modes.get(run_mode, {})
    for f in execution_config.get("required_fields", []) or []:
        if agent.fields.get(f) is None:
            return False

    return True

KB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "knowledge_base" / "vision_rule_knowledge_base.json"
RULES: List[Dict] = []


def load_rules() -> None:
    global RULES
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    RULES = data.get("rules", []) or []
    if not RULES:
        raise ValueError("Knowledge base has no rules")


load_rules()


def get_rule(rule_id: str) -> Dict:
    for rule in RULES:
        if rule.get("rule_id") == rule_id:
            return rule
    raise RuleNotFoundError(rule_id)


def compute_missing_fields(agent: AgentState, rule: Dict) -> None:
    ordered: List[str] = []

    def add(field_name: str) -> None:
        if field_name and field_name not in ordered:
            ordered.append(field_name)

    for f in rule.get("required_fields_from_user", []) or []:
        add(f)

    source_type = (agent.fields.get("source_type") or "").strip().lower()
    if source_type == "video_file":
        add("video_path")
    else:
        add("camera_id")

    zone_support = rule.get("zone_support", {})
    requires_zone = bool(zone_support.get("required", False))
    run_mode = agent.fields.get("run_mode", "continuous")
    execution_modes = rule.get("execution_modes", {})
    if run_mode not in execution_modes:
        run_mode = (rule.get("defaults") or {}).get("run_mode", "continuous")
        agent.fields["run_mode"] = run_mode
    execution_config = execution_modes.get(run_mode, {})
    execution_required = execution_config.get("required_fields", []) or []
    execution_zone_required = execution_config.get("zone_required", False)

    if execution_zone_required or requires_zone:
        add("zone")
    elif rule.get("rule_id") in ["class_count", "box_count"]:
        zone_val = agent.fields.get("zone")
        if not zone_val or (isinstance(zone_val, (dict, list)) and not zone_val):
            add("zone")

    for f in execution_required:
        add(f)

    time_window = rule.get("time_window", {})
    if time_window.get("required", False) and source_type != "video_file":
        add("start_time")
        add("end_time")

    missing = []
    rule_id = rule.get("rule_id")
    for f in ordered:
        value = agent.fields.get(f)
        if f == "zone":
            if not value or (isinstance(value, (dict, list)) and not value):
                missing.append(f)
            elif (rule.get("zone_support") or {}).get("zone_type") == "motion_rois":
                if not isinstance(value, dict) or value.get("type") != "motion_rois" or not isinstance(value.get("looms"), list) or len(value.get("looms", [])) < 1:
                    missing.append(f)
            elif (rule.get("zone_support") or {}).get("zone_type") == "polygon":
                if not isinstance(value, dict) or value.get("type") != "polygon" or not isinstance(value.get("coordinates"), list) or len(value.get("coordinates", [])) < 3:
                    missing.append(f)
        elif f == "video_path":
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(f)
        else:
            if value is None:
                missing.append(f)
    agent.missing_fields = missing


def apply_rule_defaults(agent: AgentState, rule: Dict) -> None:
    if agent.fields.get("model") is None:
        agent.fields["model"] = rule.get("model")
    for key, val in (rule.get("defaults") or {}).items():
        if agent.fields.get(key) is None:
            agent.fields[key] = val


def compute_requires_zone(rule: Dict, run_mode: str) -> bool:
    zone_support = rule.get("zone_support", {})
    if zone_support.get("required", False):
        return True
    execution_config = (rule.get("execution_modes") or {}).get(run_mode, {})
    return bool(execution_config.get("zone_required", False))


def get_current_step(agent: AgentState, rule: Dict) -> str:
    """
    Rule-aware step priority. Order is strict: camera → zone → time_window → confirmation.
    Does not rely on missing_fields ordering. Applies to all rules where zone is required.
    """
    run_mode = agent.fields.get("run_mode") or "continuous"
    source_type = (agent.fields.get("source_type") or "").strip().lower()

    # 1. Camera/video source missing → STEP_CAMERA
    if source_type == "video_file":
        video_path = (agent.fields.get("video_path") or "").strip()
        if not video_path:
            return STEP_CAMERA
    else:
        if not agent.fields.get("camera_id"):
            return STEP_CAMERA

    # 2. Zone required and zone missing → STEP_ZONE (all rules where zone is required)
    zone_required = compute_requires_zone(rule, run_mode) or rule.get("rule_id") in ("class_count", "box_count")
    if zone_required:
        zone = agent.fields.get("zone")
        if not zone or (isinstance(zone, (dict, list)) and not zone):
            return STEP_ZONE

    # 3. Time window required and (start_time or end_time) missing → STEP_TIME_WINDOW
    time_window = rule.get("time_window", {})
    if time_window.get("required", False) and source_type != "video_file":
        if agent.fields.get("start_time") is None or agent.fields.get("end_time") is None:
            return STEP_TIME_WINDOW

    # 4. All structural requirements met → STEP_CONFIRMATION
    if can_enter_confirmation(agent, rule):
        return STEP_CONFIRMATION

    # 5. Do not skip ahead (e.g. other required fields still missing)
    return STEP_TIME_WINDOW
