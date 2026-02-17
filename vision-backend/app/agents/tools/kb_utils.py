"""
Shared knowledge base utilities for agent tools.

This module provides centralized, data-driven logic for:
- Loading and querying the knowledge base
- Computing missing fields based on KB rules
- Applying rule defaults
- Validating run_mode against KB

All tools should import from here to ensure consistency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from ..exceptions import RuleNotFoundError
from ..session_state.agent_state import AgentState

logger = logging.getLogger(__name__)


# ============================================================================
# KNOWLEDGE BASE LOADING
# ============================================================================

KB_PATH = Path(__file__).resolve().parent.parent.parent / "knowledge_base" / "vision_rule_knowledge_base.json"

kb_rules: List[Dict] = []


def load_knowledge_base() -> None:
    """Load knowledge base; called at module init. Raises on missing file, invalid JSON, or empty rules."""
    global kb_rules
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {KB_PATH}")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    rules = data.get("rules", [])
    if not rules:
        raise ValueError("Knowledge base contains no rules")
    kb_rules = rules
    logger.info("Loaded %s rules from knowledge base", len(kb_rules))


load_knowledge_base()


# ============================================================================
# RULE QUERYING
# ============================================================================

def get_rule(rule_id: str) -> Dict:
    """
    Get a rule definition from the knowledge base by rule_id.

    Args:
        rule_id: The rule ID to look up (e.g., "class_presence", "gesture_detected")

    Returns:
        Rule dictionary from knowledge base

    Raises:
        RuleNotFoundError: If rule_id is not found in knowledge base
    """
    for rule in kb_rules:
        if rule.get("rule_id") == rule_id:
            return rule
    raise RuleNotFoundError(rule_id)


# ============================================================================
# FIELD COMPUTATION
# ============================================================================

def compute_missing_fields(agent: AgentState, rule: Dict) -> None:
    """
    Deterministic computation of missing fields.

    Fully data-driven from knowledge base - no hardcoding.

    Args:
        agent: The agent state to compute missing fields for
        rule: The rule definition from knowledge base
    """
    ordered_required: List[str] = []

    def add_unique(field_name: str) -> None:
        if field_name and field_name not in ordered_required:
            ordered_required.append(field_name)

    for f in rule.get("required_fields_from_user", []) or []:
        add_unique(f)

    # Source: either camera (RTSP) or video file — not both
    source_type = (agent.fields.get("source_type") or "").strip().lower()
    if source_type == "video_file":
        add_unique("video_path")
    else:
        add_unique("camera_id")

    zone_support = rule.get("zone_support", {})
    requires_zone = bool(zone_support.get("required", False))

    run_mode = agent.fields.get("run_mode", "continuous")

    execution_modes = rule.get("execution_modes", {})

    if run_mode not in execution_modes:
        defaults = rule.get("defaults", {})
        run_mode = defaults.get("run_mode", "continuous")
        agent.fields["run_mode"] = run_mode

    execution_config = execution_modes.get(run_mode, {})

    execution_required = execution_config.get("required_fields", []) or []

    execution_zone_required = execution_config.get("zone_required", False)
    if execution_zone_required or requires_zone:
        add_unique("zone")
    else:
        # Special case: For counting rules, zone is optional but we should ask about it
        # This allows users to draw a line for crossing detection
        rule_id = rule.get("rule_id")
        if rule_id in ["class_count", "box_count"]:
            zone_value = agent.fields.get("zone")
            zone_is_empty = not zone_value or (isinstance(zone_value, (dict, list)) and not zone_value)
            if zone_is_empty:
                add_unique("zone")

    for f in execution_required:
        add_unique(f)

    # For video file agents: no start/end time — agent finishes when video ends
    time_window = rule.get("time_window", {})
    if time_window.get("required", False) and source_type != "video_file":
        add_unique("start_time")
        add_unique("end_time")

    # Zone: treat None or empty dict/list as missing; video_path: None or empty string; others: None only
    missing = []
    for f in ordered_required:
        value = agent.fields.get(f)
        if f == "zone":
            if not value or (isinstance(value, (dict, list)) and not value):
                missing.append(f)
        elif f == "video_path":
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(f)
        else:
            if value is None:
                missing.append(f)
    agent.missing_fields = missing


def apply_rule_defaults(agent: AgentState, rule: Dict) -> None:
    """
    Apply deterministic defaults from the rule definition.

    Dynamically applies ALL defaults from the knowledge base - no hardcoding.

    Args:
        agent: The agent state to apply defaults to
        rule: The rule definition from knowledge base
    """
    if agent.fields.get("model") is None:
        agent.fields["model"] = rule.get("model")

    defaults = rule.get("defaults", {})
    for field_name, default_value in defaults.items():
        if agent.fields.get(field_name) is None:
            agent.fields[field_name] = default_value


def compute_requires_zone(rule: Dict, run_mode: str) -> bool:
    """
    Compute whether zone is required based on KB structure.

    This is derived state - should NOT be stored in agent.fields.

    Args:
        rule: The rule definition from knowledge base
        run_mode: Current run_mode value

    Returns:
        True if zone is required, False otherwise
    """
    zone_support = rule.get("zone_support", {})
    requires_zone = bool(zone_support.get("required", False))

    execution_modes = rule.get("execution_modes", {})
    execution_config = execution_modes.get(run_mode, {})
    execution_zone_required = execution_config.get("zone_required", False)

    return execution_zone_required or requires_zone
