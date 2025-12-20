from __future__ import annotations

import json
import os
from typing import Dict, List
from pathlib import Path

from ..session_state.agent_state import AgentState, reset_agent_state, get_agent_state


# Load knowledge base once
KB_PATH = Path(__file__).resolve().parent.parent.parent / "knowledge_base" / "vision_rule_knowledge_base.json"

with open(KB_PATH, "r", encoding="utf-8") as f:
    _KB_RULES: List[Dict] = json.load(f)["rules"]


def _get_rule(rule_id: str) -> Dict:
    for rule in _KB_RULES:
        if rule.get("rule_id") == rule_id:
            return rule
    raise ValueError(f"Unknown rule_id: {rule_id}")


def _compute_missing_fields(agent: AgentState, rule: Dict) -> None:
    """
    Deterministic computation of missing fields.
    Mirrors the previous app logic without LLM involvement.
    """
    required_fields = set(rule.get("required_fields_from_user", []))
    required_fields.add("camera_id")

    requires_zone = bool(rule.get("requires_zone", False))
    agent.fields["requires_zone"] = requires_zone

    run_mode = agent.fields.get("run_mode")
    # run_mode defaults to "continuous" during initialization, so we don't require it from user
    # User can change it if they want patrol mode

    # Patrol needs timing granularity; zone is required either by rule or patrol mode
    # For continuous mode, interval_minutes and check_duration_seconds are NOT required
    if run_mode == "patrol":
        required_fields.update({"interval_minutes", "check_duration_seconds", "zone"})
    elif requires_zone:
        required_fields.add("zone")
    # If run_mode is "continuous", do NOT require interval_minutes or check_duration_seconds

    agent.missing_fields = [
        field for field in required_fields if agent.fields.get(field) is None
    ]


def _apply_rule_defaults(agent: AgentState, rule: Dict) -> None:
    """
    Apply deterministic defaults from the rule definition.
    Dynamically applies ALL defaults from the knowledge base - no hardcoding.
    """
    # Apply model (top-level in rule, not in defaults)
    if agent.fields.get("model") is None:
        agent.fields["model"] = rule.get("model")

    # Set run_mode default to "continuous" if not provided
    if agent.fields.get("run_mode") is None:
        agent.fields["run_mode"] = "continuous"

    # Dynamically apply ALL defaults from the knowledge base
    defaults = rule.get("defaults", {})
    for field_name, default_value in defaults.items():
        # Only apply default if field is not already set
        if agent.fields.get(field_name) is None:
            agent.fields[field_name] = default_value


def initialize_state(rule_id: str, session_id: str = "default") -> Dict:
    """
    Initialize a brand-new agent state for the selected rule.

    Args:
        rule_id: The rule ID to initialize
        session_id: Session identifier for state management

    Returns a lightweight summary for the calling agent:
    {
        "rule_id": ...,
        "missing_fields": [...],
        "status": "COLLECTING"
    }
    """
    rule = _get_rule(rule_id)

    agent = reset_agent_state(session_id)
    agent.rule_id = rule_id
    agent.fields["rules"] = [{"type": rule_id}]
    _apply_rule_defaults(agent, rule)
    agent.status = "COLLECTING"

    _compute_missing_fields(agent, rule)

    return {
        "rule_id": agent.rule_id,
        "status": agent.status,
        "message": "Agent state initialized. Check CURRENT AGENT STATE in instruction for missing_fields.",
    }
