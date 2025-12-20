from __future__ import annotations

import json
import os
from typing import Dict, List
from pathlib import Path

from ..session_state.agent_state import AgentState, get_agent_state


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
    """
    required_fields = set(rule.get("required_fields_from_user", []))
    required_fields.add("camera_id")

    requires_zone = bool(rule.get("requires_zone", False))
    agent.fields["requires_zone"] = requires_zone

    run_mode = agent.fields.get("run_mode")
    if run_mode is None:
        required_fields.add("run_mode")

    if run_mode == "patrol":
        required_fields.update({"interval_minutes", "check_duration_seconds", "zone"})
    elif requires_zone:
        required_fields.add("zone")

    agent.missing_fields = [
        field for field in required_fields if agent.fields.get(field) is None
    ]


def set_field_value(field_values_json: str, session_id: str = "default") -> Dict:
    """
    Update one or more fields on the current agent state.

    Args:
        field_values_json: JSON string mapping field name -> value.
                          Using a string avoids ADK schema issues with free-form objects.
        session_id: Session identifier for state management

    Returns a concise state summary:
    {
        "updated_fields": [...],
        "status": "COLLECTING" | "CONFIRMATION",
        "message": "Fields updated. Check CURRENT AGENT STATE in instruction for current missing_fields."
    }
    """
    try:
        field_values = json.loads(field_values_json) if field_values_json else {}
    except json.JSONDecodeError:
        raise ValueError("field_values_json must be valid JSON")

    if not isinstance(field_values, dict):
        raise ValueError("field_values_json must decode to an object/dict")

    agent = get_agent_state(session_id)
    if not agent.rule_id:
        raise ValueError("Cannot set fields before rule selection. Call initialize_state first.")

    rule = _get_rule(agent.rule_id)

    # Apply user-provided fields (LLM should already convert times to ISO 8601 format)
    updated_fields = []
    for key, value in field_values.items():
        agent.fields[key] = value
        updated_fields.append(key)

    # Update rules array when relevant fields are set
    # Check if any field in rules_config_fields was updated
    rules_config_fields = rule.get("rules_config_fields", [])
    if any(key in field_values for key in rules_config_fields):
        _update_rules_field(agent, rule)

    # Auto-generate agent name if not set and we have enough information
    if not agent.fields.get("name"):
        agent.fields["name"] = f"{rule.get('rule_name')} Agent"

    # Recompute missing fields and progress state
    # Note: Defaults are already applied during initialization, so we don't apply them again here
    _compute_missing_fields(agent, rule)
    agent.status = "CONFIRMATION" if not agent.missing_fields else "COLLECTING"

    return {
        "updated_fields": updated_fields,
        "status": agent.status,
        "message": f"Updated {len(updated_fields)} field(s). Check CURRENT AGENT STATE in instruction for current missing_fields.",
    }


def _update_rules_field(agent: AgentState, rule: Dict) -> None:
    """
    Dynamically update the rules array with complete configuration based on knowledge base definition.
    
    This function reads 'rules_config_fields' from the rule definition and automatically includes
    those fields in the rules configuration. No hardcoding - fully driven by knowledge base.
    
    Special handling:
    - "label" field: If in rules_config_fields but not provided, auto-generate from "class" if available
    """
    rule_id = agent.rule_id
    rule_config = {"type": rule_id}
    
    # Get the list of fields that should be included in rules config from knowledge base
    rules_config_fields = rule.get("rules_config_fields", [])
    
    # Dynamically add each field from the knowledge base definition
    for field_name in rules_config_fields:
        # Special case: Auto-generate "label" from "class" if label is required but not provided
        if field_name == "label" and field_name not in agent.fields:
            if agent.fields.get("class"):
                class_name = agent.fields["class"].replace("_", " ").title()
                rule_config["label"] = f"{class_name} detection"
        # For all other fields, include if they exist in agent.fields
        elif field_name in agent.fields:
            value = agent.fields[field_name]
            # Only include non-None values
            if value is not None:
                rule_config[field_name] = value
    
    # Update rules array
    agent.fields["rules"] = [rule_config]
