from __future__ import annotations

import json
from typing import Dict

from ...exceptions import StateNotInitializedError, ValidationError, VisionAgentError
from ..state.agent_state import AgentState, STEP_CONFIRMATION, get_agent_state
from .knowledge_base import can_enter_confirmation, compute_missing_fields, get_current_step, get_rule


def update_rules_config(agent: AgentState, rule: Dict) -> None:
    rule_id = agent.rule_id
    config = {"type": rule_id}
    for field_name in rule.get("rules_config_fields", []):
        if field_name == "label" and field_name not in agent.fields and agent.fields.get("class"):
            config["label"] = f"{agent.fields['class'].replace('_', ' ').title()} detection"
        elif field_name in agent.fields and agent.fields[field_name] is not None:
            config[field_name] = agent.fields[field_name]
    if rule_id in ["class_count", "box_count"] and agent.fields.get("zone") is not None:
        config["zone"] = agent.fields["zone"]
    agent.fields["rules"] = [config]


def set_field_value(field_values_json: str, session_id: str = "default") -> Dict:
    try:
        field_values = json.loads(field_values_json or "{}")
    except json.JSONDecodeError:
        raise ValidationError("Invalid JSON", user_message="Invalid field values format.")
    if not isinstance(field_values, dict):
        raise ValidationError("Expected JSON object", user_message="Invalid field values format.")

    agent = get_agent_state(session_id)
    if not agent.rule_id:
        raise StateNotInitializedError(
            "No rule selected",
            user_message="Please select what you want to create first, then provide the details.",
        )
    rule = get_rule(agent.rule_id)

    try:
        if agent.fields.get("run_mode") is None:
            agent.fields["run_mode"] = "continuous"
        updated = []
        for key, value in field_values.items():
            agent.fields[key] = value
            updated.append(key)
        update_rules_config(agent, rule)
        if not agent.fields.get("name"):
            compute_missing_fields(agent, rule)
            if not agent.missing_fields:
                rule_name = rule.get("rule_name", "Agent")
                class_name = agent.fields.get("class") or agent.fields.get("gesture") or ""
                agent.fields["name"] = f"{class_name.replace('_', ' ').title()} {rule_name} Agent" if class_name else f"{rule_name} Agent"
        compute_missing_fields(agent, rule)
        if updated and not agent.missing_fields and can_enter_confirmation(agent, rule):
            agent.status = "CONFIRMATION"
            agent.current_step = STEP_CONFIRMATION
        elif updated:
            agent.status = "COLLECTING"
        if updated:
            agent.current_step = get_current_step(agent, rule)
        return {"updated_fields": updated, "status": agent.status, "message": "Updated."}
    except (ValidationError, StateNotInitializedError, VisionAgentError):
        raise
    except Exception as e:
        raise VisionAgentError(str(e), user_message="Something went wrong. Please try again.") from e
