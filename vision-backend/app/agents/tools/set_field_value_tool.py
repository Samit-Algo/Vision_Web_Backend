from __future__ import annotations

import json
from typing import Dict

from ..session_state.agent_state import AgentState, get_agent_state
from .kb_utils import compute_missing_fields, get_rule


# ============================================================================
# FIELD UPDATE
# ============================================================================

def set_field_value(field_values_json: str, session_id: str = "default") -> Dict:
    """
    Update one or more fields on the current agent state.

    Args:
        field_values_json: JSON string mapping field name -> value
        session_id: Session identifier for state management

    Returns:
        Dict with updated_fields, status, and message
    """
    try:
        field_values = json.loads(field_values_json) if field_values_json else {}
    except json.JSONDecodeError as e:
        return {
            "error": f"Invalid JSON in field_values_json: {str(e)}",
            "updated_fields": [],
            "status": "COLLECTING",
            "message": "Failed to parse field values. Please check the format."
        }

    if not isinstance(field_values, dict):
        return {
            "error": "field_values_json must decode to an object/dict",
            "updated_fields": [],
            "status": "COLLECTING",
            "message": "Invalid field values format."
        }

    print(f"[set_field_value] Called with session_id={session_id}, field_values_json={field_values_json}")
    try:
        agent = get_agent_state(session_id)
        print(f"[set_field_value] Current agent state - status: {agent.status}, rule_id: {agent.rule_id}, missing_fields: {agent.missing_fields}")

        if not agent.rule_id:
            return {
                "error": "Cannot set fields before rule selection. Call initialize_state first.",
                "updated_fields": [],
                "status": "COLLECTING",
                "message": "Agent state not initialized."
            }

        rule = get_rule(agent.rule_id)

        if agent.fields.get("run_mode") is None:
            agent.fields["run_mode"] = "continuous"

        updated_fields = []
        for key, value in field_values.items():
            agent.fields[key] = value
            updated_fields.append(key)

        _update_rules_field(agent, rule)

        if not agent.fields.get("name"):
            compute_missing_fields(agent, rule)
            if not agent.missing_fields:
                rule_name = rule.get("rule_name", "Agent")
                class_name = agent.fields.get("class") or agent.fields.get("gesture") or ""
                if class_name:
                    class_display = class_name.replace("_", " ").title()
                    agent.fields["name"] = f"{class_display} {rule_name} Agent"
                else:
                    agent.fields["name"] = f"{rule_name} Agent"

        compute_missing_fields(agent, rule)

        print(f"[set_field_value] After updating fields - updated_fields: {updated_fields}, missing_fields: {agent.missing_fields}, current_status: {agent.status}")

        if updated_fields and not agent.missing_fields:
            agent.status = "CONFIRMATION"
            print(f"[set_field_value] Status changed to CONFIRMATION - all fields collected")
        elif not updated_fields:
            print(f"[set_field_value] No fields updated - keeping status: {agent.status}")
            pass
        else:
            agent.status = "COLLECTING"
            print(f"[set_field_value] Status set to COLLECTING - still missing fields: {agent.missing_fields}")

        result = {
            "updated_fields": updated_fields,
            "status": agent.status,
            "message": f"Updated {len(updated_fields)} field(s). Check CURRENT AGENT STATE in instruction for current missing_fields.",
        }
        print(f"[set_field_value] Returning result: {result}")
        return result
    except ValueError as e:
        return {
            "error": str(e),
            "updated_fields": [],
            "status": agent.status if 'agent' in locals() else "COLLECTING",
            "message": f"Error: {str(e)}"
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "updated_fields": [],
            "status": agent.status if 'agent' in locals() else "COLLECTING",
            "message": f"An error occurred while updating fields: {str(e)}"
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _update_rules_field(agent: AgentState, rule: Dict) -> None:
    """
    Dynamically update the rules array with complete configuration based on knowledge base definition.

    Special handling:
    - "label" field: If in rules_config_fields but not provided, auto-generate from "class" if available
    """
    rule_id = agent.rule_id
    rule_config = {"type": rule_id}

    rules_config_fields = rule.get("rules_config_fields", [])

    for field_name in rules_config_fields:
        if field_name == "label" and field_name not in agent.fields:
            if agent.fields.get("class"):
                class_name = agent.fields["class"].replace("_", " ").title()
                rule_config["label"] = f"{class_name} detection"
        elif field_name in agent.fields:
            value = agent.fields[field_name]
            if value is not None:
                rule_config[field_name] = value

    agent.fields["rules"] = [rule_config]
