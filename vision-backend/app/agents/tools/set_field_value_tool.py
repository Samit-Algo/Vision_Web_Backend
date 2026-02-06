from __future__ import annotations

import json
import logging
from typing import Dict

from ..session_state.agent_state import AgentState, get_agent_state
from ..exceptions import ValidationError, StateNotInitializedError, RuleNotFoundError
from .kb_utils import compute_missing_fields, get_rule

logger = logging.getLogger(__name__)


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
        
    Raises:
        ValidationError: If JSON is invalid or field values are invalid
        StateNotInitializedError: If state is not initialized
        RuleNotFoundError: If rule is not found
    """
    logger.debug(f"set_field_value: session={session_id}, json_length={len(field_values_json) if field_values_json else 0}")
    
    # Parse JSON
    try:
        field_values = json.loads(field_values_json) if field_values_json else {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in field_values_json: {e}")
        raise ValidationError(
            f"Invalid JSON in field_values_json: {str(e)}",
            user_message="Failed to parse field values. Please check the format."
        )

    if not isinstance(field_values, dict):
        logger.error(f"field_values_json is not a dict: {type(field_values)}")
        raise ValidationError(
            "field_values_json must decode to an object/dict",
            user_message="Invalid field values format."
        )

    # Get state and validate
    try:
        agent = get_agent_state(session_id)

        if not agent.rule_id:
            logger.error(f"State not initialized for session {session_id}")
            raise StateNotInitializedError(
                "Cannot set fields before rule selection. Call initialize_state first.",
                user_message="Agent state not initialized."
            )

        rule = get_rule(agent.rule_id)
        logger.debug(f"Retrieved rule {agent.rule_id} for field update")

        # Set default run_mode if not provided
        if agent.fields.get("run_mode") is None:
            agent.fields["run_mode"] = "continuous"
            logger.debug("Set default run_mode=continuous")

        # Update fields
        updated_fields = []
        for key, value in field_values.items():
            agent.fields[key] = value
            updated_fields.append(key)
            logger.debug(f"Updated field: {key}={value}")

        # Update rules field
        _update_rules_field(agent, rule)

        # Auto-generate name if not provided and all fields collected
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
                logger.info(f"Auto-generated agent name: {agent.fields['name']}")

        # Recompute missing fields
        compute_missing_fields(agent, rule)

        # Update status based on missing fields
        if updated_fields and not agent.missing_fields:
            agent.status = "CONFIRMATION"
            logger.info(f"Session {session_id} transitioned to CONFIRMATION")
        elif updated_fields:
            agent.status = "COLLECTING"

        logger.info(
            f"Updated {len(updated_fields)} field(s) for session {session_id}: "
            f"{updated_fields}, status={agent.status}, missing={agent.missing_fields}"
        )

        return {
            "updated_fields": updated_fields,
            "status": agent.status,
            "message": f"Updated {len(updated_fields)} field(s). Check CURRENT AGENT STATE in instruction for current missing_fields.",
        }
        
    except (ValidationError, StateNotInitializedError, RuleNotFoundError):
        # Re-raise known errors
        raise
    except ValueError as e:
        logger.error(f"ValueError in set_field_value: {e}")
        raise ValidationError(
            str(e),
            user_message=f"Error: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error in set_field_value for session {session_id}: {e}")
        raise ValidationError(
            f"Unexpected error: {str(e)}",
            user_message=f"An error occurred while updating fields: {str(e)}"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _update_rules_field(agent: AgentState, rule: Dict) -> None:
    """
    Dynamically update the rules array with complete configuration based on knowledge base definition.

    Special handling:
    - "label" field: If in rules_config_fields but not provided, auto-generate from "class" if available
    - "zone" field: For counting rules (class_count, box_count), include zone in rule config if it exists
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

    # For counting rules (class_count, box_count), include zone in rule config if it exists
    if rule_id in ["class_count", "box_count"]:
        zone = agent.fields.get("zone")
        if zone is not None:
            rule_config["zone"] = zone

    agent.fields["rules"] = [rule_config]
