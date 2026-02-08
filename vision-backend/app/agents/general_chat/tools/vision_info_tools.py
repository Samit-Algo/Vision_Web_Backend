"""
Vision rules catalog tools for General Chat Agent.
Uses shared kb_utils - no duplicate KB loading.
"""

import logging
from typing import Dict, Any

from ...tools.kb_utils import get_all_rules, get_rule

logger = logging.getLogger(__name__)


def get_vision_rules_catalog() -> str:
    """Get catalog of all available vision rules. Use when user asks what agents/detections are supported."""
    try:
        rules = get_all_rules()
        if not rules:
            return "No vision rules are currently configured in the system."
        catalog = "### Available Vision Rules\n\n"
        for rule in rules:
            name = rule.get("rule_name") or rule.get("name", "Unknown")
            rule_id = rule.get("rule_id", "Unknown")
            description = rule.get("description", "No description available.")
            catalog += f"- **{name}** (`{rule_id}`): {description}\n"
        return catalog
    except Exception as e:
        logger.exception(f"get_vision_rules_catalog: {e}")
        return "An error occurred while loading vision rules."


def get_rule_details(rule_id: str) -> str:
    """Get detailed info about a specific vision rule. Use when user asks about a detection type."""
    if not rule_id:
        return "Please provide a rule ID."
    try:
        rule = get_rule(rule_id)
        details = f"### Details for: {rule.get('rule_name') or rule.get('name', 'Unknown')}\n\n"
        details += f"{rule.get('description', '')}\n\n"
        required = rule.get("required_fields_from_user", [])
        if required:
            details += "**Information needed from you:**\n"
            for field in required:
                details += f"- {field}\n"
            details += "\n"
        model = rule.get("model")
        if model:
            details += f"**Model used:** `{model}`\n"
        return details
    except ValueError:
        return f"Rule `{rule_id}` not found. Use get_vision_rules_catalog to see available rules."
    except Exception as e:
        logger.exception(f"get_rule_details: {e}")
        return "An error occurred while loading rule details."
