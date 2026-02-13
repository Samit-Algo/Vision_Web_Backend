"""
Tools for the General Chat Agent to provide information about the Vision AI system.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Constants
KB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "knowledge_base" / "vision_rule_knowledge_base.json"

def _load_kb() -> Dict[str, Any]:
    """Load the vision rule knowledge base."""
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Error loading knowledge base: %s", e)
        return {"rules": []}

def get_vision_rules_catalog() -> str:
    """
    Get a catalog of all available vision rules (AI agent types).
    Use this when the user asks what kind of agents or detections are supported.
    
    Returns:
        A markdown-formatted string listing available rules.
    """
    kb = _load_kb()
    rules = kb.get("rules", [])
    
    if not rules:
        return "No vision rules are currently configured in the system."
    
    catalog = "### Available Vision Rules\n\n"
    for rule in rules:
        name = rule.get("name", "Unknown")
        rule_id = rule.get("rule_id", "Unknown")
        description = rule.get("description", "No description available.")
        catalog += f"- **{name}** (`{rule_id}`): {description}\n"
    
    return catalog

def get_rule_details(rule_id: str) -> str:
    """
    Get detailed information about a specific vision rule.
    Use this when a user asks about a specific detection type or what information is needed for it.
    
    Args:
        rule_id: The unique ID of the rule (e.g., 'class_presence', 'intrusion_detection')
        
    Returns:
        Detailed information about the rule requirements.
    """
    kb = _load_kb()
    rules = kb.get("rules", [])
    
    selected_rule = next((r for r in rules if r.get("rule_id") == rule_id), None)
    
    if not selected_rule:
        return f"Rule with ID `{rule_id}` not found. Please ask for the catalog to see available rules."
    
    details = f"### Details for: {selected_rule.get('name')}\n\n"
    details += f"{selected_rule.get('description')}\n\n"
    
    # Required fields
    required = selected_rule.get("required_fields_from_user", [])
    if required:
        details += "**Information needed from you:**\n"
        for field in required:
            details += f"- {field}\n"
        details += "\n"
        
    # Model info
    model = selected_rule.get("model")
    if model:
        details += f"**Model used:** `{model}`\n"
        
    return details
