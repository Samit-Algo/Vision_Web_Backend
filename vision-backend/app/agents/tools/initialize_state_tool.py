from __future__ import annotations

from typing import Dict, Optional

from ..session_state.agent_state import get_agent_state, reset_agent_state
from .kb_utils import apply_rule_defaults, compute_missing_fields, get_rule


# ============================================================================
# STATE INITIALIZATION
# ============================================================================

def initialize_state(rule_id: str, session_id: str = "default", user_id: Optional[str] = None) -> Dict:
    """
    Initialize a brand-new agent state for the selected rule.

    Args:
        rule_id: The rule ID to initialize
        session_id: Session identifier for state management
        user_id: Optional user ID to store in agent state (for camera selection)

    Returns:
        Dict with rule_id, status, and message
    """
    try:
        rule = get_rule(rule_id)
    except ValueError as e:
        return {
            "error": str(e),
            "rule_id": None,
            "status": "COLLECTING",
            "message": f"Invalid rule_id: {rule_id}. Please check the knowledge base for available rules."
        }

    try:
        agent = reset_agent_state(session_id)
        agent.rule_id = rule_id
        agent.fields["rules"] = [{"type": rule_id}]

        if user_id:
            agent.user_id = user_id

        apply_rule_defaults(agent, rule)
        agent.status = "COLLECTING"

        compute_missing_fields(agent, rule)

        return {
            "rule_id": agent.rule_id,
            "status": agent.status,
            "message": "Agent state initialized. Check CURRENT AGENT STATE in instruction for missing_fields.",
        }
    except Exception as e:
        return {
            "error": f"Unexpected error initializing state: {str(e)}",
            "rule_id": None,
            "status": "COLLECTING",
            "message": f"Failed to initialize agent state: {str(e)}"
        }
