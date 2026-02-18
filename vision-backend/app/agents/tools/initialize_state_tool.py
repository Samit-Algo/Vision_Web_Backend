from __future__ import annotations

import logging
from typing import Dict, Optional

from ..exceptions import VisionAgentError
from ..session_state.agent_state import get_agent_state, reset_agent_state
from .kb_utils import apply_rule_defaults, compute_missing_fields, get_rule

logger = logging.getLogger(__name__)


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
        Dict with rule_id, status, and message (success path only).

    Raises:
        RuleNotFoundError: rule_id not in knowledge base (from get_rule).
        VisionAgentError: Unexpected error (with safe user_message).
    """
    rule = get_rule(rule_id)

    prev_state = get_agent_state(session_id)
    preserved_source_type = (prev_state.fields.get("source_type") or "").strip().lower()
    preserved_video_path = (prev_state.fields.get("video_path") or "").strip()

    try:
        agent = reset_agent_state(session_id)
        agent.rule_id = rule_id
        agent.fields["rules"] = [{"type": rule_id}]

        if user_id:
            agent.user_id = user_id

        if preserved_source_type == "video_file" and preserved_video_path:
            agent.fields["source_type"] = "video_file"
            agent.fields["video_path"] = preserved_video_path

        apply_rule_defaults(agent, rule)
        agent.status = "COLLECTING"
        compute_missing_fields(agent, rule)

        return {
            "rule_id": agent.rule_id,
            "status": agent.status,
            "message": "Agent state initialized. Check CURRENT AGENT STATE in instruction for missing_fields.",
        }
    except VisionAgentError:
        raise
    except Exception as e:
        logger.exception("Unexpected error initializing state: %s", e)
        raise VisionAgentError(
            str(e),
            user_message="Failed to initialize. Please try again.",
        ) from e
