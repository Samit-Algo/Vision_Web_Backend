from __future__ import annotations

import logging
from typing import Dict, Optional

from ..session_state.agent_state import get_agent_state, reset_agent_state
from ..exceptions import ValidationError, RuleNotFoundError, StateError
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
        Dict with rule_id, status, and message
        
    Raises:
        ValidationError: If rule_id is invalid
        RuleNotFoundError: If rule is not found
        StateError: If state initialization fails
    """
    if not rule_id:
        logger.error("initialize_state called without rule_id")
        raise ValidationError(
            "rule_id is required",
            user_message="Rule ID is required to initialize agent."
        )
    
    logger.info(f"Initializing state: rule_id={rule_id}, session={session_id}, user_id={user_id}")
    
    # Get rule from knowledge base
    try:
        rule = get_rule(rule_id)
        logger.debug(f"Retrieved rule: {rule_id}")
    except ValueError as e:
        logger.error(f"Rule not found: {rule_id}: {e}")
        raise RuleNotFoundError(rule_id)

    # Preserve source_type and video_path if set by request (e.g. "create agent for this video") before reset
    prev_state = get_agent_state(session_id)
    preserved_source_type = (prev_state.fields.get("source_type") or "").strip().lower()
    preserved_video_path = (prev_state.fields.get("video_path") or "").strip()

    # Initialize state
    try:
        agent = reset_agent_state(session_id)
        agent.rule_id = rule_id
        agent.fields["rules"] = [{"type": rule_id}]

        if user_id:
            agent.user_id = user_id
            logger.debug(f"Set user_id={user_id} for session {session_id}")

        if preserved_source_type == "video_file" and preserved_video_path:
            agent.fields["source_type"] = "video_file"
            agent.fields["video_path"] = preserved_video_path
            logger.debug(f"Preserved video file source for session {session_id}: video_path={preserved_video_path[:50]}...")

        # Apply defaults and compute missing fields
        apply_rule_defaults(agent, rule)
        agent.status = "COLLECTING"
        compute_missing_fields(agent, rule)

        logger.info(
            f"State initialized: session={session_id}, rule={rule_id}, "
            f"missing_fields={agent.missing_fields}"
        )

        return {
            "rule_id": agent.rule_id,
            "status": agent.status,
            "message": "Agent state initialized. Check CURRENT AGENT STATE in instruction for missing_fields.",
        }
        
    except Exception as e:
        logger.exception(f"Unexpected error initializing state for session {session_id}: {e}")
        raise StateError(
            f"Failed to initialize state: {str(e)}",
            user_message="Failed to initialize agent state. Please try again."
        )
