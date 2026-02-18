from __future__ import annotations

from typing import Dict, Optional

from ...exceptions import VisionAgentError
from ..state.agent_state import get_agent_state, reset_agent_state
from .knowledge_base import apply_rule_defaults, compute_missing_fields, get_current_step, get_rule


def initialize_state(
    rule_id: str,
    session_id: str = "default",
    user_id: Optional[str] = None,
) -> Dict:
    rule = get_rule(rule_id)
    prev = get_agent_state(session_id)
    preserved_source = (prev.fields.get("source_type") or "").strip().lower()
    preserved_video = (prev.fields.get("video_path") or "").strip()

    try:
        agent = reset_agent_state(session_id)
        agent.rule_id = rule_id
        agent.fields["rules"] = [{"type": rule_id}]
        if user_id:
            agent.user_id = user_id
        if preserved_source == "video_file" and preserved_video:
            agent.fields["source_type"] = "video_file"
            agent.fields["video_path"] = preserved_video
        apply_rule_defaults(agent, rule)
        agent.status = "COLLECTING"
        compute_missing_fields(agent, rule)
        agent.current_step = get_current_step(agent, rule)
        return {"rule_id": agent.rule_id, "status": agent.status, "message": "Initialized."}
    except VisionAgentError:
        raise
    except Exception as e:
        raise VisionAgentError(str(e), user_message="Failed to initialize. Please try again.") from e
