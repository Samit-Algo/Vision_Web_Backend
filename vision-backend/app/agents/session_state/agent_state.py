from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    """
    Deterministic state container for the agent-building flow.
    This acts as the single source of truth for rule selection,
    collected fields, and the overall progress status.
    """

    rule_id: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    # COLLECTING -> CONFIRMATION -> DONE
    status: str = "COLLECTING"
    user_id: Optional[str] = None  # Track which user this state belongs to


# Module-level store so the ADK-managed session can share state across turns.
# In a multi-user environment, this should be keyed by session_id or user_id
_AGENT_STATES: Dict[str, AgentState] = {}


def get_agent_state(session_id: str = "default") -> AgentState:
    """
    Retrieve the current agent state for a session, creating a fresh one if none exists.
    """
    if session_id not in _AGENT_STATES:
        _AGENT_STATES[session_id] = AgentState()
    return _AGENT_STATES[session_id]


def set_agent_state(state: AgentState, session_id: str = "default") -> None:
    """
    Replace the current agent state for a session with a new instance.
    """
    _AGENT_STATES[session_id] = state


def reset_agent_state(session_id: str = "default") -> AgentState:
    """
    Reset to a brand-new agent state for a session and return it.
    """
    _AGENT_STATES[session_id] = AgentState()
    return _AGENT_STATES[session_id]
