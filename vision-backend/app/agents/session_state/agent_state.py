from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# DATA CLASSES
# ============================================================================

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
    status: str = "COLLECTING"  # UNINITIALIZED -> COLLECTING -> CONFIRMATION -> SAVED
    user_id: Optional[str] = None
    saved_agent_id: Optional[str] = None
    saved_agent_name: Optional[str] = None


# ============================================================================
# THREAD-SAFE STATE STORAGE
# ============================================================================

_AGENT_STATES: Dict[str, AgentState] = {}
_STATE_LOCK = threading.RLock()


# ============================================================================
# STATE MANAGEMENT FUNCTIONS
# ============================================================================

def get_agent_state(session_id: str = "default") -> AgentState:
    """
    Retrieve the current agent state for a session, creating a fresh one if none exists.

    Args:
        session_id: Session identifier

    Returns:
        AgentState: Current agent state for the session
    """
    with _STATE_LOCK:
        if session_id not in _AGENT_STATES:
            _AGENT_STATES[session_id] = AgentState()
        return _AGENT_STATES[session_id]


def set_agent_state(state: AgentState, session_id: str = "default") -> None:
    """
    Replace the current agent state for a session with a new instance.

    Args:
        state: New agent state to set
        session_id: Session identifier
    """
    with _STATE_LOCK:
        _AGENT_STATES[session_id] = state


def reset_agent_state(session_id: str = "default") -> AgentState:
    """
    Reset to a brand-new agent state for a session and return it.

    Args:
        session_id: Session identifier

    Returns:
        AgentState: Fresh agent state instance
    """
    with _STATE_LOCK:
        _AGENT_STATES[session_id] = AgentState()
        return _AGENT_STATES[session_id]
