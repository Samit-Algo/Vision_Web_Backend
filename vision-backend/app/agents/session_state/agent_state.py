"""
Session-scoped agent state for the agent-creation flow.

Holds rule selection, collected fields, missing fields, and status.
Thread-safe storage keyed by session_id.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Data class
# -----------------------------------------------------------------------------


@dataclass
class AgentState:
    """
    Deterministic state container for the agent-building flow.

    Single source of truth for rule selection, collected fields, and progress.
    Status: UNINITIALIZED -> COLLECTING -> CONFIRMATION -> SAVED.
    """

    rule_id: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    status: str = "COLLECTING"
    user_id: Optional[str] = None
    saved_agent_id: Optional[str] = None
    saved_agent_name: Optional[str] = None


# -----------------------------------------------------------------------------
# Thread-safe storage
# -----------------------------------------------------------------------------

agent_states: Dict[str, AgentState] = {}
state_lock = threading.RLock()


# -----------------------------------------------------------------------------
# State API
# -----------------------------------------------------------------------------


def get_agent_state(session_id: str = "default") -> AgentState:
    """Return the current agent state for the session; create one if missing."""
    with state_lock:
        if session_id not in agent_states:
            agent_states[session_id] = AgentState()
        return agent_states[session_id]


def set_agent_state(state: AgentState, session_id: str = "default") -> None:
    """Replace the agent state for the session with the given instance."""
    with state_lock:
        agent_states[session_id] = state


def reset_agent_state(session_id: str = "default") -> AgentState:
    """Reset to a fresh agent state for the session and return it."""
    with state_lock:
        agent_states[session_id] = AgentState()
        return agent_states[session_id]
