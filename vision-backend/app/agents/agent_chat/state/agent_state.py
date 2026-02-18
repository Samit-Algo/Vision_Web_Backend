from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

STEP_CAMERA = "camera"
STEP_ZONE = "zone"
STEP_TIME_WINDOW = "time_window"
STEP_CONFIRMATION = "confirmation"
STEP_COMPLETED = "completed"

STORAGE: Dict[str, AgentState] = {}
LOCK = threading.RLock()


@dataclass
class AgentState:
    rule_id: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    status: str = "COLLECTING"
    current_step: Optional[str] = None
    user_id: Optional[str] = None
    saved_agent_id: Optional[str] = None
    saved_agent_name: Optional[str] = None


def get_agent_state(session_id: str = "default") -> AgentState:
    with LOCK:
        if session_id not in STORAGE:
            STORAGE[session_id] = AgentState()
        return STORAGE[session_id]


def set_agent_state(state: AgentState, session_id: str = "default") -> None:
    with LOCK:
        STORAGE[session_id] = state


def reset_agent_state(session_id: str = "default") -> AgentState:
    with LOCK:
        STORAGE[session_id] = AgentState()
        return STORAGE[session_id]
