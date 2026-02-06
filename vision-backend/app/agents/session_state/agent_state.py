from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..exceptions import StateError, StateLockError

logger = logging.getLogger(__name__)


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
    status: str = "COLLECTING"  # COLLECTING -> CONFIRMATION -> SAVED
    user_id: Optional[str] = None
    saved_agent_id: Optional[str] = None
    saved_agent_name: Optional[str] = None
    flow_diagram_shown: bool = False  # Track if flow diagram was already displayed
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# THREAD-SAFE STATE STORAGE
# ============================================================================

class StateManager:
    """
    Thread-safe manager for agent states.
    
    Implements:
    - Thread-safe access to states
    - Automatic cleanup of expired sessions
    - Lock management to prevent race conditions
    """
    
    def __init__(self, session_ttl_minutes: int = 60):
        """
        Initialize StateManager.
        
        Args:
            session_ttl_minutes: Time-to-live for inactive sessions (minutes)
        """
        self._states: Dict[str, AgentState] = {}
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()
        self._session_ttl = timedelta(minutes=session_ttl_minutes)
        logger.info(f"StateManager initialized with {session_ttl_minutes}min TTL")
    
    def _get_session_lock(self, session_id: str) -> threading.RLock:
        """
        Get or create a lock for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            RLock for the session
        """
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.RLock()
            return self._locks[session_id]
    
    def get_state(self, session_id: str = "default") -> AgentState:
        """
        Thread-safe retrieval of agent state.
        
        Creates a new state if none exists for the session.
        Updates last_accessed timestamp.
        
        Args:
            session_id: Session identifier
            
        Returns:
            AgentState for the session
        """
        session_lock = self._get_session_lock(session_id)
        
        with session_lock:
            if session_id not in self._states:
                logger.info(f"Creating new state for session: {session_id}")
                self._states[session_id] = AgentState()
            
            # Update last accessed time
            self._states[session_id].last_accessed = datetime.now()
            
            return self._states[session_id]
    
    def set_state(self, state: AgentState, session_id: str = "default") -> None:
        """
        Thread-safe replacement of agent state.
        
        Args:
            state: New agent state
            session_id: Session identifier
        """
        session_lock = self._get_session_lock(session_id)
        
        with session_lock:
            state.last_accessed = datetime.now()
            self._states[session_id] = state
            logger.debug(f"State updated for session: {session_id}")
    
    def reset_state(self, session_id: str = "default") -> AgentState:
        """
        Thread-safe reset of agent state to fresh instance.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Fresh AgentState instance
        """
        session_lock = self._get_session_lock(session_id)
        
        with session_lock:
            logger.info(f"Resetting state for session: {session_id}")
            new_state = AgentState()
            self._states[session_id] = new_state
            return new_state
    
    def delete_state(self, session_id: str) -> bool:
        """
        Delete a session state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if state was deleted, False if it didn't exist
        """
        session_lock = self._get_session_lock(session_id)
        
        with session_lock:
            if session_id in self._states:
                del self._states[session_id]
                logger.info(f"State deleted for session: {session_id}")
                
                # Clean up lock as well
                with self._global_lock:
                    if session_id in self._locks:
                        del self._locks[session_id]
                
                return True
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions based on TTL.
        
        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now()
        expired_sessions = []
        
        # First pass: identify expired sessions
        with self._global_lock:
            for session_id, state in self._states.items():
                if now - state.last_accessed > self._session_ttl:
                    expired_sessions.append(session_id)
        
        # Second pass: delete expired sessions
        for session_id in expired_sessions:
            self.delete_state(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_session_count(self) -> int:
        """
        Get the current number of active sessions.
        
        Returns:
            Number of active sessions
        """
        with self._global_lock:
            return len(self._states)
    
    def get_all_session_ids(self) -> List[str]:
        """
        Get list of all active session IDs.
        
        Returns:
            List of session IDs
        """
        with self._global_lock:
            return list(self._states.keys())


# ============================================================================
# GLOBAL STATE MANAGER INSTANCE
# ============================================================================

_state_manager = StateManager(session_ttl_minutes=60)


# ============================================================================
# PUBLIC API (backwards compatible)
# ============================================================================

def get_agent_state(session_id: str = "default") -> AgentState:
    """
    Retrieve the current agent state for a session, creating a fresh one if none exists.

    Thread-safe operation.

    Args:
        session_id: Session identifier

    Returns:
        AgentState: Current agent state for the session
    """
    return _state_manager.get_state(session_id)


def set_agent_state(state: AgentState, session_id: str = "default") -> None:
    """
    Replace the current agent state for a session with a new instance.

    Thread-safe operation.

    Args:
        state: New agent state to set
        session_id: Session identifier
    """
    _state_manager.set_state(state, session_id)


def reset_agent_state(session_id: str = "default") -> AgentState:
    """
    Reset to a brand-new agent state for a session and return it.

    Thread-safe operation.

    Args:
        session_id: Session identifier

    Returns:
        AgentState: Fresh agent state instance
    """
    return _state_manager.reset_state(session_id)


def cleanup_expired_sessions() -> int:
    """
    Cleanup expired sessions based on TTL.
    
    Should be called periodically (e.g., via background task).
    
    Returns:
        Number of sessions cleaned up
    """
    return _state_manager.cleanup_expired_sessions()


def get_active_session_count() -> int:
    """
    Get the number of currently active sessions.
    
    Returns:
        Number of active sessions
    """
    return _state_manager.get_session_count()
