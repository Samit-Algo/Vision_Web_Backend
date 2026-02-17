"""
Custom exception hierarchy for the Vision Agent system.

Used by agent tools, main agent, and chat use cases. All agent-related
exceptions inherit from VisionAgentError and can carry a user-facing message.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict, Optional


# -----------------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------------


class VisionAgentError(Exception):
    """Base exception for all Vision Agent errors."""

    def __init__(
        self,
        message: str,
        user_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.user_message = user_message or "An error occurred. Please try again."
        self.details = details or {}


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


class ValidationError(VisionAgentError):
    """Raised when input validation fails."""
    pass


class InvalidStateTransitionError(VisionAgentError):
    """Raised when attempting an invalid state transition."""
    pass


# -----------------------------------------------------------------------------
# State management
# -----------------------------------------------------------------------------


class StateError(VisionAgentError):
    """Base exception for state management errors."""
    pass


class StateNotInitializedError(StateError):
    """Raised when attempting to access uninitialized state."""
    pass


class StateLockError(StateError):
    """Raised when state lock cannot be acquired."""
    pass


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------


class DatabaseError(VisionAgentError):
    """Base exception for database errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        retryable: bool = True,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.retryable = retryable


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            user_message="Database connection failed. Please try again.",
            retryable=True,
            **kwargs,
        )


class DatabaseTimeoutError(DatabaseError):
    """Raised when database operation times out."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            user_message="Database operation timed out. Please try again.",
            retryable=True,
            **kwargs,
        )


# -----------------------------------------------------------------------------
# External services
# -----------------------------------------------------------------------------


class ExternalServiceError(VisionAgentError):
    """Base exception for external service errors."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        retryable: bool = True,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.retryable = retryable


class CameraServiceError(ExternalServiceError):
    """Raised when camera service operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            service_name="Camera",
            user_message="Camera service error. Please try again.",
            retryable=True,
            **kwargs,
        )


# -----------------------------------------------------------------------------
# Repository
# -----------------------------------------------------------------------------


class RepositoryNotInitializedError(VisionAgentError):
    """Raised when a repository is not initialized."""

    def __init__(self, repository_name: str):
        super().__init__(
            f"{repository_name} repository not initialized",
            user_message="System configuration error. Please contact support.",
            details={"repository": repository_name},
        )
        self.repository_name = repository_name


# -----------------------------------------------------------------------------
# Knowledge base
# -----------------------------------------------------------------------------


class KnowledgeBaseError(VisionAgentError):
    """Base exception for knowledge base errors."""
    pass


class RuleNotFoundError(KnowledgeBaseError):
    """Raised when a rule is not found in the knowledge base."""

    def __init__(self, rule_id: str):
        super().__init__(
            f"Rule not found: {rule_id}",
            user_message="The requested rule is not available.",
            details={"rule_id": rule_id},
        )
        self.rule_id = rule_id


# -----------------------------------------------------------------------------
# Safe user-facing message
# -----------------------------------------------------------------------------

def get_user_message(exc: BaseException) -> str:
    """
    Return a safe, user-facing message for any exception.
    Use this at API/use-case boundaries so internal details are never exposed.
    """
    if isinstance(exc, VisionAgentError) and getattr(exc, "user_message", None):
        return exc.user_message
    return "Something went wrong. Please try again."
