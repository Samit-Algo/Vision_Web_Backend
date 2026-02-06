"""
Custom exception hierarchy for Vision Agent system - Simplified.

Only includes exceptions that are actually used in the codebase.
"""

from typing import Optional, Dict, Any


# ============================================================================
# BASE EXCEPTION
# ============================================================================

class VisionAgentError(Exception):
    """Base exception for all Vision Agent errors."""
    
    def __init__(
        self,
        message: str,
        user_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.user_message = user_message or "An error occurred. Please try again."
        self.details = details or {}


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================

class ValidationError(VisionAgentError):
    """Raised when input validation fails."""
    pass


class InvalidStateTransitionError(VisionAgentError):
    """Raised when attempting an invalid state transition."""
    pass


# ============================================================================
# STATE MANAGEMENT EXCEPTIONS
# ============================================================================

class StateError(VisionAgentError):
    """Base exception for state management errors."""
    pass


class StateNotInitializedError(StateError):
    """Raised when attempting to access uninitialized state."""
    pass


class StateLockError(StateError):
    """Raised when state lock cannot be acquired."""
    pass


# ============================================================================
# DATABASE EXCEPTIONS
# ============================================================================

class DatabaseError(VisionAgentError):
    """Base exception for database errors."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        retryable: bool = True,
        **kwargs
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
            **kwargs
        )


class DatabaseTimeoutError(DatabaseError):
    """Raised when database operation times out."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            user_message="Database operation timed out. Please try again.",
            retryable=True,
            **kwargs
        )


# ============================================================================
# EXTERNAL SERVICE EXCEPTIONS
# ============================================================================

class ExternalServiceError(VisionAgentError):
    """Base exception for external service errors."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        retryable: bool = True,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.retryable = retryable


class JetsonRegistrationError(ExternalServiceError):
    """Raised when Jetson backend registration fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            service_name="Jetson",
            user_message="Failed to register agent with processing backend.",
            retryable=True,
            **kwargs
        )


class CameraServiceError(ExternalServiceError):
    """Raised when camera service operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            service_name="Camera",
            user_message="Camera service error. Please try again.",
            retryable=True,
            **kwargs
        )


# ============================================================================
# REPOSITORY EXCEPTIONS  
# ============================================================================

class RepositoryNotInitializedError(VisionAgentError):
    """Raised when a repository is not initialized."""
    
    def __init__(self, repository_name: str):
        super().__init__(
            f"{repository_name} repository not initialized",
            user_message="System configuration error. Please contact support.",
            details={"repository": repository_name}
        )
        self.repository_name = repository_name


# ============================================================================
# KNOWLEDGE BASE EXCEPTIONS
# ============================================================================

class KnowledgeBaseError(VisionAgentError):
    """Base exception for knowledge base errors."""
    pass


class RuleNotFoundError(KnowledgeBaseError):
    """Raised when a rule is not found in knowledge base."""
    
    def __init__(self, rule_id: str):
        super().__init__(
            f"Rule not found: {rule_id}",
            user_message="The requested rule is not available.",
            details={"rule_id": rule_id}
        )
        self.rule_id = rule_id
