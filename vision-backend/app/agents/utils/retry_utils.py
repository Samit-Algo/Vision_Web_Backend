"""
Retry utilities for resilient database and API operations - Simplified.

Provides retry decorators with exponential backoff and jitter.
Only includes functionality that is actually used.
"""

import asyncio
import functools
import logging
import time
from typing import Callable, Tuple, Type

from ..exceptions import (
    DatabaseError,
    DatabaseConnectionError,
    DatabaseTimeoutError,
    ExternalServiceError,
)


logger = logging.getLogger(__name__)


# ============================================================================
# RETRY DECORATORS
# ============================================================================

def retry_on_exception(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (DatabaseConnectionError, DatabaseTimeoutError, ExternalServiceError)
) -> Callable:
    """
    Decorator for retrying functions on specific exceptions.
    
    Implements exponential backoff with optional jitter.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorated function
        
    Example:
        @retry_on_exception(max_retries=3)
        def fetch_data():
            # This will retry up to 3 times on connection errors
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Check if exception is retryable
                    if hasattr(e, 'retryable') and not e.retryable:
                        logger.warning(f"{func.__name__}: Non-retryable error: {e}")
                        raise
                    
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                        
                        # Add jitter to prevent thundering herd
                        if jitter:
                            import random
                            delay = delay * (0.5 + random.random())
                        
                        logger.warning(
                            f"{func.__name__}: Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__}: All {max_retries + 1} attempts failed. Last error: {e}"
                        )
            
            # All retries exhausted
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def async_retry_on_exception(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (DatabaseConnectionError, DatabaseTimeoutError, ExternalServiceError)
) -> Callable:
    """
    Async version of retry_on_exception decorator.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorated async function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Check if exception is retryable
                    if hasattr(e, 'retryable') and not e.retryable:
                        logger.warning(f"{func.__name__}: Non-retryable error: {e}")
                        raise
                    
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                        
                        # Add jitter to prevent thundering herd
                        if jitter:
                            import random
                            delay = delay * (0.5 + random.random())
                        
                        logger.warning(
                            f"{func.__name__}: Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__}: All {max_retries + 1} attempts failed. Last error: {e}"
                        )
            
            # All retries exhausted
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator
