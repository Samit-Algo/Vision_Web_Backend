"""HTTP client factory for connection pooling."""
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global shared HTTP client instance
_shared_client: Optional[httpx.AsyncClient] = None


def get_shared_http_client() -> httpx.AsyncClient:
    """
    Get or create shared async HTTP client for connection pooling.
    
    This client should be reused across all services to benefit from:
    - Connection pooling
    - Keep-alive connections
    - Reduced overhead
    
    Returns:
        Shared AsyncClient instance
    """
    global _shared_client
    
    if _shared_client is None:
        _shared_client = httpx.AsyncClient(
            timeout=120.0,  # Max timeout for long operations
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0,
            ),
            http2=True,  # Enable HTTP/2 for better performance
        )
        logger.info("Created shared HTTP client for connection pooling")
    
    return _shared_client


async def close_shared_http_client() -> None:
    """
    Close shared HTTP client (call on application shutdown).
    
    This should be called during application shutdown to properly close connections.
    """
    global _shared_client
    
    if _shared_client is not None:
        await _shared_client.aclose()
        _shared_client = None
        logger.info("Closed shared HTTP client")
