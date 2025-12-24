# Standard library imports
import logging
from typing import Optional

# Local application imports
from ...core.config import get_settings

logger = logging.getLogger(__name__)


class BaseJetsonClient:
    """
    Base class for Jetson backend clients.
    
    Provides common initialization for base_url and timeout.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        """
        Initialize base Jetson client.
        
        Args:
            base_url: Base URL for Jetson backend. If None, reads from env.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.jetson_backend_url
        self.timeout = timeout

