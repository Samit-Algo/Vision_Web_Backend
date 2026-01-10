from typing import TYPE_CHECKING
from ...infrastructure.streaming import WsFmp4Service

if TYPE_CHECKING:
    from ..base_container import BaseContainer


class StreamingProvider:
    """Streaming service provider - registers WebSocket fMP4 streaming service"""
    
    @staticmethod
    def register(container: "BaseContainer") -> None:
        """
        Register streaming services.
        Services are registered as singletons.
        """
        # Register WsFmp4Service as singleton (shared across all requests/controllers)
        # Check if already registered to avoid creating multiple instances
        try:
            container.get(WsFmp4Service)
        except ValueError:
            ws_service = WsFmp4Service()
            container.register_singleton(WsFmp4Service, ws_service)

