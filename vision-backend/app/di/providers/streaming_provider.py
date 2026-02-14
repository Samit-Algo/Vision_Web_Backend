from typing import TYPE_CHECKING
from ...infrastructure.streaming import ProcessedFrameStreamService, WsFmp4Service

if TYPE_CHECKING:
    from ..base_container import BaseContainer


class StreamingProvider:
    """Streaming service provider - registers WebSocket fMP4 streaming services."""

    @staticmethod
    def register(container: "BaseContainer") -> None:
        """
        Register streaming services.
        Both services are singletons (shared across all requests/controllers).
        """
        try:
            container.get(WsFmp4Service)
        except ValueError:
            container.register_singleton(WsFmp4Service, WsFmp4Service())

        try:
            container.get(ProcessedFrameStreamService)
        except ValueError:
            container.register_singleton(
                ProcessedFrameStreamService,
                ProcessedFrameStreamService(),
            )

