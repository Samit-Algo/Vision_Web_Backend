from typing import TYPE_CHECKING
from ...domain.repositories.agent_repository import AgentRepository
from ...domain.repositories.camera_repository import CameraRepository
from ...domain.repositories.device_repository import DeviceRepository
from ...infrastructure.external.jetson_client import JetsonClient
from ...application.use_cases.chat.chat_with_agent import ChatWithAgentUseCase

if TYPE_CHECKING:
    from ..base_container import BaseContainer


class ChatProvider:
    """Chat use case provider - registers chat-related use cases"""
    
    @staticmethod
    def register(container: "BaseContainer") -> None:
        """
        Register chat use cases.
        Use cases are created on-demand via factories.
        """
        # Register JetsonClient as singleton if not already registered
        try:
            container.get(JetsonClient)
        except ValueError:
            # Not registered yet, register it now
            jetson_client = JetsonClient()
            container.register_singleton(JetsonClient, jetson_client)
        
        # Register ChatWithAgentUseCase
        container.register_factory(
            ChatWithAgentUseCase,
            lambda: ChatWithAgentUseCase(
                agent_repository=container.get(AgentRepository),
                camera_repository=container.get(CameraRepository),
                device_repository=container.get(DeviceRepository),
                jetson_client=container.get(JetsonClient)
            )
        )
