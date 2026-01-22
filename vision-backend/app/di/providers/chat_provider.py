from typing import TYPE_CHECKING
from ...domain.repositories.agent_repository import AgentRepository
from ...domain.repositories.camera_repository import CameraRepository
from ...domain.repositories.device_repository import DeviceRepository
from ...infrastructure.external.agent_client import AgentClient
from ...application.use_cases.chat.chat_with_agent import ChatWithAgentUseCase
from ...application.use_cases.chat.general_chat_use_case import GeneralChatUseCase
from ...application.services.voice_chat_service import VoiceChatService
from ...infrastructure.audio.stt_service import STTService
from ...infrastructure.audio.tts_service import TTSService
from ...infrastructure.audio.audio_processor import AudioProcessor
from ...infrastructure.cache.response_cache import ResponseCache

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
        # Register AgentClient as singleton if not already registered
        try:
            container.get(AgentClient)
        except ValueError:
            # Not registered yet, register it now
            agent_client = AgentClient()
            container.register_singleton(AgentClient, agent_client)
        
        # Register ChatWithAgentUseCase
        container.register_factory(
            ChatWithAgentUseCase,
            lambda: ChatWithAgentUseCase(
                agent_repository=container.get(AgentRepository),
                camera_repository=container.get(CameraRepository),
                device_repository=container.get(DeviceRepository),
                jetson_client=container.get(AgentClient)
            )
        )
        
        # Register GeneralChatUseCase as singleton (stateless, can be reused)
        try:
            container.get(GeneralChatUseCase)
        except ValueError:
            general_chat_use_case = GeneralChatUseCase()
            container.register_singleton(GeneralChatUseCase, general_chat_use_case)
        
        # Register VoiceChatService as singleton
        # Note: Audio services (STT, TTS, etc.) should be registered by AudioProvider first
        try:
            container.get(VoiceChatService)
        except ValueError:
            # Audio services should already be registered by AudioProvider
            voice_chat_service = VoiceChatService(
                stt_service=container.get(STTService),
                tts_service=container.get(TTSService),
                audio_processor=container.get(AudioProcessor),
                cache=container.get(ResponseCache),
                general_chat_use_case=container.get(GeneralChatUseCase),
                chat_with_agent_use_case=container.get(ChatWithAgentUseCase),
            )
            container.register_singleton(VoiceChatService, voice_chat_service)
        
        # Register VoiceChatUseCase as factory (depends on VoiceChatService)
        from ...application.use_cases.chat.voice_chat_use_case import VoiceChatUseCase
        container.register_factory(
            VoiceChatUseCase,
            lambda: VoiceChatUseCase(
                voice_chat_service=container.get(VoiceChatService)
            )
        )
