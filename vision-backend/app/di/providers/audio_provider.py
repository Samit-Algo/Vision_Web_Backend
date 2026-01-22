"""Audio services provider for dependency injection."""
import logging
from typing import TYPE_CHECKING
from ...infrastructure.audio.stt_service import STTService
from ...infrastructure.audio.tts_service import TTSService
from ...infrastructure.audio.audio_processor import AudioProcessor
from ...infrastructure.cache.response_cache import ResponseCache
from ...infrastructure.http_client_factory import get_shared_http_client

if TYPE_CHECKING:
    from ..base_container import BaseContainer

logger = logging.getLogger(__name__)


class AudioProvider:
    """Audio services provider - registers all audio-related services"""
    
    @staticmethod
    def register(container: "BaseContainer") -> None:
        """
        Register all audio services.
        Services are created as singletons for reuse.
        """
        # Get shared HTTP client for connection pooling
        http_client = get_shared_http_client()
        
        # Register AudioProcessor as singleton
        audio_processor = AudioProcessor()
        container.register_singleton(AudioProcessor, audio_processor)
        
        # Register ResponseCache as singleton
        response_cache = ResponseCache()
        container.register_singleton(ResponseCache, response_cache)
        
        # Register STTService as singleton (with shared HTTP client)
        stt_service = STTService(http_client=http_client)
        container.register_singleton(STTService, stt_service)
        
        # Register TTSService as singleton (with shared HTTP client)
        tts_service = TTSService(http_client=http_client)
        container.register_singleton(TTSService, tts_service)
        
        logger.info("Registered audio services (STT, TTS, AudioProcessor, ResponseCache)")
