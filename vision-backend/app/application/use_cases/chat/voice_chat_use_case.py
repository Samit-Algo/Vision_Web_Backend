"""Use case for voice chat with audio input/output."""
import logging
import re
from typing import Optional, Tuple, AsyncGenerator, Dict, Any

from ...dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ...services.voice_chat_service import VoiceChatService

logger = logging.getLogger(__name__)


def _markdown_to_plain_text(markdown_text: str) -> str:
    """
    Best-effort conversion from markdown to plain text for TTS.
    This keeps the content while stripping most formatting syntax.
    """
    if not markdown_text:
        return ""

    text = markdown_text

    # Images: ![alt](url) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Inline code and code blocks: `code` or ```code``` -> code
    text = re.sub(r"`{3}([\s\S]*?)`{3}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold / italic markers: **text**, __text__, *text*, _text_ -> text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Headings: "# Title" -> "Title"
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Blockquotes: "> text" -> "text"
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    # List markers: "- ", "* ", "+ " at line start -> ""
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Horizontal rules: --- or *** etc. -> ""
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Collapse excessive whitespace/newlines
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


class VoiceChatUseCase:
    """
    Use case for voice chat conversations.
    
    This use case delegates to VoiceChatService for orchestration.
    It provides the use case interface while keeping business logic in the service layer.
    """
    
    def __init__(self, voice_chat_service: Optional[VoiceChatService] = None):
        """
        Initialize voice chat use case.
        
        Args:
            voice_chat_service: Optional VoiceChatService instance.
                               If None, will be injected via DI container.
        """
        self.voice_chat_service = voice_chat_service
    
    async def execute(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[bytes, str, str]:
        """
        Process voice chat: audio input → text → LLM → audio output.
        
        Args:
            audio_bytes: Raw audio file bytes from user
            filename: Original audio filename (for format detection)
            session_id: Optional session ID for conversation continuity
            user_id: User ID for session management
            
        Returns:
            Tuple of (audio_bytes, text_response, session_id):
            - audio_bytes: Generated audio response (WAV format)
            - text_response: LLM text response (for debugging/display, may contain markdown)
            - session_id: Session ID for maintaining conversation context
            
        Raises:
            Exception: If any step in the pipeline fails
        """
        if not self.voice_chat_service:
            raise ValueError("VoiceChatService not initialized. Use DI container to inject dependencies.")
        
        # Delegate to service layer for orchestration
        return await self.voice_chat_service.process_voice_chat(
            audio_bytes=audio_bytes,
            filename=filename,
            session_id=session_id,
            user_id=user_id
        )
    
    async def stream_execute(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream voice chat processing with real-time events.
        
        Args:
            audio_bytes: Raw audio file bytes from user
            filename: Original audio filename (for format detection)
            session_id: Optional session ID for conversation continuity
            user_id: User ID for session management
            
        Yields:
            Dict with "event" and "data" keys
        """
        if not self.voice_chat_service:
            raise ValueError("VoiceChatService not initialized. Use DI container to inject dependencies.")
        
        # Delegate to service layer for streaming orchestration
        async for event in self.voice_chat_service.stream_process_voice_chat(
            audio_bytes=audio_bytes,
            filename=filename,
            session_id=session_id,
            user_id=user_id
        ):
            yield event
        
        