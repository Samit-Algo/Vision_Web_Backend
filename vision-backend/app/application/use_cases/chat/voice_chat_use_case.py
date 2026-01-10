"""Use case for voice chat with audio input/output."""
import logging
from typing import Optional, Tuple
from io import BytesIO

from ...dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ....infrastructure.external.groq_audio_service import GroqAudioService
from .general_chat_use_case import GeneralChatUseCase

logger = logging.getLogger(__name__)


class VoiceChatUseCase:
    """
    Use case for voice chat conversations.
    
    Orchestrates the flow:
    1. Audio → Text (Groq STT)
    2. Text → LLM Response (GeneralChatUseCase)
    3. LLM Response Text → Audio (Groq TTS)
    """
    
    def __init__(self):
        """Initialize voice chat use case with dependencies."""
        self.groq_audio_service = GroqAudioService()
        self.general_chat_use_case = GeneralChatUseCase()
    
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
            - text_response: LLM text response (for debugging/display)
            - session_id: Session ID for maintaining conversation context
            
        Raises:
            Exception: If any step in the pipeline fails
        """
        try:
            # Step 1: Convert audio to text using Groq STT
            logger.info("Step 1: Transcribing audio to text")
            transcribed_text = await self.groq_audio_service.transcribe(
                audio_bytes=audio_bytes,
                filename=filename
            )
            
            if not transcribed_text:
                raise ValueError("Failed to transcribe audio - empty result")
            
            logger.info(f"Transcribed text: {transcribed_text[:100]}...")
            
            # Step 2: Send transcribed text to LLM using existing chat use case
            logger.info("Step 2: Sending text to LLM")
            chat_request = ChatMessageRequest(
                message=transcribed_text,
                session_id=session_id
            )
            
            chat_response: ChatMessageResponse = await self.general_chat_use_case.execute(
                request=chat_request,
                user_id=user_id
            )
            
            llm_text_response = chat_response.response
            final_session_id = chat_response.session_id
            
            if not llm_text_response:
                raise ValueError("LLM returned empty response")
            
            logger.info(f"LLM response: {llm_text_response[:100]}...")
            
            # Step 3: Convert LLM text response to audio using Groq TTS
            logger.info("Step 3: Converting LLM response to audio")
            audio_response_bytes = await self.groq_audio_service.synthesize(
                text=llm_text_response
            )
            
            if not audio_response_bytes:
                raise ValueError("Failed to synthesize audio - empty result")
            
            logger.info(f"Generated audio: {len(audio_response_bytes)} bytes")
            
            # Return audio bytes, text response, and session ID
            return (audio_response_bytes, llm_text_response, final_session_id)
            
        except Exception as e:
            logger.error(f"Error in voice chat pipeline: {e}", exc_info=True)
            # Re-raise to let controller handle error response
            raise

