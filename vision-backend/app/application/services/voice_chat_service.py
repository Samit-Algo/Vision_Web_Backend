"""Voice chat orchestration service."""
import logging
import asyncio
import base64
from typing import Optional, Tuple, AsyncGenerator, Dict, Any
import httpx

from ...infrastructure.audio.stt_service import STTService
from ...infrastructure.audio.tts_service import TTSService
from ...infrastructure.audio.audio_processor import AudioProcessor
from ...infrastructure.cache.response_cache import ResponseCache
from ...application.use_cases.chat.general_chat_use_case import GeneralChatUseCase
from ...application.use_cases.chat.chat_with_agent import ChatWithAgentUseCase
from ..dto.chat_dto import ChatMessageRequest

logger = logging.getLogger(__name__)


def _markdown_to_plain_text(markdown_text: str) -> str:
    """
    Best-effort conversion from markdown to plain text for TTS.
    This keeps the content while stripping most formatting syntax.
    """
    import re
    
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


class VoiceChatService:
    """
    Orchestration service for voice chat pipeline.
    
    Coordinates:
    1. Audio → Text (STT)
    2. Text → LLM Response
    3. LLM Response → Audio (TTS)
    
    Provides:
    - Parallel processing where possible
    - Caching integration
    - Error handling with fallbacks
    """
    
    def __init__(
        self,
        stt_service: STTService,
        tts_service: TTSService,
        audio_processor: AudioProcessor,
        cache: ResponseCache,
        general_chat_use_case: GeneralChatUseCase,
        chat_with_agent_use_case: ChatWithAgentUseCase,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize voice chat service.
        
        Args:
            stt_service: Speech-to-Text service
            tts_service: Text-to-Speech service
            audio_processor: Audio preprocessing/postprocessing service
            cache: Response cache service
            general_chat_use_case: General chat use case for LLM
            http_client: Optional shared HTTP client (for connection pooling)
        """
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.audio_processor = audio_processor
        self.cache = cache
        self.general_chat_use_case = general_chat_use_case
        self._http_client = http_client
        self.chat_with_agent_use_case = chat_with_agent_use_case
    
    async def process_voice_chat(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[bytes, str, str]:
        """
        Process voice chat: audio input → text → LLM → audio output.
        
        This method orchestrates the entire pipeline with:
        - Caching at each stage
        - Parallel processing where possible
        - Error handling with fallbacks
        
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
            Exception: If any step in the pipeline fails after retries
        """
        try:
            # Step 1: Preprocess audio in parallel (CPU-bound operation)
            # Run preprocessing in thread pool to avoid blocking event loop
            processed_audio = await asyncio.to_thread(
                self.audio_processor.preprocess,
                audio_bytes,
                filename
            )
            
            # Step 2: Check STT cache
            audio_hash = self.stt_service.generate_cache_key(processed_audio)
            transcribed_text = self.cache.get_stt(audio_hash)
            
            if transcribed_text:
                logger.info("STT cache hit - using cached transcription")
            else:
                # Step 3: Convert audio to text using STT
                logger.info("Step 1: Transcribing audio to text")
                transcribed_text = await self.stt_service.transcribe(
                    audio_bytes=processed_audio,
                    filename=filename
                )
                
                if not transcribed_text:
                    raise ValueError("Failed to transcribe audio - empty result")
                
                # Cache STT result
                self.cache.set_stt(audio_hash, transcribed_text)
            
            logger.info(f"Transcribed text: {transcribed_text[:100]}...")
            
            # Step 4: Check LLM cache (session-aware)
            llm_response_text = None
            if session_id:
                llm_response_text = self.cache.get_llm(session_id, transcribed_text)
            
            if llm_response_text:
                logger.info("LLM cache hit - using cached response")
                final_session_id = session_id or "cached"
            else:
                # Step 5: Send transcribed text to LLM
                logger.info("Step 2: Sending text to LLM")
                chat_request = ChatMessageRequest(
                    message=transcribed_text,
                    session_id=session_id
                )
                
                chat_response = await self.general_chat_use_case.execute(
                    request=chat_request,
                    user_id=user_id
                )
                
                llm_response_text = chat_response.response
                final_session_id = chat_response.session_id
                
                if not llm_response_text:
                    raise ValueError("LLM returned empty response")
                
                # Cache LLM result (session-aware)
                if session_id:
                    self.cache.set_llm(session_id, transcribed_text, llm_response_text)
            
            logger.info(f"LLM response (raw, may contain markdown): {llm_response_text[:100]}...")
            
            # Step 6: Prepare plain-text version for TTS (strip markdown)
            tts_plain_text = _markdown_to_plain_text(llm_response_text)
            logger.info(f"TTS plain text: {tts_plain_text[:100]}...")
            
            # Step 7: Check TTS cache
            audio_response_bytes = self.cache.get_tts(tts_plain_text)
            
            if audio_response_bytes:
                logger.info("TTS cache hit - using cached audio")
            else:
                # Step 8: Convert LLM text response to audio using TTS
                # This can run in parallel with other operations if needed
                logger.info("Step 3: Converting LLM response to audio")
                audio_response_bytes = await self.tts_service.synthesize(
                    text=tts_plain_text
                )
                
                if not audio_response_bytes:
                    raise ValueError("Failed to synthesize audio - empty result")
                
                # Cache TTS result
                self.cache.set_tts(tts_plain_text, audio_response_bytes)
            
            logger.info(f"Generated audio: {len(audio_response_bytes)} bytes")
            
            # Step 9: Postprocess audio (future enhancement)
            final_audio = self.audio_processor.postprocess(audio_response_bytes)
            
            # Return audio bytes, original (markdown-capable) text response, and session ID
            return (final_audio, llm_response_text, final_session_id)
            
        except Exception as e:
            logger.error(f"Error in voice chat pipeline: {e}", exc_info=True)
            # Re-raise to let use case handle error response
            raise
    
    async def stream_process_voice_chat(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream voice chat processing with real-time events.
        
        Yields events:
        - {"event": "stt_start", "data": {}}
        - {"event": "stt_result", "data": {"text": "..."}}
        - {"event": "llm_start", "data": {}}
        - {"event": "llm_token", "data": {"delta": "..."}}
        - {"event": "llm_done", "data": {"text": "..."}}
        - {"event": "tts_start", "data": {}}
        - {"event": "tts_chunk", "data": {"audio": "<base64>"}}
        - {"event": "tts_done", "data": {}}
        - {"event": "done", "data": {"session_id": "..."}}
        - {"event": "error", "data": {"message": "..."}}
        
        Args:
            audio_bytes: Raw audio file bytes from user
            filename: Original audio filename (for format detection)
            session_id: Optional session ID for conversation continuity
            user_id: User ID for session management
            
        Yields:
            Dict with "event" and "data" keys
        """
        try:
            # Event 1: STT Start
            yield {"event": "stt_start", "data": {}}
            
            # Step 1: Preprocess audio in parallel (CPU-bound operation)
            # Run preprocessing in thread pool to avoid blocking event loop
            print("[STT] Starting audio preprocessing in parallel...")
            processed_audio = await asyncio.to_thread(
                self.audio_processor.preprocess,
                audio_bytes,
                filename
            )
            print(f"[STT] Preprocessing complete - size: {len(processed_audio)} bytes")
            
            # Step 2: Check STT cache
            audio_hash = self.stt_service.generate_cache_key(processed_audio)
            transcribed_text = self.cache.get_stt(audio_hash)
            
            if transcribed_text:
                logger.info("STT cache hit - using cached transcription")
            else:
                # Step 3: Convert audio to text using STT
                logger.info("Transcribing audio to text")
                transcribed_text = await self.stt_service.transcribe(
                    audio_bytes=processed_audio,
                    filename=filename
                )
                
                if not transcribed_text:
                    yield {"event": "error", "data": {"message": "Failed to transcribe audio - empty result"}}
                    return
                
                # Cache STT result
                self.cache.set_stt(audio_hash, transcribed_text)
            
            # Event 2: STT Result (PHASE 1 - This is what we need!)
            yield {"event": "stt_result", "data": {"text": transcribed_text}}
            logger.info(f"Transcribed text: {transcribed_text[:100]}...")
            
            # Step 4: Check LLM cache (session-aware)
            llm_response_text = None
            final_session_id = session_id
            if session_id:
                llm_response_text = self.cache.get_llm(session_id, transcribed_text)
            
            if llm_response_text:
                logger.info("LLM cache hit - using cached response")
                final_session_id = session_id or "cached"
                # Event: LLM Done (cached)
                yield {"event": "llm_done", "data": {"text": llm_response_text}}
            else:
                # Step 5: Send transcribed text to LLM (streaming)
                logger.info("Sending text to LLM")
                yield {"event": "llm_start", "data": {}}
                
                chat_request = ChatMessageRequest(
                    message=transcribed_text,
                    session_id=session_id
                )
                
                # Stream LLM tokens
                async for item in self.chat_with_agent_use_case.stream_execute(
                    request=chat_request,
                    user_id=user_id
                ):
                    # Forward LLM streaming events
                    ev = item.get("event") or "message"
                    data = item.get("data") or {}
                    
                    if ev == "token":
                        # Forward token events as llm_token
                        yield {"event": "llm_token", "data": {"delta": data.get("delta", "")}}
                    elif ev == "meta":
                        # Extract session_id from meta event if available
                        if "session_id" in data:
                            final_session_id = data.get("session_id")
                    elif ev == "done":
                        # Extract response and session_id from done event
                        # data is a ChatMessageResponse model_dump(), so it has 'response' and 'session_id'
                        llm_response_text = data.get("response", "")
                        if "session_id" in data:
                            final_session_id = data.get("session_id") or session_id
                        if llm_response_text:
                            yield {"event": "llm_done", "data": {"text": llm_response_text}}
                    elif ev == "error":
                        yield {"event": "error", "data": {"message": data.get("message", "LLM error")}}
                        return
                
                if not llm_response_text:
                    yield {"event": "error", "data": {"message": "LLM returned empty response"}}
                    return
                
                # Cache LLM result (session-aware)
                if session_id:
                    self.cache.set_llm(session_id, transcribed_text, llm_response_text)
            
            logger.info(f"LLM response: {llm_response_text[:100]}...")
            
            # Step 6: Prepare plain-text version for TTS (strip markdown)
            tts_plain_text = _markdown_to_plain_text(llm_response_text)
            logger.info(f"TTS plain text: {tts_plain_text[:100]}...")
            print(f"[TTS] Prepared text for TTS: {tts_plain_text[:100]}...")
            
            # Step 7: Check TTS cache
            print(f"[TTS] Checking cache for text: {tts_plain_text[:50]}...")
            audio_response_bytes = self.cache.get_tts(tts_plain_text)
            
            if audio_response_bytes:
                print(f"[TTS] Cache hit - audio size: {len(audio_response_bytes)} bytes")
                logger.info("TTS cache hit - using cached audio")
                # Event: TTS Done (cached - send as single chunk)
                audio_base64 = base64.b64encode(audio_response_bytes).decode('utf-8')
                print(f"[TTS] Sending cached audio chunk - base64 length: {len(audio_base64)}")
                yield {"event": "tts_chunk", "data": {"audio": audio_base64}}
                yield {"event": "tts_done", "data": {}}
                print("[TTS] Cached audio sent successfully")
            else:
                # Step 8: Convert LLM text response to audio using TTS
                print(f"[TTS] Starting synthesis for text: {tts_plain_text[:50]}...")
                logger.info("Converting LLM response to audio")
                yield {"event": "tts_start", "data": {}}
                
                audio_response_bytes = await self.tts_service.synthesize(
                    text=tts_plain_text
                )
                
                if not audio_response_bytes:
                    print("[TTS] ERROR: Synthesis returned empty result")
                    yield {"event": "error", "data": {"message": "Failed to synthesize audio - empty result"}}
                    return
                
                print(f"[TTS] Generated audio: {len(audio_response_bytes)} bytes")
                
                # Cache TTS result
                self.cache.set_tts(tts_plain_text, audio_response_bytes)
                
                # Event: TTS Chunk (send full audio as single chunk for now)
                audio_base64 = base64.b64encode(audio_response_bytes).decode('utf-8')
                print(f"[TTS] Sending audio chunk - base64 length: {len(audio_base64)}")
                yield {"event": "tts_chunk", "data": {"audio": audio_base64}}
                yield {"event": "tts_done", "data": {}}
                print("[TTS] Audio sent successfully")
            
            logger.info(f"Generated audio: {len(audio_response_bytes)} bytes")
            
            # Step 9: Postprocess audio (for future use, not sent in stream)
            # Note: We don't postprocess in streaming mode to avoid delay
            # final_audio = self.audio_processor.postprocess(audio_response_bytes)
            
            # Event: Done
            yield {"event": "done", "data": {"session_id": final_session_id}}
            
        except Exception as e:
            logger.error(f"Error in streaming voice chat pipeline: {e}", exc_info=True)
            yield {"event": "error", "data": {"message": str(e)}}
