"""Speech-to-Text service abstraction."""
import logging
import hashlib
import io
from typing import Optional
import httpx

from ...core.config import get_settings
from ...infrastructure.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class STTService:
    """
    Speech-to-Text service for converting audio to text.
    
    This service abstracts the STT provider (currently Groq) and provides:
    - Connection pooling via shared HTTP client
    - Retry logic
    - Error handling
    - Caching support (via cache key generation)
    """
    
    GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
    
    # STT Model options
    STT_MODEL_FAST = "whisper-large-v3-turbo"  # Faster processing
    STT_MODEL_ACCURATE = "whisper-large-v3"    # Higher accuracy
    
    # Supported audio formats for STT
    SUPPORTED_STT_FORMATS = {
        "flac", "mp3", "wav", "mp4", "mpeg", "mpga", "m4a", "ogg", "webm"
    }
    
    # Max file size: 25MB (Groq limit)
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        """
        Initialize STT service.
        
        Args:
            http_client: Optional shared async HTTP client for connection pooling.
                        If None, creates a new client per request (not recommended).
        """
        settings = get_settings()
        self.provider = settings.stt_provider
        self.api_key = settings.groq_stt_api_key
        self.local_model = settings.local_stt_model
        self._http_client = http_client
        self._owns_client = http_client is None
        
        # Initialize local model if using local provider
        self._local_model = None
        if self.provider == "local":
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Initializing faster_whisper with model: {self.local_model}")
                self._local_model = WhisperModel(self.local_model, device="cpu", compute_type="int8")
                logger.info("faster_whisper initialized successfully")
            except ImportError:
                logger.error("faster_whisper not installed. Install with: pip install faster-whisper")
                raise ValueError("faster_whisper is required for local STT provider")
            except Exception as e:
                logger.error(f"Failed to initialize faster_whisper: {e}")
                raise
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client (shared or create new)."""
        if self._http_client:
            return self._http_client
        
        # Create a new client if none provided (fallback, not ideal)
        return httpx.AsyncClient(timeout=30.0)
    
    async def _close_client(self, client: httpx.AsyncClient) -> None:
        """Close client if we own it."""
        if self._owns_client and client != self._http_client:
            await client.aclose()
    
    def generate_cache_key(self, audio_bytes: bytes) -> str:
        """
        Generate cache key for audio input.
        
        Args:
            audio_bytes: Audio file bytes
            
        Returns:
            Cache key string (hash of audio bytes)
        """
        return hashlib.sha256(audio_bytes).hexdigest()
    
    def validate_audio(self, audio_bytes: bytes, filename: str) -> None:
        """
        Validate audio file before processing.
        
        Args:
            audio_bytes: Audio file bytes
            filename: Original filename
            
        Raises:
            ValueError: If audio is invalid
        """
        # Validate provider-specific requirements
        if self.provider == "groq":
            if not self.api_key:
                raise ValueError("GROQ_API_KEY is not configured for Groq STT provider")
            # Validate file size (Groq limit)
            if len(audio_bytes) > self.MAX_FILE_SIZE:
                raise ValueError(
                    f"Audio file too large: {len(audio_bytes)} bytes. "
                    f"Maximum size is {self.MAX_FILE_SIZE} bytes (25MB)"
                )
        elif self.provider == "local":
            if not self._local_model:
                raise ValueError("Local STT model not initialized")
        
        # Determine content type from filename
        file_ext = filename.lower().split('.')[-1] if '.' in filename else 'wav'
        if file_ext not in self.SUPPORTED_STT_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_STT_FORMATS)}"
            )
    
    def _estimate_audio_duration(self, audio_bytes: bytes, filename: str) -> Optional[float]:
        """
        Estimate audio duration in seconds.
        
        Uses heuristics based on file size and format.
        For accurate duration, pydub would be needed, but this avoids the dependency.
        
        Args:
            audio_bytes: Audio file bytes
            filename: Original filename
            
        Returns:
            Estimated duration in seconds, or None if cannot estimate
        """
        try:
            # Try to use pydub if available for accurate duration
            try:
                from pydub import AudioSegment
                import io
                file_ext = filename.lower().split('.')[-1] if '.' in filename else 'wav'
                format_map = {'wav': 'wav', 'mp3': 'mp3', 'm4a': 'mp4', 'ogg': 'ogg', 'flac': 'flac', 'webm': 'webm'}
                audio_format = format_map.get(file_ext, 'wav')
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
                return len(audio) / 1000.0  # Convert milliseconds to seconds
            except ImportError:
                pass
            
            # Fallback: rough estimate based on file size
            # This is very approximate and format-dependent
            file_ext = filename.lower().split('.')[-1] if '.' in filename else 'wav'
            
            # Rough bitrate estimates (bytes per second) for different formats
            # These are conservative estimates
            bitrate_estimates = {
                'wav': 176400,  # 16-bit, 44.1kHz, mono ≈ 88KB/s, stereo ≈ 176KB/s
                'mp3': 16000,   # ~128kbps ≈ 16KB/s
                'm4a': 16000,   # Similar to mp3
                'ogg': 16000,   # Similar to mp3
                'flac': 50000,  # Lossless, variable
                'webm': 20000,  # Variable
            }
            
            estimated_bitrate = bitrate_estimates.get(file_ext, 20000)
            estimated_duration = len(audio_bytes) / estimated_bitrate
            
            return estimated_duration
        except Exception as e:
            logger.debug(f"Could not estimate audio duration: {e}")
            return None
    
    def _select_model(self, audio_bytes: bytes, filename: str, user_model: Optional[str] = None) -> str:
        """
        Select appropriate STT model based on audio characteristics.
        
        Strategy:
        - Short audio (< 5 seconds): Use fast model (whisper-large-v3-turbo)
        - Long audio (>= 5 seconds): Use accurate model (whisper-large-v3)
        - User-specified model: Use that
        
        Args:
            audio_bytes: Audio file bytes
            filename: Original filename
            user_model: User-specified model (takes precedence)
            
        Returns:
            Selected model name
        """
        if user_model:
            return user_model
        
        duration = self._estimate_audio_duration(audio_bytes, filename)
        
        if duration is None:
            # Cannot estimate, default to fast model
            logger.debug("Cannot estimate audio duration, using fast model")
            return self.STT_MODEL_FAST
        
        # Threshold: 5 seconds
        # Short audio: use fast model for speed
        # Long audio: use accurate model for better quality
        if duration < 5.0:
            logger.debug(f"Short audio ({duration:.1f}s), using fast model")
            return self.STT_MODEL_FAST
        else:
            logger.debug(f"Long audio ({duration:.1f}s), using accurate model")
            return self.STT_MODEL_ACCURATE
    
    async def _transcribe_local(self, audio_bytes: bytes, filename: str) -> str:
        """
        Transcribe audio using local faster_whisper model.
        
        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename
            
        Returns:
            Transcribed text string
        """
        if not self._local_model:
            raise ValueError("Local STT model not initialized")
        
        logger.info(f"Transcribing audio using faster_whisper (model: {self.local_model})")
        
        try:
            # Convert audio bytes to file-like object
            audio_file = io.BytesIO(audio_bytes)
            
            # Run transcription (faster_whisper is synchronous, so we run in thread)
            import asyncio
            loop = asyncio.get_event_loop()
            
            def _transcribe_sync():
                segments, info = self._local_model.transcribe(
                    audio_file,
                    beam_size=5,
                    language="en"  # Can be made configurable
                )
                # Collect all segments
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)
                return " ".join(text_parts).strip()
            
            transcribed_text = await loop.run_in_executor(None, _transcribe_sync)
            
            if not transcribed_text:
                logger.warning("faster_whisper returned empty transcription")
                raise ValueError("Empty transcription received from faster_whisper")
            
            logger.info(f"Successfully transcribed audio: {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error in local STT transcription: {e}", exc_info=True)
            raise Exception(f"Local STT transcription failed: {str(e)}")
    
    async def _transcribe_groq(
        self,
        audio_bytes: bytes,
        filename: str,
        model: Optional[str] = None,
        use_fallback: bool = True
    ) -> str:
        """
        Transcribe audio using Groq API.
        
        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename
            model: STT model to use
            use_fallback: If True, fallback to fast model on error
            
        Returns:
            Transcribed text string
        """
        # Select model (adaptive or user-specified)
        stt_model = self._select_model(audio_bytes, filename, model)
        
        # Determine content type from filename
        file_ext = filename.lower().split('.')[-1] if '.' in filename else 'wav'
        content_type = f"audio/{file_ext}" if file_ext != "webm" else "audio/webm"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        files = {
            "file": (filename, audio_bytes, content_type)
        }
        
        data = {
            "model": stt_model
        }
        
        client = await self._get_client()
        
        async def _transcribe_internal():
            """Internal transcription function for retry logic."""
            logger.info(f"Transcribing audio using Groq STT (model: {stt_model})")
            
            response = await client.post(
                self.GROQ_STT_URL,
                files=files,
                data=data,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            transcribed_text = result.get("text", "").strip()
            
            if not transcribed_text:
                logger.warning("Groq STT returned empty transcription")
                raise ValueError("Empty transcription received from Groq STT API")
            
            logger.info(f"Successfully transcribed audio: {len(transcribed_text)} characters")
            return transcribed_text
        
        try:
            # Retry on transient failures (timeout, 5xx errors only)
            async def _transcribe_with_retry():
                try:
                    return await _transcribe_internal()
                except httpx.HTTPStatusError as e:
                    # Don't retry on 4xx errors (client errors)
                    if 400 <= e.response.status_code < 500:
                        raise
                    raise
            
            transcribed_text = await retry_with_backoff(
                _transcribe_with_retry,
                max_retries=2,
                initial_delay=1.0,
                max_delay=10.0,
                exceptions=(httpx.TimeoutException, httpx.HTTPStatusError),
            )
            return transcribed_text
            
        except httpx.HTTPStatusError as e:
            # If using accurate model and error occurs, try fallback to fast model
            if use_fallback and stt_model == self.STT_MODEL_ACCURATE:
                logger.warning(f"Accurate model failed, falling back to fast model: {e.response.status_code}")
                return await self._transcribe_groq(
                    audio_bytes=audio_bytes,
                    filename=filename,
                    model=self.STT_MODEL_FAST,
                    use_fallback=False
                )
            
            logger.error(f"HTTP error from Groq STT API: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Groq STT API error: {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("Timeout while calling Groq STT API (all retries exhausted)")
            raise Exception("Timeout while transcribing audio. Please try again.")
        except Exception as e:
            logger.error(f"Unexpected error in Groq STT: {e}", exc_info=True)
            raise
        finally:
            await self._close_client(client)
    
    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: Optional[str] = None,
        use_fallback: bool = True
    ) -> str:
        """
        Convert audio to text using configured provider (Groq or local).
        
        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename (used to determine format)
            model: STT model to use (only for Groq provider)
            use_fallback: If True, fallback to fast model on error (only for Groq)
            
        Returns:
            Transcribed text string
            
        Raises:
            ValueError: If audio format is not supported or file is too large
            Exception: If transcription fails
        """
        self.validate_audio(audio_bytes, filename)
        
        # Route to appropriate provider
        if self.provider == "local":
            return await self._transcribe_local(audio_bytes, filename)
        elif self.provider == "groq":
            return await self._transcribe_groq(audio_bytes, filename, model, use_fallback)
        else:
            raise ValueError(f"Unknown STT provider: {self.provider}. Use 'groq' or 'local'")
