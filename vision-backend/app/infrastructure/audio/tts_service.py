"""Text-to-Speech service abstraction."""
import base64
import logging
import re
from typing import Optional, List
import httpx

from ...core.config import get_settings
from ...infrastructure.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class TTSService:
    """
    Text-to-Speech service for converting text to audio.
    
    This service abstracts the TTS provider (currently Groq) and provides:
    - Connection pooling via shared HTTP client
    - Text chunking for long responses
    - Audio concatenation
    - Retry logic
    - Error handling
    """
    
    GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"
    
    # TTS Model and voice options (can be overridden via config)
    TTS_MODEL_DEFAULT = "canopylabs/orpheus-v1-english"
    TTS_VOICE_DEFAULT = "troy"  # Default voice
    
    # Groq TTS character limit
    TTS_MAX_CHARS = 200
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        """
        Initialize TTS service.
        
        Args:
            http_client: Optional shared async HTTP client for connection pooling.
                        If None, creates a new client per request (not recommended).
        """
        settings = get_settings()
        self.provider = settings.tts_provider
        self.api_key = settings.groq_tts_api_key
        self.tts_model = settings.groq_tts_model
        self.tts_voice = settings.groq_tts_voice
        self.local_voice = settings.local_tts_voice
        self._http_client = http_client
        self._owns_client = http_client is None
        
        if self.provider == "groq" and not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")
        elif self.provider == "local":
            try:
                import edge_tts
                logger.info(f"Using edge_tts with voice: {self.local_voice}")
            except ImportError:
                logger.error("edge_tts not installed. Install with: pip install edge-tts")
                raise ValueError("edge_tts is required for local TTS provider")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client (shared or create new)."""
        if self._http_client:
            return self._http_client
        
        # Create a new client if none provided (fallback, not ideal)
        return httpx.AsyncClient(timeout=120.0)  # Increased timeout for multiple chunks
    
    async def _close_client(self, client: httpx.AsyncClient) -> None:
        """Close client if we own it."""
        if self._owns_client and client != self._http_client:
            await client.aclose()
    
    def _preprocess_text_for_tts(self, text: str) -> str:
        """
        Preprocess text for better TTS quality.
        
        Applies:
        - Better markdown handling (preserve emphasis where natural)
        - Number/date formatting for speech
        - Acronym expansion hints
        - Natural pause insertion
        
        Args:
            text: Raw text (may contain markdown)
            
        Returns:
            Preprocessed text optimized for TTS
        """
        if not text:
            return ""
        
        processed = text
        
        # Format numbers for better speech
        # Convert "2024" -> "twenty twenty four" for years (optional, can be too verbose)
        # For now, keep numbers as-is but add spacing
        
        # Format dates: "2024-01-15" -> "January 15th, 2024" (simplified)
        import re
        date_pattern = r'(\d{4})-(\d{2})-(\d{2})'
        def format_date(match):
            year, month, day = match.groups()
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            try:
                month_name = months[int(month) - 1]
                day_num = int(day)
                # Add ordinal suffix
                if 10 <= day_num <= 20:
                    suffix = 'th'
                else:
                    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day_num % 10, 'th')
                return f"{month_name} {day_num}{suffix}, {year}"
            except (ValueError, IndexError):
                return match.group(0)
        
        processed = re.sub(date_pattern, format_date, processed)
        
        # Add natural pauses after punctuation (double space = pause)
        # But don't overdo it - TTS engines handle punctuation well
        processed = re.sub(r'([.!?])\s+', r'\1 ', processed)
        
        # Expand common acronyms for better pronunciation
        acronym_map = {
            r'\bAI\b': 'A I',
            r'\bAPI\b': 'A P I',
            r'\bURL\b': 'U R L',
            r'\bHTTP\b': 'H T T P',
            r'\bHTTPS\b': 'H T T P S',
            r'\bJSON\b': 'J S O N',
            r'\bXML\b': 'X M L',
            r'\bHTML\b': 'H T M L',
            r'\bCSS\b': 'C S S',
            r'\bSQL\b': 'S Q L',
            r'\bCPU\b': 'C P U',
            r'\bGPU\b': 'G P U',
            r'\bRAM\b': 'R A M',
            r'\bGB\b': 'gigabytes',
            r'\bMB\b': 'megabytes',
            r'\bKB\b': 'kilobytes',
        }
        
        for pattern, replacement in acronym_map.items():
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        processed = re.sub(r'\s+', ' ', processed)
        processed = re.sub(r'\n{3,}', '\n\n', processed)
        
        return processed.strip()
    
    def _chunk_text(self, text: str, max_chars: int = TTS_MAX_CHARS) -> List[str]:
        """
        Split text into chunks that respect sentence boundaries when possible.
        
        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks, each <= max_chars
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        # Split by sentences first (period, exclamation, question mark)
        sentences = re.split(r'([.!?]\s+)', text)
        
        current_chunk = ""
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + punctuation
            
            # If adding this sentence would exceed limit, save current chunk and start new one
            if current_chunk and len(current_chunk) + len(full_sentence) > max_chars:
                chunks.append(current_chunk.strip())
                current_chunk = full_sentence
            else:
                current_chunk += full_sentence
            
            # If a single sentence is too long, split it by words
            if len(current_chunk) > max_chars:
                words = current_chunk.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 > max_chars:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                    else:
                        temp_chunk += " " + word if temp_chunk else word
                current_chunk = temp_chunk
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _synthesize_chunk(
        self,
        text: str,
        tts_model: str,
        tts_voice: str,
        client: httpx.AsyncClient
    ) -> bytes:
        """
        Synthesize a single text chunk to audio.
        
        Args:
            text: Text chunk to synthesize
            tts_model: TTS model to use
            tts_voice: Voice to use
            client: HTTP client instance
            
        Returns:
            Audio bytes (WAV format)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": tts_model,
            "input": text,
            "voice": tts_voice,
            "response_format": "wav"
        }
        
        async def _synthesize_internal():
            """Internal synthesis function for retry logic."""
            response = await client.post(
                self.GROQ_TTS_URL,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            # Check content type to determine response format
            content_type = response.headers.get("content-type", "").lower()
            
            if "application/json" in content_type:
                # Groq TTS returns base64-encoded audio in JSON response
                result = response.json()
                audio_base64 = result.get("audio") or result.get("data") or result.get("content")
                
                if not audio_base64:
                    raise ValueError("No audio data in Groq TTS response")
                
                # Decode base64 to bytes
                audio_bytes = base64.b64decode(audio_base64)
            else:
                # Groq TTS returns binary audio directly
                audio_bytes = response.content
                
                if not audio_bytes:
                    raise ValueError("No audio data in Groq TTS response")
            
            return audio_bytes
        
        async def _synthesize_with_retry():
            """Wrapper to handle 4xx errors (no retry) vs 5xx errors (retry)."""
            try:
                return await _synthesize_internal()
            except httpx.HTTPStatusError as e:
                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    raise  # Re-raise immediately, no retry
                # Retry on 5xx errors
                raise
        
        # Retry on transient failures (timeout, 5xx errors only)
        return await retry_with_backoff(
            _synthesize_with_retry,
            max_retries=2,  # 3 total attempts
            initial_delay=1.0,
            max_delay=10.0,
            exceptions=(httpx.TimeoutException, httpx.HTTPStatusError),
        )
    
    def _concatenate_audio_chunks(self, audio_chunks: List[bytes]) -> bytes:
        """
        Concatenate multiple WAV audio chunks into a single audio file.
        
        Uses pydub for proper audio merging if available, falls back to manual concatenation.
        
        Args:
            audio_chunks: List of audio byte chunks (WAV format)
            
        Returns:
            Combined audio bytes
        """
        if not audio_chunks:
            raise ValueError("No audio chunks to concatenate")
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        # Try to use pydub for proper audio concatenation
        try:
            from pydub import AudioSegment
            import io
            
            # Load all audio segments
            segments = []
            for chunk in audio_chunks:
                audio = AudioSegment.from_file(io.BytesIO(chunk), format="wav")
                segments.append(audio)
            
            # Concatenate with a small silence gap for natural flow (50ms)
            # This prevents audio artifacts when chunks are joined
            silence = AudioSegment.silent(duration=50)  # 50ms silence
            combined = segments[0]
            for segment in segments[1:]:
                combined = combined + silence + segment
            
            # Export to bytes
            buffer = io.BytesIO()
            combined.export(buffer, format="wav")
            logger.debug(f"Concatenated {len(audio_chunks)} audio chunks using pydub")
            return buffer.getvalue()
            
        except ImportError:
            # pydub not available, use manual concatenation
            logger.debug("pydub not available, using manual audio concatenation")
        except Exception as e:
            logger.warning(f"pydub concatenation failed: {e}. Falling back to manual method.")
        
        # Fallback: Manual concatenation (original method)
        try:
            # Try to find data chunks and concatenate
            combined_data = bytearray()
            first_header = None
            
            for i, chunk in enumerate(audio_chunks):
                # Find "data" marker (0x64617461 = "data" in ASCII)
                data_marker = b'data'
                data_pos = chunk.find(data_marker)
                
                if data_pos == -1:
                    # If no data marker found, just concatenate (fallback)
                    if i == 0:
                        first_header = chunk[:44]  # Typical WAV header size
                    combined_data.extend(chunk)
                else:
                    # Extract header (first time) and data sections
                    if i == 0:
                        # Keep header from first file
                        first_header = chunk[:data_pos + 8]  # Header + "data" + size
                        data_size = int.from_bytes(chunk[data_pos + 4:data_pos + 8], 'little')
                        combined_data.extend(chunk[data_pos + 8:data_pos + 8 + data_size])
                    else:
                        # Extract data section from subsequent chunks
                        data_size = int.from_bytes(chunk[data_pos + 4:data_pos + 8], 'little')
                        combined_data.extend(chunk[data_pos + 8:data_pos + 8 + data_size])
            
            if first_header:
                # Update file size in RIFF header (bytes 4-8)
                total_size = len(first_header) + len(combined_data) - 8
                first_header[4:8] = total_size.to_bytes(4, 'little')
                
                # Update data chunk size (in header)
                data_chunk_pos = first_header.find(b'data')
                if data_chunk_pos != -1:
                    first_header[data_chunk_pos + 4:data_chunk_pos + 8] = len(combined_data).to_bytes(4, 'little')
                
                return bytes(first_header) + bytes(combined_data)
            else:
                # Fallback: just concatenate all chunks
                return b''.join(audio_chunks)
                
        except Exception as e:
            logger.warning(f"Error concatenating audio chunks properly: {e}. Using simple concatenation.")
            # Fallback: simple concatenation (may cause audio issues but won't crash)
            return b''.join(audio_chunks)
    
    async def _synthesize_local(self, text: str, voice: Optional[str] = None) -> bytes:
        """
        Synthesize speech using local edge_tts.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (defaults to config)
            
        Returns:
            Audio bytes (MP3 format, converted to WAV)
        """
        import edge_tts
        import io
        
        tts_voice = voice or self.local_voice
        logger.info(f"Synthesizing speech using edge_tts (voice: {tts_voice})")
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text_for_tts(text)
            
            # Generate audio using edge_tts
            communicate = edge_tts.Communicate(processed_text, tts_voice)
            audio_data = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            # edge_tts returns MP3, convert to WAV if needed
            # For simplicity, we'll return MP3 bytes (browsers can play MP3)
            # Or convert to WAV using pydub if available
            try:
                from pydub import AudioSegment
                import io
                # Load MP3 from bytes
                audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
                # Export as WAV
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format="wav")
                audio_bytes = wav_buffer.getvalue()
                logger.info(f"Converted edge_tts MP3 to WAV: {len(audio_bytes)} bytes")
            except ImportError:
                # pydub not available, return MP3 (browsers can handle it)
                logger.warning("pydub not available, returning MP3 format from edge_tts")
                audio_bytes = audio_data
            
            logger.info(f"Successfully synthesized speech: {len(audio_bytes)} bytes")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Error in local TTS synthesis: {e}", exc_info=True)
            raise Exception(f"Local TTS synthesis failed: {str(e)}")
    
    async def synthesize(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None
    ) -> bytes:
        """
        Convert text to audio using configured provider (Groq or local).
        
        Args:
            text: Text to convert to speech
            model: TTS model to use (only for Groq provider)
            voice: Voice to use
            
        Returns:
            Audio bytes (WAV format)
            
        Raises:
            ValueError: If text is empty
            Exception: If synthesis fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        print(f"[TTS] synthesize() called - provider: {self.provider}, text length: {len(text)}")
        
        # Route to appropriate provider
        if self.provider == "local":
            print("[TTS] Using local provider (edge_tts)")
            result = await self._synthesize_local(text, voice)
            print(f"[TTS] Local synthesis complete - audio size: {len(result)} bytes")
            return result
        elif self.provider == "groq":
            print("[TTS] Using Groq provider")
            result = await self._synthesize_groq(text, model, voice)
            print(f"[TTS] Groq synthesis complete - audio size: {len(result)} bytes")
            return result
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}. Use 'groq' or 'local'")
    
    async def _synthesize_groq(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None
    ) -> bytes:
        """
        Synthesize speech using Groq API.
        
        Args:
            text: Text to convert to speech
            model: TTS model to use
            voice: Voice to use
            
        Returns:
            Audio bytes (WAV format)
        """
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured")
        
        # Preprocess text for better TTS quality
        processed_text = self._preprocess_text_for_tts(text)
        
        # Use specified model/voice or defaults from config
        tts_model = model or self.tts_model
        tts_voice = voice or self.tts_voice
        
        # Check if text needs chunking
        text_chunks = self._chunk_text(processed_text, max_chars=self.TTS_MAX_CHARS)
        
        if len(text_chunks) > 1:
            logger.info(
                f"Text exceeds {self.TTS_MAX_CHARS} chars, splitting into {len(text_chunks)} chunks "
                f"(total: {len(text)} chars)"
            )
        
        client = await self._get_client()
        
        try:
            audio_chunks = []
            
            for i, chunk in enumerate(text_chunks, 1):
                logger.info(
                    f"Synthesizing chunk {i}/{len(text_chunks)} "
                    f"({len(chunk)} chars): {chunk[:50]}..."
                )
                
                chunk_audio = await self._synthesize_chunk(
                    text=chunk,
                    tts_model=tts_model,
                    tts_voice=tts_voice,
                    client=client
                )
                audio_chunks.append(chunk_audio)
            
            # Concatenate audio chunks
            combined_audio = self._concatenate_audio_chunks(audio_chunks)
            
            logger.info(
                f"Successfully synthesized speech: {len(combined_audio)} bytes "
                f"from {len(text_chunks)} chunk(s)"
            )
            return combined_audio
            
        except httpx.TimeoutException:
            logger.error("Timeout while calling Groq TTS API")
            raise Exception("Timeout while synthesizing speech. Please try again.")
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                # Try to parse error response for helpful messages
                error_json = e.response.json()
                error_info = error_json.get("error", {})
                error_message = error_info.get("message", "")
                error_code = error_info.get("code", "")
                
                # Check for terms acceptance error
                if error_code == "model_terms_required" or "terms acceptance" in error_message.lower():
                    from urllib.parse import quote
                    encoded_model = quote(tts_model, safe='')
                    error_detail = (
                        f"TTS model '{tts_model}' requires terms acceptance. "
                        f"Please visit https://console.groq.com/playground?model={encoded_model} "
                        f"to accept the terms, or set GROQ_TTS_MODEL environment variable to a different model."
                    )
                else:
                    error_detail = error_message or f"Groq TTS API error: {e.response.status_code}"
            except Exception:
                # If we can't parse the error, use the raw response
                error_detail = f"Groq TTS API error: {e.response.status_code} - {e.response.text[:200]}"
            
            logger.error(
                f"HTTP error from Groq TTS API: {e.response.status_code} - {error_detail}"
            )
            raise Exception(error_detail)
        except Exception as e:
            logger.error(f"Unexpected error in Groq TTS: {e}", exc_info=True)
            raise
        finally:
            await self._close_client(client)
