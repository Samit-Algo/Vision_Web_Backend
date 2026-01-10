"""Groq Audio API service for Speech-to-Text and Text-to-Speech."""
import base64
import logging
import re
from typing import Optional, List
from urllib.parse import quote
import httpx

from ...core.config import get_settings

logger = logging.getLogger(__name__)


class GroqAudioService:
    """
    Service for interacting with Groq's Speech-to-Text and Text-to-Speech APIs.
    
    This service handles:
    - Converting audio to text using Groq STT API
    - Converting text to audio using Groq TTS API (with base64 decoding)
    """
    
    GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
    GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"
    
    # STT Model options
    STT_MODEL_FAST = "whisper-large-v3-turbo"  # Faster processing
    STT_MODEL_ACCURATE = "whisper-large-v3"    # Higher accuracy
    
    # TTS Model and voice options (can be overridden via config)
    TTS_MODEL_DEFAULT = "canopylabs/orpheus-v1-english"
    TTS_VOICE_DEFAULT = "troy"  # Default voice
    
    # Supported audio formats for STT
    SUPPORTED_STT_FORMATS = {
        "flac", "mp3", "wav", "mp4", "mpeg", "mpga", "m4a", "ogg", "webm"
    }
    
    # Max file size: 25MB (Groq limit)
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
    
    # Groq TTS character limit
    TTS_MAX_CHARS = 200
    
    def __init__(self):
        """Initialize Groq Audio Service with API key from settings."""
        settings = get_settings()
        self.api_key = settings.groq_api_key
        self.tts_model = settings.groq_tts_model
        self.tts_voice = settings.groq_tts_voice
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")
    
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
    
    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: Optional[str] = None
    ) -> str:
        """
        Convert audio to text using Groq Speech-to-Text API.
        
        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename (used to determine format)
            model: STT model to use (default: whisper-large-v3-turbo)
            
        Returns:
            Transcribed text string
            
        Raises:
            ValueError: If audio format is not supported or file is too large
            httpx.HTTPStatusError: If Groq API returns an error
        """
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured")
        
        # Validate file size
        if len(audio_bytes) > self.MAX_FILE_SIZE:
            raise ValueError(
                f"Audio file too large: {len(audio_bytes)} bytes. "
                f"Maximum size is {self.MAX_FILE_SIZE} bytes (25MB)"
            )
        
        # Determine content type from filename
        file_ext = filename.lower().split('.')[-1] if '.' in filename else 'wav'
        if file_ext not in self.SUPPORTED_STT_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_STT_FORMATS)}"
            )
        
        content_type = f"audio/{file_ext}" if file_ext != "webm" else "audio/webm"
        
        # Use specified model or default
        stt_model = model or self.STT_MODEL_FAST
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        files = {
            "file": (filename, audio_bytes, content_type)
        }
        
        data = {
            "model": stt_model
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                
        except httpx.TimeoutException:
            logger.error("Timeout while calling Groq STT API")
            raise Exception("Timeout while transcribing audio. Please try again.")
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error from Groq STT API: {e.response.status_code} - {e.response.text}"
            )
            raise Exception(f"Groq STT API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error in Groq STT: {e}", exc_info=True)
            raise
    
    def _concatenate_audio_chunks(self, audio_chunks: List[bytes]) -> bytes:
        """
        Concatenate multiple WAV audio chunks into a single audio file.
        
        This is a simplified approach that works for WAV files with the same format.
        For production, consider using pydub for proper audio merging.
        
        Args:
            audio_chunks: List of audio byte chunks (WAV format)
            
        Returns:
            Combined audio bytes
        """
        if not audio_chunks:
            raise ValueError("No audio chunks to concatenate")
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        # For WAV files, we need to:
        # 1. Keep the header from the first file
        # 2. Concatenate the data sections
        # 3. Update the file size in the header
        
        # WAV file structure: RIFF header (12 bytes) + fmt chunk + data chunk
        # Simple approach: Find "data" chunk and concatenate data sections
        # This is a simplified implementation - for production, use pydub
        
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
    
    async def synthesize(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None
    ) -> bytes:
        """
        Convert text to audio using Groq Text-to-Speech API.
        
        Automatically chunks text if it exceeds 200 characters (Groq limit)
        and concatenates the resulting audio.
        
        Args:
            text: Text to convert to speech
            model: TTS model to use (default: canopylabs/orpheus-v1-english)
            voice: Voice to use (default: troy)
            
        Returns:
            Audio bytes (WAV format) decoded from base64, concatenated if chunked
            
        Raises:
            ValueError: If text is empty
            httpx.HTTPStatusError: If Groq API returns an error
        """
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Use specified model/voice or defaults from config
        tts_model = model or self.tts_model
        tts_voice = voice or self.tts_voice
        
        # Check if text needs chunking
        text_chunks = self._chunk_text(text, max_chars=self.TTS_MAX_CHARS)
        
        if len(text_chunks) > 1:
            logger.info(
                f"Text exceeds {self.TTS_MAX_CHARS} chars, splitting into {len(text_chunks)} chunks "
                f"(total: {len(text)} chars)"
            )
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:  # Increased timeout for multiple chunks
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

