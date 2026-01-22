"""Audio preprocessing and postprocessing utilities."""
import logging
import io
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import pydub, but make it optional
try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning(
        "pydub not available. Audio preprocessing will be limited. "
        "Install pydub and ffmpeg for full functionality."
    )


class AudioProcessor:
    """
    Audio preprocessing and postprocessing service.
    
    Provides:
    - Normalization
    - Silence trimming
    - Volume adjustment
    - Format conversion (if pydub available)
    """
    
    # Silence threshold in dB (below this is considered silence)
    SILENCE_THRESHOLD_DB = -40.0
    
    # Minimum silence duration to trim (in milliseconds)
    MIN_SILENCE_DURATION_MS = 100
    
    def __init__(self, enable_preprocessing: bool = True):
        """
        Initialize audio processor.
        
        Args:
            enable_preprocessing: If False, skip preprocessing (faster but lower quality)
        """
        self.enable_preprocessing = enable_preprocessing
        self.pydub_available = PYDUB_AVAILABLE
    
    def _load_audio(self, audio_bytes: bytes, filename: str) -> Optional['AudioSegment']:
        """
        Load audio bytes into AudioSegment.
        
        Args:
            audio_bytes: Raw audio bytes
            filename: Original filename (for format detection)
            
        Returns:
            AudioSegment or None if pydub unavailable
        """
        if not self.pydub_available:
            return None
        
        try:
            # Determine format from filename
            file_ext = filename.lower().split('.')[-1] if '.' in filename else 'wav'
            
            # Map common extensions to pydub format names
            format_map = {
                'wav': 'wav',
                'mp3': 'mp3',
                'm4a': 'mp4',
                'ogg': 'ogg',
                'flac': 'flac',
                'webm': 'webm',
            }
            
            audio_format = format_map.get(file_ext, 'wav')
            
            # Load audio from bytes
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format=audio_format
            )
            
            return audio
        except Exception as e:
            logger.warning(f"Failed to load audio with pydub: {e}. Using raw bytes.")
            return None
    
    def _export_audio(self, audio: 'AudioSegment', format: str = 'wav') -> bytes:
        """
        Export AudioSegment to bytes.
        
        Args:
            audio: AudioSegment to export
            format: Output format (default: wav)
            
        Returns:
            Audio bytes
        """
        buffer = io.BytesIO()
        audio.export(buffer, format=format)
        return buffer.getvalue()
    
    def _trim_silence(self, audio: 'AudioSegment') -> 'AudioSegment':
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio: AudioSegment to trim
            
        Returns:
            Trimmed AudioSegment
        """
        if not self.pydub_available:
            return audio
        
        try:
            # Trim silence from start and end
            trimmed = audio.strip_silence(
                silence_thresh=self.SILENCE_THRESHOLD_DB,
                min_silence_len=self.MIN_SILENCE_DURATION_MS
            )
            
            logger.debug(f"Trimmed {len(audio) - len(trimmed)}ms of silence")
            return trimmed
        except Exception as e:
            logger.warning(f"Failed to trim silence: {e}. Using original audio.")
            return audio
    
    def _normalize_audio(self, audio: 'AudioSegment') -> 'AudioSegment':
        """
        Normalize audio volume.
        
        Args:
            audio: AudioSegment to normalize
            
        Returns:
            Normalized AudioSegment
        """
        if not self.pydub_available:
            return audio
        
        try:
            normalized = normalize(audio)
            logger.debug("Audio normalized")
            return normalized
        except Exception as e:
            logger.warning(f"Failed to normalize audio: {e}. Using original audio.")
            return audio
    
    def preprocess(self, audio_bytes: bytes, filename: str) -> bytes:
        """
        Preprocess audio before STT.
        
        Applies:
        - Normalization (if pydub available)
        - Silence trimming (if pydub available)
        
        Args:
            audio_bytes: Raw audio bytes
            filename: Original filename
            
        Returns:
            Preprocessed audio bytes
        """
        if not self.enable_preprocessing:
            return audio_bytes
        
        audio = self._load_audio(audio_bytes, filename)
        
        if audio is None:
            # pydub not available, return as-is
            return audio_bytes
        
        try:
            # Apply preprocessing steps
            audio = self._normalize_audio(audio)
            audio = self._trim_silence(audio)
            
            # Export back to bytes
            return self._export_audio(audio, format='wav')
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}. Using original audio.")
            return audio_bytes
    
    def postprocess(self, audio_bytes: bytes) -> bytes:
        """
        Postprocess audio after TTS.
        
        Applies:
        - Normalization (if pydub available)
        - Volume adjustment (if needed)
        
        Args:
            audio_bytes: Generated audio bytes
            
        Returns:
            Postprocessed audio bytes
        """
        if not self.enable_preprocessing:
            return audio_bytes
        
        audio = self._load_audio(audio_bytes, filename='output.wav')
        
        if audio is None:
            # pydub not available, return as-is
            return audio_bytes
        
        try:
            # Apply postprocessing steps
            audio = self._normalize_audio(audio)
            
            # Export back to bytes
            return self._export_audio(audio, format='wav')
        except Exception as e:
            logger.warning(f"Audio postprocessing failed: {e}. Using original audio.")
            return audio_bytes
