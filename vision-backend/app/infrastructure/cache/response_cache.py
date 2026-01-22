"""Response caching service for STT, TTS, and LLM responses."""
import hashlib
import json
import logging
import time
from typing import Optional, Dict, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    In-memory LRU cache for responses.
    
    Provides caching for:
    - STT results (audio hash → text)
    - TTS results (text hash → audio bytes)
    - LLM results (session + message hash → response)
    
    This is a simple in-memory implementation. For production, consider Redis.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        stt_ttl: int = 86400,  # 24 hours
        tts_ttl: int = 604800,  # 7 days
        llm_ttl: int = 3600,  # 1 hour
    ):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum number of cache entries
            stt_ttl: STT cache TTL in seconds (default: 24 hours)
            tts_ttl: TTS cache TTL in seconds (default: 7 days)
            llm_ttl: LLM cache TTL in seconds (default: 1 hour)
        """
        self.max_size = max_size
        self.stt_ttl = stt_ttl
        self.tts_ttl = tts_ttl
        self.llm_ttl = llm_ttl
        
        # LRU cache: OrderedDict with (key, (value, timestamp, cache_type))
        self._cache: OrderedDict[str, tuple[Any, float, str]] = OrderedDict()
    
    def _generate_key(self, *parts: Any) -> str:
        """Generate cache key from parts."""
        # Convert all parts to string and hash
        key_str = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float, cache_type: str) -> bool:
        """Check if cache entry is expired."""
        ttl = {
            "stt": self.stt_ttl,
            "tts": self.tts_ttl,
            "llm": self.llm_ttl,
        }.get(cache_type, 3600)
        
        return time.time() - timestamp > ttl
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry if cache is full."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry (first in OrderedDict)
            self._cache.popitem(last=False)
    
    def get_stt(self, audio_hash: str) -> Optional[str]:
        """
        Get cached STT result.
        
        Args:
            audio_hash: Hash of audio bytes
            
        Returns:
            Cached transcribed text or None
        """
        key = f"stt:{audio_hash}"
        
        if key not in self._cache:
            return None
        
        value, timestamp, cache_type = self._cache[key]
        
        if self._is_expired(timestamp, cache_type):
            del self._cache[key]
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        
        logger.debug(f"STT cache hit: {audio_hash[:16]}...")
        return value
    
    def set_stt(self, audio_hash: str, text: str) -> None:
        """
        Cache STT result.
        
        Args:
            audio_hash: Hash of audio bytes
            text: Transcribed text
        """
        key = f"stt:{audio_hash}"
        self._evict_lru()
        
        self._cache[key] = (text, time.time(), "stt")
        self._cache.move_to_end(key)
        
        logger.debug(f"STT cached: {audio_hash[:16]}...")
    
    def get_tts(self, text: str, voice: Optional[str] = None) -> Optional[bytes]:
        """
        Get cached TTS result.
        
        Args:
            text: Text to synthesize
            voice: Voice model (optional)
            
        Returns:
            Cached audio bytes or None
        """
        key = self._generate_key("tts", text, voice or "default")
        cache_key = f"tts:{key}"
        
        if cache_key not in self._cache:
            return None
        
        value, timestamp, cache_type = self._cache[cache_key]
        
        if self._is_expired(timestamp, cache_type):
            del self._cache[cache_key]
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(cache_key)
        
        logger.debug(f"TTS cache hit: {text[:30]}...")
        return value
    
    def set_tts(self, text: str, audio_bytes: bytes, voice: Optional[str] = None) -> None:
        """
        Cache TTS result.
        
        Args:
            text: Text that was synthesized
            audio_bytes: Generated audio bytes
            voice: Voice model (optional)
        """
        key = self._generate_key("tts", text, voice or "default")
        cache_key = f"tts:{key}"
        self._evict_lru()
        
        self._cache[cache_key] = (audio_bytes, time.time(), "tts")
        self._cache.move_to_end(cache_key)
        
        logger.debug(f"TTS cached: {text[:30]}...")
    
    def get_llm(self, session_id: str, message: str) -> Optional[str]:
        """
        Get cached LLM result.
        
        Args:
            session_id: Session ID
            message: User message
            
        Returns:
            Cached LLM response or None
        """
        key = self._generate_key("llm", session_id, message)
        cache_key = f"llm:{key}"
        
        if cache_key not in self._cache:
            return None
        
        value, timestamp, cache_type = self._cache[cache_key]
        
        if self._is_expired(timestamp, cache_type):
            del self._cache[cache_key]
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(cache_key)
        
        logger.debug(f"LLM cache hit: {message[:30]}...")
        return value
    
    def set_llm(self, session_id: str, message: str, response: str) -> None:
        """
        Cache LLM result.
        
        Args:
            session_id: Session ID
            message: User message
            response: LLM response
        """
        key = self._generate_key("llm", session_id, message)
        cache_key = f"llm:{key}"
        self._evict_lru()
        
        self._cache[cache_key] = (response, time.time(), "llm")
        self._cache.move_to_end(cache_key)
        
        logger.debug(f"LLM cached: {message[:30]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def invalidate_session(self, session_id: str) -> None:
        """
        Invalidate all cache entries for a session.
        
        Args:
            session_id: Session ID to invalidate
        """
        keys_to_remove = [
            key for key in self._cache.keys()
            if key.startswith(f"llm:") and session_id in str(self._cache[key])
        ]
        
        for key in keys_to_remove:
            del self._cache[key]
        
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries for session {session_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        stt_count = sum(1 for k in self._cache.keys() if k.startswith("stt:"))
        tts_count = sum(1 for k in self._cache.keys() if k.startswith("tts:"))
        llm_count = sum(1 for k in self._cache.keys() if k.startswith("llm:"))
        
        return {
            "total_entries": len(self._cache),
            "max_size": self.max_size,
            "stt_entries": stt_count,
            "tts_entries": tts_count,
            "llm_entries": llm_count,
        }
