# Standard library imports
import os
from typing import Final, Optional


class Settings:
    """
    Application settings loaded from environment variables.
    
    This class centralizes all configuration settings for the application.
    All settings are loaded from environment variables with sensible defaults.
    """
    
    def __init__(self) -> None:
        # Database Configuration
        self.mongo_uri: Final[str] = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.mongo_database_name: Final[str] = os.getenv("MONGO_DB_NAME", "algo_vision_app_cloud")
        
        # JWT Configuration
        self.jwt_secret_key: Final[str] = os.getenv("JWT_SECRET_KEY", "change_this_secret_in_production")
        self.jwt_algorithm: Final[str] = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes: Final[int] = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")
        )
        
        # Chat/LLM Configuration
        self.groq_api_key: Final[str] = os.getenv("GROQ_API_KEY", "")
        self.llm_temperature: Final[float] = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.llm_model: Final[str] = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.memory_recent_limit: Final[int] = int(os.getenv("MEMORY_RECENT_LIMIT", "12"))
        self.memory_max_chars: Final[int] = int(os.getenv("MEMORY_MAX_CHARS", "1000"))
        self.local_timezone: Final[str] = os.getenv("LOCAL_TIMEZONE", "Asia/Kolkata")
        
        # Audio Service Configuration (STT & TTS)
        # Provider options: "groq" or "local"
        self.stt_provider: Final[str] = os.getenv("STT_PROVIDER", "groq").lower()
        self.tts_provider: Final[str] = os.getenv("TTS_PROVIDER", "groq").lower()
        
        # Groq Configuration
        self.groq_stt_api_key: Final[str] = os.getenv("GROQ_API_KEY_voice", "")
        self.groq_tts_api_key: Final[str] = os.getenv("GROQ_API_KEY_voice", "")
        self.groq_tts_model: Final[str] = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
        self.groq_tts_voice: Final[str] = os.getenv("GROQ_TTS_VOICE", "autumn")
        
        # VLM Configuration (Vision Language Model for weapon detection)
        self.vlm_model: Final[str] = os.getenv("VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        
        # Local Model Configuration
        # Faster Whisper model size: tiny, base, small, medium, large, large-v2, large-v3
        self.local_stt_model: Final[str] = os.getenv("LOCAL_STT_MODEL", "base")
        # Edge TTS voice (e.g., "en-US-AriaNeural", "en-GB-SoniaNeural")
        self.local_tts_voice: Final[str] = os.getenv("LOCAL_TTS_VOICE", "en-US-AriaNeural")
        
        # Jetson Backend Configuration
        self.jetson_backend_url: Final[str] = os.getenv(
            "JETSON_BACKEND_URL",
            "http://localhost:8001"
        )
        
        # Web Backend Configuration (for Jetson backend to connect back)
        self.web_backend_url: Final[str] = os.getenv(
            "WEB_BACKEND_URL",
            "http://localhost:8000"
        )
        
        # Kafka Configuration
        self.kafka_bootstrap_servers: Final[str] = os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            "localhost:9092"
        )
        self.kafka_topic: Final[str] = os.getenv(
            "KAFKA_TOPIC",
            "vision-events"
        )
        self.kafka_consumer_group_id: Final[str] = os.getenv(
            "KAFKA_CONSUMER_GROUP_ID",
            "vision-backend-consumer"
        )
        self.kafka_auto_offset_reset: Final[str] = os.getenv(
            "KAFKA_AUTO_OFFSET_RESET",
            "latest"
        )
        self.kafka_enable_auto_commit: Final[bool] = os.getenv(
            "KAFKA_ENABLE_AUTO_COMMIT",
            "true"
        ).lower() == "true"
        
        # Event Session Configuration
        self.event_session_timeout_seconds: Final[int] = int(
            os.getenv("EVENT_SESSION_TIMEOUT_SECONDS", "30")
        )
        self.event_video_chunk_duration_seconds: Final[int] = int(
            os.getenv("EVENT_VIDEO_CHUNK_DURATION_SECONDS", "300")  # 5 minutes
        )
        self.event_session_check_interval_seconds: Final[int] = int(
            os.getenv("EVENT_SESSION_CHECK_INTERVAL_SECONDS", "5")
        )
        self.event_video_fps: Final[int] = int(
            os.getenv("EVENT_VIDEO_FPS", "5")
        )
        self.event_video_resolution_width: Final[int] = int(
            os.getenv("EVENT_VIDEO_RESOLUTION_WIDTH", "1280")
        )
        self.event_video_resolution_height: Final[int] = int(
            os.getenv("EVENT_VIDEO_RESOLUTION_HEIGHT", "720")
        )
        # Event Video File Storage
        self.event_video_save_directory: Final[str] = os.getenv(
            "EVENT_VIDEO_SAVE_DIRECTORY",
            os.path.join(os.getcwd(), "event_videos")  # Default: ./event_videos
        )
        self.event_video_save_enabled: Final[bool] = os.getenv(
            "EVENT_VIDEO_SAVE_ENABLED", "true"
        ).lower() in ("true", "1", "yes")


# Global settings instance (singleton pattern)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings (singleton pattern)
    
    Returns:
        Settings instance with all configuration values
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

