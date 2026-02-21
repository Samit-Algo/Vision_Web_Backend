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
        # uri = os.getenv("MONGO_URI", "mongodb://mongo:27017").split("#")[0].strip()
        # # On Windows, localhost can cause 30+ sec IPv6 resolution delays; use 127.0.0.1
        # if os.name == "nt" and "localhost" in uri and "127.0.0.1" not in uri:
        #     uri = uri.replace("localhost", "127.0.0.1")
        self.mongo_uri: Final[str] = "mongodb://mongo:27017"
        self.mongo_database_name: Final[str] = os.getenv("MONGO_DB_NAME", "algo_vision_app_cloud").split('#')[0].strip()
        
        # JWT Configuration
        self.jwt_secret_key: Final[str] = os.getenv("JWT_SECRET_KEY", "change_this_secret_in_production").split('#')[0].strip()
        self.jwt_algorithm: Final[str] = os.getenv("JWT_ALGORITHM", "HS256").split('#')[0].strip()
        self.access_token_expire_minutes: Final[int] = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440").split('#')[0].strip()
        )
        
        # Chat/LLM Configuration
        self.groq_api_key: Final[str] = os.getenv("GROQ_API_KEY", "").split('#')[0].strip()
        self.llm_temperature: Final[float] = float(os.getenv("LLM_TEMPERATURE", "0.2").split('#')[0].strip())
        self.llm_model: Final[str] = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile").split('#')[0].strip()
        # Agent creation chatbot model (Groq provider)
        self.agent_creation_model: Final[str] = os.getenv(
            "AGENT_CREATION_MODEL", "groq/qwen/qwen3-32b"
        ).split('#')[0].strip()
        self.memory_recent_limit: Final[int] = int(os.getenv("MEMORY_RECENT_LIMIT", "12").split('#')[0].strip())
        self.memory_max_chars: Final[int] = int(os.getenv("MEMORY_MAX_CHARS", "1000").split('#')[0].strip())
        self.local_timezone: Final[str] = os.getenv("LOCAL_TIMEZONE", "Asia/Kolkata").split('#')[0].strip()
        
        # Audio Service Configuration (STT & TTS)
        # Provider options: "groq" or "local"
        self.stt_provider: Final[str] = os.getenv("STT_PROVIDER", "groq").split('#')[0].strip().lower()
        self.tts_provider: Final[str] = os.getenv("TTS_PROVIDER", "groq").split('#')[0].strip().lower()
        
        # Groq Configuration
        self.groq_stt_api_key: Final[str] = os.getenv("GROQ_API_KEY_voice", "").split('#')[0].strip()
        self.groq_tts_api_key: Final[str] = os.getenv("GROQ_API_KEY_voice", "").split('#')[0].strip()
        self.groq_tts_model: Final[str] = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english").split('#')[0].strip()
        self.groq_tts_voice: Final[str] = os.getenv("GROQ_TTS_VOICE", "autumn").split('#')[0].strip()
        
        # VLM Configuration (Vision Language Model for weapon detection)
        self.vlm_model: Final[str] = os.getenv("VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct").split('#')[0].strip()

        # Gemini API (for static video analysis - native video understanding)
        self.gemini_api_key: Final[str] = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).split('#')[0].strip()
        # Static video agent: Groq reasoning model (qwen3-32b for better inference)
        self.static_video_agent_model: Final[str] = os.getenv(
            "STATIC_VIDEO_AGENT_MODEL", "qwen/qwen3-32b"
        ).split('#')[0].strip()
        
        # Local Model Configuration
        # Faster Whisper model size: tiny, base, small, medium, large, large-v2, large-v3
        self.local_stt_model: Final[str] = os.getenv("LOCAL_STT_MODEL", "base").split('#')[0].strip()
        # Edge TTS voice (e.g., "en-US-AriaNeural", "en-GB-SoniaNeural")
        self.local_tts_voice: Final[str] = os.getenv("LOCAL_TTS_VOICE", "en-US-AriaNeural").split('#')[0].strip()
        
        # Web Backend Configuration (used by detection_tools for API URLs)
        self.web_backend_url: Final[str] = os.getenv(
            "WEB_BACKEND_URL",
            "http://localhost:8000"
        ).split('#')[0].strip()
        
        # Kafka Configuration
        self.kafka_bootstrap_servers: Final[str] = os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            "localhost:9092"
        ).split('#')[0].strip()
        self.kafka_topic: Final[str] = os.getenv(
            "KAFKA_TOPIC",
            "vision-events"
        ).split('#')[0].strip()
        self.kafka_consumer_group_id: Final[str] = os.getenv(
            "KAFKA_CONSUMER_GROUP_ID",
            "vision-backend-consumer"
        ).split('#')[0].strip()
        self.kafka_auto_offset_reset: Final[str] = os.getenv(
            "KAFKA_AUTO_OFFSET_RESET",
            "latest"
        ).split('#')[0].strip()
        self.kafka_enable_auto_commit: Final[bool] = os.getenv(
            "KAFKA_ENABLE_AUTO_COMMIT",
            "true"
        ).split('#')[0].strip().lower() == "true"
        
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
        # Model directory - YOLO .pt files (for Docker: /app/models, local: ./models)
        self.model_dir: Final[str] = os.getenv(
            "MODEL_DIR",
            os.path.join(os.getcwd(), "models")
        ).split('#')[0].strip()

        # Event Video File Storage
        self.event_video_save_directory: Final[str] = os.getenv(
            "EVENT_VIDEO_SAVE_DIRECTORY",
            os.path.join(os.getcwd(), "event_videos")  # Default: ./event_videos
        )
        self.event_video_save_enabled: Final[bool] = os.getenv(
            "EVENT_VIDEO_SAVE_ENABLED", "true"
        ).lower() in ("true", "1", "yes")

        # Static video uploads (for "create agent for this video" in chat)
        self.static_video_upload_dir: Final[str] = os.getenv(
            "STATIC_VIDEO_UPLOAD_DIR",
            os.path.join(os.getcwd(), "static_video_uploads"),
        ).split("#")[0].strip()
        self.static_video_upload_max_mb: Final[int] = int(
            os.getenv("STATIC_VIDEO_UPLOAD_MAX_MB", "500").split("#")[0].strip()
        )

        # Static video analysis (Gemini + ChromaDB)
        self.static_video_vector_db_dir: Final[str] = os.getenv(
            "STATIC_VIDEO_VECTOR_DB_DIR",
            os.path.join(os.getcwd(), "static_video_chroma_db"),
        ).split("#")[0].strip()
        # Person gallery (reference photos for face recognition)
        self.gallery_dir: Final[str] = os.getenv(
            "GALLERY_DIR",
            os.path.join(os.getcwd(), "Gallery")
        ).split("#")[0].strip()


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

