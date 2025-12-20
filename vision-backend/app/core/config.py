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
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
        )
        
        # Chat/LLM Configuration
        self.groq_api_key: Final[str] = os.getenv("GROQ_API_KEY", "")
        self.llm_temperature: Final[float] = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.llm_model: Final[str] = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.memory_recent_limit: Final[int] = int(os.getenv("MEMORY_RECENT_LIMIT", "12"))
        self.memory_max_chars: Final[int] = int(os.getenv("MEMORY_MAX_CHARS", "1000"))
        self.local_timezone: Final[str] = os.getenv("LOCAL_TIMEZONE", "Asia/Kolkata")
        
        # Jetson Backend Configuration
        self.jetson_backend_url: Final[str] = os.getenv(
            "JETSON_BACKEND_URL",
            "http://localhost:8001"
        )


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

