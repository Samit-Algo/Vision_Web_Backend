# Standard library imports
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Camera:
    """
    Pure domain model for Camera entity - no external dependencies.
    
    This model represents a camera in the domain layer and should not
    depend on application or infrastructure layers.
    """
    id: Optional[str]
    owner_user_id: str
    name: str
    stream_url: str
    device_id: Optional[str] = None
    stream_config: Optional[Dict[str, Any]] = None
    webrtc_config: Optional[Dict[str, Any]] = None  # Per-camera WebRTC configuration
    
    def __post_init__(self) -> None:
        """Business validations"""
        if not self.owner_user_id:
            raise ValueError("Owner user ID is required")
        if not self.name or len(self.name.strip()) < 1:
            raise ValueError("Camera name is required")
        if not self.stream_url or len(self.stream_url.strip()) < 1:
            raise ValueError("Stream URL is required")

