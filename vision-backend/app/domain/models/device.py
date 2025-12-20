# Standard library imports
from dataclasses import dataclass
from typing import Optional


@dataclass
class Device:
    """
    Pure domain model for Device entity.
    
    Represents a Jetson device that can process camera streams.
    Each device has a unique ID and a backend URL where it's running.
    """
    id: Optional[str]
    owner_user_id: str
    name: str
    jetson_backend_url: str  # URL of the Jetson backend for this device
    
    def __post_init__(self) -> None:
        """Business validations"""
        if not self.owner_user_id:
            raise ValueError("Owner user ID is required")
        if not self.name or len(self.name.strip()) < 1:
            raise ValueError("Device name is required")
        if not self.jetson_backend_url or len(self.jetson_backend_url.strip()) < 1:
            raise ValueError("Jetson backend URL is required")

