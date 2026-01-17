from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from ...utils.datetime_utils import utc_now


@dataclass
class Agent:
    """Domain model for Vision Analytics Agent"""
    id: Optional[str] = None
    name: str = ""
    camera_id: str = ""
    model: str = ""
    fps: Optional[int] = None
    rules: List[Dict[str, Any]] = field(default_factory=list)
    run_mode: Optional[str] = None
    interval_minutes: Optional[int] = None
    check_duration_seconds: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    zone: Optional[Dict[str, Any]] = None
    requires_zone: bool = False
    status: str = "ACTIVE"
    created_at: Optional[datetime] = None
    owner_user_id: Optional[str] = None
    stream_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Business validations"""
        if not self.camera_id:
            raise ValueError("Camera ID is required")
        if not self.model:
            raise ValueError("Model is required")
        if not self.rules:
            raise ValueError("At least one rule is required")
        if self.created_at is None:
            # Store persisted timestamps as UTC. Mongo stores BSON dates as UTC.
            self.created_at = utc_now()
