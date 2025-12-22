from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel


class AgentResponse(BaseModel):
    """DTO for agent response"""
    id: Optional[str] = None
    name: str
    camera_id: str
    model: str
    fps: Optional[int] = None
    rules: List[Dict[str, Any]] = []
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

