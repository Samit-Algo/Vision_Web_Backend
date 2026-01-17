from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventListItemResponse(BaseModel):
    id: str
    session_id: str
    label: str
    severity: str
    rule_index: Optional[int] = None
    camera_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    device_id: Optional[str] = None
    event_ts: Optional[datetime] = None
    received_at: datetime
    has_image: bool = False


class EventListResponse(BaseModel):
    total: int = 0
    items: List[EventListItemResponse] = Field(default_factory=list)


class EventDetailResponse(BaseModel):
    id: str
    owner_user_id: Optional[str] = None
    session_id: str

    label: str
    severity: str
    rule_index: Optional[int] = None

    camera_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    device_id: Optional[str] = None

    event_ts: Optional[datetime] = None
    received_at: datetime

    has_image: bool = False
    has_json: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
