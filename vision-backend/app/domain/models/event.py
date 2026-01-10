from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, Dict


@dataclass
class Event:
    """Domain model for an Event entity"""

    id: Optional[str]
    owner_user_id: Optional[str]
    session_id: str

    label: str
    severity: str
    rule_index: Optional[int]

    camera_id: Optional[str]
    agent_id: Optional[str]
    agent_name: Optional[str]
    device_id: Optional[str]

    event_ts: Optional[datetime]
    received_at: datetime

    image_path: Optional[str]
    json_path: Optional[str]

    metadata: Optional[Dict[str, Any]] = None
