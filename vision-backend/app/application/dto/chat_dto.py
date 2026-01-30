from pydantic import BaseModel
from typing import Optional, Dict, Any


class ChatMessageRequest(BaseModel):
    """Request model for chat message"""
    message: str
    session_id: Optional[str] = None
    zone_data: Optional[Dict[str, Any]] = None  # Zone data from UI drawing
    camera_id: Optional[str] = None  # Camera ID passed from UI when user clicks "Add Agent" on a camera


class ChatMessageResponse(BaseModel):
    """Response model for chat message"""
    response: str
    session_id: str
    status: Optional[str] = None
    # Context for UI (resolved/selected camera)
    camera_id: Optional[str] = None
    # Zone UI control signals
    zone_required: bool = False  # Computed from state: requires_zone && zone is None
    awaiting_zone_input: bool = False  # LLM is currently asking for zone input
    frame_snapshot_url: Optional[str] = None  # URL to fetch camera snapshot for zone drawing
    zone_type: Optional[str] = None  # Type of zone: "line" (2 points) or "polygon" (3+ points)
    # Flow diagram data (raw JSON for frontend transformation)
    flow_diagram_data: Optional[Dict[str, Any]] = None  # Generic flow data: {"nodes": [...], "links": [...]}
