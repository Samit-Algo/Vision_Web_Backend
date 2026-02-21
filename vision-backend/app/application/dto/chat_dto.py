from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class ChatMessageRequest(BaseModel):
    """Request model for chat message"""
    message: Optional[str] = None  # Omit when sending resume (HITL approve/reject)
    session_id: Optional[str] = None
    zone_data: Optional[Dict[str, Any]] = None  # Zone data from UI drawing
    camera_id: Optional[str] = None  # Camera ID passed from UI when user clicks "Add Agent" on a camera
    video_path: Optional[str] = None  # Video file path when user creates agent for an uploaded video (no camera_id)
    # Human-in-the-loop: after backend returned pending_approval, send same session_id + resume (no message)
    resume: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Resume after approval: { "decisions": [{ "type": "approve" }] } or [{ "type": "reject" }]',
    )


class ApprovalSummarySchema(BaseModel):
    """Config snapshot for the approval UI."""
    rule_id: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class PendingApprovalSchema(BaseModel):
    """Human-in-the-loop: save_to_db is waiting for user approve/reject."""
    action_requests: List[Dict[str, Any]] = Field(default_factory=list)
    review_configs: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Optional[ApprovalSummarySchema] = None


class PendingZoneInputSchema(BaseModel):
    """Human-in-the-loop: request_zone_drawing — UI must show zone canvas; user draws and saves, then send resume with zone."""
    camera_id: Optional[str] = None
    frame_snapshot_url: Optional[str] = None
    zone_type: str = "polygon"  # "polygon" | "line"
    rule_id: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


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
    # Human-in-the-loop: when set, UI must show approve/reject then POST again with resume
    pending_approval: Optional[PendingApprovalSchema] = None
    # Human-in-the-loop: when set, UI must show zone drawing canvas; user draws and saves, then POST with resume including zone
    pending_zone_input: Optional[PendingZoneInputSchema] = None
