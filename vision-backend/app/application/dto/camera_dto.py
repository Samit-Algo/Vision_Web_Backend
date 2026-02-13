from typing import Optional, List, Dict
from pydantic import BaseModel


class WebRTCConfig(BaseModel):
    """WebRTC configuration DTO"""
    signaling_url: str
    viewer_id: str
    ice_servers: List[Dict]


class StreamConfig(BaseModel):
    """Stream configuration DTO"""
    camera_id: str
    stream_type: str
    webrtc_config: Optional[WebRTCConfig] = None


class CameraCreateRequest(BaseModel):
    """DTO for camera creation request"""
    name: str
    stream_url: str


class CameraResponse(BaseModel):
    """DTO for camera response"""
    id: str
    name: str
    stream_url: str
    device_id: Optional[str] = None
    stream_config: Optional[StreamConfig] = None
    webrtc_config: Optional[WebRTCConfig] = None  # Per-camera WebRTC configuration

