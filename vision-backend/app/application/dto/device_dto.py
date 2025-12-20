from pydantic import BaseModel, HttpUrl


class DeviceCreateRequest(BaseModel):
    """DTO for device creation request"""
    id: str  # Device ID (e.g., "DEV-790EECBE4CF2")
    name: str
    jetson_backend_url: str  # URL of the Jetson backend for this device


class DeviceResponse(BaseModel):
    """DTO for device response"""
    id: str
    owner_user_id: str
    name: str
    jetson_backend_url: str

