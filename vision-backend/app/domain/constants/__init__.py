"""Constants for domain model field names and shared media/upload rules."""

from .user_fields import UserFields
from .camera_fields import CameraFields
from .agent_fields import AgentFields
from .event_fields import EventFields
from .media_constants import (
    AGENT_VIDEO_UPLOAD_SUBDIR,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_IMAGE_MIME,
    ALLOWED_VIDEO_EXTENSIONS,
    MIN_PHOTOS_PER_PERSON,
    STATIC_VIDEO_UPLOAD_SUBDIR,
)

__all__ = [
    "UserFields",
    "CameraFields",
    "AgentFields",
    "EventFields",
    "AGENT_VIDEO_UPLOAD_SUBDIR",
    "ALLOWED_IMAGE_EXTENSIONS",
    "ALLOWED_IMAGE_MIME",
    "ALLOWED_VIDEO_EXTENSIONS",
    "MIN_PHOTOS_PER_PERSON",
    "STATIC_VIDEO_UPLOAD_SUBDIR",
]

