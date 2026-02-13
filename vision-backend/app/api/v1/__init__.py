from .auth_controller import router as auth_router
from .camera_controller import router as camera_router
from .chat_controller import router as chat_router
from .general_chat_controller import router as general_chat_router
from .notifications_controller import router as notifications_router
from .streaming_controller import router as streaming_router
from .events_controller import router as events_router
from .static_video_analysis_controller import router as static_video_analysis_router
from .video_upload_controller import router as video_upload_router
from .person_gallery_controller import router as person_gallery_router


__all__ = [
    "auth_router",
    "camera_router",
    "chat_router",
    "general_chat_router",
    "notifications_router",
    "streaming_router",
    "events_router",
    "static_video_analysis_router",
    "video_upload_router",
    "notifications_router",
    "streaming_router",
    "events_router",
    "person_gallery_router",
]

