from .auth_controller import router as auth_router
from .camera_controller import router as camera_router
from .chat_controller import router as chat_router
from .general_chat_controller import router as general_chat_router
from .device_controller import router as device_router
from .notifications_controller import router as notifications_router


__all__ = ["auth_router", "camera_router", "chat_router", "general_chat_router", "device_router", "notifications_router"]

