# Standard library imports
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Tuple, Dict, Any
import logging
import asyncio
import threading
from multiprocessing import Manager

# External package imports
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local application imports
from .api.v1 import (
    auth_router,
    camera_router,
    chat_router,
    general_chat_router,
    notifications_router,
    streaming_router,
    events_router,
    static_video_analysis_router,
    video_upload_router,
)
from .api.v1.notifications_controller import set_websocket_manager
from .infrastructure.notifications import WebSocketManager, NotificationService
from .infrastructure.streaming import WsFmp4Service
from .infrastructure.messaging import KafkaEventConsumer
from .di.container import get_container
from .processing.main_process.run_vision_app import main as run_runner
from .processing.helpers import set_shared_store
from .utils.event_session_manager import get_event_session_manager

logger = logging.getLogger(__name__)

# Global instances
_websocket_manager: Optional[WebSocketManager] = None
_notification_service: Optional[NotificationService] = None
_kafka_consumer: Optional[KafkaEventConsumer] = None
_shared_store: Optional[Any] = None
_runner_thread: Optional[threading.Thread] = None


def get_websocket_manager() -> Optional[WebSocketManager]:
    """
    Get the global WebSocketManager instance.
    
    This allows other modules (like event_notifier) to access the WebSocketManager
    to send notifications directly to connected clients.
    
    Returns:
        WebSocketManager instance if available, None otherwise
    """
    return _websocket_manager


def get_notification_service() -> Optional[NotificationService]:
    """
    Get the global NotificationService instance.
    
    Returns:
        NotificationService instance if available, None otherwise
    """
    return _notification_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.
    
    Initializes WebSocket manager, notification service,
    Event Session Manager, and YOLO processing Runner.
    """
    global _websocket_manager, _notification_service, _kafka_consumer
    global _shared_store, _runner_thread
    
    # Initialize WebSocket manager and notification service
    try:
        _websocket_manager = WebSocketManager()
        _notification_service = NotificationService()
        
        # Set global WebSocket manager for notifications controller
        set_websocket_manager(_websocket_manager)
        
        logger.info("WebSocket manager and notification service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize notification services: {e}", exc_info=True)
        _websocket_manager = None
        _notification_service = None

    # Startup: Start Kafka consumer (FastAPI side) for realtime notifications
    # IMPORTANT: no retry logic; if Kafka is down, we just log and continue.
    try:
        if _websocket_manager and _notification_service:
            if KafkaEventConsumer is not None:
                loop = asyncio.get_running_loop()
                _kafka_consumer = KafkaEventConsumer(
                    websocket_manager=_websocket_manager,
                    notification_service=_notification_service,
                    loop=loop,
                )
                _kafka_consumer.start()
                logger.info("KafkaEventConsumer started")
    except Exception as e:
        logger.warning(f"KafkaEventConsumer not started: {e}")
        _kafka_consumer = None
    
    # Startup: Initialize Event Session Manager
    try:
        session_manager = get_event_session_manager()  # This starts background workers
        logger.info("Event Session Manager initialized and started")
    except Exception as e:
        logger.error(f"Failed to initialize Event Session Manager: {e}", exc_info=True)
    
    # Startup: Create shared store for runner (frame sharing between CameraPublisher and workers)
    try:
        manager = Manager()
        _shared_store = manager.dict()
        # Expose to other modules (e.g. streaming overlays)
        set_shared_store(_shared_store)
        logger.info("Shared store created for frame sharing")
    except Exception as e:
        logger.error(f"Failed to create shared store: {e}", exc_info=True)
        _shared_store = None
    
    # Startup: Start Runner in background thread
    # Runner will start CameraPublisher for each camera and Worker for each agent
    try:
        def run_runner_with_store():
            run_runner(_shared_store)
        
        _runner_thread = threading.Thread(target=run_runner_with_store, daemon=True)
        _runner_thread.start()
        logger.info("YOLO processing Runner started in background thread")
        logger.info("Waiting for CameraPublisher to initialize...")
        
        # Give CameraPublisher time to connect and start publishing
        await asyncio.sleep(3)
        logger.info("Runner initialization complete")
    except Exception as e:
        logger.error(f"Failed to start Runner: {e}", exc_info=True)
        _runner_thread = None
    
    yield
    
    # Shutdown: Clean up live WS streams
    try:
        container = get_container()
        ws_service: WsFmp4Service = container.get(WsFmp4Service)
        await ws_service.cleanup_all_streams()
        logger.info("All live WS streams stopped during application shutdown")
    except Exception as e:
        logger.error(f"Error stopping live WS streams: {e}", exc_info=True)
    
    # Shutdown: Stop Event Session Manager
    try:
        session_manager = get_event_session_manager()
        session_manager.stop()
        logger.info("Event Session Manager stopped")
    except Exception as e:
        logger.error(f"Error stopping Event Session Manager: {e}", exc_info=True)

    # Shutdown: Stop Kafka consumer
    try:
        if _kafka_consumer:
            _kafka_consumer.stop()
            logger.info("KafkaEventConsumer stopped")
    except Exception as e:
        logger.error(f"Error stopping KafkaEventConsumer: {e}", exc_info=True)
    
    # Shutdown: Cleanup complete
    
    # Note: Runner thread is daemon, so it will terminate when main process exits
    # CameraPublisher and Worker processes are also daemon, so they will terminate automatically
    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    This function sets up the FastAPI application with:
    - Environment variable loading
    - CORS middleware configuration
    - API route registration
    
    Returns:
        Configured FastAPI application instance
    """
    # Load environment variables from .env file
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    
    # Create FastAPI app
    application = FastAPI(
        title="Vision Backend API",
        version="1.0.0",
        description="Clean Architecture Vision Backend Application",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",  # Desktop app (Electron)
            "http://localhost:8081",
            "https://spicy-garlics-wonder.loca.lt",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register API routers
    application.include_router(auth_router, prefix="/api/v1/auth")
    application.include_router(camera_router, prefix="/api/v1/cameras")
    application.include_router(chat_router, prefix="/api/v1/chat")
    application.include_router(general_chat_router, prefix="/api/v1/general-chat")
    application.include_router(notifications_router, prefix="/api/v1/notifications")
    application.include_router(events_router, prefix="/api/v1/events")
    application.include_router(streaming_router, prefix="/api/v1/streams")
    application.include_router(video_upload_router, prefix="/api/v1")
    application.include_router(static_video_analysis_router, prefix="/api/v1")

    return application


# Create application instance
app = create_application()

