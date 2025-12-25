# Standard library imports
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Tuple, Dict, Any
import logging
import asyncio
import queue

# External package imports
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local application imports
from .api.v1 import auth_router, camera_router, chat_router, general_chat_router, device_router, notifications_router
from .api.v1.notifications_controller import set_websocket_manager
from .infrastructure.messaging import KafkaEventConsumer
from .infrastructure.notifications import WebSocketManager, NotificationService

logger = logging.getLogger(__name__)

# Global instances
_kafka_consumer: Optional[KafkaEventConsumer] = None
_websocket_manager: Optional[WebSocketManager] = None
_notification_service: Optional[NotificationService] = None
_notification_queue: Optional[queue.Queue] = None


async def process_notification_queue():
    """
    Background task to process notifications from queue and send via WebSocket.
    
    This task runs continuously, processing notifications queued by the Kafka consumer
    and sending them to connected WebSocket clients.
    
    Uses asyncio.to_thread() to poll the thread-safe queue in a non-blocking way.
    """
    global _notification_queue, _websocket_manager
    
    if not _notification_queue or not _websocket_manager:
        logger.warning("Notification queue or WebSocket manager not available")
        return
    
    logger.info("Notification queue processor started")
    
    def get_from_queue():
        """Helper function to get from queue with timeout"""
        try:
            return _notification_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    while True:
        try:
            # Poll queue using asyncio.to_thread to avoid blocking the event loop
            result = await asyncio.to_thread(get_from_queue)
            
            if result is None:
                # Queue is empty (timeout), continue polling
                await asyncio.sleep(0.1)  # Small sleep to avoid busy-waiting
                continue
            
            user_id, notification = result
            
            # Send notification to user's WebSocket connections
            sent_count = await _websocket_manager.send_to_user(user_id, notification)
            
            if sent_count > 0:
                logger.debug(f"Sent notification to {sent_count} connection(s) for user {user_id}")
            
        except asyncio.CancelledError:
            logger.info("Notification queue processor cancelled")
            break
        except Exception as e:
            logger.error(f"Error processing notification from queue: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.
    
    Initializes WebSocket manager, notification service, notification queue,
    starts Kafka consumer, and background notification processor.
    """
    global _kafka_consumer, _websocket_manager, _notification_service, _notification_queue
    
    # Initialize WebSocket manager and notification service
    try:
        _websocket_manager = WebSocketManager()
        _notification_service = NotificationService()
        # Use standard library queue for thread-safe communication between Kafka thread and async event loop
        _notification_queue = queue.Queue(maxsize=1000)  # Max 1000 queued notifications
        
        # Set global WebSocket manager for notifications controller
        set_websocket_manager(_websocket_manager)
        
        logger.info("WebSocket manager and notification service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize notification services: {e}", exc_info=True)
        _websocket_manager = None
        _notification_service = None
        _notification_queue = None
    
    # Start background task to process notification queue
    notification_task = None
    if _notification_queue and _websocket_manager:
        try:
            notification_task = asyncio.create_task(process_notification_queue())
            logger.info("Notification queue processor task started")
        except Exception as e:
            logger.error(f"Failed to start notification queue processor: {e}", exc_info=True)
    
    # Startup: Start Kafka consumer
    try:
        _kafka_consumer = KafkaEventConsumer(
            websocket_manager=_websocket_manager,
            notification_service=_notification_service,
            notification_queue=_notification_queue
        )
        _kafka_consumer.start()
        logger.info("Kafka consumer started during application startup")
    except ImportError as e:
        logger.warning(
            f"Kafka consumer not available (kafka-python not installed or Kafka unavailable): {e}. "
            f"Events will not be consumed from Kafka."
        )
        _kafka_consumer = None
    except Exception as e:
        logger.error(f"Failed to start Kafka consumer: {e}", exc_info=True)
        # Don't fail app startup if Kafka is unavailable
        _kafka_consumer = None
    
    yield
    
    # Shutdown: Stop background tasks and services
    if notification_task:
        try:
            notification_task.cancel()
            try:
                await notification_task
            except asyncio.CancelledError:
                pass
            logger.info("Notification queue processor task stopped")
        except Exception as e:
            logger.error(f"Error stopping notification queue processor: {e}", exc_info=True)
    
    if _kafka_consumer:
        try:
            _kafka_consumer.stop()
            logger.info("Kafka consumer stopped during application shutdown")
        except Exception as e:
            logger.error(f"Error stopping Kafka consumer: {e}", exc_info=True)
    
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
    application.include_router(device_router, prefix="/api/v1/devices")
    application.include_router(notifications_router, prefix="/api/v1/notifications")
    
    return application


# Create application instance
app = create_application()

