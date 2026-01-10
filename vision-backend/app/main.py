# Standard library imports
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Tuple, Dict, Any
import logging
import asyncio
import queue
from datetime import datetime
from zoneinfo import ZoneInfo

# External package imports
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local application imports
from .api.v1 import auth_router, camera_router, chat_router, general_chat_router, device_router, notifications_router, streaming_router, events_router
from .api.v1.notifications_controller import set_websocket_manager
from .infrastructure.messaging import KafkaEventConsumer
from .infrastructure.notifications import WebSocketManager, NotificationService
from .infrastructure.streaming import WsFmp4Service
from .di.container import get_container
from .domain.models.event import Event
from .domain.repositories.event_repository import EventRepository

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
            
            user_id, payload, saved_paths = result

            # Persist event to DB (Mongo) in the main async loop (safe for Motor)
            event_id = None
            try:
                container = get_container()
                event_repo: EventRepository = container.get(EventRepository)

                # Build a preview notification to reuse session_id construction logic
                preview = _notification_service.format_event_notification(
                    payload,
                    saved_paths=saved_paths,
                    event_id=None,
                    include_frame_base64=False,
                )

                session_id = preview.get("session_id") or (payload.get("metadata", {}) or {}).get("session_id") or ""
                label = (payload.get("event", {}) or {}).get("label", "Event")
                rule_index = (payload.get("event", {}) or {}).get("rule_index")
                camera_id = (payload.get("agent", {}) or {}).get("camera_id")
                agent_id = (payload.get("agent", {}) or {}).get("agent_id")
                agent_name = (payload.get("agent", {}) or {}).get("agent_name")
                device_id = (payload.get("camera", {}) or {}).get("device_id")

                # Parse event timestamp
                event_ts = None
                ts_raw = (payload.get("event", {}) or {}).get("timestamp")
                try:
                    if ts_raw:
                        event_ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                except Exception:
                    event_ts = None

                # Simple backend severity inference (kept in sync with frontend heuristic)
                sev_label = str(label or "").lower()
                if any(k in sev_label for k in ["weapon", "fire", "fall", "intrusion"]):
                    severity = "critical"
                elif any(k in sev_label for k in ["violation", "restricted", "collision", "alert"]):
                    severity = "warning"
                else:
                    severity = "info"

                ev = Event(
                    id=None,
                    owner_user_id=user_id,
                    session_id=session_id,
                    label=label,
                    severity=severity,
                    rule_index=rule_index,
                    camera_id=camera_id,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    device_id=device_id,
                    event_ts=event_ts,
                    received_at=datetime.utcnow(),
                    image_path=saved_paths.get("image_path") if isinstance(saved_paths, dict) else None,
                    json_path=saved_paths.get("json_path") if isinstance(saved_paths, dict) else None,
                    metadata={
                        **((payload.get("metadata", {}) or {})),
                        **({"image_path": saved_paths.get("image_path")} if isinstance(saved_paths, dict) and saved_paths.get("image_path") else {}),
                        **({"json_path": saved_paths.get("json_path")} if isinstance(saved_paths, dict) and saved_paths.get("json_path") else {}),
                    },
                )
                event_id = await event_repo.create(ev)
            except Exception as e:
                logger.error(f"Error persisting event to DB: {e}", exc_info=True)

            # Format notification payload (now includes event_id)
            notification = _notification_service.format_event_notification(
                payload,
                saved_paths=saved_paths,
                event_id=event_id,
                include_frame_base64=False,
            )

            # If user_id is missing (e.g., Jetson payload didn't include owner_user_id),
            # fall back to broadcasting to all connected clients (dev-friendly default).
            if not user_id:
                sent_count = await _websocket_manager.broadcast_to_all(notification)
            else:
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
    
    # Shutdown: Clean up live WS streams
    try:
        container = get_container()
        ws_service: WsFmp4Service = container.get(WsFmp4Service)
        await ws_service.cleanup_all_streams()
        logger.info("All live WS streams stopped during application shutdown")
    except Exception as e:
        logger.error(f"Error stopping live WS streams: {e}", exc_info=True)
    
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
    application.include_router(device_router, prefix="/api/v1/devices")
    application.include_router(notifications_router, prefix="/api/v1/notifications")
    application.include_router(events_router, prefix="/api/v1/events")
    application.include_router(streaming_router, prefix="/api/v1/streams")
    
    return application


# Create application instance
app = create_application()

