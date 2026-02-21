"""
Application entry point for the Vision Backend API.

This module creates the FastAPI application, wires middleware and routers,
and manages startup/shutdown (WebSockets, Kafka, event session, vision runner).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Optional

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------------------------------------
# API (routers)
# -----------------------------------------------------------------------------
from .api.v1 import (
    auth_router,
    camera_router,
    chat_router,
    events_router,
    general_chat_router,
    notifications_router,
    person_gallery_router,
    static_video_analysis_router,
    streaming_router,
    video_upload_router,
)
from .api.v1.notifications_controller import set_websocket_manager

# -----------------------------------------------------------------------------
# Infrastructure
# -----------------------------------------------------------------------------
from .infrastructure.messaging import KafkaEventConsumer
from .infrastructure.notifications import NotificationService, WebSocketManager
from .infrastructure.streaming import ProcessedFrameStreamService, WsFmp4Service

# -----------------------------------------------------------------------------
# DI, processing, utils
# -----------------------------------------------------------------------------
from .di.container import get_container
from .processing.helpers import set_shared_store
from .processing.main_process.run_vision_app import main as run_vision_runner
from .utils.event_session_manager import get_event_session_manager

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants (used in lifespan)
# -----------------------------------------------------------------------------
# Seconds to wait after starting the vision runner so camera publishers can init.
RUNNER_STARTUP_DELAY_SECONDS = 3

# CORS origins allowed for the API (dev/demo).
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8081",
    "https://spicy-garlics-wonder.loca.lt",
]

# -----------------------------------------------------------------------------
# Application-global state (set during lifespan startup, cleared on shutdown)
# -----------------------------------------------------------------------------
websocket_manager: Optional[WebSocketManager] = None
notification_service: Optional[NotificationService] = None
kafka_consumer: Optional[KafkaEventConsumer] = None
shared_store: Optional[Any] = None
runner_thread: Optional[threading.Thread] = None


def get_websocket_manager() -> Optional[WebSocketManager]:
    """Return the global WebSocket manager, or None if not yet initialized."""
    return websocket_manager


def get_notification_service() -> Optional[NotificationService]:
    """Return the global notification service, or None if not yet initialized."""
    return notification_service


# -----------------------------------------------------------------------------
# Lifespan: startup and shutdown
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown.

    Startup (in order):
        1. WebSocket manager and notification service
        2. Kafka event consumer (optional; logs warning if it fails)
        3. Event session manager
        4. Shared store for frame sharing
        5. Vision runner in a background thread

    Shutdown (in order):
        1. Stop all live WebSocket streams (raw + agent processed)
        2. Stop event session manager
        3. Stop Kafka consumer
        (Runner thread is daemon; it exits with the process.)
    """
    global websocket_manager, notification_service, kafka_consumer
    global shared_store, runner_thread

    # ----- Startup -----

    # 1. WebSocket and notification service
    try:
        websocket_manager = WebSocketManager()
        notification_service = NotificationService()
        set_websocket_manager(websocket_manager)
        logger.info("WebSocket manager and notification service initialized")
    except Exception as e:
        logger.error("Failed to initialize notification services: %s", e, exc_info=True)
        websocket_manager = None
        notification_service = None

    # 2. Kafka consumer (optional)
    try:
        if websocket_manager and notification_service and KafkaEventConsumer is not None:
            loop = asyncio.get_running_loop()
            kafka_consumer = KafkaEventConsumer(
                websocket_manager=websocket_manager,
                notification_service=notification_service,
                loop=loop,
            )
            kafka_consumer.start()
            logger.info("Kafka event consumer started")
        else:
            kafka_consumer = None
    except Exception as e:
        logger.warning("Kafka consumer not started: %s", e)
        kafka_consumer = None

    # 3. Event session manager
    try:
        get_event_session_manager()
        logger.info("Event session manager initialized")
    except Exception as e:
        logger.error("Failed to initialize event session manager: %s", e, exc_info=True)

    # 4. Shared store for vision runner (frame sharing)
    try:
        manager = Manager()
        shared_store = manager.dict()
        set_shared_store(shared_store)
        logger.info("Shared store created for frame sharing")
    except Exception as e:
        logger.error("Failed to create shared store: %s", e, exc_info=True)
        shared_store = None

    # 5. Vision runner in background thread (only if shared store is available)
    if shared_store is not None:
        try:
            def run_runner():
                run_vision_runner(shared_store)

            runner_thread = threading.Thread(target=run_runner, daemon=True)
            runner_thread.start()
            logger.info("Vision runner started in background thread")
            await asyncio.sleep(RUNNER_STARTUP_DELAY_SECONDS)
            logger.info("Vision runner initialization complete")
        except Exception as e:
            logger.error("Failed to start vision runner: %s", e, exc_info=True)
            runner_thread = None
    else:
        logger.warning("Vision runner not started: shared store unavailable")
        runner_thread = None

    yield

    # ----- Shutdown -----

    # 1. Stop all live WebSocket streams (raw + agent processed)
    try:
        container = get_container()
        ws_service: WsFmp4Service = container.get(WsFmp4Service)
        await ws_service.cleanup_all_streams()
        logger.info("All live WebSocket streams stopped")

        processed_service: ProcessedFrameStreamService = container.get(
            ProcessedFrameStreamService
        )
        await processed_service.cleanup_all_streams()
        logger.info("All agent processed streams stopped")
    except Exception as e:
        logger.error("Error stopping WebSocket streams: %s", e, exc_info=True)

    # 2. Stop event session manager
    try:
        get_event_session_manager().stop()
        logger.info("Event session manager stopped")
    except Exception as e:
        logger.error("Error stopping event session manager: %s", e, exc_info=True)

    # 3. Stop Kafka consumer
    try:
        if kafka_consumer is not None:
            kafka_consumer.stop()
            logger.info("Kafka consumer stopped")
    except Exception as e:
        logger.error("Error stopping Kafka consumer: %s", e, exc_info=True)

    logger.info("Application shutdown complete")


# -----------------------------------------------------------------------------
# App factory and instance
# -----------------------------------------------------------------------------


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    - Loads environment from .env if present (optional in Docker; use runtime env)
    - Applies CORS middleware
    - Registers all v1 API routers
    - Uses lifespan for startup/shutdown
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    application = FastAPI(
        title="Vision Backend API",
        version="1.0.0",
        description="Clean Architecture Vision Backend Application",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers (v1)
    application.include_router(auth_router, prefix="/api/v1/auth")
    application.include_router(camera_router, prefix="/api/v1/cameras")
    application.include_router(chat_router, prefix="/api/v1/chat")
    application.include_router(general_chat_router, prefix="/api/v1/general-chat")
    application.include_router(notifications_router, prefix="/api/v1/notifications")
    application.include_router(events_router, prefix="/api/v1/events")
    application.include_router(streaming_router, prefix="/api/v1/streams")
    application.include_router(video_upload_router, prefix="/api/v1")
    application.include_router(static_video_analysis_router, prefix="/api/v1")
    application.include_router(person_gallery_router, prefix="/api/v1/person-gallery")

    return application


app = create_application()
