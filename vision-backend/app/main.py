# Standard library imports
from pathlib import Path

# External package imports
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local application imports
from .api.v1 import auth_router, camera_router, chat_router, general_chat_router, device_router, event_router
from .utils.event_storage import save_event_from_payload
from fastapi import HTTPException, status
from typing import Dict, Any


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
        description="Clean Architecture Vision Backend Application"
    )
    
    # Add CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
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
    application.include_router(event_router, prefix="/api/v1")
    
    # Add direct event endpoints (without /api/v1 prefix) for Jetson compatibility
    @application.post("/api/events", status_code=status.HTTP_201_CREATED)
    async def receive_event_api(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Receive event at /api/events (Jetson compatibility endpoint)."""
        try:
            saved_paths = save_event_from_payload(payload)
            return {
                "status": "success",
                "message": "Event saved successfully",
                "paths": saved_paths
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process event: {str(e)}"
            )
    
    @application.post("/events", status_code=status.HTTP_201_CREATED)
    async def receive_event_root(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Receive event at /events (Jetson compatibility endpoint)."""
        try:
            saved_paths = save_event_from_payload(payload)
            return {
                "status": "success",
                "message": "Event saved successfully",
                "paths": saved_paths
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process event: {str(e)}"
            )
    
    @application.post("/api/agents/{agent_id}/events", status_code=status.HTTP_201_CREATED)
    async def receive_event_by_agent_api(agent_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Receive event at /api/agents/{agent_id}/events (Jetson compatibility endpoint)."""
        try:
            # Override agent_id in payload if provided in path
            if "agent" not in payload:
                payload["agent"] = {}
            payload["agent"]["agent_id"] = agent_id
            
            saved_paths = save_event_from_payload(payload)
            return {
                "status": "success",
                "message": "Event saved successfully",
                "paths": saved_paths
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process event: {str(e)}"
            )
    
    return application


# Create application instance
app = create_application()

