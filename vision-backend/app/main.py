# Standard library imports
from pathlib import Path

# External package imports
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local application imports
from .api.v1 import auth_router, camera_router, chat_router, general_chat_router, device_router


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
    
    return application


# Create application instance
app = create_application()

