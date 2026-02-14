"""
Agent video upload API: POST /videos/upload for agent-creation chat flow.

Returns video_path that the chat endpoint can use in ChatMessageRequest.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import uuid
from pathlib import Path

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from fastapi import APIRouter, Depends, File, HTTPException, status, UploadFile

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...application.dto.user_dto import UserResponse
from ...core.config import get_settings
from ...domain.constants import (
    AGENT_VIDEO_UPLOAD_SUBDIR,
    ALLOWED_VIDEO_EXTENSIONS,
)

from .dependencies import get_current_user

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
router = APIRouter(tags=["video-upload"])


def get_agent_upload_dir() -> Path:
    """Return the directory for agent-creation video uploads; creates it if needed."""
    base = Path(get_settings().static_video_upload_dir)
    upload_dir = base / AGENT_VIDEO_UPLOAD_SUBDIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@router.post("/videos/upload")
async def upload_video_for_agent(
    file: UploadFile = File(...),
    current_user: UserResponse = Depends(get_current_user),
) -> dict:
    """
    Upload a video for agent-creation chat flow.

    Returns:
      {
        "video_path": "<absolute_path>",
        "filename": "<original_filename>",
        "status": "uploaded"
      }
    """
    # Auth required; dependency validates user

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing file name.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video file. Use: mp4, webm, avi, mov, mkv",
        )

    settings = get_settings()
    max_bytes = settings.static_video_upload_max_mb * 1024 * 1024

    safe_name = f"{uuid.uuid4().hex}{ext}"
    final_path = (get_agent_upload_dir() / safe_name).resolve()

    size = 0
    with open(final_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                final_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Max {settings.static_video_upload_max_mb} MB.",
                )
            f.write(chunk)

    return {
        "video_path": str(final_path),
        "filename": file.filename,
        "status": "uploaded",
    }
