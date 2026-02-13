"""
Agent video upload API (separate from static video analyzer).

Endpoint:
  POST /videos/upload

Purpose:
  Upload a video for agent-creation chat flow. Returns a concrete `video_path`
  that the chat endpoint can pass as `video_path` in ChatMessageRequest.
"""

from pathlib import Path
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ...application.dto.user_dto import UserResponse
from ...core.config import get_settings
from .dependencies import get_current_user


router = APIRouter(tags=["video-upload"])

ALLOWED_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def _agent_upload_dir() -> Path:
    base = Path(get_settings().static_video_upload_dir)
    upload_dir = base / "agent"
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
    _ = current_user  # authentication is required; user is validated by dependency

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing file name.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video file. Use: mp4, webm, avi, mov, mkv",
        )

    settings = get_settings()
    max_bytes = settings.static_video_upload_max_mb * 1024 * 1024

    safe_name = f"{uuid.uuid4().hex}{ext}"
    final_path = (_agent_upload_dir() / safe_name).resolve()

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
