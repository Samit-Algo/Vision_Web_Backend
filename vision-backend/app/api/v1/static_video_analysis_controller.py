"""
Static video analysis API: upload video, ask question, list user library.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import logging
import re
import uuid
from pathlib import Path

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, status, UploadFile

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...application.dto.static_video_dto import AskResponse
from ...application.dto.user_dto import UserResponse
from ...core.config import get_settings
from ...domain.constants import ALLOWED_VIDEO_EXTENSIONS, STATIC_VIDEO_UPLOAD_SUBDIR
from ...static_video_analysis import (
    analyze_video,
    analyze_video_full,
    get_video_path,
    has_video,
    list_user_videos,
    register_video,
    run_agent,
    store_analysis,
)

from .dependencies import get_current_user

# -----------------------------------------------------------------------------
# Router and logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
router = APIRouter(tags=["static-video-analysis"])


def safe_user_segment(user_id: str) -> str:
    """Return a filesystem-safe folder name for the user."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", str(user_id))


def get_static_upload_dir(user_id: str | None = None) -> Path:
    """Return upload directory for static videos; optionally under user subfolder."""
    base = Path(get_settings().static_video_upload_dir)
    upload_dir = base / STATIC_VIDEO_UPLOAD_SUBDIR
    if user_id:
        upload_dir = upload_dir / safe_user_segment(user_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def index_video_background(video_id: str, video_path: str, question: str | None) -> None:
    """Background task: analyze video (with optional question) and store result."""
    try:
        if has_video(video_id):
            return
        analysis = analyze_video(video_path, question)
        store_analysis(video_id, analysis)
    except Exception as e:
        logger.warning("Indexing failed for %s: %s", video_id, e)


@router.post("/videos/static")
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    question: str | None = Form(None),
    current_user: UserResponse = Depends(get_current_user),
) -> dict:
    """
    Upload video + optional question. If question: analyze with it; else use general prompt.
    Returns video_id for questions.
    """
    if not file.filename or Path(file.filename).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video file. Use: mp4, webm, avi, mov, mkv",
        )

    settings = get_settings()
    max_bytes = settings.static_video_upload_max_mb * 1024 * 1024

    video_id = uuid.uuid4().hex
    ext = Path(file.filename).suffix.lower()
    video_path = get_static_upload_dir(current_user.id) / f"{video_id}{ext}"

    size = 0
    with open(video_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                if video_path.exists():
                    video_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Max {settings.static_video_upload_max_mb} MB.",
                )
            f.write(chunk)

    register_video(video_id, str(video_path.resolve()), current_user.id)
    background_tasks.add_task(index_video_background, video_id, str(video_path.resolve()), question)

    return {"video_id": video_id, "status": "uploaded", "indexing": "started"}


@router.post("/videos/static/ask", response_model=AskResponse)
async def ask(
    video_id: str = Form(...),
    question: str = Form(...),
    current_user: UserResponse = Depends(get_current_user),
) -> AskResponse:
    """
    Ask question about a video. Uses RAG first; reanalyzes with Gemini only when needed.
    """
    video_path = get_video_path(video_id, user_id=current_user.id)
    if not video_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found. Upload first via POST /videos/static",
        )

    if not has_video(video_id):
        analysis = analyze_video_full(video_path)
        store_analysis(video_id, analysis)

    answer = run_agent(video_id, question, video_path)
    return AskResponse(answer=answer)


@router.get("/videos/static/library")
async def list_static_video_library(
    current_user: UserResponse = Depends(get_current_user),
) -> dict:
    """
    Return previously uploaded static videos for the authenticated user.
    Source of truth: user's static upload folder used by chat upload flow.
    """
    items: list[dict] = []
    user_dir = get_static_upload_dir(current_user.id)

    if user_dir.exists():
        for file_path in sorted(
            [p for p in user_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            video_id = file_path.stem
            resolved = str(file_path.resolve())
            # Keep registry in sync so /videos/static/ask works for folder-listed videos.
            register_video(video_id, resolved, current_user.id)
            items.append(
                {
                    "video_id": video_id,
                    "video_path": resolved,
                    "filename": file_path.name,
                    "created_at": None,
                }
            )

    # Backward compatibility 1: legacy flat static folder (before per-user folders).
    if not items:
        legacy_dir = get_static_upload_dir()
        if legacy_dir.resolve() != user_dir.resolve() and legacy_dir.exists():
            for file_path in sorted(
                [p for p in legacy_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                video_id = file_path.stem
                resolved = str(file_path.resolve())
                register_video(video_id, resolved, current_user.id)
                items.append(
                    {
                        "video_id": video_id,
                        "video_path": resolved,
                        "filename": file_path.name,
                        "created_at": None,
                    }
                )

    # Backward compatibility 2: include registry-based entries.
    if not items:
        items = list_user_videos(current_user.id)

    return {"videos": items}
