"""
Static video analysis — two simple APIs:

  1. POST /videos/static — upload video + optional question → returns video_id
  2. POST /videos/static/ask — video_id + question → returns answer
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, status, UploadFile

from ...application.dto.static_video_dto import AskResponse
from ...application.dto.user_dto import UserResponse
from ...core.config import get_settings
from ...static_video_analysis import (
    analyze_video,
    analyze_video_full,
    get_video_path,
    has_video,
    register_video,
    run_agent,
    store_analysis,
)
from .dependencies import get_current_user


router = APIRouter(tags=["static-video-analysis"])

ALLOWED_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def _upload_dir() -> Path:
    base = Path(get_settings().static_video_upload_dir)
    upload_dir = base / "static"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _index_video(video_id: str, video_path: str, question: str | None) -> None:
    """Background: analyze (with question or general) and store."""
    try:
        if has_video(video_id):
            return
        analysis = analyze_video(video_path, question)

        store_analysis(video_id, analysis)
    except Exception as e:
        print(f"Indexing failed for {video_id}: {e}")


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
    if not file.filename or Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video file. Use: mp4, webm, avi, mov, mkv",
        )

    settings = get_settings()
    max_bytes = settings.static_video_upload_max_mb * 1024 * 1024

    video_id = uuid.uuid4().hex
    ext = Path(file.filename).suffix.lower()
    video_path = _upload_dir() / f"{video_id}{ext}"

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
    background_tasks.add_task(_index_video, video_id, str(video_path.resolve()), question)

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
