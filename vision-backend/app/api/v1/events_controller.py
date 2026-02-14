"""
Events API: list events (with range filter), get event by ID, get event image.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...application.dto.event_dto import EventDetailResponse, EventListResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.event.get_event import GetEventUseCase
from ...application.use_cases.event.list_events import ListEventsUseCase
from ...core.config import get_settings
from ...di.container import get_container
from ...domain.repositories.event_repository import EventRepository
from ...utils.event_storage import EVENTS_BASE_DIR

from .dependencies import get_current_user

# -----------------------------------------------------------------------------
# Logging and router
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
router = APIRouter(tags=["events"])


def parse_iso_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime. Returns None if invalid."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def range_to_utc(range_name: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    Convert range selection (today/yesterday/all) into UTC datetime boundaries.
    Uses LOCAL_TIMEZONE from settings.
    """
    range_name = (range_name or "all").lower().strip()
    if range_name == "all":
        return None, None

    settings = get_settings()
    tz = ZoneInfo(settings.local_timezone)
    now_local = datetime.now(tz=tz)
    start_today_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    if range_name == "today":
        start_local = start_today_local
        end_local = start_local + timedelta(days=1)
    elif range_name == "yesterday":
        end_local = start_today_local
        start_local = end_local - timedelta(days=1)
    else:
        # Fallback to all
        return None, None

    # Convert to UTC-aware
    start_utc = start_local.astimezone(ZoneInfo("UTC"))
    end_utc = end_local.astimezone(ZoneInfo("UTC"))
    return start_utc, end_utc


@router.get("", response_model=EventListResponse)
async def list_events(
    range: str = Query("all", description="today | yesterday | all"),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    # optional: allow explicit iso override in future
    start_ts: Optional[str] = Query(None),
    end_ts: Optional[str] = Query(None),
    current_user: UserResponse = Depends(get_current_user),
) -> EventListResponse:
    """
    List events for the current user.
    - Dashboard uses range=today&limit=5
    - Events board uses range=today|yesterday|all
    """
    container = get_container()
    use_case = container.get(ListEventsUseCase)

    try:
        start_utc, end_utc = range_to_utc(range)
        if start_ts or end_ts:
            start_utc = parse_iso_timestamp(start_ts)
            end_utc = parse_iso_timestamp(end_ts)

        return await use_case.execute(
            owner_user_id=current_user.id,
            start_utc=start_utc,
            end_utc=end_utc,
            limit=limit,
            skip=skip,
        )
    except Exception as e:
        logger.error("Error listing events: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list events")


@router.get("/{event_id}", response_model=EventDetailResponse)
async def get_event(
    event_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> EventDetailResponse:
    container = get_container()
    use_case = container.get(GetEventUseCase)
    try:
        return await use_case.execute(owner_user_id=current_user.id, event_id=event_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error("Error getting event: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get event")


@router.get("/{event_id}/image")
async def get_event_image(
    event_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> FileResponse:
    """
    Serve an event image by event_id (preferred over exposing raw file paths).
    """
    container = get_container()
    repo = container.get(EventRepository)
    ev = await repo.get_by_id(owner_user_id=current_user.id, event_id=event_id)
    if not ev:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    if not ev.image_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not available for this event")

    base_dir = EVENTS_BASE_DIR.resolve()
    try:
        resolved = Path(ev.image_path).resolve()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image path")

    if base_dir not in resolved.parents and resolved != base_dir:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    if resolved.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported image type")
    return FileResponse(path=str(resolved))

