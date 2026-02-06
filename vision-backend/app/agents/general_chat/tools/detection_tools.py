"""
Detection and event related tools for the General Chat Agent.
"""

import asyncio
from datetime import timedelta
from typing import List, Dict, Any, Optional


def _run_sync(coro):
    """Run coroutine from sync code. Works inside or outside a running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


def get_recent_detections(
    user_id: str, 
    camera_id: str = "", 
    days_ago: int = 0,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get recent vision detections/events for the user.
    
    Args:
        user_id: The ID of the current user.
        camera_id: (Optional) The specific camera ID to filter by.
        days_ago: How many days ago to look (0 for today, 1 for yesterday, etc.).
        limit: Max number of results to return (default 10).
    """
    from ....di.container import get_container
    from ....domain.repositories.event_repository import EventRepository
    from ....domain.repositories.camera_repository import CameraRepository
    from ....utils.datetime_utils import now, ensure_utc
    from ....core.config import get_settings

    print(f"[Backend Tool Debug] get_recent_detections called: user_id={user_id}, days_ago={days_ago}")

    container = get_container()
    event_repo = container.get(EventRepository)
    camera_repo = container.get(CameraRepository)
    settings = get_settings()
    base_url = settings.web_backend_url.rstrip("/")
    
    local_now = now()
    target_date = local_now.date() - timedelta(days=days_ago)
    
    start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_ago)
    end_local = start_local + timedelta(days=1)
    
    start_utc = ensure_utc(start_local)
    end_utc = ensure_utc(end_local)
    
    try:
        async def _fetch():
            total, events = await event_repo.list(
                owner_user_id=user_id,
                start_utc=start_utc,
                end_utc=end_utc,
                limit=100,
                skip=0
            )
            
            # Pre-fetch all cameras to resolve names efficiently
            cameras = await camera_repo.find_by_owner(user_id)
            camera_map = {c.id: c.name for c in cameras}
            
            return events, camera_map
        
        events, camera_map = _run_sync(_fetch())
        
        print(f"[Backend Tool Debug] Found {len(events)} total events in repository.")

        filtered_events = []
        for e in events:
            if camera_id and e.camera_id != camera_id:
                continue
            
            # Resolve camera name
            camera_name = e.metadata.get("camera_name")
            if not camera_name and e.camera_id:
                camera_name = camera_map.get(e.camera_id, "Unknown")
            
            evidence_image_url = f"{base_url}/api/v1/events/{e.id}/image" if e.image_path else None
            
            print(f"[Backend Tool Debug] Event {e.id}: has_image={bool(e.image_path)}, url={evidence_image_url}")

            filtered_events.append({
                "event_id": e.id,
                "label": e.label,
                "camera_name": camera_name or "Unknown",
                "camera_id": e.camera_id,
                "timestamp": e.event_ts.strftime("%I:%M %p"),
                "severity": e.severity,
                "evidence_url": evidence_image_url
            })
            
            if len(filtered_events) >= limit:
                break
                
        return {
            "status": "success",
            "date": target_date.isoformat(),
            "events": filtered_events
        }
        
    except Exception as e:
        print(f"[Backend Tool Debug] ERROR: {str(e)}")
        return {"status": "error", "message": f"Failed to fetch events: {str(e)}"}

def get_event_details(user_id: str, event_id: str) -> Dict[str, Any]:
    """
    Get full technical details of a specific event, including raw metadata.
    """
    from ....di.container import get_container
    from ....domain.repositories.event_repository import EventRepository
    from ....domain.repositories.camera_repository import CameraRepository
    from ....core.config import get_settings

    print(f"[Backend Tool Debug] get_event_details called: event_id={event_id}")

    container = get_container()
    event_repo = container.get(EventRepository)
    camera_repo = container.get(CameraRepository)
    settings = get_settings()
    base_url = settings.web_backend_url.rstrip("/")

    try:
        async def _fetch():
            event = await event_repo.get_by_id(owner_user_id=user_id, event_id=event_id)
            if not event:
                return None, None
            
            camera = None
            if event.camera_id:
                camera = await camera_repo.find_by_id(event.camera_id)
            
            return event, camera

        event, camera = _run_sync(_fetch())

        if not event:
            print(f"[Backend Tool Debug] Event {event_id} not found.")
            return {"status": "error", "message": f"Event with ID `{event_id}` not found or access denied."}

        evidence_image_url = f"{base_url}/api/v1/events/{event.id}/image" if event.image_path else None
        print(f"[Backend Tool Debug] Event details for {event.id}: has_image={bool(event.image_path)}, url={evidence_image_url}")

        return {
            "status": "success",
            "event_id": event.id,
            "label": event.label,
            "severity": event.severity,
            "timestamp": event.event_ts.isoformat(),
            "camera": {
                "id": event.camera_id,
                "name": camera.name if camera else "Unknown"
            },
            "agent": {
                "id": event.agent_id,
                "name": event.agent_name
            },
            "metadata": event.metadata,
            "evidence_url": evidence_image_url
        }

    except Exception as e:
        print(f"[Backend Tool Debug] ERROR in get_event_details: {str(e)}")
        return {"status": "error", "message": f"Failed to fetch event details: {str(e)}"}
