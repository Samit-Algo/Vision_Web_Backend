"""
Detection and event related tools for the General Chat Agent.
"""

import logging
from datetime import timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _format_event_time(event_ts) -> str:
    """Format event time safely for UI display."""
    if event_ts is None:
        return "Unknown time"
    try:
        # datetime object case
        return event_ts.strftime("%I:%M %p")
    except Exception:
        pass
    try:
        # ISO string case
        from datetime import datetime
        dt = datetime.fromisoformat(str(event_ts).replace("Z", "+00:00"))
        return dt.strftime("%I:%M %p")
    except Exception:
        return str(event_ts)

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
    from ....utils.db import get_collection
    from ....utils.datetime_utils import now, ensure_utc
    from ....core.config import get_settings
    from ....domain.constants.event_fields import EventFields
    from ....domain.constants.camera_fields import CameraFields

    logger.debug("get_recent_detections called (user_id=%s days_ago=%s)", user_id, days_ago)

    event_collection = get_collection("events")
    camera_collection = get_collection("cameras")
    settings = get_settings()
    base_url = settings.web_backend_url.rstrip("/")
    
    local_now = now()
    target_date = local_now.date() - timedelta(days=days_ago)
    
    start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_ago)
    end_local = start_local + timedelta(days=1)
    
    start_utc = ensure_utc(start_local)
    end_utc = ensure_utc(end_local)
    
    try:
        # Match repository behavior: include user-owned + legacy events with null/missing owner.
        query = {
            "$or": [
                {EventFields.OWNER_USER_ID: user_id},
                {EventFields.OWNER_USER_ID: None},
                {EventFields.OWNER_USER_ID: {"$exists": False}},
            ],
            EventFields.EVENT_TS: {"$gte": start_utc, "$lt": end_utc},
        }

        # Fetch recent events for the date window.
        event_docs = list(
            event_collection.find(query)
            .sort([(EventFields.EVENT_TS, -1), (EventFields.RECEIVED_AT, -1)])
            .limit(100)
        )

        # Build camera name map once.
        camera_docs = list(
            camera_collection.find(
                {CameraFields.OWNER_USER_ID: user_id},
                {CameraFields.ID: 1, CameraFields.NAME: 1},
            )
        )
        camera_map = {}
        for c in camera_docs:
            cam_id = c.get(CameraFields.ID)
            if cam_id:
                camera_map[str(cam_id)] = c.get(CameraFields.NAME) or "Unknown"
        
        logger.debug("Found %d total events in repository", len(event_docs))

        filtered_events = []
        for e in event_docs:
            event_camera_id = e.get(EventFields.CAMERA_ID)
            if camera_id and event_camera_id != camera_id:
                continue
            
            # Resolve camera name
            metadata = e.get(EventFields.METADATA) or {}
            camera_name = metadata.get("camera_name")
            if not camera_name and event_camera_id:
                camera_name = camera_map.get(str(event_camera_id), "Unknown")
            
            event_id = str(e.get(EventFields.MONGO_ID))
            image_path = e.get(EventFields.IMAGE_PATH)
            evidence_image_url = f"{base_url}/api/v1/events/{event_id}/image" if image_path else None
            
            logger.debug(
                "Event %s has_image=%s evidence_url=%s",
                event_id,
                bool(image_path),
                evidence_image_url,
            )

            filtered_events.append({
                "event_id": event_id,
                "label": e.get(EventFields.LABEL) or "Event",
                "camera_name": camera_name or "Unknown",
                "camera_id": event_camera_id,
                "timestamp": _format_event_time(e.get(EventFields.EVENT_TS)),
                "severity": e.get(EventFields.SEVERITY) or "info",
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
        logger.exception("Failed to fetch recent detections: %s", e)
        return {"status": "error", "message": f"Failed to fetch events: {str(e)}"}

def get_event_details(user_id: str, event_id: str) -> Dict[str, Any]:
    """
    Get full technical details of a specific event, including raw metadata.
    """
    from bson import ObjectId
    from bson.errors import InvalidId
    from ....utils.db import get_collection
    from ....core.config import get_settings
    from ....domain.constants.event_fields import EventFields
    from ....domain.constants.camera_fields import CameraFields

    logger.debug("get_event_details called (event_id=%s)", event_id)

    event_collection = get_collection("events")
    camera_collection = get_collection("cameras")
    settings = get_settings()
    base_url = settings.web_backend_url.rstrip("/")

    try:
        try:
            object_id = ObjectId(event_id)
        except (InvalidId, ValueError, TypeError):
            return {"status": "error", "message": f"Event with ID `{event_id}` not found or access denied."}

        query = {
            EventFields.MONGO_ID: object_id,
            "$or": [
                {EventFields.OWNER_USER_ID: user_id},
                {EventFields.OWNER_USER_ID: None},
                {EventFields.OWNER_USER_ID: {"$exists": False}},
            ],
        }
        event = event_collection.find_one(query)

        if not event:
            logger.debug("Event not found: %s", event_id)
            return {"status": "error", "message": f"Event with ID `{event_id}` not found or access denied."}

        camera = None
        event_camera_id = event.get(EventFields.CAMERA_ID)
        if event_camera_id:
            camera = camera_collection.find_one({CameraFields.ID: event_camera_id})

        evidence_image_url = (
            f"{base_url}/api/v1/events/{str(event.get(EventFields.MONGO_ID))}/image"
            if event.get(EventFields.IMAGE_PATH)
            else None
        )
        logger.debug(
            "Event details %s has_image=%s evidence_url=%s",
            str(event.get(EventFields.MONGO_ID)),
            bool(event.get(EventFields.IMAGE_PATH)),
            evidence_image_url,
        )

        return {
            "status": "success",
            "event_id": str(event.get(EventFields.MONGO_ID)),
            "label": event.get(EventFields.LABEL) or "Event",
            "severity": event.get(EventFields.SEVERITY) or "info",
            "timestamp": str(event.get(EventFields.EVENT_TS)),
            "camera": {
                "id": event_camera_id,
                "name": (camera.get(CameraFields.NAME) if camera else "Unknown")
            },
            "agent": {
                "id": event.get(EventFields.AGENT_ID),
                "name": event.get(EventFields.AGENT_NAME)
            },
            "metadata": event.get(EventFields.METADATA) or {},
            "evidence_url": evidence_image_url
        }

    except Exception as e:
        logger.exception("Failed to fetch event details for %s: %s", event_id, e)
        return {"status": "error", "message": f"Failed to fetch event details: {str(e)}"}
