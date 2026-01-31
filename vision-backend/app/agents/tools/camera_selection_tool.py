from __future__ import annotations

import re
from typing import Any, Dict

from ...domain.constants.camera_fields import CameraFields
from ...utils.db import get_collection
from .save_to_db_tool import (
    get_camera_repository as _get_camera_repository_shared,
    set_camera_repository as _set_camera_repository_shared,
)


# ============================================================================
# REPOSITORY MANAGEMENT
# ============================================================================

def set_camera_repository(repository):
    """Set the camera repository for camera operations."""
    _set_camera_repository_shared(repository)
    print(f"[camera_selection_tool.set_camera_repository] Camera repository set: {type(repository)}")


def _get_camera_repository():
    """Get the shared camera repository from save_to_db_tool."""
    return _get_camera_repository_shared()


def _cameras_collection():
    """Sync PyMongo collection for cameras. No async, works in Docker and any context."""
    return get_collection("cameras")


def _doc_to_camera_item(doc: Dict[str, Any]) -> Dict[str, str]:
    """Extract id and name from a camera document."""
    camera_id = doc.get(CameraFields.ID) or (str(doc[CameraFields.MONGO_ID]) if doc.get(CameraFields.MONGO_ID) else "")
    return {"id": camera_id, "name": doc.get(CameraFields.NAME, "")}


# ============================================================================
# CAMERA OPERATIONS (sync only – no asyncio, no nest_asyncio)
# ============================================================================

def list_cameras(user_id: str, session_id: str = "default") -> Dict[str, Any]:
    """
    List all cameras owned by a user.

    Args:
        user_id: The user ID to list cameras for
        session_id: Session identifier (for consistency with other tools)

    Returns:
        Dict with cameras list:
        {
            "cameras": [
                {"id": "CAM-001", "name": "Front Gate"},
                {"id": "CAM-002", "name": "Warehouse Cam 2"}
            ]
        }
    """
    if not user_id:
        return {
            "error": "user_id is required",
            "cameras": []
        }

    try:
        coll = _cameras_collection()
        cursor = coll.find({CameraFields.OWNER_USER_ID: user_id})
        cameras = [_doc_to_camera_item(doc) for doc in cursor]
        print(f"[list_cameras] Found {len(cameras)} cameras for user {user_id}")
        return {"cameras": cameras}
    except Exception as e:
        import traceback
        print(f"[list_cameras] ERROR: Failed to list cameras for user {user_id}: {str(e)}")
        print(traceback.format_exc())
        return {"error": f"Failed to list cameras: {str(e)}", "cameras": []}


def resolve_camera(name_or_id: str, user_id: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Resolve a camera by name (partial match) or ID.

    Args:
        name_or_id: Camera name (partial match supported) or camera ID
        user_id: The user ID to filter cameras by
        session_id: Session identifier (for consistency with other tools)

    Returns:
        Dict with status: exact_match, multiple_matches, or not_found
    """
    if not name_or_id or not user_id:
        return {"status": "not_found", "error": "name_or_id and user_id are required"}

    try:
        coll = _cameras_collection()

        # Try by ID first only if it looks like ObjectId (24 hex chars)
        if len(name_or_id) == 24 and all(c in "0123456789abcdefABCDEF" for c in name_or_id):
            try:
                from bson import ObjectId
                doc = coll.find_one({CameraFields.MONGO_ID: ObjectId(name_or_id)})
                if doc and doc.get(CameraFields.OWNER_USER_ID) == user_id:
                    item = _doc_to_camera_item(doc)
                    print(f"[resolve_camera] Found exact match by ID: {item['id']} ({item['name']})")
                    return {"status": "exact_match", "camera_id": item["id"], "camera_name": item["name"]}
            except Exception:
                pass

        # Try by string id field (e.g. CAM-xxx)
        doc = coll.find_one({CameraFields.ID: name_or_id})
        if doc and doc.get(CameraFields.OWNER_USER_ID) == user_id:
            item = _doc_to_camera_item(doc)
            print(f"[resolve_camera] Found exact match by ID: {item['id']} ({item['name']})")
            return {"status": "exact_match", "camera_id": item["id"], "camera_name": item["name"]}

        # Search by name (partial, case-insensitive)
        escaped = re.escape(name_or_id)
        pattern = re.compile(f".*{escaped}.*", re.I)
        cursor = coll.find({
            CameraFields.OWNER_USER_ID: user_id,
            CameraFields.NAME: pattern
        }).limit(10)
        cameras_by_name = [_doc_to_camera_item(doc) for doc in cursor]

        if not cameras_by_name:
            return {"status": "not_found"}
        if len(cameras_by_name) == 1:
            c = cameras_by_name[0]
            print(f"[resolve_camera] Found exact match by name: {c['id']} ({c['name']})")
            return {"status": "exact_match", "camera_id": c["id"], "camera_name": c["name"]}
        print(f"[resolve_camera] Found {len(cameras_by_name)} multiple matches")
        return {"status": "multiple_matches", "cameras": cameras_by_name}
    except Exception as e:
        import traceback
        print(f"[resolve_camera] ERROR: Failed to resolve camera '{name_or_id}' for user {user_id}: {str(e)}")
        print(traceback.format_exc())
        return {"status": "not_found", "error": str(e)}
