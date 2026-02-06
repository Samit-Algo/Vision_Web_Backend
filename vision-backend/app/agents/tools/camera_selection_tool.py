from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict

from bson.errors import InvalidId as BsonInvalidId

from ...domain.constants.camera_fields import CameraFields
from ...utils.db import get_collection
from ..exceptions import (
    DatabaseError,
    DatabaseConnectionError,
    CameraServiceError,
    ValidationError,
)
from ..utils.retry_utils import retry_on_exception, async_retry_on_exception
from .save_to_db_tool import (
    get_camera_repository as _get_camera_repository_shared,
    set_camera_repository as _set_camera_repository_shared,
)

logger = logging.getLogger(__name__)


# ============================================================================
# REPOSITORY MANAGEMENT
# ============================================================================

def set_camera_repository(repository):
    """Set the camera repository for camera operations."""
    _set_camera_repository_shared(repository)


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

@retry_on_exception(max_retries=3, initial_delay=0.5)
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
        
    Raises:
        ValidationError: If user_id is invalid
        DatabaseConnectionError: If database connection fails
        CameraServiceError: If camera retrieval fails
    """
    if not user_id:
        logger.error("list_cameras called without user_id")
        raise ValidationError(
            "user_id is required",
            user_message="User authentication required to list cameras."
        )

    try:
        logger.debug(f"Listing cameras for user_id={user_id}, session={session_id}")
        coll = _cameras_collection()
        cursor = coll.find({CameraFields.OWNER_USER_ID: user_id})
        cameras = [_doc_to_camera_item(doc) for doc in cursor]
        logger.info(f"Found {len(cameras)} cameras for user_id={user_id}")
        return {"cameras": cameras}
    except DatabaseConnectionError:
        # Re-raise to trigger retry
        raise
    except Exception as e:
        logger.exception(f"Failed to list cameras for user_id={user_id}: {e}")
        raise CameraServiceError(
            f"Failed to retrieve camera list: {str(e)}",
            user_message="Unable to load cameras. Please try again."
        )


@retry_on_exception(max_retries=3, initial_delay=0.5)
def resolve_camera(name_or_id: str, user_id: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Resolve a camera by name (partial match) or ID.

    Args:
        name_or_id: Camera name (partial match supported) or camera ID
        user_id: The user ID to filter cameras by
        session_id: Session identifier (for consistency with other tools)

    Returns:
        Dict with status: exact_match, multiple_matches, or not_found
        
    Raises:
        ValidationError: If input parameters are invalid
        DatabaseConnectionError: If database connection fails
        CameraServiceError: If camera resolution fails
    """
    if not name_or_id or not user_id:
        logger.error(f"resolve_camera called with invalid params: name_or_id={name_or_id}, user_id={user_id}")
        raise ValidationError(
            "name_or_id and user_id are required",
            user_message="Camera name/ID and user authentication are required."
        )

    try:
        logger.debug(f"Resolving camera: name_or_id={name_or_id}, user_id={user_id}")
        coll = _cameras_collection()

        # Try by ID first only if it looks like ObjectId (24 hex chars)
        if len(name_or_id) == 24 and all(c in "0123456789abcdefABCDEF" for c in name_or_id):
            try:
                from bson import ObjectId
                doc = coll.find_one({CameraFields.MONGO_ID: ObjectId(name_or_id)})
                if doc and doc.get(CameraFields.OWNER_USER_ID) == user_id:
                    item = _doc_to_camera_item(doc)
                    logger.info(f"Camera resolved by ObjectId: {item['id']}")
                    return {"status": "exact_match", "camera_id": item["id"], "camera_name": item["name"]}
            except BsonInvalidId as e:
                logger.warning(f"Invalid ObjectId format: {name_or_id}: {e}")
                # Continue to try other methods
            except DatabaseConnectionError:
                # Re-raise to trigger retry
                raise
            except Exception as e:
                logger.warning(f"Error looking up camera by ObjectId: {e}")
                # Continue to try other methods

        # Try by string id field (e.g. CAM-xxx)
        doc = coll.find_one({CameraFields.ID: name_or_id})
        if doc and doc.get(CameraFields.OWNER_USER_ID) == user_id:
            item = _doc_to_camera_item(doc)
            logger.info(f"Camera resolved by ID: {item['id']}")
            return {"status": "exact_match", "camera_id": item["id"], "camera_name": item["name"]}

        # Search by name (partial, case-insensitive)
        # Validate input length to prevent ReDoS
        if len(name_or_id) > 100:
            logger.warning(f"Camera name too long: {len(name_or_id)} chars")
            raise ValidationError(
                f"Camera name too long: {len(name_or_id)} characters",
                user_message="Camera name is too long. Please use a shorter search term."
            )
        
        escaped = re.escape(name_or_id)
        pattern = re.compile(f".*{escaped}.*", re.I)
        cursor = coll.find({
            CameraFields.OWNER_USER_ID: user_id,
            CameraFields.NAME: pattern
        }).limit(10)
        cameras_by_name = [_doc_to_camera_item(doc) for doc in cursor]

        if not cameras_by_name:
            logger.info(f"No cameras found matching: {name_or_id}")
            return {"status": "not_found"}
        if len(cameras_by_name) == 1:
            c = cameras_by_name[0]
            logger.info(f"Camera resolved by name: {c['id']}")
            return {"status": "exact_match", "camera_id": c["id"], "camera_name": c["name"]}
        
        logger.info(f"Multiple cameras found matching: {name_or_id} ({len(cameras_by_name)} matches)")
        return {"status": "multiple_matches", "cameras": cameras_by_name}
        
    except (ValidationError, DatabaseConnectionError):
        # Re-raise validation and connection errors
        raise
    except Exception as e:
        logger.exception(f"Failed to resolve camera '{name_or_id}': {e}")
        raise CameraServiceError(
            f"Failed to resolve camera: {str(e)}",
            user_message="Unable to find camera. Please try again."
        )


# ============================================================================
# ASYNC WRAPPERS (non-blocking – run sync DB calls in thread pool)
# Used by Agent Creation chat to avoid blocking the event loop.
# ============================================================================


async def list_cameras_async(
    user_id: str, session_id: str = "default"
) -> Dict[str, Any]:
    """
    Non-blocking list of cameras. Runs sync MongoDB in thread pool.
    
    Args:
        user_id: The user ID to list cameras for
        session_id: Session identifier
        
    Returns:
        Dict with cameras list or error information
    """
    try:
        logger.debug(f"list_cameras_async: user_id={user_id}, session={session_id}")
        return await asyncio.to_thread(
            list_cameras, user_id=user_id, session_id=session_id
        )
    except ValidationError as e:
        logger.error(f"Validation error in list_cameras_async: {e}")
        return {"error": e.user_message, "cameras": []}
    except (DatabaseConnectionError, CameraServiceError) as e:
        logger.error(f"Service error in list_cameras_async: {e}")
        return {"error": e.user_message, "cameras": []}
    except Exception as e:
        logger.exception(f"Unexpected error in list_cameras_async: {e}")
        return {
            "error": "An unexpected error occurred while listing cameras.",
            "cameras": []
        }


async def resolve_camera_async(
    name_or_id: str, user_id: str, session_id: str = "default"
) -> Dict[str, Any]:
    """
    Non-blocking camera resolve. Runs sync MongoDB in thread pool.
    
    Args:
        name_or_id: Camera name or ID to resolve
        user_id: The user ID to filter cameras by
        session_id: Session identifier
        
    Returns:
        Dict with resolution status or error information
    """
    try:
        logger.debug(f"resolve_camera_async: name_or_id={name_or_id}, user_id={user_id}")
        return await asyncio.to_thread(
            resolve_camera,
            name_or_id=name_or_id,
            user_id=user_id,
            session_id=session_id,
        )
    except ValidationError as e:
        logger.error(f"Validation error in resolve_camera_async: {e}")
        return {"status": "not_found", "error": e.user_message}
    except (DatabaseConnectionError, CameraServiceError) as e:
        logger.error(f"Service error in resolve_camera_async: {e}")
        return {"status": "not_found", "error": e.user_message}
    except Exception as e:
        logger.exception(f"Unexpected error in resolve_camera_async: {e}")
        return {
            "status": "not_found",
            "error": "An unexpected error occurred while resolving camera."
        }
