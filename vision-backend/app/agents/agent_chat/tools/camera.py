from __future__ import annotations

import re
from typing import Any, Dict

from ....domain.constants.camera_fields import CameraFields
from ....utils.db import get_collection
from ...exceptions import CameraServiceError, ValidationError


def get_cameras_collection():
    return get_collection("cameras")


def document_to_camera_item(doc: Dict[str, Any]) -> Dict[str, str]:
    camera_id = doc.get(CameraFields.ID) or (str(doc[CameraFields.MONGO_ID]) if doc.get(CameraFields.MONGO_ID) else "")
    return {"id": camera_id, "name": doc.get(CameraFields.NAME, "")}


def list_cameras(user_id: str, session_id: str = "default") -> Dict[str, Any]:
    if not user_id:
        raise ValidationError("user_id required", user_message="User ID is required to list cameras.")
    try:
        coll = get_cameras_collection()
        cursor = coll.find({CameraFields.OWNER_USER_ID: user_id})
        cameras = [document_to_camera_item(doc) for doc in cursor]
        return {"cameras": cameras}
    except Exception as e:
        raise CameraServiceError(str(e)) from e


def resolve_camera(name_or_id: str, user_id: str, session_id: str = "default") -> Dict[str, Any]:
    if not name_or_id or not user_id:
        raise ValidationError("name_or_id and user_id required", user_message="Camera name or ID and user are required.")
    try:
        coll = get_cameras_collection()
        if len(name_or_id) == 24 and all(c in "0123456789abcdefABCDEF" for c in name_or_id):
            try:
                from bson import ObjectId
                doc = coll.find_one({CameraFields.MONGO_ID: ObjectId(name_or_id)})
                if doc and doc.get(CameraFields.OWNER_USER_ID) == user_id:
                    item = document_to_camera_item(doc)
                    return {"status": "exact_match", "camera_id": item["id"], "camera_name": item["name"]}
            except Exception:
                pass
        doc = coll.find_one({CameraFields.ID: name_or_id})
        if doc and doc.get(CameraFields.OWNER_USER_ID) == user_id:
            item = document_to_camera_item(doc)
            return {"status": "exact_match", "camera_id": item["id"], "camera_name": item["name"]}
        escaped = re.escape(name_or_id)
        pattern = re.compile(f".*{escaped}.*", re.I)
        cursor = coll.find({CameraFields.OWNER_USER_ID: user_id, CameraFields.NAME: pattern}).limit(10)
        by_name = [document_to_camera_item(doc) for doc in cursor]
        if not by_name:
            return {"status": "not_found"}
        if len(by_name) == 1:
            c = by_name[0]
            return {"status": "exact_match", "camera_id": c["id"], "camera_name": c["name"]}
        return {"status": "multiple_matches", "cameras": by_name}
    except Exception as e:
        raise CameraServiceError(str(e)) from e
