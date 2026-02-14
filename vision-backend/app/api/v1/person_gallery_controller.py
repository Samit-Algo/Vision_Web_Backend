"""
Person gallery API: upload reference photos for face recognition.

- One folder per person (Gallery/<name>/); one MongoDB document per person.
- Minimum 4 photos per person; status "active" when count >= 4.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import List, Optional

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from fastapi import APIRouter, Depends, File, Form, HTTPException, status, UploadFile

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...application.dto.user_dto import UserResponse
from ...core.config import get_settings
from ...domain.constants import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_IMAGE_MIME,
    MIN_PHOTOS_PER_PERSON,
)
from ...infrastructure.db.mongo_connection import get_person_gallery_collection
from ...utils.datetime_utils import utc_now
from ...utils.face_embedding import DEFAULT_EMBEDDING_MODEL, compute_face_embedding_for_gallery

from .dependencies import get_current_user

# -----------------------------------------------------------------------------
# Router and helpers
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
router = APIRouter(tags=["person-gallery"])


def safe_folder_name(name: str) -> str:
    """Return a folder-safe name: alphanumeric and underscores only."""
    s = "".join(c if c.isalnum() or c in " -_" else "_" for c in name.strip())
    return s.replace(" ", "_").strip("_")[:80] or "person"


def normalized_name(name: str) -> str:
    """Normalized for MongoDB lookup: lowercase, single spaces."""
    return " ".join(name.strip().lower().split())


def compute_face_embedding_for_upload(image_path: str) -> Optional[List[float]]:
    """Compute face embedding for one gallery image (front or side)."""
    return compute_face_embedding_for_gallery(image_path, model_name=DEFAULT_EMBEDDING_MODEL)


@router.post("/upload")
async def upload_reference_photos(
    name: str = Form(..., min_length=1, max_length=200),
    files: List[UploadFile] = File(...),
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Upload at least 4 reference photos for one person. Photos are stored in
    Gallery/<person_name>/ and metadata (including face embeddings) in MongoDB.
    Each image must contain at least one detectable face (front or side angle);
    if multiple faces are present, the largest is used for the embedding.
    """
    if len(files) < MIN_PHOTOS_PER_PERSON:
        detail = f"Upload at least {MIN_PHOTOS_PER_PERSON} photos per person for accurate recognition."
        logger.warning("person-gallery upload 400: %s", detail)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    settings = get_settings()
    gallery_dir = Path(settings.gallery_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    safe_name = safe_folder_name(name)
    person_dir = gallery_dir / safe_name
    person_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalized_name(name)
    display_name = name.strip()

    coll = get_person_gallery_collection()

    # Get existing person doc if any (by normalized name)
    existing = await coll.find_one({"normalized_name": normalized})
    start_index = len(existing.get("images", [])) if existing else 0

    new_images: List[str] = []
    new_file_paths: List[str] = []
    new_embeddings: List[List[float]] = []
    new_file_sizes: List[int] = []
    new_mime_types: List[str] = []
    saved_paths: List[Path] = []

    try:
        for i, file in enumerate(files):
            if not file.filename:
                detail = "One or more files have no filename."
                logger.warning("person-gallery upload 400: %s", detail)
                raise HTTPException(status_code=400, detail=detail)
            ext = Path(file.filename).suffix.lower()
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                detail = f"Allowed formats: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
                logger.warning("person-gallery upload 400: %s", detail)
                raise HTTPException(status_code=400, detail=detail)

            idx = start_index + i
            filename = f"{idx}{ext}"
            file_path = person_dir / filename

            contents = await file.read()
            file_size = len(contents)
            file_path.write_bytes(contents)
            saved_paths.append(file_path)

            embedding = compute_face_embedding_for_upload(str(file_path.resolve()))
            if embedding is None:
                for p in saved_paths:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
                detail = (
                    f"Image {i + 1}: No face could be detected. Use clear photos with one visible face "
                    "(front or side angle). Avoid heavy shadows or blur."
                )
                logger.warning("person-gallery upload 400: %s", detail)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

            content_type = (file.content_type or "").strip().lower() or "image/jpeg"
            new_images.append(filename)
            new_file_paths.append(str(file_path.resolve()))
            new_embeddings.append(embedding)
            new_file_sizes.append(file_size)
            new_mime_types.append(content_type)

        now = utc_now()
        if existing:
            await coll.update_one(
                {"normalized_name": normalized},
                {
                    "$push": {
                        "images": {"$each": new_images},
                        "file_paths": {"$each": new_file_paths},
                        "embeddings": {"$each": new_embeddings},
                        "file_sizes": {"$each": new_file_sizes},
                        "mime_types": {"$each": new_mime_types},
                    },
                    "$set": {"uploaded_at": now, "embedding_model": DEFAULT_EMBEDDING_MODEL},
                },
            )
            doc = await coll.find_one({"normalized_name": normalized})
            if not doc:
                raise HTTPException(status_code=500, detail="Failed to read back updated person record.")
            total = len(doc["embeddings"])
        else:
            doc = {
                "name": display_name,
                "normalized_name": normalized,
                "images": new_images,
                "file_paths": new_file_paths,
                "embeddings": new_embeddings,
                "file_sizes": new_file_sizes,
                "mime_types": new_mime_types,
                "uploaded_at": now,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "status": "active" if len(new_embeddings) >= MIN_PHOTOS_PER_PERSON else "incomplete",
            }
            result = await coll.insert_one(doc)
            doc["_id"] = result.inserted_id
            total = len(new_embeddings)

        status_val = "active" if total >= MIN_PHOTOS_PER_PERSON else "incomplete"
        if existing and status_val == "active":
            await coll.update_one({"normalized_name": normalized}, {"$set": {"status": "active"}})

        doc_id = str(doc["_id"])
        return {
            "id": doc_id,
            "name": display_name,
            "image_count": total,
            "uploaded_at": now.isoformat() if hasattr(now, "isoformat") else str(now),
            "status": status_val,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("person-gallery upload 500: %s", e)
        for p in saved_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_persons(
    current_user: UserResponse = Depends(get_current_user),
) -> List[dict]:
    """List all persons: name, image_count, uploaded_at, status."""
    coll = get_person_gallery_collection()
    cursor = coll.find(
        {},
        projection={"name": 1, "normalized_name": 1, "images": 1, "uploaded_at": 1, "status": 1},
    ).sort("uploaded_at", -1)
    out = []
    async for doc in cursor:
        images = doc.get("images") or []
        out.append({
            "id": str(doc["_id"]),
            "name": doc.get("name", ""),
            "image_count": len(images),
            "uploaded_at": doc.get("uploaded_at").isoformat() if hasattr(doc.get("uploaded_at"), "isoformat") else str(doc.get("uploaded_at", "")),
            "status": doc.get("status", "incomplete"),
        })
    return out
