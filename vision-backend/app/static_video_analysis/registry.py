"""Simple JSON registry for video_id -> path mapping (replaces MongoDB for static video)."""

import json
from pathlib import Path
from typing import Optional

from ..core.config import get_settings


def _registry_path() -> Path:
    db_dir = Path(get_settings().static_video_vector_db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "static_video_registry.json"


def _load_registry() -> dict:
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_registry(data: dict) -> None:
    path = _registry_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def register_video(video_id: str, path: str, user_id: str) -> None:
    """Register video_id -> path for later lookup."""
    import datetime
    data = _load_registry()
    data[video_id] = {
        "path": str(Path(path).resolve()),
        "user_id": user_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    _save_registry(data)


def get_video_path(video_id: str, user_id: Optional[str] = None) -> Optional[str]:
    """Get video path by video_id. Optional user_id for scoping."""
    data = _load_registry()
    entry = data.get(video_id)
    if not entry:
        return None
    if user_id and entry.get("user_id") != user_id:
        return None
    path = entry.get("path")
    if path and Path(path).exists():
        return path
    return None
