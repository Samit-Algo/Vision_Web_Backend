"""
Face detection configuration
-----------------------------

Parses watch_names (who to alert on), min_similarity/tolerance, cooldown, custom label.
"""
# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict, List


class FaceDetectionConfig:
    """Configuration for face detection / person identification scenario."""

    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]):
        # Names to watch: if user says "alert me if sachin appears", watch_names = ["sachin"]
        watch = config.get("watch_names") or config.get("watch_names_list") or []
        if isinstance(watch, str):
            self.watch_names: List[str] = [s.strip().lower() for s in watch.split(",") if s.strip()]
        else:
            self.watch_names = [str(s).strip().lower() for s in watch if str(s).strip()]
        # If empty, alert on any recognized person from gallery
        self.alert_cooldown_seconds = float(config.get("alert_cooldown_seconds", 10))
        # face_recognition legacy; DeepFace uses min_similarity (higher = stricter). Typical 0.5–0.7.
        self.tolerance = float(config.get("tolerance", 0.55))
        # Use tolerance as fallback so saved agents with tolerance=0.55 actually use it for matching
        self.min_similarity = float(
            config.get("min_similarity")
            or config.get("similarity_threshold")
            or self.tolerance
        )
        self.custom_label = config.get("label")
