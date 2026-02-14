"""
YOLO pose provider
------------------

Loads YOLO pose models (keypoints). Same resolution as detector (MODEL_DIR, auto-download).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import os
from typing import Optional

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from app.processing.models.data_models import Model, Provider
from app.core.config import get_settings

try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except Exception:  # noqa: BLE001
    YOLO = None  # type: ignore[assignment]
    YOLO_AVAILABLE = False

# -----------------------------------------------------------------------------
# Provider
# -----------------------------------------------------------------------------


class YOLOPoseProvider(Provider):
    """Loads YOLO pose models (e.g. yolov8n-pose.pt)."""

    def load(self, model_id: str) -> Optional[Model]:
        """Load a YOLO pose model. Resolves path; downloads standard names if missing."""
        if not YOLO_AVAILABLE:
            print("[YOLOPoseProvider] ⚠️ ultralytics not installed. Skipping.")
            return None

        model_name = model_id.strip().rstrip("/").rstrip("\\")
        is_file_path = os.sep in model_name or "/" in model_name or "\\" in model_name
        if not is_file_path:
            model_dir_path = os.path.join(get_settings().model_dir, model_name)
            if os.path.isfile(model_dir_path):
                model_name = model_dir_path

        is_standard_model = (
            (model_name.startswith("yolov8") or model_name.startswith("yolov5")
             or model_name.startswith("yolo11") or model_name.startswith("yolo10")
             or model_name.startswith("yolo9"))
            and model_name.endswith(".pt")
        )

        if is_file_path and not os.path.exists(model_name):
            print(f"[YOLOPoseProvider] ❌ Model file not found: {model_name}")
            return None

        try:
            print(f"[YOLOPoseProvider] 📥 Loading pose model: {model_name}...")
            model = YOLO(model_name)
            print(f"[YOLOPoseProvider] ✅ Loaded: {model_name}")
            return model
        except FileNotFoundError:
            if is_standard_model:
                print(f"[YOLOPoseProvider] 💡 Downloading '{model_name}'...")
                try:
                    model = YOLO(model_name)
                    print(f"[YOLOPoseProvider] ✅ Downloaded and loaded: {model_name}")
                    return model
                except Exception as e:  # noqa: BLE001
                    print(f"[YOLOPoseProvider] ❌ Download failed: {e}")
                    return None
            print(f"[YOLOPoseProvider] ❌ Model file not found: {model_name}")
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"[YOLOPoseProvider] ❌ Load failed: {exc}")
            return None
