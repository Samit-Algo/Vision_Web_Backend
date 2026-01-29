"""
Fire Detection Scenario
-----------------------

Detects fire/flames in camera feed using a fine-tuned YOLO model.
Triggers alerts when fire is detected in the frame.
"""

from app.processing.scenarios.fire_detection.scenario import FireDetectionScenario
from app.processing.scenarios.fire_detection.config import FireDetectionConfig

__all__ = [
    "FireDetectionScenario",
    "FireDetectionConfig",
]
