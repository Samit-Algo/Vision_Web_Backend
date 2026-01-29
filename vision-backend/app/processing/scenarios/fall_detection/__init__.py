"""
Fall Detection Scenario
-----------------------

Detects human falls using pose keypoint analysis.
"""

from app.processing.scenarios.fall_detection.scenario import FallDetectionScenario
from app.processing.scenarios.fall_detection.config import FallDetectionConfig

__all__ = ["FallDetectionScenario", "FallDetectionConfig"]
