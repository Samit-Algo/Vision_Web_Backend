"""
Sleep Detection Scenario
------------------------

Detects when a person near a machine is sleeping (lying or standing/nodding off).
Uses: YOLO pose → temporal consistency → VLM confirmation.
"""

from app.processing.vision_tasks.tasks.sleep_detection.scenario import SleepDetectionScenario

__all__ = ["SleepDetectionScenario"]
