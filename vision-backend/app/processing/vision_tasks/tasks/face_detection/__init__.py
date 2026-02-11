"""
Face detection / recognition scenario
-------------------------------------
Identifies persons from camera frames using reference photos stored in
person_gallery (MongoDB + Gallery folder). When watch_names are configured
(e.g. "alert me if sachin appears on camera 1"), emits events and overlays
the recognized person's name.
"""
from app.processing.vision_tasks.tasks.face_detection.scenario import FaceDetectionScenario

__all__ = ["FaceDetectionScenario"]
