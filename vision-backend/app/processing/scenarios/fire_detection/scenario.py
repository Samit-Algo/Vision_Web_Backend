"""
Fire Detection Scenario
-----------------------

Detects fire/flames in camera feed using a fine-tuned YOLO model.
Triggers critical alerts when fire is detected.

Behavior:
- Detects fire using custom fine-tuned fire_detection.pt model
- Requires confirmation across multiple frames to reduce false positives
- Triggers critical alert when fire is confirmed
- Alert cooldown prevents rapid-fire duplicate alerts
- Optional zone support to monitor specific areas
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from app.processing.scenarios.contracts import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent
)
from app.processing.scenarios.registry import register_scenario
from app.processing.scenarios.fire_detection.config import FireDetectionConfig


def is_point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: (x, y) coordinates (normalized 0-1)
        polygon: List of [x, y] coordinates (normalized 0-1)
        
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def is_box_in_zone(
    box: List[float], 
    zone_coords: List[List[float]], 
    frame_width: int, 
    frame_height: int
) -> bool:
    """
    Check if bounding box center is inside a polygon zone.
    
    Args:
        box: [x1, y1, x2, y2] bounding box in pixels
        zone_coords: List of [x, y] normalized coordinates (0-1)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        
    Returns:
        True if box center is inside zone
    """
    # Calculate box center in pixels
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Normalize to 0-1 range
    center_x_norm = center_x / frame_width
    center_y_norm = center_y / frame_height
    
    return is_point_in_polygon((center_x_norm, center_y_norm), zone_coords)


@register_scenario("fire_detection")
class FireDetectionScenario(BaseScenario):
    """
    Detects fire/flames in camera feed.
    
    Uses a fine-tuned YOLO model to detect fire and smoke.
    Triggers critical alerts when fire is confirmed across multiple frames.
    """
    
    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        
        # Load configuration
        self.config_obj = FireDetectionConfig(config, pipeline_context.task)
        
        # Initialize state
        self._state["last_alert_time"] = None
        self._state["fire_detected"] = False
        self._state["consecutive_fire_frames"] = 0
        self._state["total_fire_detections"] = 0
        self._state["fire_detection_history"] = []  # Track detection history
        
        # Track per-class detection counts
        self._state["detection_counts"] = defaultdict(int)
        
        print(f"[FIRE_DETECTION] 🔥 Initialized fire detection scenario")
        print(f"[FIRE_DETECTION]   Target classes: {self.config_obj.target_classes}")
        print(f"[FIRE_DETECTION]   Confidence threshold: {self.config_obj.confidence_threshold}")
        print(f"[FIRE_DETECTION]   Alert cooldown: {self.config_obj.alert_cooldown_seconds}s")
        print(f"[FIRE_DETECTION]   Confirm frames: {self.config_obj.confirm_frames}")
        if self.config_obj.zone_applied:
            print(f"[FIRE_DETECTION]   Zone monitoring: ENABLED")
    
    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process a single frame for fire detection.
        
        Checks for fire/smoke detections and triggers alert when confirmed.
        Always returns events when fire is detected (for bounding box visualization).
        """
        # Find fire detections in frame
        fire_detections = self._find_fire_detections(frame_context)
        
        # No fire detected in this frame
        if not fire_detections:
            # Decay consecutive count (allow some frame drops)
            if self._state["consecutive_fire_frames"] > 0:
                self._state["consecutive_fire_frames"] -= 1
            
            # Clear fire state if no detection for a while
            if self._state["consecutive_fire_frames"] == 0:
                self._state["fire_detected"] = False
            
            return []
        
        # Fire detected - increment consecutive counter
        self._state["consecutive_fire_frames"] += 1
        self._state["total_fire_detections"] += 1
        
        # Log detection
        matched_indices, matched_classes, matched_scores = fire_detections
        print(f"[FIRE_DETECTION] 🔥 Fire detected! Classes: {matched_classes}, Scores: {[f'{s:.2f}' for s in matched_scores]}")
        print(f"[FIRE_DETECTION]   Consecutive frames: {self._state['consecutive_fire_frames']}/{self.config_obj.confirm_frames}")
        
        # Calculate confidence values
        avg_confidence = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0
        max_confidence = max(matched_scores) if matched_scores else 0.0
        
        # Mark fire as detected
        self._state["fire_detected"] = True
        
        # Get current time
        now = frame_context.timestamp
        
        # Determine if this is a new alert (not in cooldown and confirmed)
        is_confirmed = self._state["consecutive_fire_frames"] >= self.config_obj.confirm_frames
        is_alert = False
        
        if is_confirmed:
            last_alert_time = self._state.get("last_alert_time")
            in_cooldown = False
            
            if last_alert_time and isinstance(last_alert_time, datetime):
                elapsed = (now - last_alert_time).total_seconds()
                in_cooldown = elapsed < self.config_obj.alert_cooldown_seconds
            
            if not in_cooldown:
                # This is a new alert!
                is_alert = True
                self._state["last_alert_time"] = now
                
                # Update detection counts
                for cls in matched_classes:
                    self._state["detection_counts"][cls] += 1
                
                # Add to history
                self._state["fire_detection_history"].append({
                    "timestamp": now.isoformat(),
                    "frame_index": frame_context.frame_index,
                    "classes": matched_classes,
                    "scores": matched_scores
                })
                
                # Keep history limited
                if len(self._state["fire_detection_history"]) > 100:
                    self._state["fire_detection_history"] = self._state["fire_detection_history"][-100:]
                
                # Print critical alert
                label = self._generate_label(len(matched_indices), matched_classes)
                print("=" * 60)
                print(f"🚨🔥 [FIRE ALERT] {label}")
                print(f"   ⚠️  CRITICAL ALERT - FIRE DETECTED!")
                print(f"   🎯 Detected classes: {', '.join(matched_classes)}")
                print(f"   📊 Confidence: avg={avg_confidence:.2f}, max={max_confidence:.2f}")
                print(f"   📍 Detection count: {len(matched_indices)}")
                print(f"   🕐 Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   📹 Frame index: {frame_context.frame_index}")
                if self.config_obj.zone_applied:
                    print(f"   📍 Zone monitoring: ACTIVE")
                print("=" * 60)
        
        # Generate label for event
        if is_alert:
            label = self._generate_label(len(matched_indices), matched_classes)
            alert_type = "critical"
            severity = "critical"
        else:
            # Detection event (for visualization) - not a full alert
            label = f"Fire: {len(matched_indices)} detection(s)"
            alert_type = "detection"
            severity = "warning" if is_confirmed else "info"
        
        # ALWAYS emit event when fire is detected (for bounding box visualization)
        event = ScenarioEvent(
            event_type="fire_detection",
            label=label,
            confidence=max_confidence,
            metadata={
                "alert_type": alert_type,
                "severity": severity,
                "is_alert": is_alert,
                "is_confirmed": is_confirmed,
                "fire_detected": True,
                "detected_classes": matched_classes,
                "detection_count": len(matched_indices),
                "avg_confidence": avg_confidence,
                "max_confidence": max_confidence,
                "consecutive_frames": self._state["consecutive_fire_frames"],
                "total_detections": self._state["total_fire_detections"],
                "zone_applied": self.config_obj.zone_applied,
                "alert_cooldown_seconds": self.config_obj.alert_cooldown_seconds
            },
            detection_indices=matched_indices,
            timestamp=frame_context.timestamp,
            frame_index=frame_context.frame_index
        )
        
        return [event]
    
    def _find_fire_detections(
        self, 
        frame_context: ScenarioFrameContext
    ) -> Optional[Tuple[List[int], List[str], List[float]]]:
        """
        Find fire-related detections in the frame.
        
        Args:
            frame_context: Frame context with detections
            
        Returns:
            Tuple of (matched_indices, matched_classes, matched_scores) or None
        """
        detections = frame_context.detections
        boxes = detections.boxes
        classes = detections.classes
        scores = detections.scores
        
        # Get frame dimensions for zone checking
        frame = frame_context.frame
        frame_height, frame_width = frame.shape[:2]
        
        matched_indices = []
        matched_classes = []
        matched_scores = []
        
        target_classes = [c.lower() for c in self.config_obj.target_classes]
        confidence_threshold = self.config_obj.confidence_threshold
        
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            # Check if class matches fire-related classes
            if isinstance(cls, str):
                cls_lower = cls.lower()
                
                # Check if it's a fire-related class
                is_fire_class = any(
                    target in cls_lower or cls_lower in target
                    for target in target_classes
                )
                
                if is_fire_class and score >= confidence_threshold:
                    # Check zone if configured
                    if self.config_obj.zone_applied and self.config_obj.zone_coordinates:
                        if not is_box_in_zone(box, self.config_obj.zone_coordinates, frame_width, frame_height):
                            continue  # Skip if not in zone
                    
                    matched_indices.append(i)
                    matched_classes.append(cls)
                    matched_scores.append(float(score))
        
        if matched_indices:
            return matched_indices, matched_classes, matched_scores
        
        return None
    
    def _generate_label(self, count: int, classes: List[str]) -> str:
        """Generate alert label."""
        custom_label = self.config_obj.custom_label
        
        if custom_label and isinstance(custom_label, str) and custom_label.strip():
            return custom_label.strip()
        
        # Generate default label based on detected classes
        unique_classes = list(set(classes))
        
        if "smoke" in [c.lower() for c in unique_classes]:
            if "fire" in [c.lower() for c in unique_classes] or "flame" in [c.lower() for c in unique_classes]:
                return f"🔥 FIRE AND SMOKE DETECTED!"
            else:
                return f"💨 SMOKE DETECTED - Possible fire!"
        
        if count == 1:
            return f"🔥 FIRE DETECTED!"
        else:
            return f"🔥 FIRE DETECTED - {count} fire sources!"
    
    def reset(self) -> None:
        """Reset scenario state."""
        print(f"[FIRE_DETECTION] Resetting fire detection scenario")
        print(f"[FIRE_DETECTION]   Total detections this session: {self._state.get('total_fire_detections', 0)}")
        
        self._state["last_alert_time"] = None
        self._state["fire_detected"] = False
        self._state["consecutive_fire_frames"] = 0
        self._state["total_fire_detections"] = 0
        self._state["fire_detection_history"] = []
        self._state["detection_counts"] = defaultdict(int)
