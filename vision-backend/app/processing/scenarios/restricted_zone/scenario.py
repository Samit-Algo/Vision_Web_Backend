"""
Restricted Zone Scenario
------------------------

Monitors a restricted zone for object presence and triggers alerts.

Behavior:
- Triggers alert when an object of specified class is INSIDE the zone
- Uses bounding box center point to determine if object is in zone
- Alert cooldown prevents rapid-fire duplicate alerts
- Zone polygon is drawn on the live streaming view
"""

from typing import List, Dict, Any
from datetime import datetime

from app.processing.scenarios.contracts import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent
)
from app.processing.scenarios.registry import register_scenario
from app.processing.scenarios.restricted_zone.config import RestrictedZoneConfig
from app.processing.scenarios.restricted_zone.zone_utils import is_box_in_zone


@register_scenario("restricted_zone")
class RestrictedZoneScenario(BaseScenario):
    """
    Monitors a restricted zone for object presence.
    
    Triggers alerts when objects of specified class are detected inside
    the restricted zone polygon.
    """
    
    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        
        # Load configuration
        self.config_obj = RestrictedZoneConfig(config, pipeline_context.task)
        
        # Initialize state
        self._state["last_alert_time"] = None
        self._state["objects_in_zone"] = False
    
    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process a single frame.
        
        Checks if any objects of the specified class are inside the zone.
        Applies alert cooldown to prevent spam.
        """
        # Early exit: No target class or zone configured
        if not self.config_obj.target_class:
            return []
        
        if not self.config_obj.zone_coordinates:
            return []
        
        # Find objects of target class that are inside the zone
        matched_indices, matched_classes = self._find_objects_in_zone(frame_context)
        
        # No objects in zone
        if not matched_indices:
            # Clear cooldown state when zone is empty
            self._state["last_alert_time"] = None
            self._state["objects_in_zone"] = False
            return []
        
        # Objects detected in zone - check cooldown
        now = frame_context.timestamp
        last_alert_time = self._state.get("last_alert_time")
        
        if last_alert_time and isinstance(last_alert_time, datetime):
            elapsed = (now - last_alert_time).total_seconds()
            if elapsed < self.config_obj.alert_cooldown_seconds:
                # Still in cooldown period - don't trigger alert
                # But still mark objects as in zone
                self._state["objects_in_zone"] = True
                return []  # No alert during cooldown
        
        # Generate alert
        label = self._generate_label(len(matched_indices))
        
        # Get confidence scores for matched objects
        detections = frame_context.detections
        scores = detections.scores
        matched_scores = [scores[i] for i in matched_indices if i < len(scores)]
        avg_confidence = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0
        
        # Print alert to console
        print(f"🚨 [RESTRICTED ZONE ALERT] {label}")
        print(f"   📍 {len(matched_indices)} object(s) detected in restricted zone")
        print(f"   🎯 Target class: {self.config_obj.target_class}")
        print(f"   📊 Confidence: avg={avg_confidence:.2f}, min={min(matched_scores):.2f}, max={max(matched_scores):.2f} (threshold={self.config_obj.confidence_threshold})")
        print(f"   🕐 Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📹 Frame index: {frame_context.frame_index}")
        
        # Update state
        self._state["last_alert_time"] = now
        self._state["objects_in_zone"] = True
        
        # Emit event
        event = ScenarioEvent(
            event_type="restricted_zone_detection",
            label=label,
            confidence=1.0,
            metadata={
                "target_class": self.config_obj.target_class,
                "objects_in_zone": len(matched_indices),
                "zone_type": "polygon",
                "alert_cooldown_seconds": self.config_obj.alert_cooldown_seconds
            },
            detection_indices=matched_indices,
            timestamp=frame_context.timestamp,
            frame_index=frame_context.frame_index
        )
        
        return [event]
    
    def _find_objects_in_zone(self, frame_context: ScenarioFrameContext) -> tuple:
        """Find objects of target class that are inside the zone with sufficient confidence."""
        detections = frame_context.detections
        boxes = detections.boxes
        classes = detections.classes
        scores = detections.scores
        
        # Get frame dimensions
        frame = frame_context.frame
        frame_height, frame_width = frame.shape[:2]
        
        matched_indices = []
        matched_classes = []
        target_class = self.config_obj.target_class.lower()
        confidence_threshold = self.config_obj.confidence_threshold
        
        # Debug: Log zone info
        print(f"[ZONE CHECK] Frame size: {frame_width}x{frame_height}")
        print(f"[ZONE CHECK] Zone coordinates: {self.config_obj.zone_coordinates}")
        print(f"[ZONE CHECK] Looking for '{target_class}' with confidence >= {confidence_threshold}")
        
        persons_checked = 0
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if isinstance(cls, str) and cls.lower() == target_class:
                # Apply confidence threshold
                if score >= confidence_threshold:
                    persons_checked += 1
                    # Calculate box center in pixels
                    x1, y1, x2, y2 = box
                    center_x_px = (x1 + x2) / 2
                    center_y_px = (y1 + y2) / 2
                    
                    # Normalize center to 0-1 range to match zone coordinates
                    center_x_norm = center_x_px / frame_width
                    center_y_norm = center_y_px / frame_height
                    
                    # Check if box center is inside the zone polygon
                    in_zone = is_box_in_zone(box, self.config_obj.zone_coordinates, frame_width, frame_height)
                    
                    print(f"[ZONE CHECK] Person #{persons_checked}:")
                    print(f"  - Box (pixels): {box}")
                    print(f"  - Center (pixels): ({center_x_px:.1f}, {center_y_px:.1f})")
                    print(f"  - Center (normalized): ({center_x_norm:.3f}, {center_y_norm:.3f})")
                    print(f"  - Score: {score:.2f}")
                    print(f"  - In zone: {in_zone}")
                    
                    if in_zone:
                        matched_indices.append(i)
                        matched_classes.append(cls)
                        print(f"[ZONE CHECK] ✅ Person #{persons_checked} IS IN ZONE!")
        
        print(f"[ZONE CHECK] Total checked: {persons_checked}, In zone: {len(matched_indices)}")
        
        return matched_indices, matched_classes
    
    def _generate_label(self, count: int) -> str:
        """Generate alert label."""
        custom_label = self.config_obj.custom_label
        
        if custom_label and isinstance(custom_label, str) and custom_label.strip():
            return custom_label.strip()
        
        # Default label
        target_class = self.config_obj.target_class.replace("_", " ").title()
        
        if count == 1:
            return f"{target_class} detected in restricted zone"
        else:
            return f"{count} {target_class}(s) detected in restricted zone"
    
    def reset(self) -> None:
        """Reset scenario state."""
        self._state["last_alert_time"] = None
        self._state["objects_in_zone"] = False
