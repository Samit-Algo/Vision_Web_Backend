"""
Restricted Zone State Management
--------------------------------

Tracks per-person state for restricted zone monitoring:
- Inside zone status
- Entry time
- Duration inside
- Optional confirmation (stability_frames=1 for instant, no wait)
- Alert levels (touch/orange, inside/red, duration/frequent)
"""

from typing import Dict, Optional
from datetime import datetime

from app.utils.datetime_utils import utc_now


class TrackZoneState:
    """State for a single tracked person in restricted zone."""
    
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.inside_zone = False  # Current inside status
        self.touches_zone = False  # Current touch status
        self.entry_time: Optional[datetime] = None  # When person entered zone
        self.last_seen: Optional[datetime] = None  # Last time person was detected
        self.inside_frame_count = 0  # Consecutive frames inside (for stability)
        self.touches_frame_count = 0  # Consecutive frames touching (for stability)
        self.outside_frame_count = 0  # Consecutive frames outside (for stability)
        self.last_alert_time: Optional[datetime] = None  # Last alert time (for cooldown)
        self.alert_level: Optional[str] = None  # 'touch', 'inside', 'duration'
        self.confirmed_inside = False  # Confirmed inside after stability check
        self.confirmed_touches = False  # Confirmed touches after stability check


class RestrictedZoneState:
    """Manages state for all tracked persons in restricted zone."""
    
    def __init__(self):
        # track_id -> TrackZoneState
        self.track_states: Dict[int, TrackZoneState] = {}
    
    def get_or_create_track_state(self, track_id: int) -> TrackZoneState:
        """Get existing track state or create new one."""
        if track_id not in self.track_states:
            self.track_states[track_id] = TrackZoneState(track_id)
        return self.track_states[track_id]
    
    def update_track_state(
        self,
        track_id: int,
        touches_zone: bool,
        inside_zone: bool,
        timestamp: datetime,
        stability_frames: int = 3,
    ) -> TrackZoneState:
        """
        Update track state with new detection.
        
        Args:
            track_id: Track ID
            touches_zone: Whether box touches zone (any corner inside or intersection)
            inside_zone: Whether majority of box is inside
            timestamp: Current timestamp
            stability_frames: Number of consecutive frames needed for confirmation
            
        Returns:
            Updated TrackZoneState
        """
        state = self.get_or_create_track_state(track_id)
        state.last_seen = timestamp
        
        # Update frame counters for stability
        if touches_zone:
            state.touches_frame_count += 1
            state.outside_frame_count = 0
        else:
            state.touches_frame_count = 0
            state.outside_frame_count += 1
        
        if inside_zone:
            state.inside_frame_count += 1
        else:
            state.inside_frame_count = 0
        
        # Confirm touches after stability_frames consecutive detections
        if state.touches_frame_count >= stability_frames:
            state.confirmed_touches = True
            state.touches_zone = True
        elif state.outside_frame_count >= stability_frames:
            state.confirmed_touches = False
            state.touches_zone = False
        
        # Confirm inside after stability_frames consecutive detections
        if state.inside_frame_count >= stability_frames:
            state.confirmed_inside = True
            state.inside_zone = True
            # Set entry time if just entered
            if state.entry_time is None:
                state.entry_time = timestamp
        elif state.inside_frame_count == 0:
            state.confirmed_inside = False
            state.inside_zone = False
            state.entry_time = None  # Reset entry time when outside
        
        return state
    
    def get_duration_inside(self, track_id: int, current_time: datetime) -> float:
        """Get duration person has been inside zone (in seconds)."""
        state = self.track_states.get(track_id)
        if not state or not state.entry_time:
            return 0.0
        return (current_time - state.entry_time).total_seconds()
    
    def cleanup_inactive_tracks(self, active_track_ids: set, max_age_seconds: float = 5.0):
        """Remove state for tracks that are no longer active."""
        current_time = utc_now()
        to_remove = []
        
        for track_id, state in self.track_states.items():
            if track_id not in active_track_ids:
                # Check if track hasn't been seen for too long
                if state.last_seen:
                    age = (current_time - state.last_seen).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(track_id)
                else:
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.track_states[track_id]
    
    def reset(self):
        """Clear all track states."""
        self.track_states.clear()
