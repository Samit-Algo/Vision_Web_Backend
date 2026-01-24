"""
Object Tracking for Scenarios
------------------------------

Simple object tracker for line-based counting.
Used by class_count and box_count scenarios.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np


class Track:
    """Represents a single tracked object."""
    
    def __init__(self, track_id: int, bbox: List[float], score: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.center = self._calculate_center()
        self.history = [self.center]  # Movement history
        self.age = 1
        self.hit_streak = 1
    
    def _calculate_center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def update(self, bbox: List[float], score: float):
        """Update track with new detection."""
        self.bbox = bbox
        self.score = score
        self.center = self._calculate_center()
        self.history.append(self.center)
        self.age += 1
        self.hit_streak += 1
        
        # Keep only last 30 positions
        if len(self.history) > 30:
            self.history = self.history[-30:]


class SimpleTracker:
    """Simple object tracker using IoU matching."""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, 
                 iou_threshold: float = 0.3, score_threshold: float = 0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.tracks: List[Track] = []
        self.track_id_counter = 0
        self.frame_count = 0
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Find intersection
        overlap_left = max(x1_1, x1_2)
        overlap_top = max(y1_1, y1_2)
        overlap_right = min(x2_1, x2_2)
        overlap_bottom = min(y2_1, y2_2)
        
        if overlap_right < overlap_left or overlap_bottom < overlap_top:
            return 0.0
        
        intersection = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _match_detections_to_tracks(
        self, 
        detections: List[Tuple[List[float], float]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to tracks using IoU."""
        if not detections or not self.tracks:
            unmatched_detections = list(range(len(detections))) if detections else []
            unmatched_tracks = list(range(len(self.tracks))) if self.tracks else []
            return [], unmatched_detections, unmatched_tracks
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d_idx, (d_bbox, _) in enumerate(detections):
            for t_idx, track in enumerate(self.tracks):
                iou_matrix[d_idx, t_idx] = self._iou(d_bbox, track.bbox)
        
        # Find matches above threshold (greedy matching)
        matches = []
        used_detections = set()
        used_tracks = set()
        
        # Sort by IoU (highest first)
        potential_matches = []
        for d_idx in range(len(detections)):
            for t_idx in range(len(self.tracks)):
                if iou_matrix[d_idx, t_idx] > self.iou_threshold:
                    potential_matches.append((d_idx, t_idx, iou_matrix[d_idx, t_idx]))
        
        potential_matches.sort(key=lambda x: x[2], reverse=True)
        
        for d_idx, t_idx, _ in potential_matches:
            if d_idx not in used_detections and t_idx not in used_tracks:
                matches.append((d_idx, t_idx))
                used_detections.add(d_idx)
                used_tracks.add(t_idx)
        
        unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        
        return matches, unmatched_detections, unmatched_tracks
    
    def update(self, detections: List[Tuple[List[float], float]]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (bbox, score) tuples
            
        Returns:
            List of confirmed active tracks
        """
        self.frame_count += 1
        
        # Filter by confidence
        valid_detections = [(bbox, score) for bbox, score in detections 
                           if score >= self.score_threshold]
        
        # Match detections to existing tracks
        matches, unmatched_dets, unmatched_tracks = self._match_detections_to_tracks(valid_detections)
        
        # Update matched tracks
        for d_idx, t_idx in matches:
            bbox, score = valid_detections[d_idx]
            self.tracks[t_idx].update(bbox, score)
        
        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            if d_idx < len(valid_detections):
                bbox, score = valid_detections[d_idx]
                new_track = Track(self.track_id_counter, bbox, score)
                self.tracks.append(new_track)
                self.track_id_counter += 1
        
        # Remove old tracks that haven't been updated
        current_frame = self.frame_count
        self.tracks = [t for t in self.tracks 
                      if (current_frame - t.age + t.hit_streak) <= self.max_age]
        
        # Return confirmed active tracks (hit_streak >= min_hits)
        return [t for t in self.tracks if t.hit_streak >= self.min_hits]


class LineCrossingCounter:
    """Counts objects crossing a line."""
    
    def __init__(self, line_coordinates: List[List[float]], direction: str = "both"):
        """
        Initialize line crossing counter.
        
        Args:
            line_coordinates: [[x1, y1], [x2, y2]] line endpoints
            direction: "entry", "exit", or "both"
        """
        if len(line_coordinates) != 2:
            raise ValueError("Line requires exactly 2 coordinates")
        if direction not in ["entry", "exit", "both"]:
            raise ValueError(f"direction must be 'entry', 'exit', or 'both', got '{direction}'")
        
        self.line_coordinates = line_coordinates
        self.direction = direction
        self.entry_count = 0
        self.exit_count = 0
        self.track_states: Dict[int, str] = {}  # {track_id: 'above'/'below'}
        
        # Calculate line midpoint Y for simple above/below detection
        self.line_y = (line_coordinates[0][1] + line_coordinates[1][1]) / 2
    
    def _get_side(self, point: Tuple[float, float]) -> str:
        """Determine which side of line the point is on."""
        return 'below' if point[1] > self.line_y else 'above'
    
    def check_crossing(self, track: Track) -> Optional[str]:
        """
        Check if track crossed the line.
        
        Args:
            track: Track to check
            
        Returns:
            'entry', 'exit', or None if no crossing detected
        """
        if len(track.history) < 2:
            return None
        
        current_pos = track.center
        previous_pos = track.history[-2]
        
        current_side = self._get_side(current_pos)
        previous_side = self._get_side(previous_pos)
        
        # Initialize track state
        if track.track_id not in self.track_states:
            self.track_states[track.track_id] = current_side
            return None
        
        # Check for crossing
        if previous_side == 'above' and current_side == 'below':
            if self.track_states.get(track.track_id) == 'above':
                self.track_states[track.track_id] = 'below'
                if self.direction in ["entry", "both"]:
                    self.entry_count += 1
                    return 'entry'
        
        elif previous_side == 'below' and current_side == 'above':
            if self.track_states.get(track.track_id) == 'below':
                self.track_states[track.track_id] = 'above'
                if self.direction in ["exit", "both"]:
                    self.exit_count += 1
                    return 'exit'
        
        self.track_states[track.track_id] = current_side
        return None
    
    def get_counts(self) -> Dict[str, int]:
        """Get current crossing counts."""
        return {
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "net_count": self.entry_count - self.exit_count
        }
    
    def reset(self):
        """Reset all counts."""
        self.entry_count = 0
        self.exit_count = 0
        self.track_states.clear()
