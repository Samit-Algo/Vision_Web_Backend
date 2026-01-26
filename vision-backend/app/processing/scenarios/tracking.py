"""
Object Tracking for Scenarios
------------------------------

Simple object tracker for line-based counting.
Used by class_count and box_count scenarios.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class Track:
    """Represents a single tracked object."""
    
    def __init__(self, track_id: int, bbox: List[float], score: float, frame_id: int):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.frame_id = frame_id  # Frame where this track was first seen
        self.center = self._calculate_center()
        self.history = [self.center]  # Movement history
        self.age = 1
        self.hit_streak = 1
    
    def _calculate_center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        center_x = float(x1 + x2) / 2.0
        center_y = float(y1 + y2) / 2.0
        return (center_x, center_y)
    
    def update(self, bbox: List[float], score: float, frame_id: int):
        """Update track with new detection."""
        self.bbox = bbox
        self.score = score
        self.frame_id = frame_id
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
        self.frame_id = 0  # Current frame number
    
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
        detections: List[Tuple[List[float], float]],
        tracks: List[Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to tracks using IoU."""
        # Edge case: No detections
        if len(detections) == 0:
            unmatched_track_indices = list(range(len(tracks)))
            return [], [], unmatched_track_indices
        
        # Edge case: No tracks
        if len(tracks) == 0:
            unmatched_detection_indices = list(range(len(detections)))
            return [], unmatched_detection_indices, []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d_idx, (d_bbox, _) in enumerate(detections):
            for t_idx, track in enumerate(tracks):
                iou_matrix[d_idx, t_idx] = self._iou(d_bbox, track.bbox)
        
        # Find matches above threshold (greedy matching)
        potential_matches = []
        for d_idx in range(len(detections)):
            for t_idx in range(len(tracks)):
                overlap_score = iou_matrix[d_idx, t_idx]
                if overlap_score > self.iou_threshold:
                    potential_matches.append((d_idx, t_idx, overlap_score))
        
        # Sort by IoU (highest first)
        potential_matches.sort(key=lambda x: x[2], reverse=True)
        
        matches = []
        used_detections = set()
        used_tracks = set()
        unmatched_track_indices = list(range(len(tracks)))
        
        for d_idx, t_idx, _ in potential_matches:
            if d_idx not in used_detections and t_idx not in used_tracks:
                matches.append((d_idx, t_idx))
                used_detections.add(d_idx)
                used_tracks.add(t_idx)
                if t_idx in unmatched_track_indices:
                    unmatched_track_indices.remove(t_idx)
        
        unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
        
        return matches, unmatched_detections, unmatched_track_indices
    
    def update(self, detections: List[Tuple[List[float], float]]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (bbox, score) tuples
            
        Returns:
            List of confirmed active tracks
        """
        self.frame_id += 1
        
        # Filter by confidence
        valid_detections = [(bbox, score) for bbox, score in detections 
                           if score >= self.score_threshold]
        
        # Separate confirmed and unconfirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.hit_streak >= self.min_hits]
        unconfirmed_tracks = [t for t in self.tracks if t.hit_streak < self.min_hits]
        
        # Match detections to confirmed tracks first
        matches, unmatched_dets, unmatched_track_indices = self._match_detections_to_tracks(
            valid_detections, confirmed_tracks
        )
        
        # Update matched confirmed tracks
        for d_idx, t_idx in matches:
            bbox, score = valid_detections[d_idx]
            confirmed_tracks[t_idx].update(bbox, score, self.frame_id)
        
        # Try to match unmatched detections to unconfirmed tracks
        unmatched_detections = [valid_detections[idx] for idx in unmatched_dets]
        remaining_unmatched_indices = set(unmatched_dets)
        
        if len(unmatched_detections) > 0 and len(unconfirmed_tracks) > 0:
            matches_2, _, _ = self._match_detections_to_tracks(
                unmatched_detections, unconfirmed_tracks
            )
            
            for d_idx, t_idx in matches_2:
                bbox, score = unmatched_detections[d_idx]
                unconfirmed_tracks[t_idx].update(bbox, score, self.frame_id)
                # Map back to original index in valid_detections
                original_idx = unmatched_dets[d_idx]
                remaining_unmatched_indices.discard(original_idx)
        
        # Create new tracks for remaining unmatched detections
        for d_idx in remaining_unmatched_indices:
            if d_idx < len(valid_detections):
                bbox, score = valid_detections[d_idx]
                new_track = Track(self.track_id_counter, bbox, score, self.frame_id)
                self.tracks.append(new_track)
                self.track_id_counter += 1
        
        # Remove old tracks that haven't been seen for too long
        active_tracks = []
        for track in self.tracks:
            # Keep track if it was updated this frame
            if track.frame_id == self.frame_id:
                active_tracks.append(track)
            # Or if it's still within max_age (might come back)
            elif self.frame_id - track.frame_id <= self.max_age:
                active_tracks.append(track)
        
        self.tracks = active_tracks
        
        # Return only confirmed active tracks that were updated this frame
        confirmed_active_tracks = [
            track for track in self.tracks 
            if track.hit_streak >= self.min_hits and track.frame_id == self.frame_id
        ]
        return confirmed_active_tracks
    
    def get_all_active_tracks(self) -> List[Track]:
        """
        Get all active tracks (both confirmed and unconfirmed) for line crossing detection.
        
        This is useful when min_hits > 1 but we still want to detect crossings
        for tracks that haven't been confirmed yet.
        
        Returns:
            List of all active tracks (updated this frame)
        """
        return [
            track for track in self.tracks 
            if track.frame_id == self.frame_id
        ]


class LineCrossingCounter:
    """
    Counts objects crossing a line.
    
    Works for any line orientation: horizontal, vertical, or diagonal.
    Uses cross product to determine which side of the line a point is on.
    
    Line coordinates are stored as percentages (0.0-1.0) and converted to
    absolute coordinates based on frame dimensions.
    """
    
    def __init__(self, line_coordinates: List[List[float]], direction: str = "both", count_mode: str = "entry_exit"):
        """
        Initialize line crossing counter.
        
        Args:
            line_coordinates: [[x1, y1], [x2, y2]] line endpoints in percentage (0.0-1.0)
            direction: "entry", "exit", or "both" (for entry_exit mode)
            count_mode: "single" (count once per track) or "entry_exit" (separate entry/exit counts)
        """
        if len(line_coordinates) != 2:
            raise ValueError("Line requires exactly 2 coordinates")
        if direction not in ["entry", "exit", "both"]:
            raise ValueError(f"direction must be 'entry', 'exit', or 'both', got '{direction}'")
        if count_mode not in ["single", "entry_exit"]:
            raise ValueError(f"count_mode must be 'single' or 'entry_exit', got '{count_mode}'")
        
        # Store line endpoints as percentages (0.0-1.0)
        self.line_start_percent = (line_coordinates[0][0], line_coordinates[0][1])
        self.line_end_percent = (line_coordinates[1][0], line_coordinates[1][1])
        self.line_coordinates = line_coordinates
        self.direction = direction
        self.count_mode = count_mode
        
        # Initialize counters
        self.entry_count = 0
        self.exit_count = 0
        self.boxes_counted = 0  # For single count mode (like old code)
        
        # Absolute coordinates (will be calculated from frame dimensions)
        self.line_start: Optional[Tuple[float, float]] = None
        self.line_end: Optional[Tuple[float, float]] = None
        self.line_vector: Optional[Tuple[float, float]] = None
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        
        # Remember which side each track is on (used for touch-based counting to determine direction)
        # Format: {track_id: 'bottom_side', 'top_side', 'left_side', or 'right_side'}
        self.track_sides: Dict[int, str] = {}
        
        # For touch-based counting: track which boxes have touched the line
        # Format: {track_id: {'touched': bool, 'last_side': str, 'counted': bool, 'direction': str}}
        # direction: 'entry' or 'exit' based on generic rules:
        #   - Horizontal line: bottom-to-top = "entry", top-to-bottom = "exit"
        #   - Vertical line: right-to-left = "entry", left-to-right = "exit"
        self._touch_tracking: Dict[int, Dict[str, Any]] = {}
        self.touch_threshold_pixels: float = 10.0  # Distance threshold for "touching" the line (increased from 5.0 for better detection)
    
    def update_frame_dimensions(self, frame_width: int, frame_height: int):
        """
        Update absolute line coordinates based on frame dimensions.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        # Only recalculate if dimensions changed
        if self.frame_width == frame_width and self.frame_height == frame_height:
            return
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Convert percentage coordinates to absolute coordinates
        self.line_start = (
            self.line_start_percent[0] * frame_width,
            self.line_start_percent[1] * frame_height
        )
        self.line_end = (
            self.line_end_percent[0] * frame_width,
            self.line_end_percent[1] * frame_height
        )
        
        # Calculate line vector (direction from start to end)
        self.line_vector = (
            self.line_end[0] - self.line_start[0],
            self.line_end[1] - self.line_start[1]
        )
    
    def _is_horizontal_line(self) -> bool:
        """
        Determine if line is mostly horizontal (within 45 degrees).
        
        Returns:
            True if line is mostly horizontal, False if mostly vertical
        """
        if self.line_vector is None:
            return False
        # Calculate angle: if |dy| < |dx|, line is more horizontal
        return abs(self.line_vector[1]) < abs(self.line_vector[0])
    
    def _get_side(self, point: Tuple[float, float]) -> str:
        """
        Determine which side of line the point is on using a generic approach.
        
        For horizontal lines:
        - Points below the line = 'bottom_side'
        - Points above the line = 'top_side'
        
        For vertical lines:
        - Points to the right of the line = 'right_side'
        - Points to the left of the line = 'left_side'
        
        Uses cross product to determine side, then maps to semantic names.
        
        Returns: 'bottom_side', 'top_side', 'right_side', or 'left_side'
        """
        if self.line_start is None or self.line_vector is None:
            raise ValueError("Line coordinates not initialized. Call update_frame_dimensions() first.")
        
        # Vector from line start to point
        point_vector = (
            point[0] - self.line_start[0],
            point[1] - self.line_start[1]
        )
        
        # Calculate cross product: (line_vector.x * point_vector.y) - (line_vector.y * point_vector.x)
        cross_product = (self.line_vector[0] * point_vector[1]) - (self.line_vector[1] * point_vector[0])
        
        # Determine side based on line orientation
        is_horizontal = self._is_horizontal_line()
        
        if is_horizontal:
            # Horizontal line: use Y-coordinate to determine top/bottom
            # In image coordinates, Y increases downward
            # For horizontal line going left-to-right: cross = dx * (py - start_y)
            #   - If point is below line (py > start_y): cross > 0 = bottom_side
            #   - If point is above line (py < start_y): cross < 0 = top_side
            if self.line_vector[0] >= 0:  # Line goes left-to-right
                # Positive cross = below line (bottom_side), negative = above line (top_side)
                return 'bottom_side' if cross_product >= 0 else 'top_side'
            else:  # Line goes right-to-left, reverse the logic
                return 'top_side' if cross_product >= 0 else 'bottom_side'
        else:
            # Vertical line: use X-coordinate to determine left/right
            # For vertical line going top-to-bottom: cross = -dy * (px - start_x)
            #   - If point is to the right (px > start_x): cross < 0 = right_side
            #   - If point is to the left (px < start_x): cross > 0 = left_side
            if self.line_vector[1] >= 0:  # Line goes top-to-bottom
                # Positive cross = left of line (left_side), negative = right of line (right_side)
                return 'left_side' if cross_product >= 0 else 'right_side'
            else:  # Line goes bottom-to-top, reverse the logic
                return 'right_side' if cross_product >= 0 else 'left_side'
    
    def is_point_on_line(self, point: Tuple[float, float], threshold_pixels: float = 5.0) -> bool:
        """
        Check if a point is on or near the line (within threshold distance).
        
        Uses the distance from point to line segment formula.
        This allows highlighting boxes when their center point touches the line.
        
        Args:
            point: Point to check (x, y) in absolute coordinates
            threshold_pixels: Maximum distance in pixels to consider "on the line" (default: 5.0)
            
        Returns:
            True if point is within threshold distance of the line, False otherwise
        """
        if self.line_start is None or self.line_end is None or self.line_vector is None:
            return False
        
        # Calculate distance from point to line segment
        # Using formula: distance = |(point - line_start) × line_vector| / |line_vector|
        # Where × is cross product
        
        # Vector from line start to point
        point_vector = (
            point[0] - self.line_start[0],
            point[1] - self.line_start[1]
        )
        
        # Calculate cross product magnitude (absolute value)
        cross_product = (self.line_vector[0] * point_vector[1]) - (self.line_vector[1] * point_vector[0])
        cross_product_abs = abs(cross_product)
        
        # Calculate line length
        line_length = (self.line_vector[0] ** 2 + self.line_vector[1] ** 2) ** 0.5
        
        if line_length == 0:
            # Degenerate line (start == end), check if point is at that location
            dist_to_start = ((point[0] - self.line_start[0]) ** 2 + (point[1] - self.line_start[1]) ** 2) ** 0.5
            return dist_to_start <= threshold_pixels
        
        # Distance from point to line = |cross_product| / line_length
        distance_to_line = cross_product_abs / line_length
        
        # Also check if point is within the line segment bounds (not just on the infinite line)
        # Project point onto line vector
        line_length_sq = line_length ** 2
        if line_length_sq == 0:
            return distance_to_line <= threshold_pixels
        
        # Dot product: (point_vector · line_vector) / |line_vector|^2
        dot_product = (point_vector[0] * self.line_vector[0] + point_vector[1] * self.line_vector[1]) / line_length_sq
        
        # Check if projection is within line segment [0, 1]
        if 0 <= dot_product <= 1:
            # Point projects onto the line segment, check distance
            return distance_to_line <= threshold_pixels
        else:
            # Point projects outside line segment, check distance to nearest endpoint
            dist_to_start = ((point[0] - self.line_start[0]) ** 2 + (point[1] - self.line_start[1]) ** 2) ** 0.5
            dist_to_end = ((point[0] - self.line_end[0]) ** 2 + (point[1] - self.line_end[1]) ** 2) ** 0.5
            return min(dist_to_start, dist_to_end) <= threshold_pixels
    
    
    def get_counts(self) -> Dict[str, int]:
        """Get current crossing counts."""
        if self.count_mode == "single":
            return {
                "boxes_counted": self.boxes_counted,
                "entry_count": self.boxes_counted,  # For compatibility
                "exit_count": 0,
                "net_count": self.boxes_counted
            }
        else:
            return {
                "entry_count": self.entry_count,
                "exit_count": self.exit_count,
                "net_count": self.entry_count - self.exit_count
            }
    
    def get_absolute_coordinates(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get absolute line coordinates for drawing.
        
        Returns:
            Tuple of ((x1, y1), (x2, y2)) in absolute coordinates, or None if not initialized
        """
        if self.line_start is None or self.line_end is None:
            return None
        return (self.line_start, self.line_end)
    
    def get_percentage_coordinates(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get percentage line coordinates (0.0-1.0).
        
        Returns:
            Tuple of ((x1, y1), (x2, y2)) in percentage coordinates
        """
        return (self.line_start_percent, self.line_end_percent)
    
    def check_touch(self, track: Track) -> Optional[str]:
        """
        Check if track crossed the line and count based on crossing (like old code).
        
        This method uses CROSSING-BASED detection (more reliable than touch-based):
        - Compares previous side vs current side
        - If sides changed = object crossed the line = COUNT
        - Determines in/out direction based on which side object came from
        
        This is more accurate than touch-based because it detects actual crossing,
        not just when center point is near the line (which can miss fast-moving objects).
        
        Args:
            track: Track to check
            
        Returns:
            'entry' if object crossed line in entry direction, 'exit' if exit direction,
            None if not crossed or already counted
        """
        if self.line_start is None or self.line_vector is None:
            return None
        
        # Need at least 2 positions in history to detect crossing
        if len(track.history) < 2:
            return None
        
        # Get current and previous positions (like old code)
        current_position = track.center
        previous_position = track.history[-2]  # 2 frames ago (like old code)
        
        # Get sides for both positions
        current_side = self._get_side(current_position)
        previous_side = self._get_side(previous_position)
        
        # Initialize tracking for this track if needed
        if track.track_id not in self._touch_tracking:
            self._touch_tracking[track.track_id] = {
                'touched': False,
                'last_side': current_side,
                'counted': False,
                'direction': None
            }
            # Store initial side
            self._touch_tracking[track.track_id]['last_side'] = current_side
            return None
        
        touch_info = self._touch_tracking[track.track_id]
        
        # Check for crossing: moved from one side to the other (like old code)
        # This is the KEY difference: we detect crossing, not just touching
        if previous_side != current_side:
            # Object crossed the line!
            
            # If already counted for this track, don't count again (prevent duplicates)
            if touch_info['counted']:
                touch_info['last_side'] = current_side
                return None
            
            # Determine direction based on line orientation and movement
            # Generic rules:
            # - Horizontal line: bottom-to-top = "in", top-to-bottom = "out"
            # - Vertical line: right-to-left = "in", left-to-right = "out"
            direction = None
            is_horizontal = self._is_horizontal_line()
            
            if is_horizontal:
                # Horizontal line: bottom-to-top = "in", top-to-bottom = "out"
                if previous_side == 'bottom_side' and current_side == 'top_side':
                    direction = 'entry'  # Moving from bottom to top = IN
                elif previous_side == 'top_side' and current_side == 'bottom_side':
                    direction = 'exit'   # Moving from top to bottom = OUT
            else:
                # Vertical line: right-to-left = "in", left-to-right = "out"
                if previous_side == 'right_side' and current_side == 'left_side':
                    direction = 'entry'  # Moving from right to left = IN
                elif previous_side == 'left_side' and current_side == 'right_side':
                    direction = 'exit'   # Moving from left to right = OUT
            
            # If direction couldn't be determined, skip (shouldn't happen)
            if direction is None:
                touch_info['last_side'] = current_side
                return None
            
            # Check if we should count based on direction setting
            should_count = False
            if self.direction == "both":
                should_count = True
            elif self.direction == "entry" and direction == "entry":
                should_count = True
            elif self.direction == "exit" and direction == "exit":
                should_count = True
            
            if should_count:
                # Count the crossing (only once per track to prevent duplicates)
                touch_info['touched'] = True
                touch_info['counted'] = True
                touch_info['direction'] = direction
                touch_info['last_side'] = current_side
                
                if direction == 'entry':
                    self.entry_count += 1
                    print(f"[CROSSING_COUNT] ✅ Track {track.track_id} CROSSED LINE (ENTRY) - Center: ({track.center[0]:.1f}, {track.center[1]:.1f}), Total IN: {self.entry_count}")
                else:
                    self.exit_count += 1
                    print(f"[CROSSING_COUNT] ✅ Track {track.track_id} CROSSED LINE (EXIT) - Center: ({track.center[0]:.1f}, {track.center[1]:.1f}), Total OUT: {self.exit_count}")
                
                return direction
        
        # Update last side for next frame
        touch_info['last_side'] = current_side
        return None
    
    def is_track_touching(self, track: Track) -> bool:
        """Check if a track's center point is currently touching the line."""
        if self.line_start is None or self.line_vector is None:
            return False
        return self.is_point_on_line(track.center, threshold_pixels=self.touch_threshold_pixels)
    
    def get_track_touch_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get touch tracking information for a track."""
        return self._touch_tracking.get(track_id)
    
    def reset(self):
        """Reset all counts and tracking data."""
        self.entry_count = 0
        self.exit_count = 0
        self.boxes_counted = 0
        self.track_sides.clear()
        self._touch_tracking.clear()
