"""
Object Tracking for Scenarios
------------------------------

Simple object tracker for line-based counting.
Used by class_count, box_count, wall_climb_detection, fall_detection.

- Kalman filter: predicts position/size when person is briefly occluded so the same
  track_id can be re-assigned when they reappear (reduces ID switches and false new IDs).
- For "same person after long absence" (many seconds): consider ByteTrack/BoT-SORT with
  Re-ID (appearance embedding) — see TRACKING_README or integration notes below.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np


# -----------------------------------------------------------------------------
# Kalman filter for bounding box (constant-velocity model)
# -----------------------------------------------------------------------------

class KalmanBoxFilter:
    """
    Kalman filter for [cx, cy, w, h, vx, vy].
    Predicts where the box will be next frame so we can match the same person
    when they reappear after brief occlusion (same track_id, fewer false positives).
    Uses only numpy; no extra dependencies.
    """
    def __init__(self, bbox: List[float], dt: float = 1.0):
        # State: [cx, cy, w, h, vx, vy]
        x1, y1, x2, y2 = bbox[:4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        self.dt = dt
        self.state = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)
        # State covariance: allow uncertainty to grow when not updated
        self.P = np.eye(6, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 2.0  # velocity more uncertain initially
        # Process noise (motion uncertainty per step)
        self.Q = np.eye(6, dtype=np.float64) * 0.5
        self.Q[0, 0] = self.Q[1, 1] = 1.0
        self.Q[4, 4] = self.Q[5, 5] = 4.0
        # Measurement noise
        self.R = np.diag([5.0, 5.0, 10.0, 10.0]).astype(np.float64)
        # F: state transition (constant velocity)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 4] = self.F[1, 5] = dt
        # H: we observe [cx, cy, w, h]
        self.H = np.zeros((4, 6), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

    def predict(self) -> np.ndarray:
        """Predict next state. Returns state vector [cx, cy, w, h, vx, vy]."""
        self.state = self.F.dot(self.state)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.state.copy()

    def update(self, bbox: List[float]) -> np.ndarray:
        """Update with new bbox measurement. Returns state after update."""
        x1, y1, x2, y2 = bbox[:4]
        z = np.array([
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            max(x2 - x1, 1.0),
            max(y2 - y1, 1.0),
        ], dtype=np.float64)
        y = z - self.H.dot(self.state)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        try:
            K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        except np.linalg.LinAlgError:
            return self.state.copy()
        self.state = self.state + K.dot(y)
        self.P = (np.eye(6) - K.dot(self.H)).dot(self.P)
        return self.state.copy()

    def get_predicted_bbox(self) -> List[float]:
        """Return [x1, y1, x2, y2] from current state (after predict)."""
        s = self.state
        cx, cy, w, h = s[0], s[1], s[2], s[3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return [float(x1), float(y1), float(x2), float(y2)]

    def get_predicted_center(self) -> Tuple[float, float]:
        """Return (cx, cy) from current state."""
        return (float(self.state[0]), float(self.state[1]))


class Track:
    """
    Represents a single tracked object.
    Uses Kalman filter for prediction so the same person can be re-matched
    when they reappear after brief occlusion (stable track_id, fewer false new IDs).
    """
    
    def __init__(self, track_id: int, bbox: List[float], score: float, frame_id: int, use_kalman: bool = True):
        self.track_id = track_id
        self.bbox = list(bbox)  # [x1, y1, x2, y2]
        self.score = score
        self.frame_id = frame_id  # Frame where this track was last updated
        self.center = self.calculate_center()
        self.history = [self.center]  # Movement history
        self.age = 1
        self.hit_streak = 1
        self.time_since_update = 0  # Frames since last successful match
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        # Kalman filter for better prediction when person reappears after occlusion
        self._kalman: Optional[KalmanBoxFilter] = KalmanBoxFilter(bbox) if use_kalman else None
        if self._kalman:
            self._kalman.update(bbox)  # initialize with first measurement
        self._cached_predicted_bbox: Optional[List[float]] = None
        self._cached_predicted_center: Optional[Tuple[float, float]] = None
    
    def advance_predict(self) -> None:
        """Call once per frame before matching: run Kalman predict and cache result."""
        if self._kalman is not None:
            self._kalman.predict()
            self._cached_predicted_bbox = self._kalman.get_predicted_bbox()
            self._cached_predicted_center = self._kalman.get_predicted_center()
        else:
            self.update_velocity()
            dx, dy = self.velocity
            x1, y1, x2, y2 = self.bbox
            self._cached_predicted_bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
            self._cached_predicted_center = (self.center[0] + dx, self.center[1] + dy)
    
    def calculate_center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        center_x = float(x1 + x2) / 2.0
        center_y = float(y1 + y2) / 2.0
        return (center_x, center_y)
    
    def update_velocity(self):
        """Update velocity based on recent movement."""
        if len(self.history) >= 2:
            # Use last 2 positions to estimate velocity
            prev = self.history[-2]
            curr = self.history[-1]
            self.velocity = (curr[0] - prev[0], curr[1] - prev[1])
        else:
            self.velocity = (0.0, 0.0)
    
    def predict_position(self) -> Tuple[float, float]:
        """Predict next position (uses cached result from advance_predict() if set)."""
        if self._cached_predicted_center is not None:
            return self._cached_predicted_center
        self.advance_predict()
        return self._cached_predicted_center or self.center
    
    def get_predicted_bbox(self) -> List[float]:
        """Get predicted bbox (uses cached result from advance_predict() if set)."""
        if self._cached_predicted_bbox is not None:
            return self._cached_predicted_bbox
        self.advance_predict()
        return self._cached_predicted_bbox or self.bbox
    
    def update(self, bbox: List[float], score: float, frame_id: int):
        """Update track with new detection."""
        self.bbox = list(bbox)
        self.score = score
        self.frame_id = frame_id
        self.center = self.calculate_center()
        self.history.append(self.center)
        self.age += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.update_velocity()
        if self._kalman is not None:
            self._kalman.update(bbox)
        self._cached_predicted_bbox = None
        self._cached_predicted_center = None
        if len(self.history) > 30:
            self.history = self.history[-30:]
    
    def mark_missed(self):
        """Mark this track as missed (not matched in current frame)."""
        self.time_since_update += 1
        self.hit_streak = 0
        self._cached_predicted_bbox = None
        self._cached_predicted_center = None


class SimpleTracker:
    """
    Enhanced object tracker for conveyor belt scenarios.
    
    Uses a multi-stage matching strategy for stable track ID assignment:
    1. Primary matching: IoU + Velocity Prediction (for fast-moving boxes)
    2. Secondary matching: Center Distance with predicted position
    3. Fallback matching: Distance-only for lost tracks (recover IDs)
    4. Each detection is assigned to at most one track
    5. Unmatched detections create new tracks
    
    Key improvements for conveyor belts:
    - Velocity-based position prediction
    - Lower IoU threshold for fast motion
    - Distance-based fallback matching
    - Better handling of temporary occlusions
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3, score_threshold: float = 0.5,
                 max_distance_threshold: float = 150.0,
                 max_distance_threshold_max: float = 350.0,
                 distance_growth_per_missed_frame: float = 8.0,
                 use_kalman: bool = True):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.tracks: List[Track] = []
        self.track_id_counter = 0
        self.frame_id = 0  # Current frame number
        self.use_kalman = use_kalman
        # Distance threshold for fallback matching (pixels); scales when track was missed (re-appearance)
        self.max_distance_threshold = max_distance_threshold
        self.max_distance_threshold_max = max_distance_threshold_max
        self.distance_growth_per_missed_frame = distance_growth_per_missed_frame
    
    def iou(self, box1: List[float], box2: List[float]) -> float:
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
    
    def center_distance(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Euclidean distance between centers of two boxes."""
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    
    def point_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    
    def get_detection_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of a detection bbox."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def match_detections_to_tracks(
        self, 
        detections: List[Tuple[List[float], float]],
        tracks: List[Track],
        use_prediction: bool = True
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Enhanced matching using IoU + Velocity Prediction + Center Distance.
        
        Matching strategy for conveyor belt scenarios:
        1. Use predicted position (from velocity) for matching
        2. Calculate IoU between detection and predicted track position
        3. Also calculate center distance to predicted position
        4. Combined score gives weight to both overlap and position
        5. Lower IoU threshold (0.15) for fast-moving boxes
        
        This ensures stable track IDs even when boxes move fast.
        """
        # Edge case: No detections
        if len(detections) == 0:
            unmatched_track_indices = list(range(len(tracks)))
            return [], [], unmatched_track_indices
        
        # Edge case: No tracks
        if len(tracks) == 0:
            unmatched_detection_indices = list(range(len(detections)))
            return [], unmatched_detection_indices, []
        
        # One Kalman predict per track so we have predicted position/bbox for matching
        if use_prediction:
            for track in tracks:
                track.advance_predict()
        
        # Calculate matching matrices
        num_dets = len(detections)
        num_tracks = len(tracks)
        
        iou_matrix = np.zeros((num_dets, num_tracks))
        iou_predicted_matrix = np.zeros((num_dets, num_tracks))  # IoU with predicted position
        dist_matrix = np.zeros((num_dets, num_tracks))
        dist_predicted_matrix = np.zeros((num_dets, num_tracks))  # Distance to predicted position
        
        for d_idx, (d_bbox, _) in enumerate(detections):
            det_center = self.get_detection_center(d_bbox)
            
            for t_idx, track in enumerate(tracks):
                # Current position matching
                iou_matrix[d_idx, t_idx] = self.iou(d_bbox, track.bbox)
                dist_matrix[d_idx, t_idx] = self.center_distance(d_bbox, track.bbox)
                
                # Predicted position matching (for fast-moving boxes)
                if use_prediction and len(track.history) >= 2:
                    predicted_bbox = track.get_predicted_bbox()
                    predicted_center = track.predict_position()
                    iou_predicted_matrix[d_idx, t_idx] = self.iou(d_bbox, predicted_bbox)
                    dist_predicted_matrix[d_idx, t_idx] = self.point_distance(det_center, predicted_center)
                else:
                    iou_predicted_matrix[d_idx, t_idx] = iou_matrix[d_idx, t_idx]
                    dist_predicted_matrix[d_idx, t_idx] = dist_matrix[d_idx, t_idx]
        
        # Use the BETTER of current vs predicted IoU
        best_iou_matrix = np.maximum(iou_matrix, iou_predicted_matrix)
        # Use the SMALLER of current vs predicted distance
        best_dist_matrix = np.minimum(dist_matrix, dist_predicted_matrix)
        
        # Normalize distances
        max_dist = np.max(best_dist_matrix) if np.max(best_dist_matrix) > 0 else 1.0
        normalized_dist = best_dist_matrix / max_dist
        
        # Combined score: IoU + (1 - normalized_distance) * weight
        # Prefer IoU when overlap is strong to reduce ID switches when objects cross paths
        distance_weight = 0.35
        combined_score = best_iou_matrix + (1.0 - normalized_dist) * distance_weight
        # Bonus for strong IoU so we keep the same track when overlap is clear
        iou_bonus = np.where(best_iou_matrix >= 0.4, 0.15, 0.0)
        combined_score = combined_score + iou_bonus
        
        # Lower IoU threshold for conveyor belts (fast-moving boxes)
        # Use 0.15 instead of self.iou_threshold for better matching
        effective_iou_threshold = min(self.iou_threshold, 0.15)
        
        # Per-track distance threshold: allow larger distance when track was missed more frames (re-appearance)
        def _distance_threshold(track: Track) -> float:
            growth = getattr(self, "distance_growth_per_missed_frame", 0.0)
            cap = getattr(self, "max_distance_threshold_max", self.max_distance_threshold)
            return min(cap, self.max_distance_threshold + track.time_since_update * growth)
        
        # Find matches (greedy matching using combined score)
        potential_matches = []
        for d_idx in range(num_dets):
            for t_idx in range(num_tracks):
                iou_score = best_iou_matrix[d_idx, t_idx]
                dist_score = best_dist_matrix[d_idx, t_idx]
                dist_thresh = _distance_threshold(tracks[t_idx])
                # Match if IoU is good enough OR distance is close enough (larger threshold for missed tracks)
                if iou_score >= effective_iou_threshold or dist_score < dist_thresh:
                    potential_matches.append((
                        d_idx, t_idx,
                        combined_score[d_idx, t_idx],
                        iou_score,
                        dist_score
                    ))
        
        # Sort by combined score (highest first)
        potential_matches.sort(key=lambda x: x[2], reverse=True)
        
        matches = []
        used_detections = set()
        used_tracks = set()
        unmatched_track_indices = list(range(num_tracks))
        
        for d_idx, t_idx, score, iou, dist in potential_matches:
            if d_idx not in used_detections and t_idx not in used_tracks:
                # Additional validation: don't match if both IoU is 0 AND distance is too far (use per-track threshold)
                if iou < 0.01 and dist > _distance_threshold(tracks[t_idx]):
                    continue
                matches.append((d_idx, t_idx))
                used_detections.add(d_idx)
                used_tracks.add(t_idx)
                if t_idx in unmatched_track_indices:
                    unmatched_track_indices.remove(t_idx)
        
        unmatched_detections = [i for i in range(num_dets) if i not in used_detections]
        
        return matches, unmatched_detections, unmatched_track_indices
    
    def fallback_distance_matching(
        self,
        detections: List[Tuple[List[float], float]],
        tracks: List[Track]
    ) -> List[Tuple[int, int]]:
        """
        Fallback matching using pure center distance.
        
        Used for tracks that couldn't be matched by IoU (e.g., after occlusion).
        This helps recover lost track IDs instead of creating new ones.
        """
        if len(detections) == 0 or len(tracks) == 0:
            return []
        
        matches = []
        used_detections = set()
        used_tracks = set()
        
        # Calculate distance from each detection to each track's predicted position
        distance_pairs = []
        for d_idx, (d_bbox, _) in enumerate(detections):
            det_center = self.get_detection_center(d_bbox)
            
            for t_idx, track in enumerate(tracks):
                predicted_center = track.predict_position()
                dist = self.point_distance(det_center, predicted_center)
                dist_thresh = min(
                    getattr(self, "max_distance_threshold_max", self.max_distance_threshold),
                    self.max_distance_threshold + track.time_since_update * getattr(self, "distance_growth_per_missed_frame", 0.0)
                )
                if dist < dist_thresh:
                    distance_pairs.append((d_idx, t_idx, dist))
        
        # Sort by distance (smallest first)
        distance_pairs.sort(key=lambda x: x[2])
        
        # Greedy matching
        for d_idx, t_idx, dist in distance_pairs:
            if d_idx not in used_detections and t_idx not in used_tracks:
                matches.append((d_idx, t_idx))
                used_detections.add(d_idx)
                used_tracks.add(t_idx)
        
        return matches
    
    def update(self, detections: List[Tuple[List[float], float]]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Enhanced update process for conveyor belt scenarios:
        1. Filter detections by confidence
        2. Match to confirmed tracks (with velocity prediction)
        3. Match remaining detections to unconfirmed tracks
        4. Fallback: Try distance-only matching for lost tracks
        5. Create new tracks for truly new detections
        6. Remove old tracks
        
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
        
        # STEP 1: Match detections to confirmed tracks (with velocity prediction)
        matches, unmatched_dets, unmatched_confirmed_indices = self.match_detections_to_tracks(
            valid_detections, confirmed_tracks, use_prediction=True
        )
        
        # Update matched confirmed tracks
        for d_idx, t_idx in matches:
            bbox, score = valid_detections[d_idx]
            confirmed_tracks[t_idx].update(bbox, score, self.frame_id)
        
        # Mark unmatched confirmed tracks as missed
        for t_idx in unmatched_confirmed_indices:
            confirmed_tracks[t_idx].mark_missed()
        
        # STEP 2: Match unmatched detections to unconfirmed tracks
        unmatched_detections = [valid_detections[idx] for idx in unmatched_dets]
        remaining_unmatched_indices = set(unmatched_dets)
        
        if len(unmatched_detections) > 0 and len(unconfirmed_tracks) > 0:
            matches_2, unmatched_dets_2, _ = self.match_detections_to_tracks(
                unmatched_detections, unconfirmed_tracks, use_prediction=True
            )
            
            for d_idx, t_idx in matches_2:
                bbox, score = unmatched_detections[d_idx]
                unconfirmed_tracks[t_idx].update(bbox, score, self.frame_id)
                # Map back to original index in valid_detections
                original_idx = unmatched_dets[d_idx]
                remaining_unmatched_indices.discard(original_idx)
        
        # STEP 3: Fallback - Try distance matching for lost confirmed tracks
        # This helps recover IDs for tracks that went out of view briefly
        if len(remaining_unmatched_indices) > 0 and len(unmatched_confirmed_indices) > 0:
            remaining_detections = [(valid_detections[idx][0], valid_detections[idx][1]) 
                                   for idx in remaining_unmatched_indices]
            lost_tracks = [confirmed_tracks[idx] for idx in unmatched_confirmed_indices]
            
            fallback_matches = self.fallback_distance_matching(remaining_detections, lost_tracks)
            
            remaining_list = list(remaining_unmatched_indices)
            for d_idx, t_idx in fallback_matches:
                original_d_idx = remaining_list[d_idx]
                original_t_idx = unmatched_confirmed_indices[t_idx]
                
                bbox, score = valid_detections[original_d_idx]
                confirmed_tracks[original_t_idx].update(bbox, score, self.frame_id)
                remaining_unmatched_indices.discard(original_d_idx)
        
        # STEP 4: Create new tracks for remaining unmatched detections
        for d_idx in remaining_unmatched_indices:
            if d_idx < len(valid_detections):
                bbox, score = valid_detections[d_idx]
                new_track = Track(self.track_id_counter, bbox, score, self.frame_id, use_kalman=self.use_kalman)
                self.tracks.append(new_track)
                self.track_id_counter += 1
        
        # STEP 5: Remove old tracks that haven't been seen for too long
        active_tracks = []
        for track in self.tracks:
            # Keep track if it was updated this frame
            if track.frame_id == self.frame_id:
                active_tracks.append(track)
            # Or if it's still within max_age (might come back - handles occlusion)
            elif self.frame_id - track.frame_id <= self.max_age:
                active_tracks.append(track)
        
        self.tracks = active_tracks
        
        # Return only confirmed active tracks that were updated this frame
        confirmed_active_tracks = [
            track for track in self.tracks 
            if track.hit_streak >= self.min_hits and track.frame_id == self.frame_id
        ]
        
        # Debug: Log tracking statistics (only when there are detections)
        if len(valid_detections) > 0:
            matched_count = len(valid_detections) - len(remaining_unmatched_indices)
            print(f"[TRACKER] 🔍 Matched: {matched_count}/{len(valid_detections)} detections | "
                  f"Active IDs: {[t.track_id for t in confirmed_active_tracks]} | "
                  f"Total tracks: {len(self.tracks)}")
        
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
        
        # For side-transition counting: track which boxes have crossed the line
        # Format: {track_id: {'initial_side': str, 'last_side': str, 'counted': bool, 'direction': str, 'count_frame': int}}
        # 
        # CONVEYOR BELT COUNTING:
        # - Each track ID is counted ONLY ONCE (counted=True is PERMANENT)
        # - Prevents jitter-based double counting
        # - direction: 'entry' (side2→side1, loading) or 'exit' (side1→side2, unloading)
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
    
    def get_side(self, point: Tuple[float, float]) -> str:
        """
        Determine which side of line the point is on.
        
        Uses cross product to determine which side of the line a point is on.
        Works for ANY line orientation (horizontal, vertical, or diagonal).
        
        ═══════════════════════════════════════════════════════════════════
        HORIZONTAL LINE (left to right):
        ═══════════════════════════════════════════════════════════════════
                    Side 2 (ABOVE)  ← negative cross product
            ─────────────────────────
                    Side 1 (BELOW)  ← positive cross product
            
            Movement: Side 2 → Side 1 (top to bottom) = ADD/ENTRY
            Movement: Side 1 → Side 2 (bottom to top) = OUT/EXIT
        
        ═══════════════════════════════════════════════════════════════════
        VERTICAL LINE (top to bottom):
        ═══════════════════════════════════════════════════════════════════
            Side 1 (LEFT)  │  Side 2 (RIGHT)
            positive cross │  negative cross
                           │
            Movement: Side 2 → Side 1 (right to left) = ADD/ENTRY
            Movement: Side 1 → Side 2 (left to right) = OUT/EXIT
        ═══════════════════════════════════════════════════════════════════
        
        Returns: 'side1' (positive cross) or 'side2' (negative cross)
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
        
        # Side determination based on cross product sign:
        # - positive cross = side1 (BELOW for horizontal, LEFT for vertical)
        # - negative cross = side2 (ABOVE for horizontal, RIGHT for vertical)
        return 'side1' if cross_product >= 0 else 'side2'
    
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
        Check if track crossed the line using side transition detection.
        
        ═══════════════════════════════════════════════════════════════════
        COUNTING LOGIC (Works for HORIZONTAL or VERTICAL lines):
        ═══════════════════════════════════════════════════════════════════
        
        HORIZONTAL LINE:
        ────────────────────────────
                Side 2 (ABOVE)
        ════════════════════════════  ← User-drawn line
                Side 1 (BELOW)
        ────────────────────────────
        
        - side2 → side1 (top to bottom) = ADD / ENTRY
        - side1 → side2 (bottom to top) = OUT / EXIT
        
        VERTICAL LINE:
        ─────────────────────────────
        Side 1 (LEFT) │ Side 2 (RIGHT)
                      │
        ─────────────────────────────
        
        - side2 → side1 (right to left) = ADD / ENTRY
        - side1 → side2 (left to right) = OUT / EXIT
        ═══════════════════════════════════════════════════════════════════
        
        IMPORTANT: Each track ID is counted ONLY ONCE.
        Once counted, it will NEVER be counted again (prevents jitter errors).
        
        Args:
            track: Track to check
            
        Returns:
            'entry' if object crossed from side2→side1, 
            'exit' if side1→side2,
            None if not crossed or already counted
        """
        if self.line_start is None or self.line_vector is None:
            return None
        
        # Need at least 3 positions in history for stable tracking
        # This ensures the track has been consistently detected for at least 3 frames
        # before we count it, reducing false counts from flickering detections
        MIN_HISTORY_FOR_COUNTING = 3
        if len(track.history) < MIN_HISTORY_FOR_COUNTING:
            return None
        
        # Get current and previous positions
        current_position = track.center
        previous_position = track.history[-2]  # Previous frame position
        
        # Get sides for both positions (returns 'side1' or 'side2')
        current_side = self.get_side(current_position)
        previous_side = self.get_side(previous_position)
        
        # Initialize tracking for this track if needed
        if track.track_id not in self._touch_tracking:
            self._touch_tracking[track.track_id] = {
                'initial_side': current_side,  # Store the FIRST side we saw this track on
                'last_side': current_side,
                'counted': False,  # PERMANENT flag - once True, never count again
                'direction': None,
                'count_frame': None  # Frame when counted (for debugging)
            }
            return None
        
        touch_info = self._touch_tracking[track.track_id]
        
        # CRITICAL: If this track has already been counted, NEVER count again
        # This prevents jitter-based double counting on conveyor belts
        if touch_info['counted']:
            touch_info['last_side'] = current_side
            return None
        
        # Check for crossing: moved from one side to the other
        if previous_side != current_side:
            # Object crossed the line!
            
            # Determine direction based on side transition
            # For conveyor loading (right to left movement):
            # - side2 (right) -> side1 (left) = ENTRY (loading)
            # - side1 (left) -> side2 (right) = EXIT (unloading)
            direction = None
            if previous_side == 'side2' and current_side == 'side1':
                direction = 'entry'  # Moving from right to left = LOADING = IN
            elif previous_side == 'side1' and current_side == 'side2':
                direction = 'exit'   # Moving from left to right = UNLOADING = OUT
            
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
                # Count the crossing - PERMANENTLY mark as counted
                touch_info['counted'] = True  # PERMANENT - this track will never be counted again
                touch_info['direction'] = direction
                touch_info['last_side'] = current_side
                touch_info['count_frame'] = track.frame_id
                
                if direction == 'entry':
                    self.entry_count += 1
                    print(f"[LINE_COUNTER] ✅ Track {track.track_id} ENTRY (side2→side1) | Total IN: {self.entry_count}")
                else:
                    self.exit_count += 1
                    print(f"[LINE_COUNTER] ✅ Track {track.track_id} EXIT (side1→side2) | Total OUT: {self.exit_count}")
                
                return direction
        
        # Update last side for next frame
        touch_info['last_side'] = current_side
        return None
    
    def is_track_touching(self, track: Track) -> bool:
        """Check if a track's center point is currently touching the line."""
        if self.line_start is None or self.line_vector is None:
            return False
        
        is_touching = self.is_point_on_line(track.center, threshold_pixels=self.touch_threshold_pixels)
        
        return is_touching
    
    def get_track_touch_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get touch tracking information for a track."""
        return self._touch_tracking.get(track_id)
    
    def reset(self):
        """Reset all counts and tracking data."""
        self.entry_count = 0
        self.exit_count = 0
        self.boxes_counted = 0
        self._touch_tracking.clear()
