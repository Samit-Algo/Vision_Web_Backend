"""
Detection Builder
-----------------

Converts raw model results (e.g., YOLO Results) into standardized DetectionPacket.
This is the only place that interprets model-specific output formats.
"""

from typing import List, Tuple
from datetime import datetime

from app.processing.detections.contracts import DetectionPacket


class DetectionBuilder:
    """
    Builds DetectionPacket from model results.
    
    Handles extraction of boxes, classes, scores, and keypoints from model outputs.
    Currently supports YOLO (ultralytics) format, but can be extended for other models.
    """
    
    @staticmethod
    def from_yolo_result(result, timestamp: datetime) -> DetectionPacket:
        """
        Extract detections from a YOLO result object.
        
        Args:
            result: YOLO result object (ultralytics Results)
            timestamp: Timestamp for the detections
        
        Returns:
            DetectionPacket with extracted detections
        """
        boxes, classes, scores = DetectionBuilder._extract_detections(result)
        keypoints = DetectionBuilder._extract_keypoints(result)
        
        return DetectionPacket(
            classes=classes,
            scores=scores,
            boxes=boxes,
            keypoints=keypoints,
            ts=timestamp
        )
    
    @staticmethod
    def _extract_detections(result) -> Tuple[List[List[float]], List[str], List[float]]:
        """
        Extract boxes, classes, and scores from YOLO result.
        
        This is extracted from worker/detections.py to keep extraction logic
        in the detections module.
        
        Args:
            result: YOLO result object
        
        Returns:
            Tuple of (boxes, classes, scores)
        """
        boxes: List[List[float]] = []
        classes: List[str] = []
        scores: List[float] = []
        
        if result is None:
            return boxes, classes, scores
        
        class_names_by_id = result.names if hasattr(result, "names") else {}
        
        if hasattr(result, "boxes"):
            try:
                xyxy = result.boxes.xyxy.tolist() if hasattr(result.boxes, "xyxy") else []
                cls_list = result.boxes.cls.tolist() if hasattr(result.boxes, "cls") else []
                conf_list = result.boxes.conf.tolist() if hasattr(result.boxes, "conf") else []
                
                for detection_index, box in enumerate(xyxy):
                    boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                    class_id = int(cls_list[detection_index]) if detection_index < len(cls_list) else -1
                    classes.append(class_names_by_id.get(class_id, str(class_id)))
                    score = float(conf_list[detection_index]) if detection_index < len(conf_list) else 0.0
                    scores.append(score)
            except Exception:  # noqa: BLE001
                pass
        
        return boxes, classes, scores
    
    @staticmethod
    def _extract_keypoints(result) -> List[List[List[float]]]:
        """
        Extract keypoints from YOLO result (for pose models).
        
        This is extracted from worker/detections.py to keep extraction logic
        in the detections module.
        
        Args:
            result: YOLO result object
        
        Returns:
            List of keypoints per person
        """
        keypoints_list: List[List[List[float]]] = []
        
        if result is None:
            return keypoints_list
        
        kp = getattr(result, "keypoints", None)
        if kp is None:
            return keypoints_list
        
        try:
            if hasattr(kp, "data") and kp.data is not None:
                data = kp.data.tolist()
                for person in data:
                    person_pts: List[List[float]] = []
                    for pt in person:
                        if pt is None or len(pt) < 2:
                            continue
                        x = float(pt[0])
                        y = float(pt[1])
                        if len(pt) >= 3:
                            c = float(pt[2])
                            person_pts.append([x, y, c])
                        else:
                            person_pts.append([x, y])
                    if person_pts:
                        keypoints_list.append(person_pts)
            elif hasattr(kp, "xy") and kp.xy is not None:
                xy = kp.xy.tolist()
                for person in xy:
                    person_pts = [[float(p[0]), float(p[1])] for p in person if p is not None and len(p) >= 2]
                    if person_pts:
                        keypoints_list.append(person_pts)
        except Exception:
            pass
        
        return keypoints_list
