"""
Event Notifier
==============

Utility to send event notifications with annotated frames.
- Saves events to MongoDB (for API queries)
- Publishes event notifications to Kafka (for real-time via FastAPI consumer)

Dual write approach: DB + Kafka.
"""
import base64
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaProducer = None  # type: ignore
    KafkaError = Exception  # type: ignore

from app.utils.datetime_utils import now, utc_now, ensure_utc, now_iso, parse_iso
from app.utils.db import get_collection
from app.domain.constants.event_fields import EventFields
from app.core.config import get_settings

# Base directory for storing event images (same pattern as event_storage.py)
EVENTS_BASE_DIR = Path("events")

# Singleton Kafka producer instance
_kafka_producer: Optional[Any] = None


def get_kafka_producer() -> Optional[Any]:
    """
    Get or create Kafka producer (singleton pattern).
    Tests connection to verify Kafka is running.
    
    Returns:
        KafkaProducer instance if available, None otherwise
    """
    global _kafka_producer
    
    if not KAFKA_AVAILABLE:
        print("[event_notifier] ❌ WARNING: kafka-python not available. Install with: pip install kafka-python")
        return None
    
    if _kafka_producer is None:
        try:
            settings = get_settings()
            print(f"[event_notifier] 🔌 Attempting to connect to Kafka: {settings.kafka_bootstrap_servers}")
            
            _kafka_producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # Reliability settings
                retries=3,
                acks='all',  # Wait for all replicas to acknowledge
                max_in_flight_requests_per_connection=1,  # Ensure ordering
                # Compression to reduce message size (important for base64 images)
                compression_type='gzip',
                # Message size settings
                max_request_size=10485760,  # 10MB
                # Timeout settings
                request_timeout_ms=30000,
                delivery_timeout_ms=120000,
            )
            
            # Test connection
            print("[event_notifier] 🔍 Testing Kafka connection...")
            try:
                metadata = _kafka_producer.list_topics(timeout=5)
                print(f"[event_notifier] ✅ [SUCCESS] Kafka producer initialized and connected: {settings.kafka_bootstrap_servers}")
                print(f"[event_notifier] 📡 Topic: {settings.kafka_topic}")
            except Exception as e:
                print(f"[event_notifier] ⚠️  [WARNING] Kafka producer created but connection test failed: {e}")
                print(f"[event_notifier] ⚠️  Kafka may not be running or unreachable")
        except Exception as e:
            print(f"[event_notifier] ❌ [ERROR] Failed to initialize Kafka producer: {e}")
            import traceback
            print(f"[event_notifier] Traceback: {traceback.format_exc()}")
            return None
    
    return _kafka_producer


def _get_event_storage_path(camera_id: str, agent_id: str) -> Path:
    """
    Get the storage path for events for a specific camera and agent.
    
    Structure: events/{camera_id}/{agent_id}/
    """
    storage_path = EVENTS_BASE_DIR / camera_id / agent_id
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path


def _save_frame_to_file(frame: np.ndarray, camera_id: str, agent_id: str) -> Optional[str]:
    """
    Save annotated frame to file.
    
    Args:
        frame: OpenCV frame (numpy array in BGR format)
        camera_id: Camera identifier
        agent_id: Agent identifier
    
    Returns:
        Path to saved image file, or None if saving fails
    """
    if cv2 is None:
        return None
    
    try:
        storage_path = _get_event_storage_path(camera_id, agent_id)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        image_path = storage_path / f"{timestamp}.jpg"
        
        # Save frame as JPEG
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return str(image_path)
    except Exception as e:
        print(f"[event_notifier] ⚠️  Error saving frame to file: {e}")
        return None


def encode_frame_to_base64(frame: np.ndarray) -> Optional[str]:
    """
    Encode a frame (numpy array) to base64 JPEG string.
    
    Args:
        frame: OpenCV frame (numpy array in BGR format)
    
    Returns:
        Base64-encoded JPEG string, or None if encoding fails
    """
    if cv2 is None:
        return None
    
    try:
        # Encode frame as JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return None
        
        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64
    except Exception as exc:
        print(f"[event_notifier] ⚠️  Error encoding frame: {exc}")
        return None


def _serialize_for_json(obj: Any) -> Any:
    """
    Recursively serialize objects for JSON encoding.
    Converts datetime objects to ISO format strings, numpy types to Python types.
    """
    if isinstance(obj, datetime):
        return obj.isoformat() + "Z" if obj.tzinfo else obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _infer_severity(label: str) -> str:
    """
    Infer event severity based on label.
    Matches vision-backend's severity inference logic.
    """
    label_lower = str(label or "").lower()
    if any(k in label_lower for k in ["weapon", "fire", "fall", "intrusion"]):
        return "critical"
    elif any(k in label_lower for k in ["violation", "restricted", "collision", "alert"]):
        return "warning"
    else:
        return "info"


def send_event_to_backend_sync(
    event: Dict[str, Any],
    annotated_frame: np.ndarray,
    agent_id: str,
    agent_name: str,
    camera_id: Optional[str] = None,
    video_timestamp: Optional[str] = None,
    detections: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> bool:
    """
    Send event notification - DB write + Kafka publish.
    
    This function:
    1. Saves event to MongoDB (for API queries)
    2. Publishes event to Kafka (FastAPI consumes and broadcasts to WebSocket)
    
    Args:
        event: Event dict with 'label' and optionally 'rule_index'
        annotated_frame: Frame with bounding boxes drawn (numpy array)
        agent_id: Agent identifier
        agent_name: Agent name
        camera_id: Camera identifier (optional)
        video_timestamp: Video timestamp string (optional)
        detections: Detection details (optional)
        session_id: Session identifier (optional)
    
    Returns:
        True if saved successfully, False otherwise
    """
    event_id = None
    db_success = False
    kafka_success = False
    
    try:
        # Get events collection (PyMongo sync)
        events_collection = get_collection("events")
        
        # Fetch camera info to get owner_user_id and device_id
        owner_user_id = None
        device_id = None
        
        if camera_id:
            try:
                cameras_collection = get_collection("cameras")
                camera_doc = cameras_collection.find_one({"id": camera_id})
                if camera_doc:
                    owner_user_id = camera_doc.get("owner_user_id")
                    device_id = camera_doc.get("device_id")
            except Exception as e:
                print(f"[event_notifier] ⚠️  Error fetching camera details: {e}")
        
        # Parse event timestamp (may be local tz-aware depending on input)
        event_ts = None
        if video_timestamp:
            event_ts = parse_iso(video_timestamp)

        # Use current time if no timestamp provided
        if event_ts is None:
            event_ts = now()

        # Persist event_ts/received_at as UTC BSON datetime for consistency
        event_ts = ensure_utc(event_ts)
        
        # Save frame to file
        image_path = None
        if camera_id and agent_id:
            image_path = _save_frame_to_file(annotated_frame, camera_id, agent_id)
        
        # Infer severity
        label = event.get("label", "Unknown")
        severity = _infer_severity(label)
        
        # Extract report from detections if present (class_count rule includes report in detections)
        report = None
        if detections and isinstance(detections, dict) and "rule_report" in detections:
            report = detections.get("rule_report")
        
        # Build event document (matches Event model structure)
        metadata = {
            "video_timestamp": video_timestamp,
            "detections": _serialize_for_json(detections) if detections else None,
            "session_id": session_id,
        }
        
        # Include report in metadata if present (for class_count and similar reporting rules)
        if report:
            metadata["report"] = _serialize_for_json(report)
        
        event_doc = {
            EventFields.OWNER_USER_ID: owner_user_id,
            EventFields.SESSION_ID: session_id or "",
            EventFields.LABEL: label,
            EventFields.SEVERITY: severity,
            EventFields.RULE_INDEX: event.get("rule_index"),
            EventFields.CAMERA_ID: camera_id,
            EventFields.AGENT_ID: agent_id,
            EventFields.AGENT_NAME: agent_name,
            EventFields.DEVICE_ID: device_id,
            EventFields.EVENT_TS: event_ts,
            EventFields.RECEIVED_AT: utc_now(),
            EventFields.IMAGE_PATH: image_path,
            EventFields.JSON_PATH: None,  # Not used
            EventFields.METADATA: metadata,
        }
        
        # 1. Insert event to MongoDB
        try:
            result = events_collection.insert_one(event_doc)
            event_id = str(result.inserted_id)
            db_success = True
            print(
                f"[event_notifier] ✅ Event saved to MongoDB: "
                f"label='{label}' | agent={agent_id} | session_id={session_id} | event_id={event_id}"
            )
        except Exception as e:
            print(f"[event_notifier] ❌ Error saving event to MongoDB: {e}")
            db_success = False
        
        # 2. Publish to Kafka (FastAPI consumes and broadcasts to WebSocket)
        producer = get_kafka_producer()
        if producer:
            try:
                # Encode frame to base64 for Kafka
                frame_base64 = encode_frame_to_base64(annotated_frame)
                
                # Build Kafka payload (same structure as original)
                payload = {
                    "event": {
                        "label": label,
                        "rule_index": event.get("rule_index"),
                        "timestamp": now_iso(),
                    },
                    "agent": {
                        "agent_id": agent_id,
                        "agent_name": agent_name,
                        "camera_id": camera_id,
                    },
                    "camera": {
                        "owner_user_id": owner_user_id,
                        "device_id": device_id,
                    },
                    "frame": {
                        "image_base64": frame_base64,
                        "format": "jpeg",
                    },
                    "metadata": {
                        "video_timestamp": video_timestamp,
                        "detections": _serialize_for_json(detections) if detections else None,
                        "session_id": session_id,
                        "event_id": event_id,  # Include MongoDB event ID
                        "image_path": image_path,  # Include image path for frontend to fetch
                    }
                }
                
                settings = get_settings()
                future = producer.send(
                    settings.kafka_topic,
                    value=payload,
                    key=agent_id.encode('utf-8') if agent_id else None
                )
                
                # Wait for send confirmation
                record_metadata = future.get(timeout=10)
                kafka_success = True
                print(
                    f"[event_notifier] ✅ Event sent to Kafka: "
                    f"label='{label}' | agent={agent_id} | topic={record_metadata.topic} | "
                    f"partition={record_metadata.partition} | offset={record_metadata.offset}"
                )
            except KafkaError as e:
                print(f"[event_notifier] ❌ Kafka error sending event: {e}")
                kafka_success = False
            except Exception as e:
                print(f"[event_notifier] ❌ Error sending event to Kafka: {e}")
                kafka_success = False
        else:
            # Kafka is optional, don't log as warning
            kafka_success = False
        
        # Return True if at least DB write succeeded
        return db_success
        
    except Exception as e:
        print(f"[event_notifier] ❌ Error in send_event_to_backend_sync: {e}")
        import traceback
        print(f"[event_notifier] Traceback: {traceback.format_exc()}")
        return False
