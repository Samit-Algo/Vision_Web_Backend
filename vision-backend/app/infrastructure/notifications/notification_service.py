"""Notification Service for formatting and preparing event notifications"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class NotificationService:
    """
    Service for formatting events into notification payloads.
    
    Converts event payloads from Kafka into structured notification messages
    that are sent to web clients via WebSocket.
    """
    
    @staticmethod
    def format_event_notification(
        event_payload: Dict[str, Any],
        saved_paths: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Format an event payload into a notification message.
        
        Args:
            event_payload: Original event payload from Kafka
            saved_paths: Optional dictionary with saved file paths (json_path, image_path)
            
        Returns:
            Formatted notification dictionary ready for WebSocket transmission
        """
        try:
            # Extract event information
            event_info = event_payload.get("event", {})
            agent_info = event_payload.get("agent", {})
            camera_info = event_payload.get("camera", {})
            frame_info = event_payload.get("frame", {})
            metadata_info = event_payload.get("metadata", {})
            
            # Extract or construct session_id
            # Session ID format: "{agent_id}_{rule_index}_{timestamp}"
            session_id = metadata_info.get("session_id")
            if not session_id:
                # Construct session_id from available information
                agent_id = agent_info.get("agent_id", "")
                rule_index = event_info.get("rule_index", 0)
                event_timestamp = event_info.get("timestamp", "")
                # Extract timestamp part (remove timezone and special chars for filename-safe format)
                if event_timestamp:
                    try:
                        # Parse ISO timestamp and format as Unix timestamp or use original
                        dt_obj = datetime.fromisoformat(event_timestamp.replace("Z", "+00:00"))
                        timestamp_str = str(int(dt_obj.timestamp()))
                    except:
                        # Fallback: use sanitized timestamp string
                        timestamp_str = event_timestamp.replace(":", "").replace("-", "").replace("T", "_").split(".")[0]
                else:
                    timestamp_str = str(int(datetime.utcnow().timestamp()))
                
                session_id = f"{agent_id}_{rule_index}_{timestamp_str}"
            
            # Build notification payload
            notification = {
                "type": "event_notification",
                "session_id": session_id,  # Include session_id for frontend to fetch video chunks
                "event": {
                    "label": event_info.get("label", "Unknown event"),
                    "timestamp": event_info.get("timestamp"),
                    "rule_index": event_info.get("rule_index"),
                },
                "agent": {
                    "agent_id": agent_info.get("agent_id"),
                    "agent_name": agent_info.get("agent_name"),
                    "camera_id": agent_info.get("camera_id"),
                },
                "camera": {
                    "owner_user_id": camera_info.get("owner_user_id"),
                    "device_id": camera_info.get("device_id"),
                },
                "frame": {
                    "image_base64": frame_info.get("image_base64"),
                    "format": frame_info.get("format", "jpeg"),
                },
                "metadata": {
                    **metadata_info,  # Include all metadata fields
                    "session_id": session_id,  # Ensure session_id is in metadata too
                },
                "received_at": datetime.utcnow().isoformat() + "Z",
            }
            
            # Add saved file paths if provided
            if saved_paths:
                if "image_path" in saved_paths:
                    notification["metadata"]["image_path"] = saved_paths["image_path"]
                if "json_path" in saved_paths:
                    notification["metadata"]["json_path"] = saved_paths["json_path"]
            
            return notification
            
        except Exception as e:
            logger.error(f"Error formatting event notification: {e}", exc_info=True)
            # Return minimal notification on error
            return {
                "type": "event_notification",
                "error": "Failed to format notification",
                "received_at": datetime.utcnow().isoformat() + "Z",
            }
    
    @staticmethod
    def extract_owner_user_id(event_payload: Dict[str, Any]) -> Optional[str]:
        """
        Extract the owner user ID from an event payload.
        
        Based on Jetson backend payload structure, owner_user_id is in camera section:
        payload["camera"]["owner_user_id"]
        
        Also checks other locations for backward compatibility.
        
        Args:
            event_payload: Event payload from Kafka
            
        Returns:
            User ID if found, None otherwise
        """
        try:
            # Primary location: camera section (as per updated Jetson backend)
            camera_info = event_payload.get("camera", {})
            if camera_info and "owner_user_id" in camera_info:
                owner_user_id = camera_info.get("owner_user_id")
                if owner_user_id:  # Check it's not None or empty
                    logger.debug(f"Found owner_user_id in camera section: {owner_user_id}")
                    return owner_user_id
            
            # Fallback: Try agent info (backward compatibility)
            agent_info = event_payload.get("agent", {})
            if agent_info and "owner_user_id" in agent_info:
                owner_user_id = agent_info.get("owner_user_id")
                if owner_user_id:
                    logger.debug(f"Found owner_user_id in agent section: {owner_user_id}")
                    return owner_user_id
            
            # Fallback: Try event info (backward compatibility)
            event_info = event_payload.get("event", {})
            if event_info and "owner_user_id" in event_info:
                owner_user_id = event_info.get("owner_user_id")
                if owner_user_id:
                    logger.debug(f"Found owner_user_id in event section: {owner_user_id}")
                    return owner_user_id
            
            # Fallback: Try top-level (backward compatibility)
            if "owner_user_id" in event_payload:
                owner_user_id = event_payload.get("owner_user_id")
                if owner_user_id:
                    logger.debug(f"Found owner_user_id at top level: {owner_user_id}")
                    return owner_user_id
            
            # Log payload structure for debugging
            logger.warning(
                f"Could not extract owner_user_id from event payload. "
                f"Payload keys: {list(event_payload.keys())}, "
                f"camera keys: {list(camera_info.keys()) if camera_info else 'N/A'}"
            )
            return None
            
        except Exception as e:
            logger.error(f"Error extracting owner_user_id: {e}", exc_info=True)
            return None
    
    @staticmethod
    def extract_device_id(event_payload: Dict[str, Any]) -> Optional[str]:
        """
        Extract the device ID from an event payload.
        
        Based on Jetson backend payload structure, device_id is in camera section:
        payload["camera"]["device_id"]
        
        Args:
            event_payload: Event payload from Kafka
            
        Returns:
            Device ID if found, None otherwise
        """
        try:
            # Primary location: camera section (as per updated Jetson backend)
            camera_info = event_payload.get("camera", {})
            if camera_info and "device_id" in camera_info:
                device_id = camera_info.get("device_id")
                if device_id:  # Check it's not None or empty
                    return device_id
            
            logger.debug(f"Could not extract device_id from event payload")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting device_id: {e}", exc_info=True)
            return None

