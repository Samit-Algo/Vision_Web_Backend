class EventFields:
    """MongoDB field names for events collection"""

    MONGO_ID = "_id"

    OWNER_USER_ID = "owner_user_id"
    SESSION_ID = "session_id"

    LABEL = "label"
    SEVERITY = "severity"
    RULE_INDEX = "rule_index"

    CAMERA_ID = "camera_id"
    AGENT_ID = "agent_id"
    AGENT_NAME = "agent_name"
    DEVICE_ID = "device_id"

    EVENT_TS = "event_ts"
    RECEIVED_AT = "received_at"

    IMAGE_PATH = "image_path"
    JSON_PATH = "json_path"

    METADATA = "metadata"
