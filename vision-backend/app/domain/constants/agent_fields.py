"""Constants for Agent model field names"""


class AgentFields:
    """Field name constants for Agent model"""
    ID = "id"
    NAME = "name"
    CAMERA_ID = "camera_id"
    MODEL = "model"
    FPS = "fps"
    RULES = "rules"
    RUN_MODE = "run_mode"
    INTERVAL_MINUTES = "interval_minutes"
    CHECK_DURATION_SECONDS = "check_duration_seconds"
    START_TIME = "start_time"
    END_TIME = "end_time"
    ZONE = "zone"
    REQUIRES_ZONE = "requires_zone"
    STATUS = "status"
    CREATED_AT = "created_at"
    OWNER_USER_ID = "owner_user_id"
    STREAM_CONFIG = "stream_config"
    
    # MongoDB specific
    MONGO_ID = "_id"  # MongoDB's internal _id field

