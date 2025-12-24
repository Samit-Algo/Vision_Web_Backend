"""Constants for Camera model field names"""


class CameraFields:
    """Field name constants for Camera model"""
    ID = "id"
    OWNER_USER_ID = "owner_user_id"
    NAME = "name"
    STREAM_URL = "stream_url"
    DEVICE_ID = "device_id"
    STREAM_CONFIG = "stream_config"
    WEBRTC_CONFIG = "webrtc_config"
    
    # MongoDB specific
    MONGO_ID = "_id"  # MongoDB's internal _id field

