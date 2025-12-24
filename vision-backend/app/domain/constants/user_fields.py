"""Constants for User model field names"""


class UserFields:
    """Field name constants for User model"""
    ID = "id"
    FULL_NAME = "full_name"
    EMAIL = "email"
    HASHED_PASSWORD = "hashed_password"
    
    # MongoDB specific
    MONGO_ID = "_id"  # MongoDB's internal _id field

