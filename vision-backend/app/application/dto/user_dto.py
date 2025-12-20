from pydantic import BaseModel, EmailStr


class UserResponse(BaseModel):
    """DTO for user response (no password)"""
    id: str
    full_name: str
    email: EmailStr

