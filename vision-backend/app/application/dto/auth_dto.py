from pydantic import BaseModel, EmailStr, Field


class UserRegistrationRequest(BaseModel):
    """DTO for user registration request"""
    full_name: str = Field(min_length=2, max_length=200)
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)


class UserLoginRequest(BaseModel):
    """DTO for user login request"""
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)


class TokenResponse(BaseModel):
    """DTO for authentication token response"""
    access_token: str
    token_type: str = "bearer"

