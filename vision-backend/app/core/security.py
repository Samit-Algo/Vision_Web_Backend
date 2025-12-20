# Standard library imports
import time
from typing import Any, Dict

# External package imports
import jwt
import bcrypt
from jwt.exceptions import InvalidTokenError, DecodeError

# Local application imports
from .config import get_settings


def hash_password(plain_password: str) -> str:
    """
    Hash a plain password using bcrypt
    
    Args:
        plain_password: The plain text password to hash
        
    Returns:
        Hashed password string
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password
    
    Args:
        plain_password: The plain text password to verify
        hashed_password: The hashed password to compare against
        
    Returns:
        True if passwords match, False otherwise
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def create_jwt_token(payload: Dict[str, Any]) -> str:
    """
    Create a JWT token with expiration
    
    Args:
        payload: Dictionary containing token claims (e.g., user_id, email)
        
    Returns:
        Encoded JWT token string
    """
    settings = get_settings()
    issued_at = int(time.time())
    # Fix: Convert minutes to seconds correctly (not * 90)
    expires_at = issued_at + (settings.access_token_expire_minutes * 60)
    
    token_payload = {
        **payload,
        "iat": issued_at,
        "exp": expires_at,
    }
    
    token = jwt.encode(
        token_payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    return token


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token
    
    Args:
        token: The JWT token string to decode
        
    Returns:
        Dictionary containing decoded token claims
        
    Raises:
        InvalidTokenError: If token is invalid or expired
        DecodeError: If token cannot be decoded
    """
    settings = get_settings()
    try:
        decoded = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        return decoded
    except (InvalidTokenError, DecodeError) as e:
        raise ValueError(f"Invalid token: {str(e)}")

