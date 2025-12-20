from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    """Pure domain model for User entity - no external dependencies"""
    id: Optional[str]
    full_name: str
    email: str
    hashed_password: str

    def __post_init__(self):
        """Business validations"""
        if not self.full_name or len(self.full_name.strip()) < 2:
            raise ValueError("Full name must be at least 2 characters")
        if not self.email or "@" not in self.email:
            raise ValueError("Invalid email format")
        if not self.hashed_password:
            raise ValueError("Password hash is required")

