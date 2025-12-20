from abc import ABC, abstractmethod
from typing import Optional
from ..models.user import User


class UserRepository(ABC):
    """Repository interface - defines contract for user data access"""
    
    @abstractmethod
    async def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email address"""
        pass
    
    @abstractmethod
    async def find_by_id(self, user_id: str) -> Optional[User]:
        """Find user by ID"""
        pass
    
    @abstractmethod
    async def save(self, user: User) -> User:
        """Save user (create or update)"""
        pass

