# Standard library imports
from typing import Optional

# Local application imports
from ....domain.repositories.user_repository import UserRepository
from ....domain.models.user import User
from ....core.security import hash_password
from ...dto.auth_dto import UserRegistrationRequest
from ...dto.user_dto import UserResponse


class RegisterUserUseCase:
    """Use case for registering a new user"""
    
    def __init__(self, user_repository: UserRepository) -> None:
        self.user_repository = user_repository
    
    async def execute(self, request: UserRegistrationRequest) -> UserResponse:
        """
        Register a new user
        
        Args:
            request: Registration request with user details
            
        Returns:
            UserResponse with created user information
            
        Raises:
            ValueError: If user with email already exists
        """
        # Check if user already exists
        existing_user = await self.user_repository.find_by_email(request.email)
        if existing_user is not None:
            raise ValueError("User with this email already exists")
        
        # Hash password
        hashed_password = hash_password(request.password)
        
        # Create domain user entity
        new_user = User(
            id=None,  # Will be set by repository
            full_name=request.full_name,
            email=request.email,
            hashed_password=hashed_password,
        )
        
        # Save user
        saved_user = await self.user_repository.save(new_user)
        
        # Return DTO
        return UserResponse(
            id=saved_user.id or "",
            full_name=saved_user.full_name,
            email=saved_user.email,
        )

