# Standard library imports
from typing import Optional

# Local application imports
from ....domain.repositories.user_repository import UserRepository
from ....domain.constants import UserFields
from ....core.security import decode_jwt_token
from ...dto.user_dto import UserResponse


class GetCurrentUserUseCase:
    """Use case for getting current authenticated user from JWT token"""
    
    def __init__(self, user_repository: UserRepository) -> None:
        self.user_repository = user_repository
    
    async def execute(self, token: str) -> UserResponse:
        """
        Get current user from JWT token
        
        Args:
            token: JWT access token
            
        Returns:
            UserResponse with user information
            
        Raises:
            ValueError: If token is invalid or user not found
        """
        try:
            # Decode token
            payload = decode_jwt_token(token)
        except Exception as exception:
            raise ValueError(f"Invalid or expired token: {str(exception)}")
        
        # Extract user ID
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise ValueError("Invalid authentication payload: missing user ID")
        
        # Find user
        user = await self.user_repository.find_by_id(user_id)
        if user is None:
            raise ValueError("User not found")
        
        # Return DTO
        return UserResponse(
            id=user.id or "",
            full_name=user.full_name,
            email=user.email,
        )

