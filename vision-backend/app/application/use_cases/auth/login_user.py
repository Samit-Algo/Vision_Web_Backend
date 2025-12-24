# Standard library imports
from typing import Optional

# Local application imports
from ....domain.repositories.user_repository import UserRepository
from ....domain.constants import UserFields
from ....core.security import verify_password, create_jwt_token
from ...dto.auth_dto import UserLoginRequest, TokenResponse


class LoginUserUseCase:
    """Use case for authenticating a user and generating JWT token"""
    
    def __init__(self, user_repository: UserRepository) -> None:
        self.user_repository = user_repository
    
    async def execute(self, request: UserLoginRequest) -> Optional[TokenResponse]:
        """
        Authenticate user and generate access token
        
        Args:
            request: Login request with email and password
            
        Returns:
            TokenResponse if authentication successful, None otherwise
        """
        # Find user by email
        user = await self.user_repository.find_by_email(request.email)
        if user is None:
            return None
        
        # Verify password
        if not verify_password(request.password, user.hashed_password):
            return None
        
        # Generate JWT token
        token = create_jwt_token({
            "sub": user.id or "",  # JWT standard claim (subject)
            UserFields.EMAIL: user.email,
        })
        
        return TokenResponse(access_token=token)

