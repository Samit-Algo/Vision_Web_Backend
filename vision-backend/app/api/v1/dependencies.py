# External package imports
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Local application imports
from ...application.use_cases.auth.get_current_user import GetCurrentUserUseCase
from ...application.dto.user_dto import UserResponse
from ...di.container import get_container


security_scheme = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> UserResponse:
    """
    FastAPI dependency to get current authenticated user from JWT token
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        UserResponse with user information
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    token: str = credentials.credentials
    
    container = get_container()
    get_current_user_use_case = container.get(GetCurrentUserUseCase)
    
    try:
        user = await get_current_user_use_case.execute(token)
        return user
    except ValueError as exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exception)
        )

