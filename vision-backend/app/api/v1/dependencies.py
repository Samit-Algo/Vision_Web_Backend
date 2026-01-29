from typing import Optional
from fastapi import Depends, HTTPException, status, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Local application imports
from ...application.use_cases.auth.get_current_user import GetCurrentUserUseCase
from ...application.dto.user_dto import UserResponse
from ...di.container import get_container



security_scheme = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    token: Optional[str] = Query(None)
) -> UserResponse:
    """
    FastAPI dependency to get current authenticated user from JWT token.
    Supports both Authorization header and 'token' query parameter.
    """
    final_token = None
    if credentials:
        final_token = credentials.credentials
    elif token:
        final_token = token
        
    if not final_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    container = get_container()
    get_current_user_use_case = container.get(GetCurrentUserUseCase)
    
    try:
        user = await get_current_user_use_case.execute(final_token)
        return user
    except ValueError as exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exception)
        )

