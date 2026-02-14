"""
FastAPI dependencies for API v1.

Provides authentication dependency (get_current_user) used by protected endpoints.
Supports JWT from Authorization header or from query parameter.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Optional

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.auth.get_current_user import GetCurrentUserUseCase
from ...di.container import get_container

# -----------------------------------------------------------------------------
# Auth scheme
# -----------------------------------------------------------------------------
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    token: Optional[str] = Query(None, description="JWT (alternative to Authorization header)"),
) -> UserResponse:
    """
    Resolve current authenticated user from JWT.

    Token is taken from Authorization header (Bearer) or from query parameter `token`.
    """
    final_token = None
    if credentials:
        final_token = credentials.credentials
    elif token is not None:
        final_token = token

    if not final_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    container = get_container()
    use_case = container.get(GetCurrentUserUseCase)

    try:
        user = await use_case.execute(final_token)
        return user
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )
