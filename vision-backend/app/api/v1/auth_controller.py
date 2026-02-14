"""
Authentication API: register, login, and current user (JWT).
"""

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from fastapi import APIRouter, Depends, HTTPException, status

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...application.dto.auth_dto import TokenResponse, UserLoginRequest, UserRegistrationRequest
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.auth.login_user import LoginUserUseCase
from ...application.use_cases.auth.register_user import RegisterUserUseCase
from ...di.container import get_container

from .dependencies import get_current_user

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
router = APIRouter(tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(request: UserRegistrationRequest) -> UserResponse:
    """Register a new user. Returns created user info."""
    container = get_container()
    use_case = container.get(RegisterUserUseCase)
    try:
        user = await use_case.execute(request)
        return user
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/login", response_model=TokenResponse)
async def login_user(request: UserLoginRequest) -> TokenResponse:
    """Authenticate user and return access token."""
    container = get_container()
    use_case = container.get(LoginUserUseCase)
    token_response = await use_case.execute(request)
    if token_response is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    return token_response


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Return current authenticated user."""
    return current_user
