# External package imports
from fastapi import APIRouter, Depends, HTTPException, status

# Local application imports
from ...application.dto.auth_dto import UserRegistrationRequest, UserLoginRequest, TokenResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.auth.register_user import RegisterUserUseCase
from ...application.use_cases.auth.login_user import LoginUserUseCase
from ...di.container import get_container
from .dependencies import get_current_user


router = APIRouter(tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(request: UserRegistrationRequest) -> UserResponse:
    """
    Register a new user
    
    Args:
        request: User registration request
        
    Returns:
        UserResponse with created user information
    """
    container = get_container()
    register_use_case = container.get(RegisterUserUseCase)
    
    try:
        user = await register_use_case.execute(request)
        return user
    except ValueError as exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exception)
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(request: UserLoginRequest) -> TokenResponse:
    """
    Authenticate user and get access token
    
    Args:
        request: User login request
        
    Returns:
        TokenResponse with access token
    """
    container = get_container()
    login_use_case = container.get(LoginUserUseCase)
    
    token_response = await login_use_case.execute(request)
    if token_response is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    return token_response


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """
    Get current authenticated user information
    
    Args:
        current_user: Current authenticated user (from dependency)
        
    Returns:
        UserResponse with user information
    """
    return current_user

