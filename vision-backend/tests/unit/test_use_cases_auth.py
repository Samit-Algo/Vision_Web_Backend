"""
Unit tests for auth use cases (Login, Register, GetCurrentUser).
"""
from unittest.mock import AsyncMock

import pytest
from app.core.security import hash_password
from app.application.use_cases.auth.login_user import LoginUserUseCase
from app.application.use_cases.auth.register_user import RegisterUserUseCase
from app.application.use_cases.auth.get_current_user import GetCurrentUserUseCase
from app.application.dto.auth_dto import UserLoginRequest, UserRegistrationRequest, TokenResponse
from app.domain.models.user import User


@pytest.fixture
def mock_user_repo():
    """Mock UserRepository with async methods."""
    repo = AsyncMock()
    return repo


class TestLoginUserUseCase:
    """Tests for LoginUserUseCase"""

    @pytest.mark.asyncio
    async def test_login_success(self, mock_user_repo, mock_settings):
        user = User(
            id="usr-123",
            full_name="Test User",
            email="test@example.com",
            hashed_password=hash_password("validpass123"),
        )
        mock_user_repo.find_by_email.return_value = user

        use_case = LoginUserUseCase(mock_user_repo)
        result = await use_case.execute(
            UserLoginRequest(email="test@example.com", password="validpass123")
        )
        assert result is not None
        assert isinstance(result, TokenResponse)
        assert hasattr(result, "access_token")
        assert len(result.access_token) > 0

    @pytest.mark.asyncio
    async def test_login_user_not_found(self, mock_user_repo):
        mock_user_repo.find_by_email.return_value = None
        use_case = LoginUserUseCase(mock_user_repo)
        result = await use_case.execute(
            UserLoginRequest(email="unknown@example.com", password="anypass123")
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, mock_user_repo, mock_settings):
        from app.core.security import hash_password

        user = User(
            id="usr-1",
            full_name="Test",
            email="test@example.com",
            hashed_password=hash_password("correctpass"),
        )
        mock_user_repo.find_by_email.return_value = user

        use_case = LoginUserUseCase(mock_user_repo)
        result = await use_case.execute(
            UserLoginRequest(email="test@example.com", password="wrongpassword")
        )
        assert result is None


class TestRegisterUserUseCase:
    """Tests for RegisterUserUseCase"""

    @pytest.mark.asyncio
    async def test_register_success(self, mock_user_repo):
        mock_user_repo.find_by_email.return_value = None
        saved_user = User(
            id="usr-new",
            full_name="New User",
            email="new@example.com",
            hashed_password="$2b$12$hashed",
        )
        mock_user_repo.save.return_value = saved_user

        use_case = RegisterUserUseCase(mock_user_repo)
        result = await use_case.execute(
            UserRegistrationRequest(
                full_name="New User",
                email="new@example.com",
                password="password123",
            )
        )
        assert result.id == "usr-new"
        assert result.full_name == "New User"
        assert result.email == "new@example.com"
        mock_user_repo.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_duplicate_email_raises(self, mock_user_repo):
        existing = User(
            id="usr-1",
            full_name="Existing",
            email="existing@example.com",
            hashed_password="hash",
        )
        mock_user_repo.find_by_email.return_value = existing

        use_case = RegisterUserUseCase(mock_user_repo)
        with pytest.raises(ValueError, match="already exists"):
            await use_case.execute(
                UserRegistrationRequest(
                    full_name="Duplicate",
                    email="existing@example.com",
                    password="password123",
                )
            )


class TestGetCurrentUserUseCase:
    """Tests for GetCurrentUserUseCase"""

    @pytest.mark.asyncio
    async def test_get_current_user_success(self, mock_user_repo, mock_settings):
        from app.core.security import create_jwt_token

        token = create_jwt_token({"sub": "usr-123", "email": "user@example.com"})
        user = User(
            id="usr-123",
            full_name="Current User",
            email="user@example.com",
            hashed_password="hash",
        )
        mock_user_repo.find_by_id.return_value = user

        use_case = GetCurrentUserUseCase(mock_user_repo)
        result = await use_case.execute(token)
        assert result.id == "usr-123"
        assert result.email == "user@example.com"

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token_raises(self, mock_user_repo, mock_settings):
        use_case = GetCurrentUserUseCase(mock_user_repo)
        with pytest.raises(ValueError, match="Invalid"):
            await use_case.execute("invalid.jwt.token")

    @pytest.mark.asyncio
    async def test_get_current_user_not_found_raises(self, mock_user_repo, mock_settings):
        from app.core.security import create_jwt_token

        token = create_jwt_token({"sub": "nonexistent", "email": "x@x.com"})
        mock_user_repo.find_by_id.return_value = None

        use_case = GetCurrentUserUseCase(mock_user_repo)
        with pytest.raises(ValueError, match="User not found"):
            await use_case.execute(token)
