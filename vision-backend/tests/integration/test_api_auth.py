"""
Integration tests for auth API endpoints.
Uses TestClient with mocked use cases (no real DB).
Note: Runs full app lifespan (slower). Use: pytest tests/unit/ for fast unit-only runs.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.integration
from fastapi.testclient import TestClient

from app.application.dto.auth_dto import TokenResponse
from app.application.dto.user_dto import UserResponse
from app.application.use_cases.auth.login_user import LoginUserUseCase
from app.application.use_cases.auth.register_user import RegisterUserUseCase


@pytest.fixture
def mock_register_use_case():
    uc = AsyncMock(spec=RegisterUserUseCase)
    return uc


@pytest.fixture
def mock_login_use_case():
    uc = AsyncMock(spec=LoginUserUseCase)
    return uc


@pytest.fixture
def mock_container(mock_register_use_case, mock_login_use_case):
    container = MagicMock()
    container.get.side_effect = lambda cls: {
        RegisterUserUseCase: mock_register_use_case,
        LoginUserUseCase: mock_login_use_case,
    }.get(cls, None)
    return container


@pytest.fixture
def client(mock_container):
    """Create test client with mocked container."""
    from unittest.mock import patch
    from app.main import app

    with patch("app.api.v1.auth_controller.get_container", return_value=mock_container):
        with TestClient(app) as c:
            yield c


class TestAuthAPI:
    """Tests for /api/v1/auth endpoints"""

    def test_register_success(self, client, mock_register_use_case):
        mock_register_use_case.execute.return_value = UserResponse(
            id="usr-1",
            full_name="Test User",
            email="test@example.com",
        )
        response = client.post(
            "/api/v1/auth/register",
            json={
                "full_name": "Test User",
                "email": "test@example.com",
                "password": "password123",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["full_name"] == "Test User"

    def test_register_duplicate_returns_400(self, client, mock_register_use_case):
        mock_register_use_case.execute.side_effect = ValueError(
            "User with this email already exists"
        )
        response = client.post(
            "/api/v1/auth/register",
            json={
                "full_name": "Test",
                "email": "existing@example.com",
                "password": "password123",
            },
        )
        assert response.status_code == 400

    def test_login_success(self, client, mock_login_use_case):
        mock_login_use_case.execute.return_value = TokenResponse(
            access_token="jwt.token.here"
        )
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "validpass123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["access_token"] == "jwt.token.here"

    def test_login_invalid_returns_401(self, client, mock_login_use_case):
        mock_login_use_case.execute.return_value = None
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "wrongpass123"},
        )
        assert response.status_code == 401
