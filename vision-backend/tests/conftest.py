"""
Shared pytest fixtures for vision-backend tests.
"""
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_env():
    """Fixture to set common test environment variables."""
    env_vars = {
        "MONGO_URI": "mongodb://localhost:27017",
        "MONGO_DB_NAME": "test_vision_db",
        "JWT_SECRET_KEY": "test_secret_key_for_testing_only",
        "GROQ_API_KEY": "test_groq_key_placeholder",
        "LOCAL_TIMEZONE": "UTC",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def mock_settings():
    """Fixture to mock get_settings for tests. Patches all modules that use it."""
    mock = MagicMock()
    mock.mongo_uri = "mongodb://localhost:27017"
    mock.mongo_database_name = "test_db"
    mock.jwt_secret_key = "test_jwt_secret"
    mock.jwt_algorithm = "HS256"
    mock.access_token_expire_minutes = 1440
    mock.groq_api_key = "test_groq_key"
    mock.local_timezone = "UTC"

    # Patch at source and at use sites (modules import get_settings at load time)
    with patch("app.core.config.get_settings", return_value=mock), patch(
        "app.core.security.get_settings", return_value=mock
    ), patch("app.utils.datetime_utils.get_settings", return_value=mock):
        yield mock
