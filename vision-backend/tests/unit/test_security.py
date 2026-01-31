"""
Unit tests for app.core.security
"""
import pytest
from app.core.security import (
    hash_password,
    verify_password,
    create_jwt_token,
    decode_jwt_token,
)


class TestHashPassword:
    """Tests for hash_password"""

    def test_returns_non_empty_string(self):
        result = hash_password("mypassword")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_different_salts_per_call(self):
        """Each hash should use a new salt, so hashes differ."""
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2

    def test_hash_not_equal_to_plain(self):
        result = hash_password("secret123")
        assert result != "secret123"


class TestVerifyPassword:
    """Tests for verify_password"""

    def test_matching_password_returns_true(self):
        hashed = hash_password("correct")
        assert verify_password("correct", hashed) is True

    def test_wrong_password_returns_false(self):
        hashed = hash_password("correct")
        assert verify_password("wrong", hashed) is False

    def test_empty_password(self):
        hashed = hash_password("")
        assert verify_password("", hashed) is True
        assert verify_password("x", hashed) is False


class TestJwtToken:
    """Tests for create_jwt_token and decode_jwt_token"""

    def test_create_and_decode_roundtrip(self, mock_settings):
        payload = {"sub": "user-123", "email": "test@example.com"}
        token = create_jwt_token(payload)
        decoded = decode_jwt_token(token)
        assert decoded["sub"] == "user-123"
        assert decoded["email"] == "test@example.com"
        assert "iat" in decoded
        assert "exp" in decoded

    def test_decode_invalid_token_raises(self, mock_settings):
        with pytest.raises(ValueError) as exc_info:
            decode_jwt_token("invalid.jwt.token")
        assert "Invalid token" in str(exc_info.value)

    def test_decode_tampered_token_raises(self, mock_settings):
        payload = {"sub": "user-1"}
        token = create_jwt_token(payload)
        tampered = token[:-5] + "xxxxx"
        with pytest.raises(ValueError):
            decode_jwt_token(tampered)
