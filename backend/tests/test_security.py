"""
Test suite for security module to verify JWT functionality.
"""
import pytest
from datetime import timedelta
from app.core.security import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password
)
from app.core.config import settings


class TestSecurityConfiguration:
    """Test that security settings are loaded from config."""
    
    def test_secret_key_from_settings(self):
        """Verify SECRET_KEY is loaded from settings, not hardcoded."""
        # The SECRET_KEY should match the one in settings
        assert SECRET_KEY == settings.SECRET_KEY
        # And it should NOT be the old hardcoded placeholder
        assert SECRET_KEY != "your-secret-key-change-in-production-use-env-variable"
    
    def test_algorithm_from_settings(self):
        """Verify ALGORITHM is loaded from settings."""
        assert ALGORITHM == settings.ALGORITHM
    
    def test_token_expiry_from_settings(self):
        """Verify token expiry times are loaded from settings."""
        assert ACCESS_TOKEN_EXPIRE_MINUTES == settings.ACCESS_TOKEN_EXPIRE_MINUTES
        assert REFRESH_TOKEN_EXPIRE_DAYS == settings.REFRESH_TOKEN_EXPIRE_DAYS


class TestJWTTokens:
    """Test JWT token creation and validation."""
    
    def test_create_access_token(self):
        """Test access token creation with user data."""
        data = {"sub": 123, "email": "test@example.com"}
        token = create_access_token(data=data)
        
        # Token should be a non-empty string
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_access_token_with_expiry(self):
        """Test access token creation with custom expiry."""
        data = {"sub": 123}
        expires_delta = timedelta(minutes=15)
        token = create_access_token(data=data, expires_delta=expires_delta)
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_decode_access_token(self):
        """Test decoding a valid access token."""
        user_id = 123
        data = {"sub": user_id}
        token = create_access_token(data=data)
        
        # Decode the token
        payload = decode_access_token(token)
        
        # Verify the payload contains the user ID
        assert payload["sub"] == user_id
        assert payload["type"] == "access"
        assert "exp" in payload
    
    def test_decode_invalid_token(self):
        """Test decoding an invalid token raises HTTPException."""
        from fastapi import HTTPException
        
        invalid_token = "invalid.token.here"
        
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(invalid_token)
        
        assert exc_info.value.status_code == 401
    
    def test_token_consistency(self):
        """Test that tokens created with same data can be decoded."""
        data = {"sub": 456, "role": "admin"}
        token = create_access_token(data=data)
        payload = decode_access_token(token)
        
        # The decoded payload should contain our original data
        assert payload["sub"] == data["sub"]
        assert payload["role"] == data["role"]


class TestPasswordHashing:
    """Test password hashing and verification."""
    
    def test_password_hash(self):
        """Test password hashing."""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        # Hash should be different from password
        assert hashed != password
        # Hash should be non-empty
        assert len(hashed) > 0
    
    def test_password_verification(self):
        """Test password verification with correct password."""
        password = "my_secure_password"
        hashed = get_password_hash(password)
        
        # Correct password should verify
        assert verify_password(password, hashed) is True
    
    def test_password_verification_wrong_password(self):
        """Test password verification with wrong password."""
        password = "correct_password"
        wrong_password = "wrong_password"
        hashed = get_password_hash(password)
        
        # Wrong password should not verify
        assert verify_password(wrong_password, hashed) is False
    
    def test_different_passwords_different_hashes(self):
        """Test that different passwords produce different hashes."""
        password1 = "password1"
        password2 = "password2"
        
        hash1 = get_password_hash(password1)
        hash2 = get_password_hash(password2)
        
        # Different passwords should have different hashes
        assert hash1 != hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
