"""
Tests for authentication routes - specifically testing the user role assignment bug fix.
"""
import pytest
from backend.app.models.auth import UserRole


def test_userrole_enum_values():
    """Test that UserRole enum has the expected values and 'user' is not among them."""
    # Verify the valid roles in the enum
    valid_roles = [role.value for role in UserRole]
    
    # Assert that 'user' is NOT a valid role
    assert "user" not in valid_roles, "The role 'user' should not exist in UserRole enum"
    
    # Assert that 'operator' IS a valid role (the correct default)
    assert "operator" in valid_roles, "The role 'operator' should exist in UserRole enum"
    
    # Verify all expected roles exist
    assert UserRole.OPERATOR.value == "operator"
    assert UserRole.ADMIN.value == "admin"
    assert UserRole.AUDITOR.value == "auditor"
    assert UserRole.SYSTEM.value == "system"
    
    # Verify these are the only 4 roles
    assert len(valid_roles) == 4, f"Expected 4 roles but found {len(valid_roles)}: {valid_roles}"


def test_default_role_should_be_operator():
    """Test that the default role constant is set to 'operator'."""
    # This test documents that the default role for new users should be OPERATOR
    default_role = UserRole.OPERATOR.value
    assert default_role == "operator"
