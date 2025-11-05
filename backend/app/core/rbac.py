"""
Role-Based Access Control (RBAC) for The Reasoner AI Platform.

Implements operator/admin/auditor roles with proper permissions.
"""
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from functools import wraps
from typing import List, Callable

from app.core.security import get_current_user
from app.core.database import get_db
from app.models.auth import User, UserRole


class PermissionDenied(HTTPException):
    """Exception raised when user lacks required permissions."""
    def __init__(self, message: str = "Permission denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=message
        )


def require_role(required_roles: List[UserRole]):
    """
    Decorator to require specific roles for an endpoint.

    Usage:
        @router.get("/admin-only")
        @require_role([UserRole.ADMIN])
        async def admin_endpoint(current_user: User = Depends(get_current_user)):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from kwargs
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )

            # Check role
            if current_user.role not in required_roles:
                raise PermissionDenied(
                    f"This action requires one of: {', '.join([r.value for r in required_roles])}"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Role-specific dependencies
async def get_current_operator(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require operator role or higher."""
    if current_user.role not in [UserRole.OPERATOR, UserRole.ADMIN]:
        raise PermissionDenied("This action requires operator privileges")
    return current_user


async def get_current_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require admin role."""
    if current_user.role != UserRole.ADMIN:
        raise PermissionDenied("This action requires admin privileges")
    return current_user


async def get_current_auditor(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require auditor role or higher."""
    if current_user.role not in [UserRole.AUDITOR, UserRole.ADMIN]:
        raise PermissionDenied("This action requires auditor privileges")
    return current_user


def can_execute_formula(user: User, formula_tier: int) -> bool:
    """
    Check if user can execute a formula based on tier.

    Rules:
    - Operators can only execute Tier 1 (certified) formulas
    - Admins can execute any tier
    """
    if user.role == UserRole.ADMIN:
        return True

    if user.role == UserRole.OPERATOR:
        # Operators can only use Tier 1 certified formulas
        return formula_tier == 1

    return False


def can_certify_formula(user: User) -> bool:
    """Check if user can certify formulas (promote tiers)."""
    return user.role == UserRole.ADMIN


def can_submit_correction(user: User) -> bool:
    """Check if user can submit corrections."""
    return user.role in [UserRole.OPERATOR, UserRole.ADMIN]


def can_review_correction(user: User) -> bool:
    """Check if user can review corrections."""
    return user.role == UserRole.ADMIN


def can_view_audit_log(user: User) -> bool:
    """Check if user can view audit logs."""
    return user.role in [UserRole.AUDITOR, UserRole.ADMIN]


def can_manage_users(user: User) -> bool:
    """Check if user can manage other users."""
    return user.role == UserRole.ADMIN


class RBACChecker:
    """Helper class for checking permissions."""

    @staticmethod
    def check_formula_execution(user: User, formula_tier: int):
        """Check formula execution permission and raise exception if denied."""
        if not can_execute_formula(user, formula_tier):
            raise PermissionDenied(
                f"Operators can only execute Tier 1 (certified) formulas. "
                f"This formula is Tier {formula_tier}."
            )

    @staticmethod
    def check_formula_certification(user: User):
        """Check formula certification permission."""
        if not can_certify_formula(user):
            raise PermissionDenied("Only admins can certify formulas")

    @staticmethod
    def check_correction_submission(user: User):
        """Check correction submission permission."""
        if not can_submit_correction(user):
            raise PermissionDenied("You do not have permission to submit corrections")

    @staticmethod
    def check_correction_review(user: User):
        """Check correction review permission."""
        if not can_review_correction(user):
            raise PermissionDenied("Only admins can review corrections")

    @staticmethod
    def check_audit_log_access(user: User):
        """Check audit log access permission."""
        if not can_view_audit_log(user):
            raise PermissionDenied("You do not have permission to view audit logs")

    @staticmethod
    def check_user_management(user: User):
        """Check user management permission."""
        if not can_manage_users(user):
            raise PermissionDenied("Only admins can manage users")


# Create global instance
rbac = RBACChecker()
