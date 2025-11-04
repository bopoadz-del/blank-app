"""
Admin API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.security import get_current_admin_user
from app.models.auth import User
from app.models.chat import Project, Conversation, Message
from app.schemas.auth_schemas import UserResponse
from pydantic import BaseModel

router = APIRouter()


class SystemMetricsResponse(BaseModel):
    totalUsers: int
    totalProjects: int
    totalConversations: int
    totalMessages: int
    activeUsers24h: int


class UserUpdateRequest(BaseModel):
    role: str


@router.get("/users", response_model=List[UserResponse])
async def get_all_users(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get all users (admin only)."""
    users = db.query(User).all()

    return [
        UserResponse(
            id=str(u.id),
            email=u.email,
            username=u.username,
            role=u.role,
            createdAt=u.created_at
        ) for u in users
    ]


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user_role(
    user_id: int,
    update_data: UserUpdateRequest,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Update user role (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if update_data.role not in ["user", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role"
        )

    user.role = update_data.role
    db.commit()
    db.refresh(user)

    return UserResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        role=user.role,
        createdAt=user.created_at
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Delete a user (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Prevent admin from deleting themselves
    if user.id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    db.delete(user)
    db.commit()

    return {"message": "User deleted successfully"}


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get system metrics (admin only)."""
    total_users = db.query(User).count()
    total_projects = db.query(Project).count()
    total_conversations = db.query(Conversation).count()
    total_messages = db.query(Message).count()

    # Active users in last 24 hours (users with messages in last 24h)
    yesterday = datetime.utcnow() - timedelta(days=1)
    active_users = db.query(Message.conversation_id).filter(
        Message.created_at >= yesterday
    ).distinct().count()

    return SystemMetricsResponse(
        totalUsers=total_users,
        totalProjects=total_projects,
        totalConversations=total_conversations,
        totalMessages=total_messages,
        activeUsers24h=active_users
    )
