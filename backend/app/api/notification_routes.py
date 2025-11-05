"""
Notification API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.auth import User
from app.models.notifications import Notification, NotificationPreferences, NotificationType, NotificationChannel
from app.services.notification_service import notification_service

router = APIRouter()


class NotificationResponse(BaseModel):
    id: str
    type: str
    channel: str
    title: str
    message: str
    read: bool
    createdAt: str
    metadata: dict

    class Config:
        from_attributes = True


class NotificationPreferencesResponse(BaseModel):
    emailEnabled: bool
    pushEnabled: bool
    slackEnabled: bool
    inAppEnabled: bool

    class Config:
        from_attributes = True


class NotificationPreferencesUpdate(BaseModel):
    emailEnabled: Optional[bool] = None
    pushEnabled: Optional[bool] = None
    slackEnabled: Optional[bool] = None
    inAppEnabled: Optional[bool] = None


class UnreadCountResponse(BaseModel):
    count: int


@router.get("/notifications", response_model=List[NotificationResponse])
async def get_notifications(
    read: Optional[bool] = None,
    notification_type: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get notifications for the current user."""
    # Convert string type to enum if provided
    type_filter = None
    if notification_type:
        try:
            type_filter = NotificationType[notification_type.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid notification type: {notification_type}"
            )

    notifications = notification_service.get_user_notifications(
        db=db,
        user_id=current_user.id,
        read=read,
        notification_type=type_filter,
        limit=limit
    )

    return [
        NotificationResponse(
            id=str(n.id),
            type=n.type.value,
            channel=n.channel.value,
            title=n.title,
            message=n.message,
            read=n.read,
            createdAt=n.created_at.isoformat(),
            metadata=n.metadata or {}
        ) for n in notifications
    ]


@router.get("/notifications/unread-count", response_model=UnreadCountResponse)
async def get_unread_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get unread notification count."""
    count = notification_service.get_unread_count(db, current_user.id)
    return UnreadCountResponse(count=count)


@router.patch("/notifications/{notification_id}/read")
async def mark_notification_as_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark a notification as read."""
    success = notification_service.mark_as_read(db, notification_id, current_user.id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )

    return {"message": "Notification marked as read"}


@router.post("/notifications/read-all")
async def mark_all_as_read(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark all notifications as read."""
    count = notification_service.mark_all_as_read(db, current_user.id)
    return {"message": f"Marked {count} notifications as read", "count": count}


@router.delete("/notifications/{notification_id}")
async def delete_notification(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a notification."""
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    ).first()

    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )

    db.delete(notification)
    db.commit()

    return {"message": "Notification deleted successfully"}


@router.get("/notifications/preferences", response_model=NotificationPreferencesResponse)
async def get_notification_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get notification preferences for the current user."""
    preferences = notification_service.get_preferences(db, current_user.id)

    return NotificationPreferencesResponse(
        emailEnabled=preferences.email_enabled,
        pushEnabled=preferences.push_enabled,
        slackEnabled=preferences.slack_enabled,
        inAppEnabled=preferences.in_app_enabled
    )


@router.patch("/notifications/preferences", response_model=NotificationPreferencesResponse)
async def update_notification_preferences(
    update_data: NotificationPreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update notification preferences."""
    preferences = notification_service.update_preferences(
        db=db,
        user_id=current_user.id,
        email_enabled=update_data.emailEnabled,
        push_enabled=update_data.pushEnabled,
        slack_enabled=update_data.slackEnabled,
        in_app_enabled=update_data.inAppEnabled
    )

    return NotificationPreferencesResponse(
        emailEnabled=preferences.email_enabled,
        pushEnabled=preferences.push_enabled,
        slackEnabled=preferences.slack_enabled,
        inAppEnabled=preferences.in_app_enabled
    )
