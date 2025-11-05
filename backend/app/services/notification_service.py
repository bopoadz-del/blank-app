"""
Notification service for managing in-app, email, and push notifications.
"""
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from app.models.notifications import (
    Notification,
    NotificationPreferences,
    NotificationType,
    NotificationChannel
)
from app.models.auth import User

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for managing notifications."""

    @staticmethod
    def create_notification(
        db: Session,
        user_id: int,
        title: str,
        message: str,
        type: NotificationType = NotificationType.INFO,
        channel: NotificationChannel = NotificationChannel.IN_APP,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Create a new notification."""
        notification = Notification(
            user_id=user_id,
            type=type,
            channel=channel,
            title=title,
            message=message,
            metadata=metadata
        )
        db.add(notification)
        db.commit()
        db.refresh(notification)

        logger.info(f"Created {channel.value} notification for user {user_id}: {title}")
        return notification

    @staticmethod
    def get_user_notifications(
        db: Session,
        user_id: int,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Notification]:
        """Get notifications for a user."""
        query = db.query(Notification).filter(Notification.user_id == user_id)

        if unread_only:
            query = query.filter(Notification.read == False)

        notifications = query.order_by(
            Notification.created_at.desc()
        ).limit(limit).all()

        return notifications

    @staticmethod
    def mark_as_read(db: Session, notification_id: int, user_id: int) -> bool:
        """Mark a notification as read."""
        notification = db.query(Notification).filter(
            Notification.id == notification_id,
            Notification.user_id == user_id
        ).first()

        if notification:
            notification.read = True
            notification.read_at = datetime.utcnow()
            db.commit()
            return True

        return False

    @staticmethod
    def mark_all_as_read(db: Session, user_id: int) -> int:
        """Mark all notifications as read for a user."""
        count = db.query(Notification).filter(
            Notification.user_id == user_id,
            Notification.read == False
        ).update({
            "read": True,
            "read_at": datetime.utcnow()
        })
        db.commit()
        return count

    @staticmethod
    def delete_notification(db: Session, notification_id: int, user_id: int) -> bool:
        """Delete a notification."""
        notification = db.query(Notification).filter(
            Notification.id == notification_id,
            Notification.user_id == user_id
        ).first()

        if notification:
            db.delete(notification)
            db.commit()
            return True

        return False

    @staticmethod
    def get_unread_count(db: Session, user_id: int) -> int:
        """Get count of unread notifications."""
        return db.query(Notification).filter(
            Notification.user_id == user_id,
            Notification.read == False
        ).count()

    @staticmethod
    def get_preferences(db: Session, user_id: int) -> NotificationPreferences:
        """Get user notification preferences."""
        prefs = db.query(NotificationPreferences).filter(
            NotificationPreferences.user_id == user_id
        ).first()

        if not prefs:
            # Create default preferences
            prefs = NotificationPreferences(user_id=user_id)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)

        return prefs

    @staticmethod
    def update_preferences(
        db: Session,
        user_id: int,
        **kwargs
    ) -> NotificationPreferences:
        """Update user notification preferences."""
        prefs = NotificationService.get_preferences(db, user_id)

        for key, value in kwargs.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)

        db.commit()
        db.refresh(prefs)

        logger.info(f"Updated notification preferences for user {user_id}")
        return prefs

    @staticmethod
    def send_system_notification(
        db: Session,
        user_id: int,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Send a system notification."""
        return NotificationService.create_notification(
            db=db,
            user_id=user_id,
            title=title,
            message=message,
            type=NotificationType.SYSTEM,
            metadata=metadata
        )

    @staticmethod
    def notify_new_message(
        db: Session,
        user_id: int,
        conversation_title: str,
        message_preview: str
    ) -> Notification:
        """Notify user of a new message."""
        return NotificationService.create_notification(
            db=db,
            user_id=user_id,
            title=f"New message in {conversation_title}",
            message=message_preview,
            type=NotificationType.MESSAGE,
            metadata={"conversation_title": conversation_title}
        )


notification_service = NotificationService()
