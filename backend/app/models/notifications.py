"""
Notification models for in-app, email, and push notifications.
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.database import Base
import enum


class NotificationType(enum.Enum):
    """Notification types."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    MESSAGE = "message"
    SYSTEM = "system"


class NotificationChannel(enum.Enum):
    """Notification delivery channels."""
    IN_APP = "in_app"
    EMAIL = "email"
    PUSH = "push"
    SLACK = "slack"


class Notification(Base):
    """Notification model for all notification types."""
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(SQLEnum(NotificationType), nullable=False, default=NotificationType.INFO)
    channel = Column(SQLEnum(NotificationChannel), nullable=False, default=NotificationChannel.IN_APP)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    metadata = Column(JSON, nullable=True)  # Additional data (links, actions, etc.)
    read = Column(Boolean, default=False)
    read_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class NotificationPreferences(Base):
    """User notification preferences."""
    __tablename__ = "notification_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    email_enabled = Column(Boolean, default=True)
    push_enabled = Column(Boolean, default=True)
    slack_enabled = Column(Boolean, default=False)

    # Notification type preferences
    message_notifications = Column(Boolean, default=True)
    system_notifications = Column(Boolean, default=True)
    error_notifications = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class SlackIntegration(Base):
    """Slack integration configuration."""
    __tablename__ = "slack_integrations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    workspace_name = Column(String, nullable=False)
    webhook_url = Column(String, nullable=False)  # Encrypted in production
    channel = Column(String, nullable=False)  # Default channel
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Report(Base):
    """Generated reports."""
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    report_type = Column(String, nullable=False)  # conversation, project, system, custom
    format = Column(String, nullable=False)  # pdf, excel, csv
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=True)
    parameters = Column(JSON, nullable=True)  # Report generation parameters
    created_at = Column(DateTime(timezone=True), server_default=func.now())
