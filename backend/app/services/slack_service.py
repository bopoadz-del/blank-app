"""
Slack integration service for sending messages to Slack channels.
"""
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from datetime import datetime
import httpx
import logging
import json

from app.models.notifications import SlackIntegration

logger = logging.getLogger(__name__)


class SlackService:
    """Service for Slack integration."""

    @staticmethod
    async def send_message(
        webhook_url: str,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Send a message to Slack via webhook."""
        try:
            payload = {"text": text}

            if blocks:
                payload["blocks"] = blocks

            if attachments:
                payload["attachments"] = attachments

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )

                if response.status_code == 200:
                    logger.info(f"Successfully sent message to Slack")
                    return True
                else:
                    logger.error(f"Failed to send Slack message: {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"Error sending Slack message: {str(e)}")
            return False

    @staticmethod
    async def send_notification(
        webhook_url: str,
        title: str,
        message: str,
        color: str = "#36a64f",
        fields: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        """Send a formatted notification to Slack."""
        attachment = {
            "color": color,
            "title": title,
            "text": message,
            "footer": "ML Framework",
            "ts": int(datetime.utcnow().timestamp())
        }

        if fields:
            attachment["fields"] = fields

        return await SlackService.send_message(
            webhook_url=webhook_url,
            text=title,
            attachments=[attachment]
        )

    @staticmethod
    def create_integration(
        db: Session,
        user_id: int,
        workspace_name: str,
        webhook_url: str,
        channel: str
    ) -> SlackIntegration:
        """Create a new Slack integration."""
        # Check if integration already exists
        existing = db.query(SlackIntegration).filter(
            SlackIntegration.user_id == user_id
        ).first()

        if existing:
            # Update existing integration
            existing.workspace_name = workspace_name
            existing.webhook_url = webhook_url
            existing.channel = channel
            existing.enabled = True
            db.commit()
            db.refresh(existing)
            logger.info(f"Updated Slack integration for user {user_id}")
            return existing
        else:
            # Create new integration
            integration = SlackIntegration(
                user_id=user_id,
                workspace_name=workspace_name,
                webhook_url=webhook_url,
                channel=channel,
                enabled=True
            )
            db.add(integration)
            db.commit()
            db.refresh(integration)
            logger.info(f"Created Slack integration for user {user_id}")
            return integration

    @staticmethod
    def get_integration(db: Session, user_id: int) -> Optional[SlackIntegration]:
        """Get Slack integration for a user."""
        return db.query(SlackIntegration).filter(
            SlackIntegration.user_id == user_id
        ).first()

    @staticmethod
    def delete_integration(db: Session, user_id: int) -> bool:
        """Delete Slack integration for a user."""
        integration = db.query(SlackIntegration).filter(
            SlackIntegration.user_id == user_id
        ).first()

        if integration:
            db.delete(integration)
            db.commit()
            logger.info(f"Deleted Slack integration for user {user_id}")
            return True

        return False

    @staticmethod
    def toggle_integration(db: Session, user_id: int, enabled: bool) -> bool:
        """Enable or disable Slack integration."""
        integration = db.query(SlackIntegration).filter(
            SlackIntegration.user_id == user_id
        ).first()

        if integration:
            integration.enabled = enabled
            db.commit()
            logger.info(f"{'Enabled' if enabled else 'Disabled'} Slack integration for user {user_id}")
            return True

        return False

    @staticmethod
    async def test_webhook(webhook_url: str) -> bool:
        """Test a Slack webhook URL."""
        return await SlackService.send_message(
            webhook_url=webhook_url,
            text="✅ Slack integration test successful! Your webhook is working correctly."
        )

    @staticmethod
    async def notify_conversation_message(
        db: Session,
        user_id: int,
        conversation_title: str,
        message: str,
        username: str
    ) -> bool:
        """Send conversation message notification to Slack."""
        integration = SlackService.get_integration(db, user_id)

        if not integration or not integration.enabled:
            return False

        return await SlackService.send_notification(
            webhook_url=integration.webhook_url,
            title=f"💬 New message in {conversation_title}",
            message=f"{username}: {message[:200]}{'...' if len(message) > 200 else ''}",
            color="#4A90E2"
        )

    @staticmethod
    async def notify_file_upload(
        db: Session,
        user_id: int,
        filename: str,
        file_type: str,
        username: str
    ) -> bool:
        """Send file upload notification to Slack."""
        integration = SlackService.get_integration(db, user_id)

        if not integration or not integration.enabled:
            return False

        icon = "📎"
        if file_type.startswith("image"):
            icon = "🖼️"
        elif file_type.startswith("video"):
            icon = "🎥"
        elif file_type.startswith("audio"):
            icon = "🎵"
        elif "zip" in file_type or "compressed" in file_type:
            icon = "📦"
        elif "xer" in filename.lower() or "mpp" in filename.lower():
            icon = "🗓️"
        elif any(ext in filename.lower() for ext in ["dwg", "dxf", "rvt", "ifc"]):
            icon = "📐"

        return await SlackService.send_notification(
            webhook_url=integration.webhook_url,
            title=f"{icon} File uploaded",
            message=f"{username} uploaded: {filename}",
            color="#F5A623",
            fields=[
                {"title": "Filename", "value": filename, "short": True},
                {"title": "Type", "value": file_type, "short": True}
            ]
        )


slack_service = SlackService()
