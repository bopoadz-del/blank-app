"""
Slack integration API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.auth import User
from app.services.slack_service import slack_service

router = APIRouter()


class SlackIntegrationRequest(BaseModel):
    workspaceName: str
    webhookUrl: str
    channel: str


class SlackIntegrationResponse(BaseModel):
    id: str
    workspaceName: str
    webhookUrl: str
    channel: str
    enabled: bool
    createdAt: str

    class Config:
        from_attributes = True


class SlackToggleRequest(BaseModel):
    enabled: bool


class SlackTestResponse(BaseModel):
    success: bool
    message: str


@router.post("/slack/integration", response_model=SlackIntegrationResponse)
async def create_or_update_slack_integration(
    integration_data: SlackIntegrationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create or update Slack integration."""
    integration = slack_service.create_integration(
        db=db,
        user_id=current_user.id,
        workspace_name=integration_data.workspaceName,
        webhook_url=integration_data.webhookUrl,
        channel=integration_data.channel
    )

    return SlackIntegrationResponse(
        id=str(integration.id),
        workspaceName=integration.workspace_name,
        webhookUrl=integration.webhook_url,
        channel=integration.channel,
        enabled=integration.enabled,
        createdAt=integration.created_at.isoformat()
    )


@router.get("/slack/integration", response_model=Optional[SlackIntegrationResponse])
async def get_slack_integration(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current Slack integration."""
    integration = slack_service.get_integration(db, current_user.id)

    if not integration:
        return None

    return SlackIntegrationResponse(
        id=str(integration.id),
        workspaceName=integration.workspace_name,
        webhookUrl=integration.webhook_url,
        channel=integration.channel,
        enabled=integration.enabled,
        createdAt=integration.created_at.isoformat()
    )


@router.delete("/slack/integration")
async def delete_slack_integration(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete Slack integration."""
    success = slack_service.delete_integration(db, current_user.id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Slack integration not found"
        )

    return {"message": "Slack integration deleted successfully"}


@router.patch("/slack/toggle")
async def toggle_slack_integration(
    toggle_data: SlackToggleRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enable or disable Slack integration."""
    success = slack_service.toggle_integration(
        db=db,
        user_id=current_user.id,
        enabled=toggle_data.enabled
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Slack integration not found"
        )

    return {
        "message": f"Slack integration {'enabled' if toggle_data.enabled else 'disabled'}",
        "enabled": toggle_data.enabled
    }


@router.post("/slack/test", response_model=SlackTestResponse)
async def test_slack_webhook(
    integration_data: SlackIntegrationRequest,
    current_user: User = Depends(get_current_user)
):
    """Test Slack webhook URL."""
    success = await slack_service.test_webhook(integration_data.webhookUrl)

    return SlackTestResponse(
        success=success,
        message="Webhook test successful! Check your Slack channel." if success else "Webhook test failed. Please check your URL."
    )
