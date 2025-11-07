"""
API routes for data ingestion from Google Drive.

Handles OAuth authentication and file syncing.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from loguru import logger

from app.core.config import settings
from app.core.database import get_db
from app.models.auth import User
from app.api.auth_routes import get_current_user

# OAuth imports
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

router = APIRouter(prefix="/drive", tags=["data-ingestion"])


# Store OAuth state temporarily (in production, use Redis/DB)
oauth_states = {}


@router.get("/authorize")
async def google_drive_authorize(
    current_user: User = Depends(get_current_user)
):
    """
    Initiate Google Drive OAuth flow.

    Redirects user to Google consent screen.
    """
    if not settings.GOOGLE_OAUTH_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth not configured. Set GOOGLE_OAUTH_CLIENT_ID."
        )

    # Create OAuth flow
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.GOOGLE_OAUTH_REDIRECT_URI]
            }
        },
        scopes=[
            'https://www.googleapis.com/auth/drive.readonly',
            'https://www.googleapis.com/auth/drive.metadata.readonly'
        ]
    )

    flow.redirect_uri = settings.GOOGLE_OAUTH_REDIRECT_URI

    # Generate authorization URL
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
    )

    # Store state for verification
    oauth_states[state] = {
        'user_id': current_user.id,
        'flow': flow
    }

    logger.info(f"User {current_user.id} initiated Google Drive OAuth")

    return {
        "authorization_url": authorization_url,
        "state": state,
        "message": "Redirect user to authorization_url"
    }


@router.get("/callback")
async def google_drive_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db)
):
    """
    Handle Google OAuth callback.

    Exchanges authorization code for access token.
    """
    if state not in oauth_states:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired OAuth state"
        )

    oauth_data = oauth_states[state]
    flow = oauth_data['flow']

    try:
        # Exchange code for token
        flow.fetch_token(code=code)

        credentials = flow.credentials

        # Store credentials securely (in production, encrypt and store in DB)
        # For now, log success
        logger.info(f"User {oauth_data['user_id']} authorized Google Drive access")

        # Clean up state
        del oauth_states[state]

        # Redirect to frontend dashboard
        frontend_url = settings.CORS_ORIGINS.split(',')[0] if settings.CORS_ORIGINS != "*" else "http://localhost:3000"

        return RedirectResponse(
            url=f"{frontend_url}/dashboard?gdrive_connected=true"
        )

    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to complete OAuth flow: {str(e)}"
        )


@router.post("/sync")
async def trigger_sync(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Manually trigger Google Drive file sync.

    Syncs files from configured folder to local cache.
    """
    if not settings.GOOGLE_DRIVE_FOLDER_ID:
        raise HTTPException(
            status_code=400,
            detail="Google Drive folder not configured"
        )

    # Import here to avoid issues if deps not installed
    try:
        from app.services.data_ingestion import GoogleDriveConnector
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data ingestion service not available: {e}"
        )

    def sync_task():
        """Background task to sync files."""
        try:
            connector = GoogleDriveConnector(
                credentials_path=settings.GOOGLE_DRIVE_CREDENTIALS_PATH,
                folder_id=settings.GOOGLE_DRIVE_FOLDER_ID
            )

            synced_files = connector.sync_folder(
                local_cache_dir="/tmp/drive_cache"
            )

            logger.info(f"Synced {len(synced_files)} files from Google Drive")

        except Exception as e:
            logger.error(f"Sync task failed: {e}")

    # Queue sync task
    background_tasks.add_task(sync_task)

    return {
        "message": "Sync task started",
        "status": "processing"
    }


@router.get("/files")
async def list_synced_files(
    current_user: User = Depends(get_current_user)
):
    """
    List files synced from Google Drive.

    Returns metadata of cached files.
    """
    if not settings.GOOGLE_DRIVE_FOLDER_ID:
        raise HTTPException(
            status_code=400,
            detail="Google Drive folder not configured"
        )

    try:
        from app.services.data_ingestion import GoogleDriveConnector

        connector = GoogleDriveConnector(
            credentials_path=settings.GOOGLE_DRIVE_CREDENTIALS_PATH,
            folder_id=settings.GOOGLE_DRIVE_FOLDER_ID
        )

        files = connector.list_files(limit=100)

        return {
            "files": files,
            "count": len(files),
            "folder_id": settings.GOOGLE_DRIVE_FOLDER_ID
        }

    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files: {str(e)}"
        )


@router.post("/parse/{file_id}")
async def parse_file(
    file_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Parse a specific file from Google Drive.

    Downloads, parses, and extracts data from the file.
    """
    try:
        from app.services.data_ingestion import (
            GoogleDriveConnector,
            FileParser,
            DataExtractor
        )

        # Download file
        connector = GoogleDriveConnector(
            credentials_path=settings.GOOGLE_DRIVE_CREDENTIALS_PATH,
            folder_id=settings.GOOGLE_DRIVE_FOLDER_ID
        )

        content = connector.download_file(file_id)

        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Parse file
        parsed_data = FileParser.parse_file(tmp_path)

        # Extract numerical data
        numerical_data = DataExtractor.extract_numerical_data(parsed_data)

        # Extract context hints
        context_hints = DataExtractor.extract_context_hints(parsed_data)

        # Clean up temp file
        import os
        os.unlink(tmp_path)

        return {
            "file_id": file_id,
            "parsed_data": parsed_data,
            "numerical_data": numerical_data,
            "context_hints": context_hints
        }

    except Exception as e:
        logger.error(f"Failed to parse file {file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse file: {str(e)}"
        )


@router.get("/status")
async def ingestion_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get data ingestion configuration status.
    """
    return {
        "configured": bool(settings.GOOGLE_DRIVE_FOLDER_ID),
        "credentials_path": settings.GOOGLE_DRIVE_CREDENTIALS_PATH,
        "folder_id": settings.GOOGLE_DRIVE_FOLDER_ID,
        "sync_interval": settings.DATA_INGESTION_INTERVAL,
        "supported_file_types": settings.SUPPORTED_FILE_TYPES,
        "oauth_configured": bool(settings.GOOGLE_OAUTH_CLIENT_ID)
    }
