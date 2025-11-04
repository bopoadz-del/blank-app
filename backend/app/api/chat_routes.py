"""
Chat, conversation, and project API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import shutil
import os
from pathlib import Path

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.auth import User
from app.models.chat import Project, Conversation, Message, FileAttachment
from app.schemas.chat_schemas import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ConversationCreate,
    ConversationResponse,
    MessageCreate,
    MessageResponse,
    SendMessageRequest,
    FileAttachmentResponse
)

router = APIRouter()

# File upload settings
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ==================== PROJECTS ====================

@router.post("/projects", response_model=ProjectResponse)
async def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new project."""
    project = Project(
        user_id=current_user.id,
        name=project_data.name,
        description=project_data.description,
        color=project_data.color
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    return ProjectResponse(
        id=str(project.id),
        name=project.name,
        description=project.description,
        color=project.color,
        conversations=[],
        createdAt=project.created_at,
        updatedAt=project.updated_at or project.created_at
    )


@router.get("/projects", response_model=List[ProjectResponse])
async def get_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all projects for the current user."""
    projects = db.query(Project).filter(Project.user_id == current_user.id).all()

    return [
        ProjectResponse(
            id=str(p.id),
            name=p.name,
            description=p.description,
            color=p.color,
            conversations=[
                ConversationResponse(
                    id=str(c.id),
                    title=c.title,
                    messages=[
                        MessageResponse(
                            id=str(m.id),
                            role=m.role,
                            content=m.content,
                            timestamp=m.created_at,
                            attachments=[]
                        ) for m in c.messages
                    ],
                    createdAt=c.created_at,
                    updatedAt=c.updated_at or c.created_at
                ) for c in p.conversations
            ],
            createdAt=p.created_at,
            updatedAt=p.updated_at or p.created_at
        ) for p in projects
    ]


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a project."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    if project_data.name is not None:
        project.name = project_data.name
    if project_data.description is not None:
        project.description = project_data.description
    if project_data.color is not None:
        project.color = project_data.color

    db.commit()
    db.refresh(project)

    return ProjectResponse(
        id=str(project.id),
        name=project.name,
        description=project.description,
        color=project.color,
        conversations=[],
        createdAt=project.created_at,
        updatedAt=project.updated_at or project.created_at
    )


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a project."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    db.delete(project)
    db.commit()

    return {"message": "Project deleted successfully"}


# ==================== CONVERSATIONS ====================

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    conversation_data: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new conversation."""
    conversation = Conversation(
        user_id=current_user.id,
        project_id=conversation_data.project_id,
        title=conversation_data.title
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return ConversationResponse(
        id=str(conversation.id),
        title=conversation.title,
        messages=[],
        createdAt=conversation.created_at,
        updatedAt=conversation.updated_at or conversation.created_at
    )


@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all conversations for the current user."""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).all()

    return [
        ConversationResponse(
            id=str(c.id),
            title=c.title,
            messages=[
                MessageResponse(
                    id=str(m.id),
                    role=m.role,
                    content=m.content,
                    timestamp=m.created_at,
                    attachments=[
                        FileAttachmentResponse(
                            id=str(a.id),
                            filename=a.filename,
                            file_type=a.file_type,
                            file_size=a.file_size,
                            mime_type=a.mime_type,
                            created_at=a.created_at
                        ) for a in m.attachments
                    ]
                ) for m in c.messages
            ],
            createdAt=c.created_at,
            updatedAt=c.updated_at or c.created_at
        ) for c in conversations
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific conversation."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    return ConversationResponse(
        id=str(conversation.id),
        title=conversation.title,
        messages=[
            MessageResponse(
                id=str(m.id),
                role=m.role,
                content=m.content,
                timestamp=m.created_at,
                attachments=[
                    FileAttachmentResponse(
                        id=str(a.id),
                        filename=a.filename,
                        file_type=a.file_type,
                        file_size=a.file_size,
                        mime_type=a.mime_type,
                        created_at=a.created_at
                    ) for a in m.attachments
                ]
            ) for m in conversation.messages
        ],
        createdAt=conversation.created_at,
        updatedAt=conversation.updated_at or conversation.created_at
    )


# ==================== MESSAGES ====================

@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(
    conversation_id: int,
    message_data: SendMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message in a conversation."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # Create user message
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=message_data.content,
        metadata={"internet_enabled": message_data.internet_enabled}
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    # Generate AI response (placeholder - integrate with actual ML model)
    ai_response_content = f"I received your message. This is a demo response."

    if message_data.internet_enabled:
        ai_response_content += " Internet search is enabled."

    ai_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=ai_response_content
    )
    db.add(ai_message)
    db.commit()
    db.refresh(ai_message)

    return MessageResponse(
        id=str(ai_message.id),
        role=ai_message.role,
        content=ai_message.content,
        timestamp=ai_message.created_at,
        attachments=[]
    )


# ==================== FILE UPLOADS ====================

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a file (images, videos, audio, documents, CAD, XER, ZIP, etc.)."""
    # Create user-specific upload directory
    user_upload_dir = UPLOAD_DIR / str(current_user.id)
    user_upload_dir.mkdir(exist_ok=True)

    # Save file
    file_path = user_upload_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "file_path": str(file_path),
        "file_size": file_path.stat().st_size,
        "mime_type": file.content_type
    }
