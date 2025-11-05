"""
Pydantic schemas for chat, conversations, and projects.
"""
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict, Any


class MessageBase(BaseModel):
    role: str
    content: str


class MessageCreate(MessageBase):
    metadata: Optional[Dict[str, Any]] = None


class FileAttachmentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    mime_type: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class MessageResponse(MessageBase):
    id: str
    timestamp: datetime
    attachments: List[FileAttachmentResponse] = []

    class Config:
        from_attributes = True


class ConversationBase(BaseModel):
    title: str


class ConversationCreate(ConversationBase):
    project_id: Optional[int] = None


class ConversationResponse(ConversationBase):
    id: str
    messages: List[MessageResponse] = []
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    color: Optional[str] = "blue"


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None


class ProjectResponse(ProjectBase):
    id: str
    conversations: List[ConversationResponse] = []
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True


class SendMessageRequest(BaseModel):
    content: str
    internet_enabled: Optional[bool] = False
    files: Optional[List[str]] = []  # List of uploaded file IDs
