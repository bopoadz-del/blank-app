"""
Report generation API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import logging

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.auth import User
from app.models.notifications import ReportType, ReportFormat
from app.services.report_service import report_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Report storage directory
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


class ReportGenerateRequest(BaseModel):
    type: str  # "conversation" or "project"
    format: str  # "pdf", "excel", or "csv"
    resourceId: int  # conversation_id or project_id


class ReportResponse(BaseModel):
    id: str
    title: str
    type: str
    format: str
    fileSize: int
    createdAt: str

    class Config:
        from_attributes = True


@router.post("/reports/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate a new report."""
    # Validate type
    if request.type not in ["conversation", "project"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid report type. Must be 'conversation' or 'project'"
        )

    # Validate format
    if request.format not in ["pdf", "excel", "csv"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid format. Must be 'pdf', 'excel', or 'csv'"
        )

    try:
        # Generate report based on type and format
        report_data = None
        title = ""

        if request.type == "conversation":
            if request.format == "pdf":
                report_data = report_service.generate_conversation_pdf(
                    db, request.resourceId, current_user.id
                )
                title = f"Conversation Report"
            elif request.format == "excel":
                report_data = report_service.generate_conversation_excel(
                    db, request.resourceId, current_user.id
                )
                title = f"Conversation Report"
            elif request.format == "csv":
                report_data = report_service.generate_conversation_csv(
                    db, request.resourceId, current_user.id
                )
                title = f"Conversation Report"

        elif request.type == "project":
            if request.format == "pdf":
                report_data = report_service.generate_project_pdf(
                    db, request.resourceId, current_user.id
                )
                title = f"Project Report"
            elif request.format == "excel":
                report_data = report_service.generate_project_excel(
                    db, request.resourceId, current_user.id
                )
                title = f"Project Report"
            elif request.format == "csv":
                report_data = report_service.generate_project_csv(
                    db, request.resourceId, current_user.id
                )
                title = f"Project Report"

        if not report_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate report"
            )

        # Save report to file
        user_reports_dir = REPORTS_DIR / str(current_user.id)
        user_reports_dir.mkdir(exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        extension = "pdf" if request.format == "pdf" else ("xlsx" if request.format == "excel" else "csv")
        filename = f"{request.type}_{timestamp}.{extension}"
        file_path = user_reports_dir / filename

        with open(file_path, "wb") as f:
            f.write(report_data)

        # Create report record
        report_type = ReportType.CONVERSATION if request.type == "conversation" else ReportType.PROJECT
        report_format = ReportFormat[request.format.upper()]

        report = report_service.create_report(
            db=db,
            user_id=current_user.id,
            title=title,
            report_type=report_type,
            report_format=report_format,
            file_path=str(file_path),
            file_size=len(report_data),
            metadata={
                "resource_id": request.resourceId,
                "generated_at": datetime.utcnow().isoformat()
            }
        )

        return ReportResponse(
            id=str(report.id),
            title=report.title,
            type=report.type.value,
            format=report.format.value,
            fileSize=report.file_size,
            createdAt=report.created_at.isoformat()
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating report: {str(e)}"
        )


@router.get("/reports", response_model=List[ReportResponse])
async def get_reports(
    report_type: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's reports."""
    type_filter = None
    if report_type:
        if report_type not in ["conversation", "project", "system"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid report type"
            )
        type_filter = ReportType[report_type.upper()]

    reports = report_service.get_user_reports(
        db=db,
        user_id=current_user.id,
        report_type=type_filter,
        limit=limit
    )

    return [
        ReportResponse(
            id=str(r.id),
            title=r.title,
            type=r.type.value,
            format=r.format.value,
            fileSize=r.file_size,
            createdAt=r.created_at.isoformat()
        ) for r in reports
    ]


@router.get("/reports/{report_id}/download")
async def download_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download a report."""
    from app.models.notifications import Report

    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id
    ).first()

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )

    file_path = Path(report.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found"
        )

    # Determine content type and extension
    content_type_map = {
        "pdf": "application/pdf",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv"
    }
    extension_map = {
        "pdf": "pdf",
        "excel": "xlsx",
        "csv": "csv"
    }

    content_type = content_type_map.get(report.format.value, "application/octet-stream")
    extension = extension_map.get(report.format.value, "bin")

    with open(file_path, "rb") as f:
        content = f.read()

    return Response(
        content=content,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{report.title.replace(" ", "_")}.{extension}"'
        }
    )


@router.delete("/reports/{report_id}")
async def delete_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a report."""
    success = report_service.delete_report(db, report_id, current_user.id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )

    return {"message": "Report deleted successfully"}
