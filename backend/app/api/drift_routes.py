"""
Drift Detection and Auto-Retrain API routes.
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ..core.database import get_db
from ..models.edge_devices import DriftMetric, RetrainJob
from ..services.drift_detection_service import DriftDetectionService
from ..services.retrain_service import RetrainService
from ..core.rbac import get_current_admin, get_current_operator


router = APIRouter(prefix="/api/v1", tags=["Drift Detection & Retrain"])


# Pydantic Schemas
class DriftMetricResponse(BaseModel):
    id: int
    formula_id: int
    device_id: Optional[int]
    window_start: datetime
    window_end: datetime
    correction_rate: float
    executions_count: int
    corrections_count: int
    drift_detected: bool
    drift_score: Optional[float]
    baseline_correction_rate: Optional[float]
    correction_rate_change: Optional[float]
    retrain_triggered: bool

    class Config:
        from_attributes = True


class DriftSummaryResponse(BaseModel):
    total_checks: int
    drift_detected_count: int
    latest_correction_rate: float
    avg_correction_rate: float
    drift_percentage: float
    retrains_triggered: int
    latest_drift_score: float


class TriggerDriftCheck(BaseModel):
    formula_id: int
    time_window_hours: int = Field(default=24, ge=1, le=168)
    device_id: Optional[int] = None
    auto_trigger_retrain: bool = True


class TriggerRetrain(BaseModel):
    formula_id: int
    config: Optional[dict] = None


class RetrainJobResponse(BaseModel):
    id: int
    job_id: str
    formula_id: int
    trigger_type: str
    status: str
    corrections_used_count: Optional[int]
    training_samples_count: Optional[int]
    validation_samples_count: Optional[int]
    new_model_id: Optional[int]
    new_model_version: Optional[str]
    mlflow_run_id: Optional[str]
    training_accuracy: Optional[float]
    validation_accuracy: Optional[float]
    scheduled_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class RetrainStatsResponse(BaseModel):
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    avg_training_accuracy: float
    avg_validation_accuracy: float
    total_corrections_used: int
    latest_model_version: Optional[str]
    latest_mlflow_run: Optional[str]


# Routes

@router.post("/drift/check", response_model=DriftMetricResponse)
async def check_drift(
    check_data: TriggerDriftCheck,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Manually trigger drift detection for a formula (admin only).
    Analyzes correction rate over time window and detects concept drift.
    """
    metric = DriftDetectionService.calculate_drift_metrics(
        db=db,
        formula_id=check_data.formula_id,
        time_window_hours=check_data.time_window_hours,
        device_id=check_data.device_id
    )

    if not metric:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No execution data available for drift analysis"
        )

    # Auto-trigger retrain if drift detected and enabled
    if metric.drift_detected and check_data.auto_trigger_retrain and not metric.retrain_triggered:
        job = RetrainService.trigger_retrain(
            db=db,
            formula_id=check_data.formula_id,
            trigger_type="drift_detected",
            drift_metric_id=metric.id,
            user_id=current_user.id
        )

        if job:
            metric.retrain_triggered = True
            metric.retrain_job_id = job.job_id
            db.commit()
            db.refresh(metric)

            # Execute retrain in background
            background_tasks.add_task(
                RetrainService.execute_retrain_job,
                db,
                job.job_id
            )

    return metric


@router.get("/drift/formulas/{formula_id}/history", response_model=List[DriftMetricResponse])
async def get_drift_history(
    formula_id: int,
    days: int = 30,
    current_user = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    Get drift detection history for a formula.
    """
    metrics = DriftDetectionService.get_drift_history(
        db=db,
        formula_id=formula_id,
        days=days
    )

    return metrics


@router.get("/drift/formulas/{formula_id}/summary", response_model=DriftSummaryResponse)
async def get_drift_summary(
    formula_id: int,
    current_user = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    Get drift summary statistics for a formula.
    """
    summary = DriftDetectionService.get_drift_summary(
        db=db,
        formula_id=formula_id
    )

    return summary


@router.post("/retrain/trigger", response_model=RetrainJobResponse, status_code=status.HTTP_201_CREATED)
async def trigger_retrain(
    retrain_data: TriggerRetrain,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Manually trigger a retrain job for a formula (admin only).
    Uses approved corrections to fine-tune and create a new Tier 4 model.
    """
    job = RetrainService.trigger_retrain(
        db=db,
        formula_id=retrain_data.formula_id,
        trigger_type="manual",
        user_id=current_user.id,
        config=retrain_data.config
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot trigger retrain. Not enough approved corrections (minimum 10 required)."
        )

    # Execute retrain in background
    background_tasks.add_task(
        RetrainService.execute_retrain_job,
        db,
        job.job_id
    )

    return job


@router.get("/retrain/jobs", response_model=List[RetrainJobResponse])
async def list_retrain_jobs(
    formula_id: Optional[int] = None,
    status_filter: Optional[str] = None,
    limit: int = 100,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    List retrain jobs with optional filters (admin only).
    """
    jobs = RetrainService.get_retrain_jobs(
        db=db,
        formula_id=formula_id,
        status=status_filter,
        limit=limit
    )

    return jobs


@router.get("/retrain/jobs/{job_id}", response_model=RetrainJobResponse)
async def get_retrain_job(
    job_id: str,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific retrain job (admin only).
    """
    job = db.query(RetrainJob).filter(RetrainJob.job_id == job_id).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Retrain job {job_id} not found"
        )

    return job


@router.get("/retrain/formulas/{formula_id}/stats", response_model=RetrainStatsResponse)
async def get_retrain_stats(
    formula_id: int,
    current_user = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    Get retrain statistics for a formula.
    """
    stats = RetrainService.get_retrain_stats(
        db=db,
        formula_id=formula_id
    )

    return stats


@router.post("/retrain/scheduled/run", response_model=dict)
async def run_scheduled_retrain(
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Manually trigger scheduled retrain for all eligible formulas (admin only).
    Normally this would run on a schedule (e.g., nightly).
    """
    jobs = RetrainService.run_scheduled_retrain(db=db)

    # Execute jobs in background
    for job in jobs:
        background_tasks.add_task(
            RetrainService.execute_retrain_job,
            db,
            job.job_id
        )

    return {
        "jobs_created": len(jobs),
        "job_ids": [job.job_id for job in jobs]
    }


@router.post("/drift/check-all", response_model=dict)
async def check_all_formulas_for_drift(
    time_window_hours: int = 24,
    auto_trigger_retrain: bool = True,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Check all formulas for drift (admin only).
    This would normally run on a schedule.
    """
    drift_metrics = DriftDetectionService.check_all_formulas_for_drift(
        db=db,
        time_window_hours=time_window_hours,
        auto_trigger_retrain=auto_trigger_retrain
    )

    return {
        "formulas_checked": len(drift_metrics),
        "drift_detected": len([m for m in drift_metrics if m.drift_detected]),
        "retrains_triggered": len([m for m in drift_metrics if m.retrain_triggered]),
        "metrics": [
            {
                "formula_id": m.formula_id,
                "drift_score": m.drift_score,
                "correction_rate": m.correction_rate,
                "retrain_triggered": m.retrain_triggered
            }
            for m in drift_metrics
        ]
    }
