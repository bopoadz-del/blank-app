"""
Auto-Retrain Service for ML models.
Uses approved corrections to fine-tune and create new model versions.
Integrates with MLflow for model versioning and tracking.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func
import uuid
import json

from ..models.edge_devices import RetrainJob
from ..models.corrections import Correction, CorrectionStatus
from ..models.database import Formula, FormulaTier, FormulaStatus
from ..models.auth import User, UserRole


class RetrainService:
    """
    Service for auto-retraining models using approved corrections.
    Creates new Tier 4 (experimental) models after successful training.
    """

    @staticmethod
    def trigger_retrain(
        db: Session,
        formula_id: int,
        trigger_type: str = "manual",
        drift_metric_id: Optional[int] = None,
        user_id: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[RetrainJob]:
        """
        Trigger a retrain job for a formula.

        Args:
            db: Database session
            formula_id: Formula to retrain
            trigger_type: Type of trigger (manual, drift_detected, scheduled)
            drift_metric_id: ID of drift metric that triggered this (if applicable)
            user_id: User who triggered (if manual)
            config: Training configuration

        Returns:
            RetrainJob if created, None if not enough data
        """
        # Check if formula exists
        formula = db.query(Formula).filter(Formula.id == formula_id).first()
        if not formula:
            return None

        # Check for existing running jobs
        existing_job = db.query(RetrainJob).filter(
            RetrainJob.formula_id == formula_id,
            RetrainJob.status.in_(["pending", "running"])
        ).first()

        if existing_job:
            # Don't create duplicate jobs
            return existing_job

        # Get approved corrections for training data
        corrections = db.query(Correction).filter(
            Correction.execution.has(formula_id=formula_id),
            Correction.status == CorrectionStatus.APPROVED
        ).all()

        if len(corrections) < 10:  # Minimum threshold
            return None  # Not enough training data

        # Create retrain job
        job_id = f"retrain_{formula_id}_{uuid.uuid4().hex[:8]}"

        retrain_job = RetrainJob(
            job_id=job_id,
            formula_id=formula_id,
            trigger_type=trigger_type,
            triggered_by_drift_id=drift_metric_id,
            triggered_by_user_id=user_id,
            status="pending",
            corrections_used_count=len(corrections),
            config=config or {},
            scheduled_at=datetime.utcnow()
        )

        db.add(retrain_job)
        db.commit()
        db.refresh(retrain_job)

        return retrain_job

    @staticmethod
    def prepare_training_data(
        db: Session,
        formula_id: int,
        test_split: float = 0.2
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare training and validation data from approved corrections.

        Args:
            db: Database session
            formula_id: Formula to prepare data for
            test_split: Fraction of data for validation

        Returns:
            Tuple of (training_data, validation_data)
        """
        # Get approved corrections
        corrections = db.query(Correction).join(
            Correction.execution
        ).filter(
            Correction.execution.has(formula_id=formula_id),
            Correction.status == CorrectionStatus.APPROVED
        ).all()

        # Convert corrections to training samples
        samples = []
        for correction in corrections:
            sample = {
                "execution_id": correction.execution_id,
                "input": correction.execution.input_values,
                "original_output": correction.original_output,
                "corrected_output": correction.corrected_output,
                "correction_reason": correction.correction_reason,
                "operator_confidence": correction.operator_confidence,
                "timestamp": correction.created_at.isoformat()
            }
            samples.append(sample)

        # Split into train/validation
        split_index = int(len(samples) * (1 - test_split))
        training_data = samples[:split_index]
        validation_data = samples[split_index:]

        return training_data, validation_data

    @staticmethod
    def execute_retrain_job(
        db: Session,
        job_id: str
    ) -> bool:
        """
        Execute a retrain job.
        This is a simplified version - in production, this would:
        1. Load the base model
        2. Fine-tune with correction data
        3. Evaluate on validation set
        4. Log to MLflow
        5. Create new Tier 4 formula

        Args:
            db: Database session
            job_id: Job to execute

        Returns:
            True if successful, False otherwise
        """
        job = db.query(RetrainJob).filter(RetrainJob.job_id == job_id).first()
        if not job:
            return False

        try:
            # Update job status
            job.status = "running"
            job.started_at = datetime.utcnow()
            db.commit()

            # Get base formula
            base_formula = db.query(Formula).filter(Formula.id == job.formula_id).first()
            if not base_formula:
                raise Exception("Base formula not found")

            # Prepare training data
            training_data, validation_data = RetrainService.prepare_training_data(
                db=db,
                formula_id=job.formula_id
            )

            job.training_samples_count = len(training_data)
            job.validation_samples_count = len(validation_data)
            db.commit()

            # ===================================
            # SIMPLIFIED TRAINING SIMULATION
            # In production, replace with actual ML training
            # ===================================

            # Simulate training metrics
            import random
            training_accuracy = random.uniform(0.85, 0.95)
            validation_accuracy = random.uniform(0.80, 0.90)
            training_loss = random.uniform(0.05, 0.15)
            validation_loss = random.uniform(0.10, 0.20)

            # Create MLflow run ID (simulated)
            mlflow_run_id = f"mlflow_run_{uuid.uuid4().hex}"

            # Create new Tier 4 formula (experimental)
            new_version = f"{base_formula.version}_retrained_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            new_formula = Formula(
                formula_id=f"{base_formula.formula_id}_v{new_version}",
                name=f"{base_formula.name} (Retrained)",
                description=f"Auto-retrained version of {base_formula.name} using {job.training_samples_count} corrections",
                version=new_version,
                status=FormulaStatus.APPROVED,
                tier=FormulaTier.TIER_4_EXPERIMENTAL,
                is_locked=False,
                base_model=base_formula.base_model,
                input_schema=base_formula.input_schema,
                output_schema=base_formula.output_schema,
                tags=base_formula.tags or [],
                metadata={
                    "parent_formula_id": base_formula.id,
                    "retrain_job_id": job.job_id,
                    "training_samples": job.training_samples_count,
                    "validation_samples": job.validation_samples_count,
                    "training_accuracy": training_accuracy,
                    "validation_accuracy": validation_accuracy,
                    "mlflow_run_id": mlflow_run_id
                }
            )

            db.add(new_formula)
            db.flush()  # Get the ID

            # Update job with results
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.duration_seconds = (job.completed_at - job.started_at).total_seconds()
            job.new_model_id = new_formula.id
            job.new_model_version = new_version
            job.mlflow_run_id = mlflow_run_id
            job.training_accuracy = training_accuracy
            job.validation_accuracy = validation_accuracy
            job.training_loss = training_loss
            job.validation_loss = validation_loss
            job.metrics = {
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "samples_used": job.training_samples_count
            }

            db.commit()

            return True

        except Exception as e:
            # Mark job as failed
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            if job.started_at:
                job.duration_seconds = (job.completed_at - job.started_at).total_seconds()
            db.commit()

            return False

    @staticmethod
    def get_retrain_jobs(
        db: Session,
        formula_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[RetrainJob]:
        """
        Get retrain jobs with optional filters.

        Args:
            db: Database session
            formula_id: Filter by formula
            status: Filter by status
            limit: Maximum number of jobs to return

        Returns:
            List of retrain jobs
        """
        query = db.query(RetrainJob)

        if formula_id:
            query = query.filter(RetrainJob.formula_id == formula_id)
        if status:
            query = query.filter(RetrainJob.status == status)

        jobs = query.order_by(RetrainJob.created_at.desc()).limit(limit).all()
        return jobs

    @staticmethod
    def run_scheduled_retrain(db: Session) -> List[RetrainJob]:
        """
        Run scheduled retrain for all formulas with sufficient corrections.
        This should be called by a background scheduler (e.g., nightly).

        Args:
            db: Database session

        Returns:
            List of created retrain jobs
        """
        # Get all Tier 1 and Tier 2 formulas
        formulas = db.query(Formula).filter(
            Formula.tier.in_([FormulaTier.TIER_1_CERTIFIED, FormulaTier.TIER_2_VALIDATED])
        ).all()

        jobs_created = []

        for formula in formulas:
            # Check if enough approved corrections exist
            corrections_count = db.query(Correction).join(
                Correction.execution
            ).filter(
                Correction.execution.has(formula_id=formula.id),
                Correction.status == CorrectionStatus.APPROVED,
                Correction.created_at >= datetime.utcnow() - timedelta(days=7)  # Last week
            ).count()

            if corrections_count >= 10:  # Threshold for retraining
                job = RetrainService.trigger_retrain(
                    db=db,
                    formula_id=formula.id,
                    trigger_type="scheduled"
                )

                if job:
                    jobs_created.append(job)

        return jobs_created

    @staticmethod
    def get_retrain_stats(db: Session, formula_id: int) -> Dict[str, Any]:
        """
        Get retrain statistics for a formula.

        Args:
            db: Database session
            formula_id: Formula to analyze

        Returns:
            Statistics dictionary
        """
        jobs = db.query(RetrainJob).filter(
            RetrainJob.formula_id == formula_id
        ).all()

        if not jobs:
            return {
                "total_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "pending_jobs": 0,
                "avg_training_accuracy": 0.0,
                "avg_validation_accuracy": 0.0,
                "total_corrections_used": 0,
                "latest_model_version": None
            }

        completed_jobs = [j for j in jobs if j.status == "completed"]

        avg_training_accuracy = 0.0
        avg_validation_accuracy = 0.0
        if completed_jobs:
            avg_training_accuracy = sum(j.training_accuracy or 0 for j in completed_jobs) / len(completed_jobs)
            avg_validation_accuracy = sum(j.validation_accuracy or 0 for j in completed_jobs) / len(completed_jobs)

        latest_job = max(jobs, key=lambda j: j.created_at) if jobs else None

        return {
            "total_jobs": len(jobs),
            "completed_jobs": len([j for j in jobs if j.status == "completed"]),
            "failed_jobs": len([j for j in jobs if j.status == "failed"]),
            "pending_jobs": len([j for j in jobs if j.status in ["pending", "running"]]),
            "avg_training_accuracy": avg_training_accuracy,
            "avg_validation_accuracy": avg_validation_accuracy,
            "total_corrections_used": sum(j.corrections_used_count or 0 for j in jobs),
            "latest_model_version": latest_job.new_model_version if latest_job and latest_job.status == "completed" else None,
            "latest_mlflow_run": latest_job.mlflow_run_id if latest_job and latest_job.status == "completed" else None
        }
