"""
Edge Device models for Jetson and other edge deployments.
Supports device registration, heartbeat tracking, model deployments, and drift detection.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime, Text, Enum as SQLEnum, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
import enum

from .database import Base


class DeviceStatus(str, enum.Enum):
    """Edge device status."""
    ONLINE = "online"              # Device is actively sending heartbeats
    OFFLINE = "offline"            # Device hasn't sent heartbeat recently
    PROVISIONING = "provisioning"  # Device is being set up
    ERROR = "error"                # Device reported an error
    MAINTENANCE = "maintenance"    # Device is in maintenance mode


class DeviceType(str, enum.Enum):
    """Type of edge device."""
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER_NX = "jetson_xavier_nx"
    JETSON_AGX_XAVIER = "jetson_agx_xavier"
    JETSON_ORIN = "jetson_orin"
    RASPBERRY_PI = "raspberry_pi"
    GENERIC = "generic"


class DeploymentStatus(str, enum.Enum):
    """Model deployment status."""
    PENDING = "pending"          # Deployment scheduled
    DOWNLOADING = "downloading"  # Model being downloaded
    DEPLOYED = "deployed"        # Model successfully deployed
    ACTIVE = "active"            # Model is currently active
    FAILED = "failed"            # Deployment failed
    SUPERSEDED = "superseded"    # Replaced by newer version


class EdgeDevice(Base):
    """
    Represents an edge device (Jetson, etc.) running formulas in headless mode.
    Devices periodically send heartbeats and can receive OTA model updates.
    """
    __tablename__ = "edge_devices"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(255), unique=True, nullable=False, index=True)  # Unique device identifier (MAC, UUID, etc.)
    device_name = Column(String(255), nullable=False)
    device_type = Column(SQLEnum(DeviceType), nullable=False)
    status = Column(SQLEnum(DeviceStatus), default=DeviceStatus.PROVISIONING, nullable=False)

    # Device hardware info
    hardware_info = Column(JSON, nullable=True)  # CPU, GPU, RAM, storage info
    firmware_version = Column(String(50), nullable=True)
    os_version = Column(String(100), nullable=True)

    # Network info
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    mac_address = Column(String(17), nullable=True)
    location = Column(String(255), nullable=True)  # Physical location description

    # Operational metadata
    registered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_heartbeat_at = Column(DateTime, nullable=True, index=True)
    last_deployment_at = Column(DateTime, nullable=True)
    last_error_at = Column(DateTime, nullable=True)
    last_error_message = Column(Text, nullable=True)

    # Configuration
    heartbeat_interval_seconds = Column(Integer, default=30, nullable=False)
    auto_update_enabled = Column(Boolean, default=True, nullable=False)

    # Operational stats
    total_executions = Column(Integer, default=0, nullable=False)
    total_uptime_seconds = Column(Integer, default=0, nullable=False)

    # Metadata
    metadata = Column(JSON, nullable=True)  # Custom key-value pairs
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    heartbeats = relationship("DeviceHeartbeat", back_populates="device", cascade="all, delete-orphan")
    deployments = relationship("ModelDeployment", back_populates="device", cascade="all, delete-orphan")
    executions = relationship("FormulaExecution", back_populates="edge_device", foreign_keys="FormulaExecution.edge_device_id")

    # Indexes
    __table_args__ = (
        Index('idx_device_status_heartbeat', 'status', 'last_heartbeat_at'),
    )

    def is_online(self, timeout_seconds: int = 60) -> bool:
        """Check if device is online based on last heartbeat."""
        if not self.last_heartbeat_at:
            return False
        time_since_heartbeat = (datetime.utcnow() - self.last_heartbeat_at).total_seconds()
        return time_since_heartbeat < timeout_seconds


class DeviceHeartbeat(Base):
    """
    Records heartbeat pings from edge devices.
    Used to monitor device health and connectivity.
    """
    __tablename__ = "device_heartbeats"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("edge_devices.id", ondelete="CASCADE"), nullable=False, index=True)

    # Heartbeat data
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    status = Column(SQLEnum(DeviceStatus), nullable=False)

    # System metrics at heartbeat time
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_percent = Column(Float, nullable=True)
    disk_usage_percent = Column(Float, nullable=True)
    gpu_usage_percent = Column(Float, nullable=True)
    temperature_celsius = Column(Float, nullable=True)

    # Active model info
    active_model_id = Column(Integer, ForeignKey("formulas.id"), nullable=True)
    active_model_version = Column(String(50), nullable=True)

    # Network info
    ip_address = Column(String(45), nullable=True)

    # Additional metadata
    metadata = Column(JSON, nullable=True)

    # Relationships
    device = relationship("EdgeDevice", back_populates="heartbeats")

    # Indexes
    __table_args__ = (
        Index('idx_heartbeat_device_time', 'device_id', 'timestamp'),
    )


class ModelDeployment(Base):
    """
    Tracks model deployments to edge devices (OTA updates).
    Each deployment represents a model version pushed to a specific device.
    """
    __tablename__ = "model_deployments"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("edge_devices.id", ondelete="CASCADE"), nullable=False, index=True)
    formula_id = Column(Integer, ForeignKey("formulas.id", ondelete="CASCADE"), nullable=False, index=True)

    # Deployment metadata
    deployment_status = Column(SQLEnum(DeploymentStatus), default=DeploymentStatus.PENDING, nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_tier = Column(Integer, nullable=False)  # Tier at deployment time

    # Deployment lifecycle
    scheduled_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    activated_at = Column(DateTime, nullable=True)  # When model became active
    deactivated_at = Column(DateTime, nullable=True)  # When model was replaced

    # Deployment details
    deployment_method = Column(String(50), default="ota", nullable=False)  # ota, manual, rollback
    model_size_bytes = Column(Integer, nullable=True)
    download_duration_seconds = Column(Float, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)

    # Performance after deployment
    executions_count = Column(Integer, default=0, nullable=False)
    avg_execution_time_ms = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=True)  # Percentage

    # Metadata
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    device = relationship("EdgeDevice", back_populates="deployments")
    formula = relationship("Formula", foreign_keys=[formula_id])

    # Indexes
    __table_args__ = (
        Index('idx_deployment_device_status', 'device_id', 'deployment_status'),
        Index('idx_deployment_formula_device', 'formula_id', 'device_id'),
    )


class DriftMetric(Base):
    """
    Tracks drift detection metrics for models deployed on edge devices.
    Uses ADWIN (Adaptive Windowing) algorithm to detect concept drift.
    """
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, index=True)
    formula_id = Column(Integer, ForeignKey("formulas.id", ondelete="CASCADE"), nullable=False, index=True)
    device_id = Column(Integer, ForeignKey("edge_devices.id", ondelete="CASCADE"), nullable=True, index=True)

    # Time window
    window_start = Column(DateTime, nullable=False, index=True)
    window_end = Column(DateTime, nullable=False, index=True)

    # Drift metrics
    correction_rate = Column(Float, nullable=False)  # Percentage of executions corrected
    executions_count = Column(Integer, nullable=False)
    corrections_count = Column(Integer, nullable=False)

    # ADWIN drift detection
    drift_detected = Column(Boolean, default=False, nullable=False, index=True)
    drift_score = Column(Float, nullable=True)  # Confidence score for drift
    drift_threshold = Column(Float, default=0.05, nullable=False)  # Detection threshold

    # Performance metrics
    avg_confidence = Column(Float, nullable=True)
    avg_execution_time_ms = Column(Float, nullable=True)
    error_rate = Column(Float, nullable=True)

    # Statistical data
    baseline_correction_rate = Column(Float, nullable=True)  # Historical baseline
    correction_rate_change = Column(Float, nullable=True)  # Change from baseline

    # Action taken
    retrain_triggered = Column(Boolean, default=False, nullable=False)
    retrain_job_id = Column(String(255), nullable=True)  # Reference to retrain job

    # Metadata
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    formula = relationship("Formula", foreign_keys=[formula_id])
    device = relationship("EdgeDevice", foreign_keys=[device_id])

    # Indexes
    __table_args__ = (
        Index('idx_drift_formula_time', 'formula_id', 'window_start'),
        Index('idx_drift_detected', 'drift_detected', 'retrain_triggered'),
    )


class RetrainJob(Base):
    """
    Tracks auto-retrain jobs triggered by drift detection or scheduled retraining.
    Jobs query approved corrections and fine-tune models using MLflow.
    """
    __tablename__ = "retrain_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), unique=True, nullable=False, index=True)
    formula_id = Column(Integer, ForeignKey("formulas.id", ondelete="CASCADE"), nullable=False, index=True)

    # Job trigger
    trigger_type = Column(String(50), nullable=False)  # scheduled, drift_detected, manual
    triggered_by_drift_id = Column(Integer, ForeignKey("drift_metrics.id"), nullable=True)
    triggered_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Job status
    status = Column(String(50), default="pending", nullable=False, index=True)  # pending, running, completed, failed

    # Training data
    corrections_used_count = Column(Integer, nullable=True)
    training_samples_count = Column(Integer, nullable=True)
    validation_samples_count = Column(Integer, nullable=True)

    # Training results
    new_model_id = Column(Integer, ForeignKey("formulas.id"), nullable=True)  # New Tier 4 formula
    new_model_version = Column(String(50), nullable=True)
    mlflow_run_id = Column(String(255), nullable=True)  # MLflow tracking

    # Performance metrics
    training_accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    training_loss = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)

    # Timing
    scheduled_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Metadata
    config = Column(JSON, nullable=True)  # Training configuration
    metrics = Column(JSON, nullable=True)  # Additional metrics
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    formula = relationship("Formula", foreign_keys=[formula_id])
    new_model = relationship("Formula", foreign_keys=[new_model_id])
    drift_metric = relationship("DriftMetric", foreign_keys=[triggered_by_drift_id])
    triggered_by_user = relationship("User", foreign_keys=[triggered_by_user_id])

    # Indexes
    __table_args__ = (
        Index('idx_retrain_formula_status', 'formula_id', 'status'),
        Index('idx_retrain_trigger', 'trigger_type', 'created_at'),
    )
