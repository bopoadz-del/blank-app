"""
Edge Device API routes.
Handles device registration, heartbeat tracking, and OTA updates.
"""
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ..core.database import get_db
from ..models.edge_devices import (
    EdgeDevice, DeviceHeartbeat, ModelDeployment,
    DeviceStatus, DeviceType, DeploymentStatus
)
from ..models.database import Formula, FormulaTier
from ..services.audit_service import AuditService
from ..core.rbac import get_current_admin


router = APIRouter(prefix="/api/v1/edge-devices", tags=["Edge Devices"])


# Pydantic Schemas
class DeviceRegistration(BaseModel):
    device_id: str = Field(..., description="Unique device identifier (MAC, UUID, etc.)")
    device_name: str
    device_type: str
    hardware_info: Optional[dict] = None
    firmware_version: Optional[str] = None
    os_version: Optional[str] = None
    mac_address: Optional[str] = None
    location: Optional[str] = None


class DeviceHeartbeatData(BaseModel):
    status: str = "online"
    cpu_usage_percent: Optional[float] = None
    memory_usage_percent: Optional[float] = None
    disk_usage_percent: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    temperature_celsius: Optional[float] = None
    active_model_version: Optional[str] = None
    ip_address: Optional[str] = None
    metadata: Optional[dict] = None


class DeviceResponse(BaseModel):
    id: int
    device_id: str
    device_name: str
    device_type: str
    status: str
    last_heartbeat_at: Optional[datetime] = None
    registered_at: datetime
    total_executions: int
    is_online: bool

    class Config:
        from_attributes = True


class HeartbeatResponse(BaseModel):
    success: bool
    message: str
    pending_updates: bool
    available_deployment: Optional[dict] = None


class OTAUpdateCheck(BaseModel):
    current_model_version: Optional[str] = None
    current_tier: Optional[int] = None


class DeploymentResponse(BaseModel):
    id: int
    formula_id: int
    model_version: str
    model_tier: int
    deployment_status: str
    scheduled_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Routes

@router.post("/register", response_model=DeviceResponse, status_code=status.HTTP_201_CREATED)
async def register_device(
    device_data: DeviceRegistration,
    db: Session = Depends(get_db)
):
    """
    Register a new edge device (Jetson, etc.)
    Devices must register before they can receive heartbeats and deployments.
    """
    # Check if device already registered
    existing_device = db.query(EdgeDevice).filter(
        EdgeDevice.device_id == device_data.device_id
    ).first()

    if existing_device:
        # Update existing device
        existing_device.device_name = device_data.device_name
        existing_device.device_type = DeviceType(device_data.device_type)
        existing_device.hardware_info = device_data.hardware_info
        existing_device.firmware_version = device_data.firmware_version
        existing_device.os_version = device_data.os_version
        existing_device.mac_address = device_data.mac_address
        existing_device.location = device_data.location
        existing_device.status = DeviceStatus.ONLINE
        existing_device.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(existing_device)

        return {
            **existing_device.__dict__,
            "is_online": existing_device.is_online()
        }

    # Create new device
    new_device = EdgeDevice(
        device_id=device_data.device_id,
        device_name=device_data.device_name,
        device_type=DeviceType(device_data.device_type),
        hardware_info=device_data.hardware_info,
        firmware_version=device_data.firmware_version,
        os_version=device_data.os_version,
        mac_address=device_data.mac_address,
        location=device_data.location,
        status=DeviceStatus.ONLINE
    )

    db.add(new_device)
    db.commit()
    db.refresh(new_device)

    # Create audit log
    AuditService.log_action(
        db=db,
        action="device_registered",
        entity_type="edge_device",
        entity_id=new_device.id,
        description=f"Edge device {new_device.device_name} registered",
        after_state={
            "device_id": new_device.device_id,
            "device_type": new_device.device_type.value
        }
    )

    return {
        **new_device.__dict__,
        "is_online": new_device.is_online()
    }


@router.post("/{device_id}/heartbeat", response_model=HeartbeatResponse)
async def send_heartbeat(
    device_id: str,
    heartbeat_data: DeviceHeartbeatData,
    db: Session = Depends(get_db)
):
    """
    Send heartbeat from edge device.
    Devices should send heartbeats every 30 seconds to indicate they're online.
    Returns information about pending OTA updates.
    """
    # Find device
    device = db.query(EdgeDevice).filter(EdgeDevice.device_id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device {device_id} not found. Please register first."
        )

    # Update device status
    device.last_heartbeat_at = datetime.utcnow()
    device.status = DeviceStatus(heartbeat_data.status)
    if heartbeat_data.ip_address:
        device.ip_address = heartbeat_data.ip_address

    # Create heartbeat record
    heartbeat = DeviceHeartbeat(
        device_id=device.id,
        timestamp=datetime.utcnow(),
        status=DeviceStatus(heartbeat_data.status),
        cpu_usage_percent=heartbeat_data.cpu_usage_percent,
        memory_usage_percent=heartbeat_data.memory_usage_percent,
        disk_usage_percent=heartbeat_data.disk_usage_percent,
        gpu_usage_percent=heartbeat_data.gpu_usage_percent,
        temperature_celsius=heartbeat_data.temperature_celsius,
        active_model_version=heartbeat_data.active_model_version,
        ip_address=heartbeat_data.ip_address,
        metadata=heartbeat_data.metadata
    )

    db.add(heartbeat)
    db.commit()

    # Check for pending deployments (OTA updates)
    pending_deployment = db.query(ModelDeployment).filter(
        ModelDeployment.device_id == device.id,
        ModelDeployment.deployment_status == DeploymentStatus.PENDING
    ).first()

    response = {
        "success": True,
        "message": "Heartbeat received",
        "pending_updates": pending_deployment is not None,
        "available_deployment": None
    }

    if pending_deployment:
        response["available_deployment"] = {
            "deployment_id": pending_deployment.id,
            "formula_id": pending_deployment.formula_id,
            "model_version": pending_deployment.model_version,
            "model_tier": pending_deployment.model_tier,
            "download_url": f"/api/v1/formulas/{pending_deployment.formula_id}/download"
        }

    return response


@router.get("", response_model=List[DeviceResponse])
async def list_devices(
    status_filter: Optional[str] = None,
    device_type: Optional[str] = None,
    limit: int = 100,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    List all registered edge devices (admin only).
    """
    query = db.query(EdgeDevice)

    if status_filter:
        query = query.filter(EdgeDevice.status == DeviceStatus(status_filter))
    if device_type:
        query = query.filter(EdgeDevice.device_type == DeviceType(device_type))

    devices = query.order_by(EdgeDevice.last_heartbeat_at.desc()).limit(limit).all()

    return [
        {
            **device.__dict__,
            "is_online": device.is_online()
        }
        for device in devices
    ]


@router.get("/{device_id}", response_model=DeviceResponse)
async def get_device(
    device_id: str,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific edge device (admin only).
    """
    device = db.query(EdgeDevice).filter(EdgeDevice.device_id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device {device_id} not found"
        )

    return {
        **device.__dict__,
        "is_online": device.is_online()
    }


@router.get("/{device_id}/deployments", response_model=List[DeploymentResponse])
async def get_device_deployments(
    device_id: str,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Get all model deployments for a specific device (admin only).
    """
    device = db.query(EdgeDevice).filter(EdgeDevice.device_id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device {device_id} not found"
        )

    deployments = db.query(ModelDeployment).filter(
        ModelDeployment.device_id == device.id
    ).order_by(ModelDeployment.scheduled_at.desc()).all()

    return deployments


@router.post("/{device_id}/check-updates")
async def check_for_updates(
    device_id: str,
    update_check: OTAUpdateCheck,
    db: Session = Depends(get_db)
):
    """
    Check for available OTA updates for this device.
    Returns the latest Tier 1 certified formula if different from current version.
    """
    device = db.query(EdgeDevice).filter(EdgeDevice.device_id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device {device_id} not found"
        )

    if not device.auto_update_enabled:
        return {
            "update_available": False,
            "message": "Auto-update is disabled for this device"
        }

    # Find latest Tier 1 certified formula
    latest_tier1 = db.query(Formula).filter(
        Formula.tier == FormulaTier.TIER_1_CERTIFIED,
        Formula.is_locked == True
    ).order_by(Formula.updated_at.desc()).first()

    if not latest_tier1:
        return {
            "update_available": False,
            "message": "No Tier 1 certified formulas available"
        }

    # Check if device already has this version
    if update_check.current_model_version == latest_tier1.version:
        return {
            "update_available": False,
            "message": "Device is already on the latest version"
        }

    # Check if deployment already scheduled
    existing_deployment = db.query(ModelDeployment).filter(
        ModelDeployment.device_id == device.id,
        ModelDeployment.formula_id == latest_tier1.id,
        ModelDeployment.deployment_status.in_([DeploymentStatus.PENDING, DeploymentStatus.DOWNLOADING])
    ).first()

    if existing_deployment:
        return {
            "update_available": True,
            "deployment_id": existing_deployment.id,
            "formula_id": latest_tier1.id,
            "model_version": latest_tier1.version,
            "model_tier": latest_tier1.tier.value,
            "download_url": f"/api/v1/formulas/{latest_tier1.id}/download",
            "message": "Update already scheduled"
        }

    # Create new deployment
    new_deployment = ModelDeployment(
        device_id=device.id,
        formula_id=latest_tier1.id,
        deployment_status=DeploymentStatus.PENDING,
        model_version=latest_tier1.version,
        model_tier=latest_tier1.tier.value,
        deployment_method="ota"
    )

    db.add(new_deployment)
    db.commit()
    db.refresh(new_deployment)

    # Create audit log
    AuditService.log_action(
        db=db,
        action="deployment_scheduled",
        entity_type="model_deployment",
        entity_id=new_deployment.id,
        description=f"OTA deployment scheduled for device {device.device_name}",
        after_state={
            "formula_id": latest_tier1.id,
            "model_version": latest_tier1.version
        }
    )

    return {
        "update_available": True,
        "deployment_id": new_deployment.id,
        "formula_id": latest_tier1.id,
        "model_version": latest_tier1.version,
        "model_tier": latest_tier1.tier.value,
        "download_url": f"/api/v1/formulas/{latest_tier1.id}/download",
        "message": "New update available"
    }


@router.post("/{device_id}/deployments/{deployment_id}/confirm")
async def confirm_deployment(
    device_id: str,
    deployment_id: int,
    success: bool,
    error_message: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Confirm deployment completion from edge device.
    Device calls this after downloading and activating the new model.
    """
    device = db.query(EdgeDevice).filter(EdgeDevice.device_id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device {device_id} not found"
        )

    deployment = db.query(ModelDeployment).filter(
        ModelDeployment.id == deployment_id,
        ModelDeployment.device_id == device.id
    ).first()

    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )

    if success:
        deployment.deployment_status = DeploymentStatus.DEPLOYED
        deployment.completed_at = datetime.utcnow()
        deployment.activated_at = datetime.utcnow()
        device.last_deployment_at = datetime.utcnow()

        # Deactivate other deployments for this device
        db.query(ModelDeployment).filter(
            ModelDeployment.device_id == device.id,
            ModelDeployment.id != deployment_id,
            ModelDeployment.deployment_status == DeploymentStatus.ACTIVE
        ).update({"deployment_status": DeploymentStatus.SUPERSEDED})

        deployment.deployment_status = DeploymentStatus.ACTIVE

        message = "Deployment confirmed and activated"
    else:
        deployment.deployment_status = DeploymentStatus.FAILED
        deployment.error_message = error_message
        deployment.retry_count += 1
        device.last_error_at = datetime.utcnow()
        device.last_error_message = error_message

        message = "Deployment failed"

    db.commit()

    # Create audit log
    AuditService.log_action(
        db=db,
        action="deployment_confirmed",
        entity_type="model_deployment",
        entity_id=deployment.id,
        description=f"Deployment {deployment.id} {'succeeded' if success else 'failed'} on device {device.device_name}",
        after_state={
            "status": deployment.deployment_status.value,
            "error": error_message
        }
    )

    return {
        "success": True,
        "message": message
    }
