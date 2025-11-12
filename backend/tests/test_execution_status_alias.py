"""Test ExecutionStatus backwards-compatibility alias."""
import pytest


def test_execution_status_import_with_enum_suffix():
    """Test that ExecutionStatusEnum can be imported."""
    from app.models.schemas import ExecutionStatusEnum
    
    assert ExecutionStatusEnum.QUEUED == "queued"
    assert ExecutionStatusEnum.RUNNING == "running"
    assert ExecutionStatusEnum.COMPLETED == "completed"
    assert ExecutionStatusEnum.FAILED == "failed"
    assert ExecutionStatusEnum.TIMEOUT == "timeout"


def test_execution_status_import_without_enum_suffix():
    """Test that ExecutionStatus alias can be imported."""
    from app.models.schemas import ExecutionStatus
    
    assert ExecutionStatus.QUEUED == "queued"
    assert ExecutionStatus.RUNNING == "running"
    assert ExecutionStatus.COMPLETED == "completed"
    assert ExecutionStatus.FAILED == "failed"
    assert ExecutionStatus.TIMEOUT == "timeout"


def test_execution_status_alias_is_same_as_enum():
    """Test that ExecutionStatus is an alias for ExecutionStatusEnum."""
    from app.models.schemas import ExecutionStatus, ExecutionStatusEnum
    
    # They should be the exact same object
    assert ExecutionStatus is ExecutionStatusEnum
    
    # They should have the same values
    assert ExecutionStatus.QUEUED == ExecutionStatusEnum.QUEUED
    assert ExecutionStatus.RUNNING == ExecutionStatusEnum.RUNNING
    assert ExecutionStatus.COMPLETED == ExecutionStatusEnum.COMPLETED
    assert ExecutionStatus.FAILED == ExecutionStatusEnum.FAILED
    assert ExecutionStatus.TIMEOUT == ExecutionStatusEnum.TIMEOUT


def test_execution_status_can_be_used_in_type_hints():
    """Test that ExecutionStatus works in type hints and isinstance checks."""
    from app.models.schemas import ExecutionStatus, ExecutionStatusEnum
    
    # Should work with both names
    def check_status_enum(status: ExecutionStatusEnum) -> bool:
        return status in ExecutionStatusEnum
    
    def check_status_alias(status: ExecutionStatus) -> bool:
        return status in ExecutionStatus
    
    # Both should work with values from either
    status = ExecutionStatusEnum.QUEUED
    assert check_status_enum(status)
    assert check_status_alias(status)
    
    status = ExecutionStatus.RUNNING
    assert check_status_enum(status)
    assert check_status_alias(status)
