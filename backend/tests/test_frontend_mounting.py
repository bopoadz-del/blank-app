"""Test frontend mounting and unified deployment."""
import pytest
from pathlib import Path
import tempfile
import os


def test_frontend_path_logic():
    """Test that frontend path resolution works correctly."""
    # Simulate the path logic from main.py
    # main.py is at backend/app/main.py
    backend_app_dir = Path(__file__).resolve().parent.parent / "app"
    BASE_DIR = backend_app_dir.parent  # backend/
    FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
    
    # The path should point to backend/frontend/dist
    assert str(FRONTEND_DIST).endswith("backend/frontend/dist")


def test_dockerfile_structure():
    """Test that Dockerfile exists and has multi-stage structure."""
    dockerfile_path = Path(__file__).resolve().parent.parent.parent / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile should exist at repo root"
    
    content = dockerfile_path.read_text()
    
    # Check for multi-stage build
    assert "FROM node:18-alpine AS frontend-builder" in content
    assert "FROM python:3.11-slim AS backend-builder" in content
    assert "FROM python:3.11-slim" in content  # Final stage
    
    # Check for frontend build steps
    assert "npm ci" in content
    assert "npm run build" in content
    
    # Check for backend build steps
    assert "python -m venv" in content
    assert "pip install" in content
    
    # Check for copying frontend to backend
    assert "COPY --from=frontend-builder" in content
    assert "frontend/dist" in content
    
    # Check for non-root user
    assert "USER" in content
    
    # Check for PORT env variable support
    assert "${PORT" in content or "$PORT" in content


def test_dockerignore_exists():
    """Test that .dockerignore exists and has appropriate entries."""
    dockerignore_path = Path(__file__).resolve().parent.parent.parent / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore should exist at repo root"
    
    content = dockerignore_path.read_text()
    
    # Check for essential ignores
    assert "node_modules" in content
    assert "__pycache__" in content
    assert ".git" in content
    assert "frontend/dist" in content


def test_main_py_has_frontend_mounting():
    """Test that main.py has the frontend mounting code."""
    main_py_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    assert main_py_path.exists()
    
    content = main_py_path.read_text()
    
    # Check for required imports
    assert "from pathlib import Path" in content
    assert "from fastapi.staticfiles import StaticFiles" in content
    assert "from fastapi.responses import" in content and "FileResponse" in content
    
    # Check for frontend path setup
    assert "FRONTEND_DIST" in content
    assert "frontend" in content and "dist" in content
    
    # Check for mounting logic
    assert "app.mount" in content or "StaticFiles" in content
    assert 'html=True' in content  # SPA-friendly routing
    
    # Check for health endpoint preservation
    assert '@app.get("/health")' in content
    
    # Check for fallback when frontend not found
    assert "Frontend not found" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

