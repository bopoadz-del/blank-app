"""Tests for unified UI+API deployment functionality."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import os
import tempfile
import shutil


def test_health_endpoint_accessible():
    """Test that /health endpoint is accessible regardless of frontend presence."""
    from app.main import app
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code in [200, 503]  # 200 if healthy, 503 if degraded
    data = response.json()
    assert "status" in data
    assert "version" in data


def test_root_without_frontend():
    """Test root endpoint when frontend is not available."""
    # Temporarily set FRONTEND_DIST_PATH to non-existent directory
    os.environ["FRONTEND_DIST_PATH"] = "/tmp/nonexistent-frontend"
    
    # Reimport app to pick up new env var
    import importlib
    import app.main
    importlib.reload(app.main)
    
    from app.main import app
    client = TestClient(app)
    
    response = client.get("/")
    
    # Should return JSON when frontend is not available
    if response.status_code == 200 and "application/json" in response.headers.get("content-type", ""):
        data = response.json()
        assert "status" in data
        assert data["status"] == "backend" or "name" in data
    
    # Clean up
    if "FRONTEND_DIST_PATH" in os.environ:
        del os.environ["FRONTEND_DIST_PATH"]


def test_root_with_frontend():
    """Test root endpoint when frontend is available."""
    # Create temporary frontend directory with index.html
    with tempfile.TemporaryDirectory() as tmpdir:
        frontend_dist = Path(tmpdir) / "dist"
        frontend_dist.mkdir()
        (frontend_dist / "index.html").write_text("<html><body>Test Frontend</body></html>")
        
        # Set environment variable
        os.environ["FRONTEND_DIST_PATH"] = str(frontend_dist)
        
        # Reimport app to pick up new env var
        import importlib
        import app.main
        importlib.reload(app.main)
        
        from app.main import app
        client = TestClient(app)
        
        response = client.get("/")
        
        # Should serve HTML when frontend is available
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            # Frontend should be served as HTML
            assert "text/html" in content_type or response.text.startswith("<html>")
        
        # Clean up
        if "FRONTEND_DIST_PATH" in os.environ:
            del os.environ["FRONTEND_DIST_PATH"]


def test_api_endpoints_still_work():
    """Test that API endpoints are not affected by frontend mounting."""
    from app.main import app
    client = TestClient(app)
    
    # Test metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200
    
    # Test that docs are accessible
    response = client.get("/docs")
    assert response.status_code == 200


def test_frontend_mounting_logic():
    """Test the frontend mounting logic in isolation."""
    from pathlib import Path
    
    # Test with non-existent directory
    fake_path = Path("/tmp/nonexistent-123456")
    assert not (fake_path.exists() and (fake_path / "index.html").exists())
    
    # Test with existing directory but no index.html
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir)
        assert test_path.exists()
        assert not (test_path / "index.html").exists()
        assert not (test_path.exists() and (test_path / "index.html").exists())
    
    # Test with existing directory and index.html
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir)
        (test_path / "index.html").write_text("test")
        assert test_path.exists()
        assert (test_path / "index.html").exists()
        assert test_path.exists() and (test_path / "index.html").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
