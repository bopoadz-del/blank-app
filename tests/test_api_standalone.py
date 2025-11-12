"""Test that the API works standalone without frontend dependencies"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_api_root_returns_correct_info():
    """Test that root endpoint returns API information without requiring frontend"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["name"] == "Formula Execution API"
    assert data["version"] == "1.0.0"
    assert data["environment"] == "development"
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


def test_swagger_docs_accessible():
    """Test that Swagger UI documentation is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200
    assert b"swagger-ui" in response.content.lower()


def test_openapi_json_accessible():
    """Test that OpenAPI JSON schema is accessible"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    data = response.json()
    assert "info" in data
    assert data["info"]["title"] == "Formula Execution API"
    assert data["info"]["version"] == "1.0.0"


def test_health_endpoint_no_frontend_dependency():
    """Test health endpoint works without frontend"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["environment"] == "development"
    assert "timestamp" in data


def test_api_works_without_redis():
    """Test that API functions without Redis (uses in-memory fallback)"""
    # This test verifies the API starts and runs without Redis
    # The TestClient automatically handles the lifespan events
    response = client.get("/health")
    assert response.status_code == 200


def test_formula_list_endpoint_exists():
    """Test that formula list endpoint is accessible (requires auth)"""
    # Without API key, should return 403
    response = client.get("/api/v1/formulas/list")
    assert response.status_code == 403


def test_formula_execute_endpoint_exists():
    """Test that formula execute endpoint is accessible (requires auth)"""
    # Without API key, should return 403
    response = client.post(
        "/api/v1/formulas/execute",
        json={
            "formula_id": "test",
            "input_values": {}
        }
    )
    assert response.status_code == 403


def test_api_provides_complete_documentation():
    """Test that API provides comprehensive documentation via OpenAPI"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    
    # Verify key API paths are documented
    assert "/health" in schema["paths"]
    assert "/" in schema["paths"]
    assert "/api/v1/formulas/execute" in schema["paths"]
    assert "/api/v1/formulas/list" in schema["paths"]
    
    # Verify schemas are documented
    assert "components" in schema
    assert "schemas" in schema["components"]


def test_api_is_self_sufficient():
    """Test that API is self-sufficient and doesn't require external UI"""
    # Test 1: API info is available
    root_response = client.get("/")
    assert root_response.status_code == 200
    
    # Test 2: Health check works
    health_response = client.get("/health")
    assert health_response.status_code == 200
    
    # Test 3: Documentation is self-hosted
    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
    
    # Test 4: OpenAPI schema is available
    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    
    # All core functionality works without external dependencies
    assert True  # All assertions passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
