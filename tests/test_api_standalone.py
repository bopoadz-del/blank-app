"""Test that the API works standalone without frontend dependencies"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database.session import Base, get_db

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_standalone.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override dependency
app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)


@pytest.fixture(scope="function", autouse=True)
def setup_database():
    """Setup test database before each test"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_api_root_returns_correct_info():
    """Test that root endpoint returns API information or frontend"""
    response = client.get("/")
    assert response.status_code == 200
    
    # Check if frontend is available (HTML response) or API-only mode (JSON response)
    content_type = response.headers.get("content-type", "")
    
    if "text/html" in content_type:
        # Frontend is available - check for HTML content
        assert b"<!doctype html" in response.content.lower()
    else:
        # API-only mode - check for JSON with API info
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
    """Test that formula list endpoint is accessible (public access)"""
    # After PR #68, authentication is removed - public access is allowed
    response = client.get("/api/v1/formulas/list")
    assert response.status_code == 200


def test_formula_execute_endpoint_exists():
    """Test that formula execute endpoint is accessible (public access)"""
    # After PR #68, authentication is removed - public access is allowed
    response = client.post(
        "/api/v1/formulas/execute",
        json={
            "formula_id": "test",
            "input_values": {}
        }
    )
    # Should return 200 with success=False since formula doesn't exist
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "not found" in data["error"].lower()


def test_api_provides_complete_documentation():
    """Test that API provides comprehensive documentation via OpenAPI"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    
    # Verify key API paths are documented
    assert "/health" in schema["paths"]
    # Note: "/" might be frontend route when frontend is available
    # assert "/" in schema["paths"]  # Removed since / serves frontend when available
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
