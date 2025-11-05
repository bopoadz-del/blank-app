"""Tests for the FastAPI application"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test API key
TEST_API_KEY = "test-api-key"


def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "docs" in data


def test_execute_formula_without_api_key():
    """Test formula execution without API key"""
    response = client.post(
        "/api/v1/formulas/execute",
        json={
            "formula_id": "beam_deflection_simply_supported",
            "input_values": {"w": 10, "L": 5, "E": 200, "I": 0.0001}
        }
    )
    assert response.status_code == 403


def test_list_formulas_without_api_key():
    """Test listing formulas without API key"""
    response = client.get("/api/v1/formulas/list")
    assert response.status_code == 403


def test_formula_service_beam_deflection():
    """Test beam deflection formula calculation"""
    from app.services.formula_service import formula_service

    result, unit = formula_service.execute(
        "beam_deflection_simply_supported",
        {"w": 10, "L": 5, "E": 200, "I": 0.0001}
    )

    assert result > 0
    assert unit == "m"
    # Expect result from implemented formula (5*w*L^4)/(384*E*1e9*I)
    expected = (5 * 10 * 5**4) / (384 * 200 * 1e9 * 0.0001)
    assert abs(result - expected) < 1e-12


def test_formula_service_invalid_formula():
    """Test formula service with invalid formula ID"""
    from app.services.formula_service import formula_service

    with pytest.raises(ValueError, match="Formula .* not found"):
        formula_service.execute(
            "invalid_formula",
            {"w": 10}
        )


def test_formula_service_missing_parameters():
    """Test formula service with missing parameters"""
    from app.services.formula_service import formula_service

    with pytest.raises(ValueError, match="Missing parameters"):
        formula_service.execute(
            "beam_deflection_simply_supported",
            {"w": 10, "L": 5}  # Missing E and I
        )


def test_formula_service_list_formulas():
    """Test listing all formulas"""
    from app.services.formula_service import formula_service

    formulas = formula_service.list_formulas()

    assert len(formulas) > 0
    assert all("formula_id" in f for f in formulas)
    assert all("name" in f for f in formulas)
    assert all("parameters" in f for f in formulas)


def test_formula_service_get_info():
    """Test getting formula information"""
    from app.services.formula_service import formula_service

    info = formula_service.get_formula_info("beam_deflection_simply_supported")

    assert info["formula_id"] == "beam_deflection_simply_supported"
    assert "name" in info
    assert "description" in info
    assert "parameters" in info
    assert "unit" in info


def test_reynolds_number_formula():
    """Test Reynolds number calculation"""
    from app.services.formula_service import formula_service

    result, unit = formula_service.execute(
        "reynolds_number",
        {"rho": 1000, "v": 2, "L": 0.5, "mu": 0.001}
    )

    assert result == 1000000
    assert unit == "dimensionless"


def test_spring_deflection_formula():
    """Test spring deflection calculation"""
    from app.services.formula_service import formula_service

    result, unit = formula_service.execute(
        "spring_deflection",
        {"F": 100, "k": 1000}
    )

    assert result == 0.1
    assert unit == "m"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
