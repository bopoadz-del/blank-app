"""End-to-end integration tests"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database.session import Base, get_db
from app.models.formula_execution import FormulaExecution

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
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

# Test API key
TEST_API_KEY = "test-api-key-12345"


@pytest.fixture(scope="function", autouse=True)
def setup_database():
    """Setup test database before each test"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_end_to_end_formula_execution():
    """Test complete formula execution flow"""
    # 1. Execute formula
    response = client.post(
        "/api/v1/formulas/execute",
        headers={"X-API-Key": TEST_API_KEY},
        json={
            "formula_id": "beam_deflection_simply_supported",
            "input_values": {
                "w": 10,
                "L": 5,
                "E": 200,
                "I": 0.0001
            }
        }
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert data["success"] is True
    assert data["formula_id"] == "beam_deflection_simply_supported"
    assert data["result"] is not None
    assert data["unit"] == "m"
    assert "execution_id" in data
    assert data["execution_time_ms"] > 0

    execution_id = data["execution_id"]

    # 2. Verify database write
    db = TestingSessionLocal()
    execution = db.query(FormulaExecution).filter_by(id=execution_id).first()

    assert execution is not None
    assert execution.formula_id == "beam_deflection_simply_supported"
    assert execution.success is True
    assert execution.result is not None
    assert execution.unit == "m"

    db.close()

    # 3. Get recent executions
    response = client.get(
        "/api/v1/formulas/history/recent",
        headers={"X-API-Key": TEST_API_KEY},
        params={"limit": 5}
    )

    assert response.status_code == 200
    history = response.json()
    assert len(history) > 0
    assert history[0]["id"] == execution_id


def test_unit_conversion():
    """Test unit conversion functionality"""
    # Execute with unit conversion
    response = client.post(
        "/api/v1/formulas/execute",
        headers={"X-API-Key": TEST_API_KEY},
        json={
            "formula_id": "beam_deflection_simply_supported",
            "input_values": {
                "w": 10,
                "L": 5,
                "E": 200,
                "I": 0.0001
            },
            "convert_to_unit": "mm"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["unit"] == "mm"
    assert data["original_unit"] == "m"

    # Verify conversion (m to mm = *1000)
    assert data["result"] > 0.5  # Should be ~0.65104 mm


def test_error_scenarios():
    """Test various error scenarios"""

    # 1. Test invalid formula ID
    response = client.post(
        "/api/v1/formulas/execute",
        headers={"X-API-Key": TEST_API_KEY},
        json={
            "formula_id": "nonexistent_formula",
            "input_values": {"x": 10}
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "not found" in data["error"].lower()

    # 2. Test missing parameters
    response = client.post(
        "/api/v1/formulas/execute",
        headers={"X-API-Key": TEST_API_KEY},
        json={
            "formula_id": "beam_deflection_simply_supported",
            "input_values": {"w": 10, "L": 5}  # Missing E and I
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "missing parameters" in data["error"].lower()

    # 3. Test invalid unit conversion
    response = client.post(
        "/api/v1/formulas/execute",
        headers={"X-API-Key": TEST_API_KEY},
        json={
            "formula_id": "beam_deflection_simply_supported",
            "input_values": {"w": 10, "L": 5, "E": 200, "I": 0.0001},
            "convert_to_unit": "Pa"  # Incompatible unit (pressure vs length)
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "incompatible" in data["error"].lower() or "cannot convert" in data["error"].lower()

    # 4. Test unauthenticated access - authentication is not required for public endpoints
    response = client.post(
        "/api/v1/formulas/execute",
        json={
            "formula_id": "beam_deflection_simply_supported",
            "input_values": {"w": 10, "L": 5, "E": 200, "I": 0.0001}
        }
    )

    # After PR #68, authentication is removed - public access is allowed
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_database_persistence():
    """Test that executions are persisted correctly"""

    # Execute multiple formulas
    formulas_to_test = [
        ("beam_deflection_simply_supported", {"w": 10, "L": 5, "E": 200, "I": 0.0001}),
        ("spring_deflection", {"F": 100, "k": 1000}),
        ("reynolds_number", {"rho": 1000, "v": 2, "L": 0.5, "mu": 0.001})
    ]

    execution_ids = []

    for formula_id, input_values in formulas_to_test:
        response = client.post(
            "/api/v1/formulas/execute",
            headers={"X-API-Key": TEST_API_KEY},
            json={
                "formula_id": formula_id,
                "input_values": input_values
            }
        )

        assert response.status_code == 200
        data = response.json()
        execution_ids.append(data["execution_id"])

    # Verify all executions are in database
    db = TestingSessionLocal()

    for exec_id, (formula_id, _) in zip(execution_ids, formulas_to_test):
        execution = db.query(FormulaExecution).filter_by(id=exec_id).first()
        assert execution is not None
        assert execution.formula_id == formula_id
        assert execution.success is True

    # Verify count
    total_count = db.query(FormulaExecution).count()
    assert total_count == len(formulas_to_test)

    db.close()


def test_formula_list_and_info():
    """Test formula listing and info endpoints"""

    # 1. List all formulas
    response = client.get(
        "/api/v1/formulas/list",
        headers={"X-API-Key": TEST_API_KEY}
    )

    assert response.status_code == 200
    formulas = response.json()
    assert len(formulas) >= 8  # We have 8 formulas

    # 2. Get info for specific formula
    response = client.get(
        "/api/v1/formulas/beam_deflection_simply_supported",
        headers={"X-API-Key": TEST_API_KEY}
    )

    assert response.status_code == 200
    info = response.json()
    assert info["formula_id"] == "beam_deflection_simply_supported"
    assert "parameters" in info
    assert "w" in info["parameters"]
    assert "L" in info["parameters"]

    # 3. Test non-existent formula
    response = client.get(
        "/api/v1/formulas/nonexistent",
        headers={"X-API-Key": TEST_API_KEY}
    )

    assert response.status_code == 404


def test_multiple_unit_conversions():
    """Test various unit conversions"""

    test_cases = [
        # (formula_id, inputs, target_unit, should_succeed)
        ("beam_deflection_simply_supported", {"w": 10, "L": 5, "E": 200, "I": 0.0001}, "mm", True),
        ("beam_deflection_simply_supported", {"w": 10, "L": 5, "E": 200, "I": 0.0001}, "cm", True),
        ("beam_stress", {"M": 1000, "c": 0.05, "I": 0.0001}, "MPa", True),
        ("beam_stress", {"M": 1000, "c": 0.05, "I": 0.0001}, "GPa", True),
    ]

    for formula_id, inputs, target_unit, should_succeed in test_cases:
        response = client.post(
            "/api/v1/formulas/execute",
            headers={"X-API-Key": TEST_API_KEY},
            json={
                "formula_id": formula_id,
                "input_values": inputs,
                "convert_to_unit": target_unit
            }
        )

        assert response.status_code == 200
        data = response.json()

        if should_succeed:
            assert data["success"] is True
            assert data["unit"] == target_unit
            assert data["original_unit"] is not None
        else:
            assert data["success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
