"""Tests for unit conversion"""

import pytest
from app.utils.unit_converter import unit_converter


def test_length_conversions():
    """Test length unit conversions"""
    # Meters to millimeters
    result, unit = unit_converter.convert(1.0, "m", "mm")
    assert result == 1000.0
    assert unit == "mm"

    # Meters to centimeters
    result, unit = unit_converter.convert(1.0, "m", "cm")
    assert result == 100.0
    assert unit == "cm"

    # Meters to inches
    result, unit = unit_converter.convert(1.0, "m", "in")
    assert abs(result - 39.3701) < 0.01

    # Meters to feet
    result, unit = unit_converter.convert(1.0, "m", "ft")
    assert abs(result - 3.28084) < 0.01


def test_pressure_conversions():
    """Test pressure unit conversions"""
    # Pascals to Megapascals
    result, unit = unit_converter.convert(1000000.0, "Pa", "MPa")
    assert result == 1.0
    assert unit == "MPa"

    # Megapascals to Gigapascals
    result, unit = unit_converter.convert(1000.0, "MPa", "GPa")
    assert result == 1.0
    assert unit == "GPa"

    # Pascals to PSI
    result, unit = unit_converter.convert(6894.76, "Pa", "psi")
    assert abs(result - 1.0) < 0.01


def test_force_conversions():
    """Test force unit conversions"""
    # Newtons to kilonewtons
    result, unit = unit_converter.convert(1000.0, "N", "kN")
    assert result == 1.0
    assert unit == "kN"

    # Newtons to pound-force
    result, unit = unit_converter.convert(4.44822, "N", "lb")
    assert abs(result - 1.0) < 0.01


def test_same_unit_conversion():
    """Test conversion with same source and target unit"""
    result, unit = unit_converter.convert(5.0, "m", "m")
    assert result == 5.0
    assert unit == "m"


def test_incompatible_units():
    """Test conversion between incompatible units"""
    with pytest.raises(ValueError, match="Incompatible|cannot convert"):
        unit_converter.convert(1.0, "m", "Pa")

    with pytest.raises(ValueError, match="Incompatible|cannot convert"):
        unit_converter.convert(1.0, "N", "m/s")


def test_unknown_units():
    """Test conversion with unknown units"""
    with pytest.raises(ValueError, match="Unknown"):
        unit_converter.convert(1.0, "invalid_unit", "m")

    with pytest.raises(ValueError, match="Unknown"):
        unit_converter.convert(1.0, "m", "invalid_unit")


def test_dimensionless():
    """Test dimensionless unit handling"""
    # Same dimensionless
    result, unit = unit_converter.convert(5.0, "dimensionless", "dimensionless")
    assert result == 5.0
    assert unit == "dimensionless"

    # Cannot convert to/from dimensionless
    with pytest.raises(ValueError):
        unit_converter.convert(1.0, "dimensionless", "m")

    with pytest.raises(ValueError):
        unit_converter.convert(1.0, "m", "dimensionless")


def test_is_compatible():
    """Test unit compatibility checking"""
    # Compatible units
    assert unit_converter.is_compatible("m", "mm") is True
    assert unit_converter.is_compatible("Pa", "MPa") is True
    assert unit_converter.is_compatible("N", "kN") is True

    # Same unit
    assert unit_converter.is_compatible("m", "m") is True

    # Incompatible units
    assert unit_converter.is_compatible("m", "Pa") is False
    assert unit_converter.is_compatible("N", "m/s") is False

    # Dimensionless
    assert unit_converter.is_compatible("dimensionless", "dimensionless") is True
    assert unit_converter.is_compatible("dimensionless", "m") is False


def test_get_supported_units():
    """Test getting list of supported units"""
    units = unit_converter.get_supported_units()

    assert "m" in units
    assert "mm" in units
    assert "Pa" in units
    assert "N" in units
    assert "m/s" in units
    assert "dimensionless" in units

    assert len(units) >= 15  # Should have at least 15 supported units


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
