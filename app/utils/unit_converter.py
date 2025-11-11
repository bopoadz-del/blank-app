"""Unit conversion utilities without external dependencies."""

from typing import Tuple


class _UnitGroup:
    """Helper structure for linear unit conversions."""

    def __init__(self, base_unit: str, factors: dict[str, float]):
        self.base_unit = base_unit
        self.factors = factors

    def to_base(self, value: float, unit: str) -> float:
        if unit not in self.factors:
            raise ValueError(f"Unknown source unit: {unit}")
        return value * self.factors[unit]

    def from_base(self, value: float, unit: str) -> float:
        if unit not in self.factors:
            raise ValueError(f"Unknown target unit: {unit}")
        return value / self.factors[unit]


class UnitConverter:
    """Handle unit conversions for formula results"""

    # Supported unit conversions
    _LENGTH = _UnitGroup("m", {
        "m": 1.0,
        "mm": 0.001,
        "cm": 0.01,
        "in": 0.0254,
        "ft": 0.3048,
    })
    _PRESSURE = _UnitGroup("Pa", {
        "Pa": 1.0,
        "MPa": 1_000_000.0,
        "GPa": 1_000_000_000.0,
        "psi": 6_894.757293168361,
        "ksi": 6_894_757.293168361,
    })
    _FORCE = _UnitGroup("N", {
        "N": 1.0,
        "kN": 1_000.0,
        "lb": 4.4482216152605,
    })
    _VELOCITY = _UnitGroup("m/s", {
        "m/s": 1.0,
        "ft/s": 0.3048,
    })

    UNIT_GROUPS = {
        "m": _LENGTH,
        "mm": _LENGTH,
        "cm": _LENGTH,
        "in": _LENGTH,
        "ft": _LENGTH,
        "Pa": _PRESSURE,
        "MPa": _PRESSURE,
        "GPa": _PRESSURE,
        "psi": _PRESSURE,
        "ksi": _PRESSURE,
        "N": _FORCE,
        "kN": _FORCE,
        "lb": _FORCE,
        "m/s": _VELOCITY,
        "ft/s": _VELOCITY,
        "dimensionless": None,
    }

    @classmethod
    def convert(cls, value: float, from_unit: str, to_unit: str) -> Tuple[float, str]:
        """
        Convert a value from one unit to another

        Args:
            value: Numerical value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Tuple of (converted_value, unit_string)

        Raises:
            ValueError: If units are incompatible or unknown
        """
        if from_unit == to_unit:
            return value, to_unit

        # Handle dimensionless
        if from_unit == "dimensionless" or to_unit == "dimensionless":
            if from_unit != to_unit:
                raise ValueError("Cannot convert dimensionless to/from other units")
            return value, "dimensionless"

        from_group = cls.UNIT_GROUPS.get(from_unit)
        to_group = cls.UNIT_GROUPS.get(to_unit)

        if from_group is None:
            raise ValueError(f"Unknown source unit: {from_unit}")
        if to_group is None:
            raise ValueError(f"Unknown target unit: {to_unit}")
        if from_group is not to_group:
            raise ValueError(f"Incompatible units: cannot convert {from_unit} to {to_unit}")

        base_value = from_group.to_base(value, from_unit)
        converted_value = to_group.from_base(base_value, to_unit)
        return converted_value, to_unit

    @classmethod
    def is_compatible(cls, unit1: str, unit2: str) -> bool:
        """
        Check if two units are compatible for conversion

        Args:
            unit1: First unit
            unit2: Second unit

        Returns:
            True if units are compatible
        """
        if unit1 == unit2:
            return True

        if "dimensionless" in [unit1, unit2]:
            return unit1 == unit2

        try:
            cls.convert(1.0, unit1, unit2)
            return True
        except ValueError:
            return False

    @classmethod
    def get_supported_units(cls) -> list:
        """
        Get list of supported units

        Returns:
            List of unit strings
        """
        return list(cls.UNIT_GROUPS.keys())


unit_converter = UnitConverter()
