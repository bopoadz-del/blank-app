"""Unit conversion utilities using Pint"""

from pint import UnitRegistry
from typing import Tuple, Optional

# Initialize unit registry
ureg = UnitRegistry()


class UnitConverter:
    """Handle unit conversions for formula results"""

    # Supported unit conversions
    UNIT_MAPPINGS = {
        "m": ureg.meter,
        "mm": ureg.millimeter,
        "cm": ureg.centimeter,
        "in": ureg.inch,
        "ft": ureg.foot,
        "Pa": ureg.pascal,
        "MPa": ureg.megapascal,
        "GPa": ureg.gigapascal,
        "psi": ureg.psi,
        "ksi": ureg.ksi,
        "N": ureg.newton,
        "kN": ureg.kilonewton,
        "lb": ureg.pound_force,
        "m/s": ureg.meter / ureg.second,
        "ft/s": ureg.foot / ureg.second,
        "dimensionless": ureg.dimensionless,
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

        # Get pint units
        from_pint = cls.UNIT_MAPPINGS.get(from_unit)
        to_pint = cls.UNIT_MAPPINGS.get(to_unit)

        if from_pint is None:
            raise ValueError(f"Unknown source unit: {from_unit}")
        if to_pint is None:
            raise ValueError(f"Unknown target unit: {to_unit}")

        try:
            # Create quantity and convert
            quantity = value * from_pint
            converted = quantity.to(to_pint)
            return converted.magnitude, to_unit
        except Exception as e:
            raise ValueError(f"Incompatible units: cannot convert {from_unit} to {to_unit}: {str(e)}")

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
        except (ValueError, Exception):
            return False

    @classmethod
    def get_supported_units(cls) -> list:
        """
        Get list of supported units

        Returns:
            List of unit strings
        """
        return list(cls.UNIT_MAPPINGS.keys())


unit_converter = UnitConverter()
