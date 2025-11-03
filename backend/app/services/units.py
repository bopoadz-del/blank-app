"""
Universal Unit Service
Handles unit conversions and dimensional analysis using Pint
"""
from pint import UnitRegistry, DimensionalityError, UndefinedUnitError
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from loguru import logger


class UnitService:
    """Universal unit conversion and validation service"""
    
    def __init__(self, custom_definitions_path: Optional[str] = None):
        """
        Initialize unit service with custom definitions
        
        Args:
            custom_definitions_path: Path to custom unit definitions file
        """
        self.ureg = UnitRegistry()
        
        # Load custom definitions if provided
        if custom_definitions_path and Path(custom_definitions_path).exists():
            try:
                self.ureg.load_definitions(custom_definitions_path)
                logger.info(f"Loaded custom unit definitions from {custom_definitions_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom definitions: {e}")
        
        # Cache for commonly used conversions
        self._conversion_cache: Dict[str, float] = {}
    
    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        context: Optional[str] = None
    ) -> Tuple[float, bool, str]:
        """
        Convert value between units
        
        Args:
            value: Numerical value to convert
            from_unit: Source unit string
            to_unit: Target unit string
            context: Optional context (construction, energy, finance)
            
        Returns:
            (converted_value, success, error_message)
        """
        # Check cache
        cache_key = f"{from_unit}_{to_unit}_{context}"
        if cache_key in self._conversion_cache:
            return (value * self._conversion_cache[cache_key], True, "")
        
        try:
            # Apply context if specified
            if context:
                self.ureg.enable_contexts(context)
            
            # Perform conversion
            quantity = value * self.ureg(from_unit)
            converted = quantity.to(to_unit)
            
            # Cache conversion factor
            conversion_factor = converted.magnitude / value
            self._conversion_cache[cache_key] = conversion_factor
            
            # Disable context
            if context:
                self.ureg.disable_contexts(context)
            
            return (converted.magnitude, True, "")
            
        except DimensionalityError:
            return (
                value,
                False,
                f"Cannot convert {from_unit} to {to_unit}: incompatible dimensions"
            )
        except UndefinedUnitError as e:
            return (value, False, f"Undefined unit: {e}")
        except Exception as e:
            return (value, False, f"Conversion error: {str(e)}")
    
    def normalize(
        self,
        value: float,
        unit: str,
        system: str = "SI"
    ) -> Tuple[float, str]:
        """
        Normalize value to base units in specified system
        
        Args:
            value: Value to normalize
            unit: Current unit
            system: Target system (SI, imperial, etc.)
            
        Returns:
            (normalized_value, normalized_unit)
        """
        try:
            quantity = value * self.ureg(unit)
            base = quantity.to_base_units()
            return (base.magnitude, str(base.units))
        except:
            return (value, unit)
    
    def check_dimensional_consistency(
        self,
        inputs: Dict[str, Tuple[float, str]],
        output_unit: str,
        formula_expr: str
    ) -> Dict[str, Any]:
        """
        Check if formula is dimensionally consistent
        
        Args:
            inputs: Dict of {var: (value, unit)}
            output_unit: Expected output unit
            formula_expr: Formula expression
            
        Returns:
            Dict with consistency check results
        """
        try:
            # Parse all input units
            input_quantities = {}
            for var, (value, unit) in inputs.items():
                try:
                    input_quantities[var] = self.ureg(unit)
                except Exception as e:
                    return {
                        "consistent": False,
                        "error": f"Invalid unit for {var}: {unit}",
                        "details": str(e)
                    }
            
            # Check output unit is valid
            try:
                output_quantity = self.ureg(output_unit)
            except Exception as e:
                return {
                    "consistent": False,
                    "error": f"Invalid output unit: {output_unit}",
                    "details": str(e)
                }
            
            # For MVP, we verify all units are parseable
            # Full dimensional analysis requires parsing formula AST
            return {
                "consistent": True,
                "input_units": {k: str(v) for k, v in input_quantities.items()},
                "output_unit": str(output_quantity),
                "note": "Full dimensional analysis requires formula parsing"
            }
            
        except Exception as e:
            return {
                "consistent": False,
                "error": str(e)
            }
    
    def get_unit_info(self, unit_str: str) -> Dict[str, Any]:
        """Get information about a unit"""
        try:
            unit = self.ureg(unit_str)
            return {
                "valid": True,
                "dimensionality": str(unit.dimensionality),
                "base_units": str(unit.to_base_units().units),
                "system": self._determine_system(unit_str)
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    def _determine_system(self, unit_str: str) -> str:
        """Determine if unit is SI, imperial, or other"""
        imperial_units = ['ft', 'inch', 'mile', 'yard', 'lb', 'gallon', 'psi', 'Btu']
        
        unit_lower = unit_str.lower()
        for imp_unit in imperial_units:
            if imp_unit.lower() in unit_lower:
                return "imperial"
        
        return "SI"
    
    def list_available_units(self, dimension: Optional[str] = None) -> List[str]:
        """
        List available units, optionally filtered by dimension
        
        Args:
            dimension: Optional dimension filter (length, mass, time, etc.)
        """
        if dimension:
            # Filter by dimension
            units = [
                str(u) for u in self.ureg._units.keys()
                if str(self.ureg(u).dimensionality) == dimension
            ]
            return sorted(units)
        else:
            # Return all units
            return sorted([str(u) for u in self.ureg._units.keys()])
    
    def parse_unit_expression(self, expr: str) -> Dict[str, Any]:
        """
        Parse a unit expression to understand its components
        
        Args:
            expr: Unit expression like "kg*m/s^2"
            
        Returns:
            Dict with parsed components
        """
        try:
            quantity = self.ureg(expr)
            return {
                "valid": True,
                "dimensionality": str(quantity.dimensionality),
                "base_form": str(quantity.to_base_units().units),
                "components": self._extract_components(quantity)
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    def _extract_components(self, quantity) -> Dict[str, float]:
        """Extract dimensional components (mass, length, time, etc.)"""
        try:
            dims = quantity.dimensionality
            return {str(dim): power for dim, power in dims.items()}
        except:
            return {}


# Global instance
unit_service = UnitService(
    custom_definitions_path="config/unit_definitions.txt"
)
