"""Formula execution service with engineering formulas"""

from typing import Dict, Any, Tuple
import math


class FormulaService:
    """Service for executing engineering formulas"""

    # Available formulas
    FORMULAS = {
        "beam_deflection_simply_supported": {
            "name": "Simply Supported Beam Deflection",
            "description": "Calculate maximum deflection of a simply supported beam with uniform load",
            "parameters": {
                "w": "Uniform load (N/m)",
                "L": "Beam length (m)",
                "E": "Young's modulus (GPa)",
                "I": "Second moment of area (m^4)"
            },
            "unit": "m",
            "category": "Structural Engineering",
            "formula": lambda w, L, E, I: (5 * w * L**4) / (384 * E * 1e9 * I)
        },
        "beam_deflection_cantilever": {
            "name": "Cantilever Beam Deflection",
            "description": "Calculate maximum deflection of a cantilever beam with uniform load",
            "parameters": {
                "w": "Uniform load (N/m)",
                "L": "Beam length (m)",
                "E": "Young's modulus (GPa)",
                "I": "Second moment of area (m^4)"
            },
            "unit": "m",
            "category": "Structural Engineering",
            "formula": lambda w, L, E, I: (w * L**4) / (8 * E * 1e9 * I)
        },
        "beam_stress": {
            "name": "Beam Bending Stress",
            "description": "Calculate maximum bending stress in a beam",
            "parameters": {
                "M": "Bending moment (N·m)",
                "c": "Distance from neutral axis to outer fiber (m)",
                "I": "Second moment of area (m^4)"
            },
            "unit": "Pa",
            "category": "Structural Engineering",
            "formula": lambda M, c, I: (M * c) / I
        },
        "column_buckling": {
            "name": "Euler Column Buckling Load",
            "description": "Calculate critical buckling load for a column",
            "parameters": {
                "E": "Young's modulus (GPa)",
                "I": "Second moment of area (m^4)",
                "L": "Column length (m)",
                "K": "Effective length factor"
            },
            "unit": "N",
            "category": "Structural Engineering",
            "formula": lambda E, I, L, K: (math.pi**2 * E * 1e9 * I) / (K * L)**2
        },
        "pressure_vessel_stress": {
            "name": "Thin-Walled Pressure Vessel Stress",
            "description": "Calculate hoop stress in a thin-walled cylindrical pressure vessel",
            "parameters": {
                "P": "Internal pressure (Pa)",
                "r": "Radius (m)",
                "t": "Wall thickness (m)"
            },
            "unit": "Pa",
            "category": "Mechanical Engineering",
            "formula": lambda P, r, t: (P * r) / t
        },
        "spring_deflection": {
            "name": "Spring Deflection",
            "description": "Calculate deflection of a helical spring",
            "parameters": {
                "F": "Applied force (N)",
                "k": "Spring constant (N/m)"
            },
            "unit": "m",
            "category": "Mechanical Engineering",
            "formula": lambda F, k: F / k
        },
        "reynolds_number": {
            "name": "Reynolds Number",
            "description": "Calculate Reynolds number for fluid flow",
            "parameters": {
                "rho": "Fluid density (kg/m³)",
                "v": "Flow velocity (m/s)",
                "L": "Characteristic length (m)",
                "mu": "Dynamic viscosity (Pa·s)"
            },
            "unit": "dimensionless",
            "category": "Fluid Mechanics",
            "formula": lambda rho, v, L, mu: (rho * v * L) / mu
        },
        "flow_velocity": {
            "name": "Flow Velocity",
            "description": "Calculate flow velocity from volumetric flow rate",
            "parameters": {
                "Q": "Volumetric flow rate (m³/s)",
                "A": "Cross-sectional area (m²)"
            },
            "unit": "m/s",
            "category": "Fluid Mechanics",
            "formula": lambda Q, A: Q / A
        }
    }

    @classmethod
    def get_formula_info(cls, formula_id: str) -> Dict[str, Any]:
        """
        Get information about a specific formula

        Args:
            formula_id: ID of the formula

        Returns:
            Dictionary with formula information

        Raises:
            ValueError: If formula_id is not found
        """
        if formula_id not in cls.FORMULAS:
            raise ValueError(f"Formula '{formula_id}' not found")

        formula = cls.FORMULAS[formula_id]
        return {
            "formula_id": formula_id,
            "name": formula["name"],
            "description": formula["description"],
            "parameters": formula["parameters"],
            "unit": formula["unit"],
            "category": formula["category"]
        }

    @classmethod
    def list_formulas(cls) -> list:
        """
        List all available formulas

        Returns:
            List of formula information dictionaries
        """
        return [cls.get_formula_info(fid) for fid in cls.FORMULAS.keys()]

    @classmethod
    def execute(cls, formula_id: str, input_values: Dict[str, float]) -> Tuple[float, str]:
        """
        Execute a formula with given input values

        Args:
            formula_id: ID of the formula to execute
            input_values: Dictionary of parameter names to values

        Returns:
            Tuple of (result, unit)

        Raises:
            ValueError: If formula not found or parameters are invalid
        """
        if formula_id not in cls.FORMULAS:
            raise ValueError(f"Formula '{formula_id}' not found")

        formula = cls.FORMULAS[formula_id]
        required_params = set(formula["parameters"].keys())
        provided_params = set(input_values.keys())

        if required_params != provided_params:
            missing = required_params - provided_params
            extra = provided_params - required_params
            error_msg = []
            if missing:
                error_msg.append(f"Missing parameters: {', '.join(missing)}")
            if extra:
                error_msg.append(f"Unexpected parameters: {', '.join(extra)}")
            raise ValueError(". ".join(error_msg))

        try:
            result = formula["formula"](**input_values)
            return result, formula["unit"]
        except Exception as e:
            raise ValueError(f"Error executing formula: {str(e)}")


formula_service = FormulaService()
