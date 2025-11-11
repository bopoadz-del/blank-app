"""FastAPI Formula Execution Backend."""

from .compat import install_shims as _install_shims

_install_shims()

__version__ = "1.0.0"

__all__ = ["__version__"]
