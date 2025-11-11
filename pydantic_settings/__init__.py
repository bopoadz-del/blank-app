"""Expose the real pydantic-settings package when available, otherwise use the shim."""

import importlib.util
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Temporarily remove project root from sys.path to check for real pydantic-settings
_temp_path = [p for p in sys.path if Path(p).resolve() != _PROJECT_ROOT]
_has_real = False

_saved_path = sys.path[:]
sys.path[:] = _temp_path
try:
    import pydantic_settings as _real
    _has_real = True
except ImportError:
    pass
finally:
    sys.path[:] = _saved_path

if _has_real:
    # Real pydantic-settings exists - replace this module
    sys.path[:] = _temp_path
    
    if 'pydantic_settings' in sys.modules:
        del sys.modules['pydantic_settings']
    import pydantic_settings as _real_pydantic_settings
    
    sys.modules[__name__] = _real_pydantic_settings
    
    sys.path[:] = _saved_path
else:
    # No real pydantic-settings - use shim
    import importlib
    _module = importlib.import_module("app.shims.pydantic_settings")
    sys.modules[__name__] = _module
