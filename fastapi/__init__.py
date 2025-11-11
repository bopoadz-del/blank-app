"""Fallback FastAPI loader.

The offline test environment for this project does not have FastAPI available,
so we provide a tiny shim under :mod:`app.shims.fastapi`.  When the real
framework is installed (for example in production or security scans) this
module defers to it by importing the real package with the project root
temporarily removed from sys.path, then restoring it.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Temporarily remove project root from sys.path to check for real FastAPI
_temp_path = [p for p in sys.path if Path(p).resolve() != _PROJECT_ROOT]
_has_real = False

_saved_path = sys.path[:]
sys.path[:] = _temp_path
try:
    import fastapi as _real
    _has_real = True
except ImportError:
    pass
finally:
    sys.path[:] = _saved_path

if _has_real:
    # Real FastAPI exists - we need to ensure it's used instead of local shim
    # The trick is to replace this module in sys.modules and ensure
    # submodule lookups also go to the real package
    
    # Temporarily remove project root again to import real fastapi
    sys.path[:] = _temp_path
    
    # Import the real fastapi package
    if 'fastapi' in sys.modules:
        del sys.modules['fastapi']  # Clear any cached import
    import fastapi as _real_fastapi
    
    # Replace this local module with the real one
    sys.modules[__name__] = _real_fastapi
    
    # Pre-load common submodules from real package to prevent local shim loading
    try:
        import fastapi.testclient
        sys.modules['fastapi.testclient'] = fastapi.testclient
    except ImportError:
        pass
    
    try:
        import fastapi.middleware
        sys.modules['fastapi.middleware'] = fastapi.middleware
    except ImportError:
        pass
    
    # Restore original sys.path
    sys.path[:] = _saved_path
else:
    # No real FastAPI - use shim
    import importlib
    _module = importlib.import_module("app.shims.fastapi")
    
    # Ensure nested imports resolve to the shim
    sys.modules[__name__ + ".middleware"] = importlib.import_module("app.shims.fastapi.middleware")
    sys.modules[__name__ + ".middleware.cors"] = importlib.import_module("app.shims.fastapi.middleware.cors")
    sys.modules[__name__ + ".security"] = importlib.import_module("app.shims.fastapi.security")
    sys.modules[__name__ + ".testclient"] = importlib.import_module("app.shims.fastapi.testclient")
    
    sys.modules[__name__] = _module
