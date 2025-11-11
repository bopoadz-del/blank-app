"""Compatibility loader for SQLAlchemy.

Searches for an installed SQLAlchemy distribution outside of the project root
and falls back to the lightweight shim when none is present.
"""

import importlib.util
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Temporarily remove project root from sys.path to check for real sqlalchemy
_temp_path = [p for p in sys.path if Path(p).resolve() != _PROJECT_ROOT]
_has_real = False

_saved_path = sys.path[:]
sys.path[:] = _temp_path
try:
    import sqlalchemy as _real
    _has_real = True
except ImportError:
    pass
finally:
    sys.path[:] = _saved_path

if _has_real:
    # Real sqlalchemy exists - replace this module
    sys.path[:] = _temp_path
    
    if 'sqlalchemy' in sys.modules:
        del sys.modules['sqlalchemy']
    import sqlalchemy as _real_sqlalchemy
    
    sys.modules[__name__] = _real_sqlalchemy
    
    # Pre-load common submodules
    try:
        import sqlalchemy.orm
        sys.modules['sqlalchemy.orm'] = sqlalchemy.orm
    except ImportError:
        pass
    
    sys.path[:] = _saved_path
else:
    # No real sqlalchemy - use shim
    import importlib
    _module = importlib.import_module("app.shims.sqlalchemy")
    
    sys.modules[__name__ + ".exc"] = importlib.import_module("app.shims.sqlalchemy.exc")
    sys.modules[__name__ + ".ext"] = importlib.import_module("app.shims.sqlalchemy.ext")
    sys.modules[__name__ + ".ext.declarative"] = importlib.import_module("app.shims.sqlalchemy.ext.declarative")
    sys.modules[__name__ + ".orm"] = importlib.import_module("app.shims.sqlalchemy.orm")
    sys.modules[__name__ + ".orm.session"] = importlib.import_module("app.shims.sqlalchemy.orm.session")
    sys.modules[__name__ + ".sql"] = importlib.import_module("app.shims.sqlalchemy.sql")
    
    sys.modules[__name__] = _module
