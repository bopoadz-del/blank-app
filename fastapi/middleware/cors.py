"""CORS middleware stub for compatibility."""


class CORSMiddleware:  # pragma: no cover - behavior not needed in tests
    def __init__(self, app=None, **kwargs):
        self.app = app
        self.options = kwargs
