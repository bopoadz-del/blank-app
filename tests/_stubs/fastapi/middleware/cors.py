"""Minimal CORS middleware stand-in."""

class CORSMiddleware:
    def __init__(self, app=None, **kwargs):
        self.app = app
        self.options = kwargs
