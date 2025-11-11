"""Exception hierarchy for the SQLAlchemy stub."""


class OperationalError(Exception):
    def __init__(self, message: str, params: object = None, orig: object = None) -> None:
        super().__init__(message)
        self.params = params
        self.orig = orig


__all__ = ["OperationalError"]
