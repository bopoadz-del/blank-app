"""SQL helpers for the SQLAlchemy stub."""
from __future__ import annotations

from .._support import func


class TextClause:
    def __init__(self, text: str) -> None:
        self.text = text


def text(value: str) -> TextClause:
    return TextClause(value)


__all__ = ["TextClause", "func", "text"]
