from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["App"]

if TYPE_CHECKING:
    from .apps import App as App


def __getattr__(name: str) -> Any:
    if name == "App":
        from .apps import App as _App

        return _App
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
