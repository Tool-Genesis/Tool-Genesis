from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["call_embedding"]

if TYPE_CHECKING:
    from .llm_embedding import call_embedding as call_embedding


def __getattr__(name: str) -> Any:
    if name == "call_embedding":
        try:
            from .llm_embedding import call_embedding as _call_embedding
        except Exception:
            from .llm import call_embedding as _call_embedding

        return _call_embedding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
