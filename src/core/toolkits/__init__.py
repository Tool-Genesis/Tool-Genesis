from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "BaseToolkit": (".base", "BaseToolkit"),
    "FunctionTool": (".function_tool", "FunctionTool"),
    "WebSearchToolkit": (".webSearch_toolkit", "WebSearchToolkit"),
    "SandboxToolkit": (".sandbox_toolkit", "SandboxToolkit"),
    "MathToolkit": (".math_toolkit", "MathToolkit"),
    "WebSearchWithSummaryToolkit": (
        ".webSearchWithSummary_toolkit",
        "WebSearchWithSummaryToolkit",
    ),
    "MCPServerToolsToolkit": (
        ".mcp_sandbox_bridge_toolkit",
        "MCPServerToolsToolkit",
    ),
}
_OPTIONAL_EXPORTS = {"SandboxToolkit", "MCPServerToolsToolkit"}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    try:
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
    except Exception:
        if name in _OPTIONAL_EXPORTS:
            value = None
        else:
            raise
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
