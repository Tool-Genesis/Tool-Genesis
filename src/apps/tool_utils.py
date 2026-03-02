"""
Utilities for App tool registration and wrapping in the simulation package.

Provides:
- ToolAttributeName: Enum of attribute markers used to discover tools on App classes.
- AppTool: Lightweight callable wrapper with a name and optional description.
- build_tool: Factory to bind class/instance methods to App as AppTool, with optional failure injection.
"""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


class ToolAttributeName(Enum):
    """Markers used by App.get_tools_with_attribute to discover tools.

    Methods intended as tools should set one of these attributes (string values)
    on the function object, e.g.:

        def my_tool(...):
            ...
        my_tool.app_tool = True

    The App class checks for these attributes by name.
    """

    APP = "app_tool"
    USER = "user_tool"
    ENV = "env_tool"
    DATA = "data_tool"


@dataclass
class AppTool:
    """Lightweight wrapper for a tool function bound to an App instance.

    - name: Fully-qualified tool name, usually "<AppName>__<method_name>".
    - func: Callable that executes the underlying method with correct binding.
    - description: Optional docstring to surface tool purpose.
    """

    name: str
    func: Callable[..., Any]
    description: Optional[str] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def get_openai_tool_schema(self) -> dict:
        """Minimal OpenAI tool schema derived from signature + docstring.

        This is provided for compatibility with callers that expect a schema-like
        object; not all simulation paths require it.
        """
        # Tool display name without app prefix for schema (keep simple)
        simple_name = self.name.split("__", 1)[-1]
        sig = inspect.signature(self.func)
        params: dict[str, dict[str, Any]] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            # Basic type mapping; default to string when unknown
            ann = p.annotation
            if ann in (int, float, bool, str):
                t = ann.__name__
            else:
                t = "string"
            params[pname] = {"type": t}
            if p.default is inspect._empty:
                required.append(pname)
        return {
            "type": "function",
            "function": {
                "name": simple_name,
                "description": self.description or "",
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required,
                },
            },
        }


def build_tool(app: Any, method: Callable[..., Any], failure_probability: Optional[float] = None) -> AppTool:
    """Create an AppTool for a given App and method.

    - Binds the (possibly unbound) method to the provided app instance.
    - Prefixes tool name with the app name for disambiguation.
    - Optionally injects random failure when `failure_probability` is provided.
    """

    method_name = getattr(method, "__name__", "tool")
    full_name = f"{getattr(app, 'name', app.__class__.__name__)}__{method_name}"
    doc = inspect.getdoc(method) or ""

    def _bound_call(*args: Any, **kwargs: Any) -> Any:
        if failure_probability is not None:
            if not (0.0 <= float(failure_probability) <= 1.0):
                # Ignore invalid values silently to avoid surprising crashes
                pass
            else:
                if random.random() < float(failure_probability):
                    raise RuntimeError("Simulated tool failure")

        # Always re-fetch the bound method from the instance to honor overrides
        bound = getattr(app, method_name, None)
        if callable(bound):
            return bound(*args, **kwargs)
        # Fallback: call original method binding app explicitly
        return method(app, *args, **kwargs)

    return AppTool(name=full_name, func=_bound_call, description=doc)


# Optional helpers: decorators to mark tools on methods
def _make_marker(attr_name: str):
    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, attr_name, True)
        return fn
    return _decorator


app_tool = _make_marker(ToolAttributeName.APP.value)
user_tool = _make_marker(ToolAttributeName.USER.value)
env_tool = _make_marker(ToolAttributeName.ENV.value)
data_tool = _make_marker(ToolAttributeName.DATA.value)

__all__ = [
    "ToolAttributeName",
    "AppTool",
    "build_tool",
    "app_tool",
    "user_tool",
    "env_tool",
    "data_tool",
]