from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Core classes
    "PersistentEnvironmentSandbox": (".persistent_sandbox", "PersistentEnvironmentSandbox"),
    "EnvironmentSandbox": (".persistent_sandbox", "EnvironmentSandbox"),
    "SandboxSessionManager": (".session_manager", "SandboxSessionManager"),
    # API helpers
    "exec_bash": (".api", "exec_bash"),
    "run_environment_safely": (".api", "run_environment_safely"),
    "create_true_sandbox": (".api", "create_true_sandbox"),
    "get_sandbox_session": (".api", "get_sandbox_session"),
    "list_sandbox_sessions": (".api", "list_sandbox_sessions"),
    "cleanup_sandbox_session": (".api", "cleanup_sandbox_session"),
    "get_or_create_session": (".api", "get_or_create_session"),
    "execute_in_sandbox": (".api", "execute_in_sandbox"),
    "cleanup_all_sandbox_sessions": (".api", "cleanup_all_sandbox_sessions"),
    "get_sandbox_session_stats": (".api", "get_sandbox_session_stats"),
    "extend_sandbox_session_timeout": (".api", "extend_sandbox_session_timeout"),
    "get_sandbox_stats": (".api", "get_sandbox_stats"),
    # Persistent aliases (defined in api.py)
    "create_persistent_sandbox": (".api", "create_persistent_sandbox"),
    "create_true_persistent_sandbox": (".api", "create_true_persistent_sandbox"),
    "get_true_persistent_session": (".api", "get_true_persistent_session"),
    "list_true_persistent_sessions": (".api", "list_true_persistent_sessions"),
    "cleanup_true_persistent_session": (".api", "cleanup_true_persistent_session"),
    "execute_in_persistent_sandbox": (".api", "execute_in_persistent_sandbox"),
    "list_persistent_sessions": (".api", "list_persistent_sessions"),
    "cleanup_persistent_session": (".api", "cleanup_persistent_session"),
    "cleanup_all_persistent_sessions": (".api", "cleanup_all_persistent_sessions"),
    "get_persistent_session_stats": (".api", "get_persistent_session_stats"),
    "extend_persistent_session_timeout": (".api", "extend_persistent_session_timeout"),
    # Utility exports
    "cleanup_all_sandboxes": (".core", "cleanup_all_sandboxes"),
    "get_requirements": (".utils", "get_requirements"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
