"""Environment generation public API with lazy imports.

Avoid importing heavy dependencies (LLM/sandbox/toolkit stacks) at package import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__version__ = "1.0.0"
__author__ = "AgentGen Team"

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Generation functions
    "generate_simple_env": (".single_turn", "generate_environment_from_mcp"),
    "generate_agentic_env": (
        ".agentic_single_turn",
        "generate_environment_from_single_agent",
    ),
    "generate_multi_agent_env": (
        ".agentic_multi_turn",
        "generate_environment_from_multi_agent",
    ),
    "generate_environment_json_and_code": (
        ".main_env_generate",
        "generate_environment_json_and_code",
    ),
    # Framework
    "initialize_agent": (".agentic_framework", "initialize_agent"),
    # Evaluation
    "check_executability": (".env_eval", "check_executability"),
    "schema_fidelity_score": (".env_eval", "schema_fidelity_score"),
    "functionality_score": (".env_eval", "functionality_score"),
    "semantic_fidelity_score": (".env_eval", "semantic_fidelity_score"),
    "truthfulness_score": (".env_eval", "truthfulness_score"),
    # Utilities
    "load_server_def": (".utils.load_data", "load_server_def"),
    "save_environment": (".utils.save_environment", "save_environment"),
    "extract_tool_defs": (".utils.extract_tool_defs", "extract_tool_defs"),
    # Prompt system
    "BasePrompt": (".prompt.base_prompt", "BasePrompt"),
    "ToolGenesisPrompt": (".prompt.tool_genesis_prompt", "ToolGenesisPrompt"),
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
