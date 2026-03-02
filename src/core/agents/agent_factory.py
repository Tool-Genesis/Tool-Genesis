# src/factories/agent_factory.py
from __future__ import annotations

import importlib
import re
from typing import Any, Dict, List, Optional, Type

from src.core.configs.agents import AgentConfig
from src.core.toolkits import FunctionTool
from src.core.toolkits.utils import convert_to_function_tool
from src.core.models import ModelFactory


AGENT_CLASS_REGISTRY: Dict[str, str] = {
    "chat": "src.core.agents.chat_agent:ChatAgent",
    "deep_research": "src.core.agents.deep_research_agent:DeepResearchAgent",
    "player": "src.core.agents.player_agent:PlayerAgent",
}


def _camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _load_class(path_or_name: str) -> Type:
    """
    Load a class by one of:
      - 'package.module:ClassName'
      - 'package.module.ClassName'
      - 'ClassName'  (resolved under 'src.core.toolkits' first, then 'src.core.toolkits.<snake_case>')
    """
    # A) 'package.module:ClassName'
    if ":" in path_or_name:
        module_name, class_name = path_or_name.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    # B) 'package.module.ClassName'
    if "." in path_or_name:
        parts = path_or_name.split(".")
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    # C) 'ClassName' — resolve in 'src.core.toolkits' namespace
    class_name = path_or_name

    # Try re-export from src.core.toolkits/__init__.py
    try:
        mod = importlib.import_module("src.core.toolkits")
        return getattr(mod, class_name)
    except Exception:
        pass

    # Try snake-case module under src.core.toolkits
    snake = _camel_to_snake(class_name)
    try:
        mod = importlib.import_module(f"src.core.toolkits.{snake}")
        return getattr(mod, class_name)
    except Exception as e:
        tried = [
            "src.core.toolkits (re-export)",
            f"src.core.toolkits.{snake} (module file)",
        ]
        raise ImportError(
            f"Cannot resolve class '{path_or_name}'. Tried: {', '.join(tried)}. "
            f"Consider using fully-qualified 'package.module:ClassName'."
        ) from e


def _canonical_key(cls: Type) -> str:
    """Return a stable key 'module:ClassName' for kwargs lookup."""
    return f"{cls.__module__}:{cls.__name__}"


def _build_tools(
    toolkit_imports: List[str],
    toolkit_kwargs: Dict[str, Dict[str, Any]],
) -> List[FunctionTool]:
    """
    Instantiate toolkits from class identifiers and collect their tools.
    `toolkit_imports` accepts:
      - 'package.module:ClassName'
      - 'package.module.ClassName'
      - 'ClassName' (resolved under src.core.toolkits)
    Kwargs lookup tries (in order):
      - the original identifier string
      - the canonical 'module:ClassName'
      - the bare class name
    """
    tools: List[FunctionTool] = []
    for ident in toolkit_imports:
        cls = _load_class(ident)
        canon = _canonical_key(cls)

        # find kwargs with multiple keys fallback
        kwargs = (
            toolkit_kwargs.get(ident)
            or toolkit_kwargs.get(canon)
            or toolkit_kwargs.get(cls.__name__)
            or {}
        )

        toolkit = cls(**kwargs) if kwargs else cls()
        for t in toolkit.get_tools():
            tools.append(t if isinstance(t, FunctionTool) else convert_to_function_tool(t))
    return tools


def _resolve_agent_cls(cfg: AgentConfig) -> Type:
    """
    Resolve the agent class from the config's agent_type.
    Uses the AGENT_CLASS_REGISTRY to map agent types to class paths.
    """
    agent_type = cfg.agent_type
    if agent_type not in AGENT_CLASS_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}. Available types: {list(AGENT_CLASS_REGISTRY.keys())}")
    
    class_path = AGENT_CLASS_REGISTRY[agent_type]
    return _load_class(class_path)


class AgentFactory:
    """
    Stateless factory that builds an Agent from an AgentConfig.
    Supports runtime overrides for model params and toolkits.
    """

    @staticmethod
    def build_from_config(
        cfg: AgentConfig,
        *,
        model_override: Optional[Dict[str, Any]] = None,
        extra_toolkits: Optional[List[str]] = None,
        disable_tools_by_name: Optional[List[str]] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 1) Build model (allow overrides on top of config)
        merged_model_params = {**cfg.model_params, **(model_override or {})}
        model = ModelFactory.create(
            model_platform=cfg.model_platform,
            model_type=cfg.model_type,
            model_config_dict=merged_model_params,
        )

        # 2) Build tools (base + optional extras)
        toolkit_imports = list(cfg.toolkit_imports) + list(extra_toolkits or [])
        tools = _build_tools(toolkit_imports, cfg.toolkit_kwargs)
        if disable_tools_by_name:
            deny = set(disable_tools_by_name)
            tools = [t for t in tools if t.get_function_name() not in deny]

        # 3) Resolve and instantiate the agent class
        AgentCls = _resolve_agent_cls(cfg)
        init_kwargs = dict(
            system_message=cfg.system_message,
            model=model,
            tools=tools,
            auto_save=cfg.auto_save,
            results_base_dir=cfg.results_base_dir,
        )
        if agent_kwargs:
            init_kwargs.update(agent_kwargs)

        return AgentCls(**init_kwargs)

    @staticmethod
    def build_from_args(args: Dict[str, Any]):
        """
        Optional convenience: build directly from a packed dict
        (e.g., config.to_factory_args()).
        """
        AgentCls = _load_class(args["agent_cls_path"])
        model = ModelFactory.create(
            model_platform=args["model_platform"],
            model_type=args["model_type"],
            model_config_dict=args["model_params"],
        )
        tools = _build_tools(args["toolkit_imports"], args["toolkit_kwargs"])
        return AgentCls(model=model, tools=tools, **args["agent_kwargs"])
