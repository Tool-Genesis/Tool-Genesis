import os
import sys
from typing import Dict, Optional, Tuple, Union

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.agents import BaseAgent, ChatAgent
from src.core.models import ModelFactory
from src.core.types import ModelPlatformType, ModelType
from src.core.toolkits import (
    FunctionTool,
    WebSearchWithSummaryToolkit,
    SandboxToolkit,
)
from src.core.sandbox.managers.workspace_manager import PersistPolicy
from src.utils.llm import extract_code

from .prompt import BasePrompt

def _normalize_platform(platform: Union[ModelPlatformType, str]) -> ModelPlatformType:
    if isinstance(platform, ModelPlatformType):
        return platform
    raw = str(platform).strip()
    if not raw:
        return ModelPlatformType.DEFAULT
    lowered = raw
    try:
        return ModelPlatformType.from_name(lowered)
    except Exception:
        pass
    try:
        return ModelPlatformType[raw.upper()]
    except Exception:
        return ModelPlatformType.DEFAULT


def map_model_to_platform_and_type(
    model: str,
    platform: Optional[Union[ModelPlatformType, str]] = None,
) -> Tuple[ModelPlatformType, ModelType]:
    """
    Map an input model name to ModelPlatformType and ModelType.
    - Supports plain model names like "gpt-4.1-mini", "deepseek-chat".
    - Supports prefixed names like "openai/gpt-4.1-mini" or "openai:gpt-4.1-mini".
    - Falls back to DEFAULT platform if no specific platform can be inferred.
    """
    raw = model.strip()
    platform_prefix = None
    name_part = raw
    if "/" in raw:
        parts = raw.split("/", 1)
        platform_prefix, name_part = parts[0], parts[1]
    elif ":" in raw:
        parts = raw.split(":", 1)
        platform_prefix, name_part = parts[0], parts[1]
    else:
        name_part = raw
    try:
        model_type = ModelType.from_name(raw)
    except Exception:
        # Try full raw string if value already matches enum
        try:
            model_type = ModelType.from_name(raw)
        except Exception:
            # Sensible defaults
            model_type = ModelType.GPT_4_1_MINI
    if platform is not None:
        model_platform = _normalize_platform(platform)
    elif platform_prefix is not None and str(platform_prefix).strip():
        model_platform = _normalize_platform(platform_prefix)
    else:
        model_platform = ModelPlatformType.OPENAI
    return model_platform, model_type


def initialize_code_agent(
    model_platform: ModelPlatformType,
    model_type: ModelType,
    results_base_dir: str = "temp/agentic",
    sandbox_session_id: str = None,
) -> Dict[str, BaseAgent]:
    
    tool_schmea_model = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type,
        model_config_dict={"temperature": 0},
    )
    
    code_model = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type,
        model_config_dict={"temperature": 0},
    )
    
    sandbox_toolkit = SandboxToolkit(
        default_file_map={
            "src/apps/*": "src/apps/",
        },
        default_requirements=["pytest","pytest-asyncio"],
        timeout_minutes=120,
        persist_policy=PersistPolicy.CONTINUOUS_MOUNT,
        cleanup_paths_on_close=["venv"],
        mount_dir=results_base_dir + "/sandbox",
        session_id=sandbox_session_id,
        mcp_server_mode=True,
    )
    
    tool_schema_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=tool_schmea_model,
        tools=[],
        auto_save=True,
        results_base_dir=results_base_dir + "/toolgen/",
    )
    
    code_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=code_model,
        tools=[
            FunctionTool(sandbox_toolkit.file_tool),
            FunctionTool(sandbox_toolkit.run_pytest_with_analysis)
            ],
        auto_save=True,
        results_base_dir=results_base_dir + "/codegen/",
    )
    
    return {"code": code_agent, "tool_schema": tool_schema_agent, "sandbox": sandbox_toolkit}

def initialize_agent(
    results_base_dir: str = "temp/agentic",
    model_platform: ModelPlatformType = ModelPlatformType.DEEPSEEK,
    model_type: ModelType = ModelType.DEEPSEEK_CHAT,
) -> Dict[str, BaseAgent]:
    
    tool_schema_model = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type,
        model_config_dict={"temperature": 0},
    )
    
    code_model = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type,
        model_config_dict={"temperature": 0},
    )
    
    web_search_toolkit = WebSearchWithSummaryToolkit(preferred_provider="serper")
    
    tool_schema_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=tool_schema_model,
        tools=[
            FunctionTool(web_search_toolkit.browser_open),
            FunctionTool(web_search_toolkit.browser_search),
        ],
        auto_save=True,
        results_base_dir=results_base_dir + "/toolgen/",
    )
    
    code_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=code_model,
        tools=[],
        auto_save=True,
        results_base_dir=results_base_dir + "/codegen/",
    )
    return {"code": code_agent, "toolgen": tool_schema_agent}

def generate_env_code(
    agents: Dict[str, BaseAgent],
    mcp_prompt: BasePrompt,
) -> Tuple[str, str]:
    
    tool_schema_prompt = mcp_prompt.build_tool_schema_prompt(
        mcp_prompt.task_description
    )
    tool_schema_resp = agents["toolgen"].step(tool_schema_prompt.as_single())
    tool_schema = extract_code(tool_schema_resp.content)

    
    env_code_prompt = mcp_prompt.build_env_code_prompt(tool_schema)
    env_code_resp = agents["code"].step(env_code_prompt.as_single())
    env_code = extract_code(env_code_resp.content)
    
    return tool_schema, env_code
