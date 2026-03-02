import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.core.agents import ChatAgent
from src.core.models import ModelFactory
from src.core.types import ModelPlatformType, ModelType
from typing import Dict
from src.core.prompts.user_agent_prompt import USER_AGENT_SYSTEM_PROMPT


def initialize_user_agent(base_path: Path) -> ChatAgent:
    
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENROUTER,
        model_type=ModelType.OPENROUTER_GPT_4_1_MINI,
    )

    user_agent = ChatAgent(
        system_message=USER_AGENT_SYSTEM_PROMPT,
        model=model,
        tools=[],
        auto_save=True,
        results_base_dir=base_path / "user_agent_trajectories",
    )
    return user_agent


def initialize_task_execution_agent(
    base_path: Path,
    tools: List[Dict[str, Any]],
    servers: List[str],
) -> ChatAgent:
    
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENROUTER,
        model_type=ModelType.OPENROUTER_GPT_4_1_MINI,
    )

    agent = ChatAgent(
        system_message=(
            "You are a helpful assistant with access to various tools and data sources. "
            "When users ask questions, you MUST use the available tools."
        ),
        model=model,
        tools=tools,
        auto_save=True,
        results_base_dir=base_path / "temp",
    )
    logger.info(
        f"Successfully initialized agent for servers {servers} with {len(tools)} tools"
    )
    return agent


def initialize_filter_agent(results_base_dir: str) -> ChatAgent:
    
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENROUTER,
        model_type=ModelType.OPENROUTER_GPT_4_1_MINI,
    )
    
    agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=model,
        results_base_dir=results_base_dir + "/evaluation",
        auto_save=True,
    )
    return agent


def initialize_task_gen_agent() -> Dict[str, ChatAgent]:
    
    open_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENROUTER,
        model_type=ModelType.OPENROUTER_GPT_4_1_MINI,
        model_config_dict={"temperature": 0.8},
    )
    rigorous_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENROUTER,
        model_type=ModelType.OPENROUTER_GPT_4_1_MINI,
        model_config_dict={"temperature": 0.2},
    )
    
    open_agent = ChatAgent(
        system_message="You are a helpful assistant",
        model=open_model,
    )
    rigorous_agent = ChatAgent(
        system_message="You are a helpful assistant",
        model=rigorous_model,
    )
    
    return {"open": open_agent, "rigorous": rigorous_agent}
