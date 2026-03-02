import os
import sys
from typing import Dict, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.agents import BaseAgent, ChatAgent
from src.core.models import ModelFactory
from src.core.types import ModelPlatformType, ModelType
from src.core.toolkits import (
    FunctionTool,
    MCPServerToolsToolkit,
)

def initialize_agent(
    server_name: str,
    simulation_toolkit: MCPServerToolsToolkit,
    results_base_dir: str = "temp/agentic"
) -> Dict[str, BaseAgent]:
    
    ## initialize models
    simulate_solver_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENROUTER,
        model_type=ModelType.QWEN_3_14B,
        model_config_dict={"temperature": 0},
    )
    
    judge_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GLM_OSS_120B,
        model_config_dict={"temperature": 0},
    )
    
    ## initialize toolkits
    
    
    
    ## initialize agents
        
    simulate_solver_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=simulate_solver_model,
        tools=simulation_toolkit.get_tools(),
        auto_save=True,
        results_base_dir=results_base_dir + "/simulate_solve/",
    )
    
    judge_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=judge_model,
        tools=[],
        auto_save=True,
        results_base_dir=results_base_dir + "/judge/",
    )
    return {"simulate_solve": simulate_solver_agent, "judge": judge_agent}
