from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .agentic_framework import (
    initialize_agent,
    map_model_to_platform_and_type,
    generate_env_code,
)
from .prompt import ToolGenesisPrompt

COMBINED_JSON_PATH = Path("data/tools/combined_tools.json")

BASE_ENV_TEMPLATE_PATH = "src/apps/app.py"
TEST_ENV_TEMPLATE_PATH = "src/apps/test/test_template.py"
ENV_STATE_TEMPLATE_PATH = "src/apps/template/env_state_template.py"
DATACLASS_TEMPLATE_PATH = "src/apps/template/dataclass_template.py"

REG_FILE = Path("data/tools/env_registry.json")

def generate_environment_from_multi_agent(
    task_description: str,
    model: str = "gpt-4.1-mini",
    output_dir: str = "temp/agentic/envs",
    platform: Optional[Union[str]] = None,
) -> Tuple[str, str]:
    model_platform, model_type = map_model_to_platform_and_type(
        model=model,
        platform=platform,
    )
    base_dir = output_dir
    agents = initialize_agent(
        results_base_dir=base_dir,
        model_platform=model_platform,
        model_type=model_type,
    )
    
    mcp_prompt = ToolGenesisPrompt(task_description)
    
    tool_schema, env_code = generate_env_code(
        agents,
        mcp_prompt,
    )
    
    return tool_schema, env_code
