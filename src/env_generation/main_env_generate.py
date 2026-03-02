from typing import Optional, Tuple, Union

from .agentic_multi_turn import generate_environment_from_multi_agent
from .coder_agent import generate_environment_from_code_agent
from .consistency import (
    direct_generate,
)


def generate_environment_json_and_code(
    task_description: str,
    model: str = "gpt-4.1-mini",
    output_dir: str = "temp/agentic/envs",
    strategy: str = "multi_agent",
    platform: Optional[Union[str]] = None,
) -> Tuple[str, str]:
    """
    Main function to generate environment from task description.
    
    Args:
        task_description: Description of the task to be performed
        model: Model to use for generation
        output_dir: Directory to save generated environment
        platform: Optional model platform name or enum
        
    Returns:
        Tuple of (generated_code, environment_name)
    """
    tool_schema, env_code = None, None
    
    if strategy == "multi_agent":
        tool_schema, env_code = generate_environment_from_multi_agent(
            task_description=task_description,
            model=model,
            output_dir=output_dir,
            platform=platform,
        )
    elif strategy == "coder_agent":
        tool_schema, env_code = generate_environment_from_code_agent(
            task_description=task_description,
            model=model,
            output_dir=output_dir,
            sandbox_session_id=None,
            platform=platform,
        )
    elif strategy == "direct":
        tool_schema, env_code = direct_generate(task_description, model, output_dir, platform=platform)
    else:
        raise ValueError("Invalid strategy")

    return tool_schema, env_code
