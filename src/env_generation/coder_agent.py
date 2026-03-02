from typing import Tuple, Optional, Union
from loguru import logger
from src.utils.llm import extract_code
from .agentic_framework import initialize_code_agent, map_model_to_platform_and_type
from .prompt import CodeAgentPrompt

def generate_environment_from_code_agent(
    task_description: str,
    model: str = "gpt-4.1-mini",
    output_dir: str = "temp/agentic",
    sandbox_session_id: str = None,
    platform: Optional[Union[str]] = None,
) -> Tuple[str, str]:
    model_platform, model_type = map_model_to_platform_and_type(
        model=model,
        platform=platform,
    )
    logger.info(f"Using model {model_type} on platform {model_platform}")
    base_dir = output_dir
    agents = initialize_code_agent(
        model_platform=model_platform,
        model_type=model_type,
        results_base_dir=base_dir,
        sandbox_session_id=sandbox_session_id,
    )
    prompt = CodeAgentPrompt(
        task_description=task_description,
        final_language="python",
    )
    tool_schema_prompt = prompt.build_tool_schema_prompt(task_description).as_single()
    tool_schema_resp = agents["tool_schema"].step(tool_schema_prompt)
    tool_schema = extract_code(tool_schema_resp.content)

    env_code_prompt = prompt.build_env_code_prompt(tool_schema).as_single()

    env_code = ""
    max_retries = 2
    logger.info(f"Generating env code")
    for _ in range(max_retries):
        agents["code"].reset()
        try:
            env_code_resp = agents["code"].step(env_code_prompt)
            env_code = extract_code(env_code_resp.content)
        except Exception:
            env_code = ""
        if env_code.strip():
            break
    agents["sandbox"].cleanup()
    return tool_schema, env_code
