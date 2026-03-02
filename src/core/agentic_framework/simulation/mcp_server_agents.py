from typing import Dict

from src.core.agents import BaseAgent, ChatAgent
from src.core.models import ModelFactory
from src.core.types import ModelPlatformType, ModelType
from src.core.toolkits import SandboxToolkit, WebSearchWithSummaryToolkit


def initialize_mcp_server_agent(
    results_base_dir: str = "temp/agentic",
    persist_sandbox: bool = False,
    sandbox_session_id: str | None = None,
) -> Dict[str, BaseAgent]:
    simulation_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENROUTER,
        model_type=ModelType.OPENROUTER_GPT_4_1_MINI,
        model_config_dict={"temperature": 0},
    )

    # web_search_toolkit = WebSearchWithSummaryToolkit(preferred_provider="serper")
    # sandbox_toolkit = SandboxToolkit(
    #     default_file_map={
    #         "src/simulation/*": "src/simulation/",
    #     },
    #     default_requirements=["pytest", "pytest-asyncio"],
    #     timeout_minutes=120,
    # )

    agents: Dict[str, BaseAgent] = {}

    system_message = "You are a helpful assistant."
    agents["mcp_server"] = ChatAgent(
        model=simulation_model,
        system_message=system_message,
        tools=[],
        auto_save=True,
    )

    return agents

