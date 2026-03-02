# src/configs/agents/deep_research_agent_config.py
from typing_extensions import Literal
from src.core.configs.agents.base_config import AgentConfig

class DeepResearchAgentConfig(AgentConfig):
    """Concrete config with sensible defaults for a DeepResearchAgent."""
    agent_type: Literal["deep_research"] = "deep_research"

    @classmethod
    def default(cls) -> "DeepResearchAgentConfig":
        from src.core.prompts import DeepResearchPromptTemplateDict
        # You can pass *string values*; Pydantic will coerce them to enums,
        # or you can import the enums here and pass the enum members explicitly.
        return cls(
            system_message=DeepResearchPromptTemplateDict.build(),
            model_platform="openai",
            model_type="gpt-4o-mini",
            model_params={"temperature": 0.0},
            toolkit_imports=["src.core.toolkits:WebSearchToolkit"],
            toolkit_kwargs={},
            auto_save=True,
            results_base_dir="./results/deep_research",
        )