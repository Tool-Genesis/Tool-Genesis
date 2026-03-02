# src/configs/agents/base_config.py
from typing import Any, Dict, List, Optional, ClassVar
from pydantic import BaseModel, Field
from typing_extensions import Literal

class AgentConfig(BaseModel):
    """
    Serializable config for building an Agent.
    """
    # Choose one of the two built-in agents
    agent_type: Literal["chat", "deep_research"] = "deep_research"

    # Optional hard override (advanced). If set, it wins over agent_type.
    # Format: "package.module:ClassName"
    agent_cls_path: Optional[str] = None

    # Core prompt & model
    system_message: Optional[str] = None
    model_platform: Optional[str] = None
    model_type: Optional[str] = None
    model_params: Dict[str, Any] = Field(default_factory=dict)

    # Toolkits to load dynamically by class path
    toolkit_imports: List[str] = Field(default_factory=list)
    toolkit_kwargs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Persistence & misc
    auto_save: bool = True
    results_base_dir: str = "./results/"
    
    @classmethod
    def get_platform_enum(cls):
        """Get ModelPlatformType enum class"""
        from src.core.types.enums import ModelPlatformType
        return ModelPlatformType
    
    @classmethod
    def get_model_enum(cls):
        """Get ModelType enum class"""
        from src.core.types.enums import ModelType
        return ModelType