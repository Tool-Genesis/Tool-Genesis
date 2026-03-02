# src/configs/models/__init__.py
from __future__ import annotations

from .base_config import BaseConfig
from .openai_config import ChatGPTConfig, OPENAI_API_PARAMS
from .vllm_config import VLLMConfig,VLLM_API_PARAMS


def make_model_config():
    pass

__all__ = [
    "BaseConfig",
    "ChatGPTConfig",
    "VLLMConfig",
    "OPENAI_API_PARAMS",
    "VLLM_API_PARAMS",
]