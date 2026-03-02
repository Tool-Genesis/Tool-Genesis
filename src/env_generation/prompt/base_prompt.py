# base_prompt.py
from __future__ import annotations
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from textwrap import dedent


class PromptPack:
    """
    A lightweight container for prompts that returns a single string.
    Since system_prompt is fixed, this only provides an interface to combine user_prompt.
    """
    
    def __init__(self, prompt: str):
        self.prompt = prompt
    
    def as_single(self) -> str:
        """Return the prompt as a single string."""
        return self.prompt.strip()


class BasePrompt(ABC):
    """
    Minimal prompt base class for two stages:
      1) Research (structure and validation planning)
      2) Code generation (final MCP server implementation)

    Subclasses implement the two abstract builders to produce PromptPack objects.
    """

    def __init__(self, *, final_language: str = "python") -> None:
        self.final_language = final_language

    # ---------- Abstract builders ----------
    @abstractmethod
    def build_env_code_prompt(
        self,
        tool_schema:str
    ) -> PromptPack:
        """
        Build a codegen-stage prompt.
        Expected to ask for codegen strategy, codegen constraints, and shared utilities.
        """

    @abstractmethod
    def build_tool_schema_prompt(
        self,
        task_describe: str,
    ) -> PromptPack:
        """
        Build a tool-schema-generation-stage prompt.
        Expected to ask for tool schema generation strategy, tool schema validation strategy, and shared utilities.
        """
