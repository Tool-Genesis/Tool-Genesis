# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional, List
import json
import hashlib
from abc import ABC, abstractmethod
import logging
import random
from enum import Enum

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

from src.apps.time_manager import TimeManager
from src.apps.tools import ToolDefinition
from src.apps.utils import SkippableDeepCopy, add_reset

class Protocol(Enum):
    FILE_SYSTEM = "FILE_SYSTEM"

@add_reset
class App(ABC, SkippableDeepCopy):
    """
    MCP Server base class for users/generators:
    - Provides basic state management and MCP server initialization
    - Adds MCP initialization and common tools (server_spec/state_* and extra_* indexing)
    - Subclasses only need to:
        1) Implement _set_environment_state(self, initial_state: Dict[str, Any])
        2) Implement save_state(self, path: str) and load_state(self, path: str) methods
        3) Register business tools with @mcp.tool at the end of _initialize_mcp_server()
           (After registration, call _record_tool(...) to record in catalog if needed in server_spec)
    """

    def __init__(
        self,
        server_name: Optional[str] = None,
        server_description: str = "",
        tools: Optional[List[ToolDefinition]] = None,
    ):
        super().__init__()
        # Basic properties
        self.server_name = server_name or self.__class__.__name__
        self.server_description = server_description
        self.tools = tools or []
        self.time_manager = TimeManager()
        self.set_seed(0)
        
        
    def set_seed(self, seed: int) -> None:
        # Derive a new seed from the combination of the input seed and app name
        # This ensures each app instance gets a unique but deterministic seed
        combined_seed = f"{seed}_{self.server_name}"
        self.seed = int(hashlib.sha256(combined_seed.encode("utf-8")).hexdigest()[:8], 16)
        self.rng = random.Random(self.seed)
    
    def _initialize_mcp_server(self):
        if not FastMCP:
            raise RuntimeError("FastMCP not available. Install `modelcontextprotocol[fastmcp]`.")

        mcp = FastMCP(self.server_name)

        # —— Subclasses continue to register business tools here (using @mcp.tool), and call _record_tool to record in catalog ——
        # Example:
        # @mcp.tool(name="email_create", description="Create an email")
        # def email_create(...): ...
        # self._record_tool("email_create", "Create an email")

        mcp._env = self
        return mcp

    def _execute_action(self, action: str) -> dict:
        # Parse the action and call the corresponding tool
        action_data = json.loads(action)
        tool_name = action_data["tool"]
        tool_args = action_data["args"]
        tool = getattr(self.mcp_server, tool_name)
        return tool(**tool_args)
    
    def load_state(self, state_dict: dict[str, Any]):
        """
        Abstract method for loading state from dictionary, must be implemented by subclasses
        Args:
            state_dict: State dictionary to load
        """
        pass
    
    def get_state(self) -> dict[str, Any] | None:
        """
        Abstract method for getting current state, must be implemented by subclasses
        Returns:
            Current state as dictionary or None
        """
        pass

    def reset_state(self) -> None:
        """
        Abstract method for resetting state, must be implemented by subclasses
        """
        pass

    def env_name(self) -> str:
        """Get environment name"""
        return self.server_name

    def reset(self) -> None:
        """Reset environment to initial state"""
        self.reset_state()
    
    def get_implemented_protocols(self) -> list[Protocol]:
        """
        App can provide protocols, e.g. FileSystem which could be used by other apps
        Returns a list of protocol names that the app implements.
        """
        return []

    def connect_to_protocols(self, protocols: dict[Protocol, Any]) -> None:
        """
        App can connect to other apps via protocols.
        Args:
            protocols: Dictionary mapping protocol types to protocol implementations
        """
        pass
        
    def pause_env(self) -> None:
        """Pause the environment"""
        pass

    def resume_env(self) -> None:
        """Resume the environment"""
        pass
