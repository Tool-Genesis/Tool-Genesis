import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger

from src.core.toolkits.base import BaseToolkit
from src.core.toolkits.mcp_function_tool import MCPFunctionTool
from src.env_generation.utils.load_data import load_server_def
from .mcp_server_engine import SimulationEngine


class MCPServerSimulationToolkit(BaseToolkit):
    """
    Toolkit that discovers and simulates MCP server tools from a local directory.
    """

    name = "MCPServerSimulationToolkit"
    description = "Toolkit for discovering and simulating MCP server tools"

    def __init__(
        self,
        server_names: list[str],
        init_states: Optional[dict] = None,
        envs_dir: str = "temp/agentic/envs",
        timeout: Optional[float] = None,
    ):
        """
        Initialize the MCP Server Simulation Toolkit.

        Args:
            server_names: A list of server names to load tools from.
            init_states: An optional dictionary to initialize the state of the environments.
            envs_dir: The directory containing the simulation environments.
            timeout: Optional timeout for toolkit operations.
        """
        super().__init__(timeout=timeout)
        self.server_names = server_names
        self.init_states = init_states or {}
        self.envs_dir = envs_dir
        self._discovered_tools: Optional[List[MCPFunctionTool]] = None
        self.engines: List[SimulationEngine] = []
        self.get_tools()

    def update_results_base_dir(self, base_dir: str) -> None:
        """Update the base directory for results storage."""
        for engine in self.engines:
            engine.update_results_base_dir(base_dir)
    
    def reset_histories(self):
        """清空所有已创建模拟引擎的交互历史，用于按任务隔离上下文。"""
        for engine in self.engines:
            if hasattr(engine, "clear_history"):
                engine.cleanup()
                

    def get_tools(self) -> List[MCPFunctionTool]:
        """
        Get all available simulated MCP server tools as MCPFunctionTool instances.

        Returns:
            List[MCPFunctionTool]: List of simulated MCP tools wrapped as MCPFunctionTool instances.
        """
        if self._discovered_tools is not None:
            return self._discovered_tools

        tools = []
        for server_name in self.server_names:
            try:
                tools_data = load_server_def(server_name)
                engine = SimulationEngine(
                    env_name=server_name,
                    init_state=self.init_states.get(server_name),
                    tools=tools_data.get("tools", []),
                )
                self.engines.append(engine)

                for tool_info in tools_data.get("tools", []):
                    wrapper_func = self._create_mcp_tool_wrapper(engine, tool_info)
                    mcp_tool = MCPFunctionTool(
                        func=wrapper_func,
                        mcp_schema=tool_info,
                        server_name=server_name,
                        tool_name=tool_info["name"],
                    )
                    tools.append(mcp_tool)

            except FileNotFoundError as e:
                logger.warning(f"Skipping server {server_name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error loading tools for server {server_name}: {e}")
                continue

        self._discovered_tools = tools
        return tools

    def _create_mcp_tool_wrapper(
        self, engine: SimulationEngine, tool_info: Dict[str, Any]
    ) -> callable:
        """
        Create a wrapper function for a simulated MCP tool.

        Args:
            engine: The simulation engine for the MCP server.
            tool_info: Tool information from the tools.json file.

        Returns:
            A wrapper function for the simulated MCP tool.
        """
        tool_name = tool_info["name"]

        def wrapper_func(**kwargs):
            """Simulated MCP tool wrapper function."""
            try:
                result = engine.execute_tool(tool_name, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error simulating MCP tool {engine.env_name}.{tool_name}: {e}")
                raise

        wrapper_func.__name__ = f"{engine.env_name}_{tool_name}"
        wrapper_func.__doc__ = tool_info.get(
            "description", f"Simulated MCP tool {tool_name} from server {engine.env_name}"
        )
        return wrapper_func

    def cleanup(self):
        """Clean up any resources used by the toolkit."""
        for engine in self.engines:
            engine.cleanup()