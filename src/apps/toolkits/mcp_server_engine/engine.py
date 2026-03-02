"""
Simulation Engine for MCP Tools

This module provides the core simulation engine for MCP tools.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .prompt.base_prompt import BasePrompt, SimulationPrompt, PromptContext
from src.core.agentic_framework.simulation import initialize_mcp_server_agent
from src.core.agents import BaseAgent

SIM_ENVS_PATH = "temp/agentic/envs"

class SimulationEngine:
    def __init__(self, env_name: str, init_state: Optional[Dict[str, Any]] = None,tools:List[Dict[str, Any]] = []):
        self.env_name = env_name
        self.env_path = os.path.join(SIM_ENVS_PATH, self.env_name)
        self.tools = tools
        self.init_state = init_state
        self.init_state_file_path = "env_state.json"
        # 交互历史：跨工具调用保持，用于提供上下文
        self.interaction_history: List[Dict[str, Any]] = []
        self._init_simulation_env()
        

    def _init_simulation_env(self):
        self.env_state_content = self._read_file(os.path.join(self.env_path, "env_state.json"))
        self.research_report_content = self._read_file(os.path.join(self.env_path, "research_report.md"))
        self.agents = initialize_mcp_server_agent()
        self.prompt_builder = SimulationPrompt()
        self.agents["sandbox"].file_tool("save",self.init_state_file_path,json.dumps(self.init_state))
        
        
    
    def _read_file(self, file_path):
        with open(file_path, "r") as f:
            return f.read()
    
    def load_env_state(self):
        return self.agents["sandbox"].file_tool("read",self.init_state_file_path)
    
    def execute_tool(self, tool_name: str, **kwargs):
        # In a real implementation, this would be populated with actual data
        new_env_state = self.agents["sandbox"].file_tool("read",self.init_state_file_path)
        context = PromptContext(
            research_report=self.research_report_content,
            tools_json=json.dumps(self.tools),
            env_state=new_env_state,
            interaction_history=self.interaction_history,
            tool_call={"name": tool_name, "arguments": kwargs}
        )

        prompt = self.prompt_builder.build(context)
        raw_content = self.agents["mcp_server"].step(prompt)
        content = raw_content.content
        # 记录到交互历史，便于后续上下文使用
        try:
            self.interaction_history.append({
                "tool": tool_name,
                "arguments": kwargs,
                "result": content,
            })
        except Exception:
            # 历史记录不应阻断流程
            pass
        self.agents["mcp_server"].update_results_base_dir(self.base_dir)
        # In a real implementation, this would involve calling an LLM
        return content

    def clear_history(self):
        """清空本引擎的交互历史，用于任务切换时隔离上下文。"""
        self.interaction_history.clear()

    def cleanup(self):
        for agent in self.agents.values():
            if hasattr(agent, "cleanup") and callable(agent.cleanup):
                agent.cleanup()
        self.clear_history()
    def update_results_base_dir(self, base_dir: str) -> None:
        """Update the base directory for results storage."""
        self.agents["mcp_server"].update_results_base_dir(base_dir)
        self.base_dir = base_dir
        self.interaction_history.clear()
