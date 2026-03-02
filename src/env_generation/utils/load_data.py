import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List,Literal

COMBINED_JSON_PATH = Path("data/tools/combined_tools.json")
SELECTED_SERVER_PATH   = Path("data/tools/selected_server.txt")
ENV_REGISTRY_PATH = Path("data/tools/env_registry.json")
def load_server_def(server_name: str, combined_path: Path = COMBINED_JSON_PATH) -> Dict[str, Any]:
    """from combined_tools.json get server def by server_name"""
    if not combined_path.is_file():
        raise FileNotFoundError(f"Combined file not found: {combined_path}")
    combined = json.load(combined_path.open("r", encoding="utf-8"))

    for srv in combined.get("servers", []):
        if srv.get("server_name") == server_name or srv.get("server_name").replace("-", "_").replace(" ", "_") == server_name:
            srv["server_name"] = server_name
            return srv

    raise ValueError(f"Server '{server_name}' not found in {combined_path}")

def load_selected_server(selected_server_path: Path = SELECTED_SERVER_PATH) -> List[str]:
    """load selected server from selected_server_path"""
    if not selected_server_path.is_file():
        raise FileNotFoundError(f"Selected server file not found: {selected_server_path}")
    return [line.strip() for line in selected_server_path.read_text().splitlines() if line.strip()]

def load_env_registry(env_registry_path: Path = ENV_REGISTRY_PATH) -> Dict[str, Any]:
    """load env registry from env_registry_path"""
    if not env_registry_path.is_file():
        raise FileNotFoundError(f"Env registry file not found: {env_registry_path}")
    return json.load(env_registry_path.open("r", encoding="utf-8"))

def tool_catalog(
    servers: List[Dict[str, Any]],
    mode: Literal["compact", "detailed"] = "detailed"
) -> str:
    """
    Build a clean, LLM-friendly catalogue of tools for prompt use.

    Parameters
    ----------
    servers : list
        List of MCP-server dicts (same shape as in combined_tools.json).
    mode : {"compact", "detailed"}, default "detailed"
        * "compact"   – Each tool: tool_name(param1:type, ...) — description.
        * "detailed"  – Includes parameters and parameter descriptions.

    Returns
    -------
    str
        Formatted catalogue, ready for direct embedding into prompts.
    """
    lines: List[str] = []
    for srv in servers:
        lines.append(f"{srv['server_name']}:")
        for tool in srv["tools"]:
            # Tool signature line
            param_list = tool.get("parameters", [])
            if param_list:
                params_str = ", ".join(f"{p['name']}:{p.get('type', 'any')}" for p in param_list)
                sig = f"{tool['name']}({params_str})"
            else:
                sig = f"{tool['name']}()"
            # Main line: tool signature + description
            desc = tool.get("description", "").strip()
            lines.append(f"  - {sig}: {desc}")
            # If detailed, list parameter explanations (indented)
            if mode == "detailed" and param_list:
                for p in param_list:
                    pname = p["name"]
                    ptype = p.get("type", "any")
                    pdesc = p.get("description", "")
                    lines.append(f"      • {pname} ({ptype}): {pdesc}")
        lines.append("")  # Blank line for readability between servers
    return "\n".join(lines)