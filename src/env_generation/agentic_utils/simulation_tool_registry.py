# simulation_tool_registry.py

from pathlib import Path
import json
from typing import Dict, List

SIMULATION_TOOL_REGISTRY = {
    "call_llm": {
        "description": (
            "Use for text tasks that require LLM support, such as summarization, rewriting, or dialogue simulation. "
            "Use this tool only when realistic simulation is not possible via code or library."
        ),
        "import": "from src.apps.llm import call_llm",
        "parameters": [
            {"name": "text", "required": True, "description": "The main input text to process."},
            {"name": "system_prompt", "required": True, "description": "The system instruction describing the task (e.g., summarization, rewriting)."},
        ],
        "usage": '''response = call_llm(
    text="Text to process",
    system_prompt="Summarize the following in 2 short sentences:"
)''',
        "notes": '''Only text and system_prompt must be supplied.''',
        "simulation": '''
        Use only when the task truly requires natural language understanding/generation, OR when an MCP-server tool cannot realistically call its real backend (e.g., missing API keys, offline/sandbox, rate/cost limits, dependency not ready). In such cases, use `call_llm` as a controlled stand-in so the tool contract and pipeline remain runnable.

        Stand-in scope examples:
        - Replace a real model or external API.
        - Retrieval / RAG / ranking chain (query rewrite, re-ranking, multi-source fusion).
        - Evaluation / quality control / reward-model judging (scoring, safety/compliance checks, A/B arbitration).
        - Decision & policy engine (routing/splitting, guardrail decisions, fallback/rollback planning).
        - Business mini-models (intent/sentiment/topic classification, tag suggestions, style consistency checks).

        Requirements:
        - Be **explicit and concrete** in the `system_prompt`. 
        ''',
    },
    # 可继续添加其它工具
}

def get_raw_simulation_tool_info(tool_names: list[str]) -> dict:
    """
    Returns a dict of tool_name:tool_info for each tool in tool_names, if it exists in the registry.
    """
    return {name: SIMULATION_TOOL_REGISTRY[name] for name in tool_names if name in SIMULATION_TOOL_REGISTRY}

def tools_info_to_xml(tool_info: dict) -> str:
    """
    Convert the tool info dict (tool_name:tool_info) to XML string for LLM prompt or documentation.
    """
    lines = ["<tools>"]
    for name, info in tool_info.items():
        lines.append(f'  <tool name="{name}">')
        lines.append(f'    <description>{info["description"]}</description>')
        lines.append(f'    <import>{info["import"]}</import>')
        lines.append(f'    <parameters>')
        for param in info["parameters"]:
            lines.append(
                f'      <param name="{param["name"]}" required="{str(param["required"]).lower()}">{param["description"]}</param>'
            )
        lines.append(f'    </parameters>')
        lines.append(f'    <usage>')
        lines.append(f'    ```python')
        for l in info["usage"].splitlines():
            lines.append("      " + l)
        lines.append(f'    ```')
        lines.append(f'    </usage>')
        lines.append(f'    <notes>')
        for l in info["notes"].splitlines():
            lines.append("      " + l)
        lines.append(f'    </notes>')
        lines.append(f'    <simulation>{info["simulation"]}</simulation>')
        lines.append(f'  </tool>')
    lines.append("</tools>")
    return "\n".join(lines)

def get_simulation_tool_info(tool_names: List[str]) -> str:
    info = get_raw_simulation_tool_info(tool_names)
    return tools_info_to_xml(info)


if __name__ == "__main__":
    selected = ["call_llm", "extract_document_content"]
    xml_str = get_simulation_tool_info(selected)
    print(xml_str)
