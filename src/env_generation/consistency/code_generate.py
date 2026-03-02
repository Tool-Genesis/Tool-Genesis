from typing import Tuple
from src.utils.llm import call_llm
import re

MAX_TURN = 3

def _build_tool_schema_prompt(task_describe: str) -> str:
    return f"""
## Task
Generate a MCP server tools schema for the document.

## Naming & Schema Conventions
- tool-name: MUST be lowercase, start with a letter, and contain only letters, numbers, and underscores.
- type:object (and object items in arrays):
  - MUST explicitly declare "additionalProperties": false unless the shape is truly dynamic.
  - MUST include a fully defined "properties": {{}} object, even if it is empty.
  - "required": [] MUST list all mandatory properties for that object.

## Core rules
- Preserve existing "metadata". If "server_name" exists, keep it; else if metadata.name exists, set "server_name" = metadata.name.
- Replace "tools" with the converted list. Do not rename original input parameter names.
- Every property you emit MUST include both "type" and "description".

## Per-tool structure
- name: Stable tool name.
- description: One concise line on what the tool does and when to use it.
- inputSchema (JSON Schema object):
  {{
    "type": "object",
    "additionalProperties": false,
    "properties": {{
      "<param_name>": {{ "type": "...", "description": "..." }},
      ...
    }},
    "required": [ ... ]
  }}

### Notes
- For arrays, "items" MUST include its own "type" and "description"; if "items" is an object, define its "properties" and "required".
- For objects, define "properties" with type+description. If the shape is truly dynamic, set "additionalProperties": true and explain keys/values.
- If a tool has no inputs, emit: {{ "type": "object", "additionalProperties": false, "properties": {{}}, "required": [] }}.
- Type inference: array -> "array"; dict/JSON -> "object"; boolean -> "boolean"; integer-like -> "integer"; numeric -> "number"; otherwise -> "string".

## Output format
Return exactly one fenced code block:
```json
{{"metadata": ..., "server_name": "...", "tools": [ ... ]}}
```

## Task description
{task_describe}
""".strip()

def _build_env_code_prompt(task_describe: str,tool_schema: str) -> str:
    return f"""
## Task
Generate an executable environment class python code based on the given tool schema.

## IMPLEMENTATION GUIDELINES (Minimal)
- Implement a production-ready Python class that imports `App` via `from src.apps import App`, imports `FastMCP` via `from mcp.server.fastmcp import FastMCP`, and inherits from `App`.
- Constructor must be: `def __init__(self, spec: Optional[Dict[str, Any]] = None)`
- Override `_initialize_mcp_server()` and register every tool using `@mcp.tool()`.
- Tool names and parameter names must exactly match the MCP SERVER SPECIFICATION; every parameter must have an explicit type annotation.
- Implement `get_state() -> Dict[str, Any]` and `load_state(state: Dict[str, Any]) -> None` with JSON-serializable state (use direct attribute assignment).
- Tool logic must use all declared parameters and current state; do not use randomness or fixed placeholder outputs.

## Output format
Return exactly one fenced code block:
```python
# environment class code
```

## Task description
{task_describe}

## Tool Schema
```json
{tool_schema}
```
""".strip()

def _extract_json(resp: str) -> str:
    s = resp if isinstance(resp, str) else str(resp)
    j = re.findall(r"```json\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    return j[-1].strip() if j else s

def _extract_python(resp: str) -> str:
    s = resp if isinstance(resp, str) else str(resp)
    p = re.findall(r"```python\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    return p[-1].strip() if p else s

def direct_generate(task_describe: str, model: str | None = None, output_dir: str | None = None, platform: str | None = None) -> Tuple[str, str]:
    schema_prompt = _build_tool_schema_prompt(task_describe)
    schema_resp = call_llm(
        schema_prompt,
        system_prompt="You are a helpful assistant that generates JSON schemas.",
        max_tokens=4096,
        temperature=0.2,
        model=(model or "openai/gpt-4.1-mini"),
        platform=(platform or "bailian"),
    )
    tool_schema = _extract_json(schema_resp)
    env_prompt = _build_env_code_prompt(task_describe,tool_schema)
    env_resp = call_llm(
        env_prompt,
        system_prompt="You are a helpful assistant that generates Python code.",
        max_tokens=8192,
        temperature=0.2,
        model=(model or "openai/gpt-4.1-mini"),
        platform=(platform or "bailian"),
    )
    env_code = _extract_python(env_resp)
    return tool_schema, env_code
