from .base_prompt import PromptPack, BasePrompt
from .base_template import MCP_TOOL_REGISTRATION

class ToolGenesisPrompt(BasePrompt):
    def __init__(self, task_description: str, final_language: str = "python") -> None:
        super().__init__(final_language=final_language)
        self.task_description = task_description

    def build_tool_schema_prompt(self, task_describe: str) -> PromptPack:
        user_prompt = f"""
## Task
Generate a MCP server tools schema for the document

## Naming & Schema Conventions
- `tool-name`: MUST be all lowercase and MUST start with a letter and contain only letters, numbers, and underscores. No spaces or hyphens.
- `type:object` or object in type:List :
  - MUST explicitly declare `"additionalProperties": false`.
  - MUST have a fully defined `"properties": {{}}` object, even if it is empty.
  - The `"required": []` array MUST list all mandatory properties for that object.

## Core rules
- Preserve existing "metadata". If "server_name" exists, keep it; else if metadata.name exists, set "server_name" = metadata.name.
- Replace "tools" with the converted list. Do not rename original input parameter names.
- Every property you emit MUST include both "type" and "description".

### Per-tool structure
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
    "required": [ ... ]    // subset of the above properties; use [] if none
  }}
  Notes:
  - For arrays, "items" MUST include its own "type" and "description"; if "items" is an object, define its "properties" and "required".
  - For objects, define "properties" with type+description. If the shape is truly dynamic, set "additionalProperties": true and explain what keys/values represent.
  - If a tool has no inputs, emit: {{ "type": "object", "additionalProperties": false, "properties": {{}}, "required": [] }}.
  - Type inference: array -> "array"; dict/JSON -> "object"; boolean -> "boolean"; integer-like -> "integer"; numeric -> "number"; otherwise -> "string".
  - Map each legacy parameter entry directly to a property with the same name (do NOT rename inputs).

## Output format:
- JSON object with "metadata", "server_name", and "tools" keys.
- "tools" is an array of tool objects, each following the per-tool structure.
  
## Task description:
{task_describe}

"""

        return PromptPack(user_prompt)


    def build_env_code_prompt(self, tool_schema: str) -> PromptPack:
        user_prompt = f"""
## Task
Generate an executable environment class {self.final_language} code based on the given tool schema.

## IMPLEMENTATION GUIDELINES

**CLASS AND IMPORT REQUIREMENTS**:
    - Use `from src.apps import App` and inherit from `App`.
    - Import `FastMCP` from `mcp.server.fastmcp`.
    - FORBIDDEN: Do not use `from __future__ import annotations`.
    - Constructor MUST be `def __init__(self, spec: Optional[Dict[str, Any]] = None)`; parse known keys from `spec` to initialize state.
**TOOL DEFINITION AND REGISTRATION**:
    - Override `_initialize_mcp_server()` to register all MCP tools.
    - Use the `@mcp.tool()` decorator for each tool, ensuring robust parameter validation.
    - Tool and parameter names MUST be identical to the `MCP SERVER SPECIFICATION`.
    - CRITICAL: All tool function parameters MUST have explicit type annotations (e.g., `param: str`).
**STATE AND DATA INITIALIZATION**:
    - Update state via direct property assignment (e.g., `self.gender = 1`).
    - Use realistic default values for dataclass fields (e.g., `name: str = 'John Doe'`).
    - Implement `load_state(state: Dict[str, Any]) -> None` to restore internal state from a JSON-serializable dict.
    - Implement `get_state() -> Dict[str, Any]` to return the full JSON-serializable state dict for auto-save.
**TOOL IMPLEMENTATION LOGIC**:
    - **Dynamic Output**: Outputs must be derived from inputs and state (no hard-coded or random values).
    - **Parameter Usage**: All declared parameters must be used in the tool's logic.
    - **Output Relevance**: Outputs must be contextually relevant and realistic.
**GENERAL REQUIREMENTS**:
    - **Language**: All code, comments, and text must be in English.
    - **Deliverables**: Provide a complete, production-ready MCP server class.
MCP TOOL REGISTRATION EXAMPLE:
{MCP_TOOL_REGISTRATION}

## Output format:
```{self.final_language}
You final MCP server class code
```

## Tool Schema:
```json
{tool_schema}
```
        """
        return PromptPack(user_prompt)
