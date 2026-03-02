from .base_prompt import PromptPack, BasePrompt
from .base_template import MCP_TOOL_REGISTRATION, TESTING_SECTION

class CodeAgentPrompt(BasePrompt):
    """
    Tool Genesis Prompt
    """
    
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

## Task description:
{self.task_description}


## Output format:
- JSON object with "metadata", "server_name", and "tools" keys.
- "tools" is an array of tool objects, each following the per-tool structure.

"""


        return PromptPack(user_prompt)


    def build_env_code_prompt(self, tool_schema: str) -> PromptPack:
        user_prompt = f"""
## Task
Generate a deterministic, benchmark-valid MCP executable environment class in Python that implements EXACTLY the tools in the provided Tool Schema.

## Objective
- Determinism: No randomness; no time-based calls (datetime.now/time.time); no network; no reading external state unless explicitly provided in INPUTS.
- Semantic grounding: Outputs must be derived from inputs and/or internal state (no unrelated hard-coded results).
- Parameter usage: Every declared tool parameter MUST affect at least one returned field.
- Stable schema: For the same tool, keep keys stable across calls (deterministic output schema).
- MCP return type: Every MCP tool MUST return a JSON-serializable dict at the top level (never list/str/int).

## WORKFLOW
Follow this execution protocol strictly (finite-state, do not skip steps):
1) WRITE_SERVER: Implement the MCP server class with tools registered via FastMCP.
2) WRITE_TESTS : Write pytest tests (2 per tool: OK + INVALID).
3) RUN_TESTS: run pytest.
4) FIX: If pytest fails, apply minimal patches only, then go back to RUN_TESTS.
5) FINAL: Exit only when a stop condition is met.

Stop conditions:
- pytest passes, OR
- max_fix_iters is reached, OR
- required execution tools are unavailable.


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
    - Must implement:
      - `load_state(state: Dict[str, Any]) -> None` (restore internal state)
      - `get_state() -> Dict[str, Any]` (return full JSON-serializable state)
    - State must remain JSON-serializable (no bytes/custom classes/non-serializable objects).

**OUTPUT CONTRACT**:
    - Return a JSON-serializable dict at the top level (never list/tuple/str/int).
    - If the natural result is a list, wrap it (e.g., {{"items":[...], "count": N, "meta": {...}}}).
    - All input parameters MUST affect at least one field in the returned dict.
    - Keep key names stable across calls for the same tool.

MCP TOOL REGISTRATION EXAMPLE:
{MCP_TOOL_REGISTRATION}

## Testing Guidance
{TESTING_SECTION}

## Never DO
- Do not use randomness or time-based values (datetime.now/time.time).
- Do not hard-code outputs unrelated to inputs/state (no fixed semantic constants).
- Do not ignore any tool parameter: each parameter must influence the returned dict.
- Do not return non-dict top-level values (no list/str/int).
- Do not return non-JSON-serializable objects (e.g., bytes, custom classes, TextContent).


## Tool Schema:
```json
{tool_schema}
```

## Task Description:
{self.task_description}

## Output format:
```{self.final_language}
You final MCP server class code
```
    """
        return PromptPack(user_prompt)
