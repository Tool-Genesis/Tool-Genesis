
#!/usr/bin/env python3
"""
Naive Single Turn Environment Generator

This script generates a basic environment implementation from MCP server tool definitions.
It takes a JSON file containing MCP tool definitions and generates a Python environment
class that extends the UnifiedBaseEnv template.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

from .utils import (
    save_environment,
    load_server_def
    )

from .test_utils import validate_and_refine

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.llm import call_llm,extract_code
from src.env_generation.simulation_env_planner import generate_simulation_env_plan,SIMULATION_PLAN_SCHEME
from src.apps.file_system import FileSystem as MCPFileSystem

COMBINED_JSON_PATH = Path("data/tools/combined_tools.json")   
REG_FILE = Path("data/tools/env_registry.json")
EXTENRAL_DATA_PATH = Path("temp/data")

def build_plan_prompt(
    plan: Dict[str, Any],
) -> str:
    """
    Build the full system prompt given a simulation_plan and the base env template.

    Args:
        plan:               The simulation_plan dict returned by the planner.

    Returns:
        A formatted multi-line system prompt string.
    """
    tool_strategies = plan.get("tool_strategies", {})
    init_hint = plan.get("init_hint", "")
    init_strategy = plan.get("init_strategy", {})

    # Build files_block_text
    files_block_text = ""
    if plan.get("files"):
        files_block_text = "EXTERNAL FILES:\n"
        for spec, path in zip(plan["files"], plan["files_path"]):
            files_block_text += (
                f"- {spec['name']}: {spec['description']}\n"
                f"    stored at: `{path}`\n"
            )

    # Build tool_strategy_text
    tool_strategy_text = ""
    if tool_strategies:
        tool_strategy_text = (
            "TOOL STRATEGIES:\n"
            + "\n".join([f"- {tool}: {desc}" for tool, desc in tool_strategies.items()])
        )

    # Build init_strategy_text
    init_strategy_text = ""
    if init_strategy:
        init_strategy_text = "INIT STRATEGY:\n"
        for field, desc in init_strategy.items():
            if isinstance(desc, dict):
                init_strategy_text += f"- {field}:\n"
                for subk, subv in desc.items():
                    init_strategy_text += f"    • {subk}: {subv}\n"
            else:
                init_strategy_text += f"- {field}: {desc}\n"

    # -------- external_usage block (docs/media via MCPFileSystem) --------------
    external_usage = ""
    if plan.get("files"):
        ext_tools = [
            t for t, steps in tool_strategies.items()
            if any("FILE_" in step for step in steps)
        ]
        ext_list = ", ".join(ext_tools) if ext_tools else "<none>"

        external_usage = f"""
    EXTERNAL FILES USAGE
    --------------------
    Instantiate once in __init__:
    ```python
    from pathlib import Path
    from src.file_system import MCPFileSystem
    self.fs = MCPFileSystem(Path(fs_root))
    ```
    
    Typical operations:
    
    # read / slice
    ```python
    txt  = self.fs.load_text(file_id)
    info = self.fs.load_info(file_id)
    part = self.fs.slice_text(file_id, 0, 100)
    ```
    
    # create / overwrite text
    ```python
    self.fs.save_text(file_id, fmt="docx", text=content,
                    meta={{"page_count": 1}})
    ```
    
    # create binary placeholder
    ```python
    self.fs.save_binary(img_id, fmt="png")
    ```
    
    # update metadata only
    ```python
    self.fs.update_meta(file_id, metadata_patch={{"author": "Alice"}})
    ```
    # remove entry
    ```python
    self.fs.delete_file(file_id)
    ```
    
    All FILE_* DSL steps must be implemented with these methods –
    never open files directly with the built-in open().

    Tools using MCPFileSystem: {ext_list}
"""

    # LLM simulation usage block
    llm_usage = ""
    sim_tools = [
        t
        for t, steps in tool_strategies.items()
        if any("LLM_CALL" in step for step in steps)
    ]
    if sim_tools:
        sim_list = ", ".join(sim_tools) if sim_tools else "<none>"
        llm_usage = f'''
LLM_SIMULATION FUNCTION USAGE:
- Make sure to import the LLM call utility:
    ```python
    from src.utils.llm import call_llm
    ```
- Define the target Python function:
    ```python
    def extract_key_points(text: str, max_points: int) -> List[str]:
        """Extract up to `max_points` salient key points from the input `text`."""
        # Simulated by AI
    ```

- Tools using LLM simulation: {sim_list}

- For any tool marked "llm_simulation", 
1. **System Prompt**:  
   You are an AI assistant simulating the `{{tool_name}}` function: {{tool_description}}.
2. **User Prompt**:  
   Simulate `{{tool_name}}` with: {{param1}}={{param1}}, {{param2}}={{param2}}, … .  
   Return only the function’s result.
3. **State Update**:  
   If this tool mutates the environment state, after obtaining the result include:  
   “Then update the environment state accordingly.”
use this template:
    ```python
    @self.mcp_server.tool()
    def {{tool_name}}({{param_list}}) -> {{return_type}}:
        """{{desc}}"""
        from src.utils.llm import call_llm
        system_prompt = """{{system_p}}"""
        user_prompt = f"""{{user_p}}"""
        result = call_llm(
            text=user_prompt,
            system_prompt=system_prompt,
            max_tokens=1500,
            temperature=0.2
        )
        # If this tool modifies environment state, update it here:
        # e.g., self._state['...'] = result or reload_data()
        return result
'''

    # Compose final prompt
    prompt = f"""

Simulation Plan Details:
  Tools_Strategy  : {tool_strategy_text}
  Snapshot files  : {', '.join(plan.get('files_path', []))}
  Init hint       : {init_hint}
  Init strategy   : {', '.join(init_strategy.keys())}

{files_block_text if files_block_text else ""}
INIT HINT:
{init_hint}

{init_strategy_text if init_strategy_text else ""}

{external_usage}

{llm_usage}

"""

    return prompt


def generate_environment_code(mcp_data: Dict[str, Any],
                              env_plan: Dict[str, Any]
                              ) -> str:
    """
    Ask an LLM to emit a fully-functional simulated MCP environment that follows
    the given simulation_plan (state_only / external_files / llm_simulation).

    Parameters
    ----------
    mcp_data  : dict   – original server spec (metadata + tools)
    env_plan  : dict   – simulation_plan generated by generate_simulation_plan()

    Returns
    -------
    str  – raw Python source code for the environment (no markdown)
    """
    
    # Extract tool information
    tools_info = mcp_data.get('tools', [])
    metadata = mcp_data.get('metadata', {})
    
    # Read the UnifiedBaseEnv template
    base_env_path = os.path.join(
        os.path.dirname(__file__), "..", "utils", "unified_base_env.py"
    )
    with open(base_env_path, "r", encoding="utf-8") as f:
        base_env_template = f.read()
    
    
    api_md = MCPFileSystem.describe_api([
    "file_exists","save_text", "save_binary", "load_text",
    "load_info","update_meta", "delete_file", "list_files"
    ])
    
    # Create the prompt for LLM
    system_prompt = f"""You are an expert Python developer specialized in building reinforcement-learning environments for MCP servers. Follow the **simulation_plan** strictly—no external network calls, only in-memory state or provided snapshot files.

TEMPLATE SIMULATION PLAN:
{SIMULATION_PLAN_SCHEME}

TEMPLATE CODE TO EXTEND:
```python
{base_env_template}
```

MCP INTEGRATION REQUIREMENTS:
1. Use FastMCP to create a simulated MCP server
2. Register MCP tools using @mcp.tool() decorator
3. For each tool, implement the following in terms of its internal DSL:
    - **STATE_**: Operate on in-memory structures and commit any create/update/delete operations by modifying those structures.
    - **FILE_**: Read/write local snapshot files in `plan["files_path"]`, then reload or patch the in-memory state so future calls see the updates.
    - **LLM_CALL**: Call `call_llm()` to produce results, then **only** apply to internal state (no writing to files).

3a. State mutations:
    - Any tool may read **and**modify the internal state of the environment.
    - **STATE_** tools must modify in-memory data structures directly.
    - **FILE_** tools must also persist mutations to snapshot files, then update the in-memory state.
    - **LLM_CALL** tools should only apply any returned changes to the in-memory state (never write to disk).
    - After any mutation, ensure `_get_environment_state()` reflects the new state.
4. Maintain internal state for tool operations
5. Handle tool parameter validation
6. Provide meaningful and realistic tool responses
7. Parameter usage requirement – Every declared parameter must be consumed. The logic of each tool has to reference and meaningfully use all parameters provided in its signature.

IMPLEMENTATION REQUIREMENTS:
1. Create a class that inherits from UnifiedBaseEnv, accepting `tools: List[ToolDefinition]` in its constructor.
2. **Import & use the lightweight file system**
   {api_md}
3. Override `_initialize_mcp_server()` to register MCP tools via `@mcp.tool()` decorators.
4. Implement all abstract methods: _execute_action, _calculate_reward, _is_task_complete, _reset_environment_state, _set_environment_state,_get_environment_state
5. Create MCP tool implementations for each tool in the specification using @mcp.tool() decorator
6. Design appropriate task scenarios and reward functions
7. Maintain internal state for the environment simulation
8. Use proper error handling and validation
9. Follow Python best practices and include docstrings


MCP TOOL PATTERN:
```python
@self.mcp_server.tool()
def tool_name(param1: type, param2: type) -> return_type:
    \"\"\"Tool description\"\"\"
    # Implementation logic
    return result
```
    
The generated environment should be a realistic MCP server simulation functional for training RL agents.
Return ONLY the Python code without any explanations or markdown formatting.
All identifiers, comments, and docstrings must be written in English only."""

    # Create detailed prompt with tool information
    tools_description = []
    for tool in tools_info:
        tool_desc = f"- {tool['name']}: {tool['description']}"
        if 'parameters' in tool:
            params = [f"{p['name']} ({p['type']})" for p in tool['parameters']]
            tool_desc += f" Parameters: {', '.join(params)}"
        tools_description.append(tool_desc)
    
    strategy_plan = build_plan_prompt(env_plan)
    
    user_prompt = f"""Generate a Python environment class that simulates the following MCP server:

MCP SERVER METADATA:
- Name: {metadata.get('name', 'Unknown')}
- Description: {metadata.get('description', 'No description')}
- URL: {metadata.get('url', 'No URL')}

AVAILABLE MCP TOOLS:
{chr(10).join(tools_description)}

SIMULATION STRATEGY:
{strategy_plan}

IMPLEMENTATION GUIDELINES:
0. import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..")) must be added to the beginning of the file.
1. from src.utils.unified_base_env import UnifiedBaseEnv, ToolDefinition 
2. from mcp.server.fastmcp import FastMCP ; from src.file_system import MCPFileSystem 
3. from src.utils.llm import call_llm if the strategy is llm_simulation
4. Create a class that extends UnifiedBaseEnv
5. Override _initialize_mcp_server() method to create and register MCP tools
6. For each tool, decide implementation approach:
   - Use direct code simulation for simple, deterministic tools (math operations, data processing)
   - Use LLM simulation (via utils.llm.call_llm) for complex tools requiring external services
7. Each tool should be registered using @mcp.tool() decorator
8. Implement realistic simulation behavior for each tool
9. Create meaningful task scenarios that require multi-step tool usage
10. Implement reward functions that encourage correct tool usage
11. Maintain environment state that reflects tool operations
12. Handle parameter validation and error cases

EXAMPLE MCP TOOL REGISTRATION:
```python
def _initialize_mcp_server(self):
    if not FastMCP:
        return None
    mcp = FastMCP({metadata.get('name', 'Unknown')})
    
    @mcp.tool()
    def tool_name(param: str) -> str:
        \"\"\"Tool description\"\"\"
        # Simulate tool behavior
        return result
    mcp._env = self
    return mcp
```

```python
def _execute_action(self, action: str) -> dict:
    # Parse action and execute the corresponding tool
    action_data = json.loads(action)
    tool_name = action_data["tool"]
    tool_args = action_data["args"]
    tool = getattr(self.mcp_server, tool_name)
    return tool(**tool_args)

```

13. At the end of the file, append this exact executable entry point:
   ```python
    if __name__ == "__main__":
        import json
        COMBINED = "{COMBINED_JSON_PATH}"
        SERVER   = "{metadata["name"]}"
        data = json.load(open(COMBINED, "r", encoding="utf-8"))
        srv  = next(s for s in data["servers"] if s["server_name"] == SERVER)
        tools = srv["tools"]
        fs_root = f"temp/data/<SERVER.replace(' ', '_').replace('-', '_')>"
        mcp_server = <YourClassName>(tools,fs_root)._initialize_mcp_server()
        mcp_server.run()
    ```

Generate a complete, functional Python environment class with MCP server simulation."""

    # Call LLM to generate the code
    generated_code = call_llm(
        text=user_prompt,
        system_prompt=system_prompt,
        max_tokens=3000,
        temperature=0.3
    )
    generated_code = extract_code(generated_code)
    
    return generated_code,system_prompt

def generate_environment_from_mcp(server_name: str, output_dir: str = "temp/generated_envs") -> str:
    """
    Main function to generate environment from MCP tool definition.
    
    Args:
        server_name: Name of the MCP server
        output_dir: Directory to save generated environment
        
    Returns:
        Path to generated environment file
    """
    
    # Load MCP tool definitions
    mcp_data = load_server_def(server_name)
    
    # Determine environment name
    env_name = mcp_data.get('metadata', {}).get('name', 'unknown').replace('-', '_').replace(" ", "_")
    
    # Generate simulation plan
    # print("Generating simulation plan...")
    json_path = EXTENRAL_DATA_PATH / env_name
    plan = generate_simulation_env_plan(mcp_data, json_path)
        
    # Generate environment code
    # print("Generating environment code...")
    env_code,gen_code_system_prompt = generate_environment_code(mcp_data,plan)
        
    # Validate and refine
    # print("Validating and refining environment code...")
    env_code = validate_and_refine(env_code, gen_code_system_prompt, plan,mcp_data,print_info=False)
    
    # Save the environment
    # print("Saving environment...")
    output_path = save_environment(env_code, output_dir, env_name)

    return output_path
