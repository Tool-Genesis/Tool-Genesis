
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

from src.utils.llm import call_llm, extract_code

COMBINED_JSON_PATH = Path("data/tools/combined_tools.json")   
REG_FILE = Path("data/tools/single_env_registry.json")

from .utils import (
    load_server_def
    )
from .test_utils import only_validate

def generate_environment_code(mcp_data: Dict[str, Any]) -> str:
    """Generate environment code using LLM based on MCP tool definitions."""
    
    # Extract tool information
    tools_info = mcp_data.get('tools', [])
    metadata = mcp_data.get('metadata', {})
    
    # Read the UnifiedBaseEnv template
    base_env_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'unified_base_env.py')
    with open(base_env_path, 'r', encoding='utf-8') as f:
        base_env_template = f.read()
    
    # Create the prompt for LLM
    system_prompt = f"""You are an expert Python developer specialized in creating reinforcement learning environments with MCP (Model Context Protocol) server integration. 
    
Your task is to generate a complete Python environment class that extends UnifiedBaseEnv and implements simulated MCP server functionality.

TEMPLATE CODE TO EXTEND:
```python
{base_env_template}
```

MCP INTEGRATION REQUIREMENTS:
1. Use FastMCP to create a simulated MCP server
2. Register MCP tools using @mcp.tool() decorator
3. For each tool, choose the appropriate implementation approach:
   - Direct implementation: For tools with simple, deterministic behavior (math, file ops, data processing)
   - LLM simulation: For tools requiring external services or complex behavior (web APIs, search engines, proprietary data)
4. Maintain internal state for tool operations
5. Handle tool parameter validation
6. Provide meaningful and realistic tool responses

IMPLEMENTATION REQUIREMENTS:
1. Create a class that inherits from UnifiedBaseEnv
2. Override _initialize_mcp_server() to register MCP tools
3. Implement all abstract methods: _execute_action, _calculate_reward, _is_task_complete, _reset_environment_state, _get_environment_state
4. Create MCP tool implementations for each tool in the specification using @mcp.tool() decorator
5. Design appropriate task scenarios and reward functions
6. Maintain internal state for the environment simulation
7. Use proper error handling and validation
8. Follow Python best practices and include docstrings

MCP TOOL PATTERN:
```python
@self.mcp_server.tool()
def tool_name(param1: type, param2: type) -> return_type:
    \"\"\"Tool description\"\"\"
    # Implementation logic
    return result
```

9. **In the end of code please add a standard executable entry‐point**:
   ```python
    if __name__ == "__main__":
        import json
        COMBINED = "{COMBINED_JSON_PATH}"
        SERVER   = "{metadata["name"]}"
        data = json.load(open(COMBINED, "r", encoding="utf-8"))
        srv  = next(s for s in data["servers"] if s["server_name"] == SERVER)
        tools = srv["tools"]
        mcp_server = <YourClassName>(tools)._initialize_mcp_server()
        mcp_server.run()
    ```
The generated environment should be a realistic MCP server simulation functional for training RL agents.
Return ONLY the Python code without any explanations or markdown formatting."""

    # Create detailed prompt with tool information
    tools_description = []
    for tool in tools_info:
        tool_desc = f"- {tool['name']}: {tool['description']}"
        if 'parameters' in tool:
            params = [f"{p['name']} ({p['type']})" for p in tool['parameters']]
            tool_desc += f" Parameters: {', '.join(params)}"
        tools_description.append(tool_desc)
    
    user_prompt = f"""Generate a Python environment class that simulates the following MCP server:

MCP SERVER METADATA:
- Name: {metadata.get('name', 'Unknown')}
- Description: {metadata.get('description', 'No description')}
- URL: {metadata.get('url', 'No URL')}

AVAILABLE MCP TOOLS:
{chr(10).join(tools_description)}

IMPLEMENTATION GUIDELINES:
0. import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
1. from src.utils.unified_base_env import UnifiedBaseEnv, ToolDefinition 
2. Import FastMCP for MCP server simulation
3. Create a class that extends UnifiedBaseEnv
4. Override _initialize_mcp_server() method to create and register MCP tools
5. For each tool, decide implementation approach:
   - Use direct code simulation for simple, deterministic tools (math operations, data processing)
   - Use LLM simulation (via utils.llm.call_llm) for complex tools requiring external services
6. Each tool should be registered using @mcp.tool() decorator
7. Implement realistic simulation behavior for each tool
8. Create meaningful task scenarios that require multi-step tool usage
9. Implement reward functions that encourage correct tool usage
10. Maintain environment state that reflects tool operations
11. Handle parameter validation and error cases

EXAMPLE MCP TOOL REGISTRATION:
```python
def _initialize_mcp_server(self):
    if not FastMCP:
        return None
    mcp = FastMCP(self.mcp_server_name)
    
    @mcp.tool()
    def tool_name(param: str) -> str:
        \"\"\"Tool description\"\"\"
        # Simulate tool behavior
        return result
    
    return mcp
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
    
    return generated_code

def generate_environment_from_mcp(server_name: str) -> str:
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
    
    # Generate environment code
    env_code = generate_environment_code(mcp_data)
    
    # Determine environment name
    env_name = mcp_data.get('metadata', {}).get('name', 'unknown').replace('-', '_')
    
    # Generate the test python script
    # only_validate(env_code, mcp_data)
    return env_code,env_name
