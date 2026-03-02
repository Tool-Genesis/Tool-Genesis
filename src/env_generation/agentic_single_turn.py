import os
import sys
import re
import asyncio
import json
from pathlib import Path

from .utils import (
    load_server_def
    )
from .agentic_utils import (
    get_simulation_tool_info,
)

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

COMBINED_JSON_PATH = Path("data/tools/combined_tools.json")   
REG_FILE = Path("data/tools/env_registry.json")

from src.utils.llm import extract_code
from src.core.configs.agents import DeepResearchAgentConfig


async def single_agent_code_gen(single_agent,
                                mcp_data: str) -> str:
    """
    Use a single agent to both research environment implementation details
    and generate a complete, self-contained Python environment class.
    """
    
    # Read the UnifiedBaseEnv template
    base_env_path = os.path.join(
    os.path.dirname(__file__), "..", "utils", "unified_base_env.py"
)
    with open(base_env_path, "r", encoding="utf-8") as f:
        base_env_template = f.read()
    
    base_test_env_path = os.path.join(
        os.path.dirname(__file__), "test_utils", "unified_agentic_env_test.py"
    )
    with open(base_test_env_path, "r", encoding="utf-8") as f:
        base_test_env_template = f.read()
    
    # Extract tool information
    tools_info = mcp_data.get('tools', [])
    metadata = mcp_data.get('metadata', {})

    
    # Create detailed prompt with tool information
    tools_description = []
    for tool in tools_info:
        tool_desc = f"- {tool['name']}: {tool['description']}"
        if 'parameters' in tool:
            params = [f"{p['name']} ({p['type']})" for p in tool['parameters']]
            tool_desc += f" Parameters: {', '.join(params)}"
        tools_description.append(tool_desc)
    
    selected = ["call_llm", "extract_document_content"]
    used_simulation_tool_info = get_simulation_tool_info(selected)
    
    full_prompt = f"""
Your task is to develop a complete, standalone Python class that implements the MCP server described above. This process consists of three strictly independent phases: research and code generation and code verification. Do not output any implementation code until the research phase is complete. And do not output the final code before the code has been verified. Make sure that the code has been verified layer by layer before outputting it.

<Research Requirements>
Before generating any code, MUST use the provided tools (`browser_search` and `browser_open`) to research:
- Identify conventions for tool registration, error handling, internal state management design.
- Carefully review whether the Tool-scheme is complete and logically correct. If you find the Tool-scheme is incomplete, unclear, or missing any important information (such as parameter types or expected behaviors), search for relevant information and propose specific ways to improve or supplement it.
- For each MCP tool, research and list a realistic simulation logic, and choose either direct code simulation or LLM simulation as appropriate. You are also encouraged to search for third-party libraries that may assist with tool implementation or enhance simulation realism.
- You must NOT output any code, pseudocode, or implementation details before completing this research step.
- At the end of this step, clearly summarize your Tool-scheme integrity check and list any improvements or completion actions you suggest.
</Research Requirements>

<Implementation Requirements>

Extend this base environment:

{base_env_template}

**Real-Simulation Rule **
> Every MCP tool must return a value that can be _deterministically_ derived  
> from its input parameters and the current environment state.  
> - NO hard-coded placeholders such as `"result": "ok"`  
> - NO purely random or constant outputs  
> - NO “TODO” / `pass` / empty stubs 

Implementation Steps:
1.  Do not redefine Tool Definition and Unified BaseEnv. Always use from src.utils.unified_base_env import UnifiedBaseEnv, ToolDefinition (if a class definition is generated, this import must be changed)
2.	Import FastMCP from mcp.server.fastmcp
4.	Implement a class that inherits from UnifiedBaseEnv
5.	Override _initialize_mcp_server() and register all MCP tools with realistic simulation logic
6.	For each tool, choose the simulation method (direct code, LLM, or third-party library) as recommended by your research, with strict parameter and exception handling.
7.	Register each tool using the @mcp.tool() decorator.
8.	Provide realistic simulated feedback for each tool.
9.	Design meaningful multi-step task scenarios.
10.	Maintain and synchronize internal environment state as needed.
11.	Ensure robust parameter validation and error handling throughout.

MCP TOOL REGISTRATION EXAMPLE:

```python
def _initialize_mcp_server(self):
    if not FastMCP:
        return None
    mcp = FastMCP({metadata.get('name', 'Unknown')})

    @mcp.tool()
    def tool_name(param: str) -> str:
        \"\"\"Tool description\"\"\"
        # Simulate tool behavior here
        return result
    mcp._env = self
    return mcp

def _execute_action(self, action: str) -> dict:
    # Parse the action and call the corresponding tool
    action_data = json.loads(action)
    tool_name = action_data["tool"]
    tool_args = action_data["args"]
    tool = getattr(self.mcp_server, tool_name)
    return tool(**tool_args)
```

12.	At the end of the file, Must make sure the following standard executable entry point is appended:

```python
if __name__ == "__main__":
    import json
    COMBINED = "{COMBINED_JSON_PATH}"
    SERVER = "{metadata["name"]}"
    data = json.load(open(COMBINED, "r", encoding="utf-8"))
    srv = next(s for s in data["servers"] if s["server_name"] == SERVER)
    tools = srv["tools"]
    fs_root = f"temp/data/{metadata["name"].replace(' ', '_').replace('-', '_')}"
    mcp_server = <YourClassName>(tools, fs_root)._initialize_mcp_server()
    mcp_server.run()
```

Replace <YourClassName> with your actual class name.


<LLM Simulation Tools>
Whenever possible, implement each tool’s behavior directly in Python—leveraging standard libraries or well-chosen third-party packages (after confirming they provide the required functionality).
Only if those options cannot deliver an accurate, deterministic simulation should you fall back to the auxiliary tools listed below.

```xml
{used_simulation_tool_info}
```

</LLM Simulation Tools>

</Implementation Requirements>


<Code-Testing Requirements>

After writing your environment implementation, you must execute multiple rounds of automated self-tests using the code_execution tool.

Testing rounds MUST include:

1. **Initial Validation Tests**
    - Execute basic scenarios to ensure all MCP tools respond correctly given valid, standard inputs.
    - Ensure tests pass without any errors.

2. **Robustness and Edge Case Tests**
    - Execute comprehensive scenarios, including but not limited to:
        - Invalid parameters, missing required arguments, incorrect formats.
        - Boundary values and edge conditions.
    - Explicitly confirm proper error handling and error message accuracy.
    - Ensure robustness tests pass without errors.

You MUST NOT produce a `<final>` response until you explicitly confirm BOTH rounds of tests above have been executed and passed successfully. Completing only the initial validation test does NOT satisfy the conditions for `<final>` output.

Each MCP tool must be validated with at least:
- One successful test scenario (valid input)
- One failing/error-triggering scenario (invalid input)

Ensure that no final implementation code is output until the entire testing process described above has been successfully completed and explicitly confirmed.

TEMPLATE TEST CODE TO EXTEND:
```python
{base_test_env_template}
```

MCP server test code integration requirements
• Keep the same file headers and fixtures as defined in the base template.
• Use the same import statements and DummyMCP patch fixtures without modification.
• Do not add or remove any auxiliary functions: `_get_underlying_env`, `_exec`, `_state`, and `reset_state`.
• Generate a test function named `test_<tool_name>` for each tool in the environment tool list.
• Each test function must follow the following three-step pattern:
    1. Get a random state through `s = _state(env)`.
    2. Construct the `args` dictionary with the values ​​required by the tool-scheme and the functions implemented in the Environment code.
    3. Call the tool using `_exec(env, "<tool_name>", args)` and assert the expected keys. The tool_name here must be the same as the tool function in the Environment code
    4. Only can use _exec to call <tool_name> like: out = _exec(env, <tool_name>, rgs)
• Use double quotes only for strings and JSON payloads.
• Leave exactly two blank lines after the auxiliary fixture section and before the first test.
• Leave exactly one blank line between subsequent test functions.
• Do not include any extra imports, comments, or text outside of boilerplate code and test definitions.

**Additional result validation rules (apply inside each test function)**
• If the tool returns a scalar (str/int/bool), assert that it is exactly equal to the documented literal.
• If the tool returns a dictionary:
    – Assert that all required top-level keys are present.
    – For each key, assert its value type (e.g. list/str/int).
    – **Use** `out.get("<key>") == <expected>` **rather than** `out == <expected dict>`  
    – When the spec defines a deterministic value (e.g. "status": "OK"), assert equality.
    – When the spec allows multiple items (e.g. "members": [...]):
* When the input is guaranteed to match, assert that the list is non-empty.
* Assert that each element satisfies the filter implied by the input status or arguments.
    • Never query external state to verify side effects; assertions are strictly limited to the returned payload.
    • Use only built-in `assert` and `isinstance`, no need to use additional libraries.
    • All identifiers, comments, and docstrings must be written in English only.
    • File usage rule – Any argument representing a file ID or path **must**
    come from the list provided in EVAL FILES PATH JSON.
    Never invent new filenames or directories that are not in that list.

</Code-Testing Requirements>

<MCP SERVER DESCRIBE>

- Name: {metadata.get('name', 'Unknown')}
- Description: {metadata.get('description', 'No description')}
- URL: {metadata.get('url', 'No URL')}

AVAILABLE MCP TOOLS:
{chr(10).join(tools_description)}

</MCP SERVER DESCRIBE>

<OUTPUT FORMAT>
Once all research is summarised, all tests pass, and you are ready to deliver,
return exactly one code block—that is, the complete environment
implementation *plus* the required executable entry point.
**Do NOT include any of your test code, fixtures, or helper scripts.**

<final>
```python
# Place your final, fully functional code here
```

</final>

**Important:**

*Emit the <final> section only after both validation and robustness
tests have been executed and passed.*
*The <final> block must contain implementation code only—no draft
fragments, no test suites, no explanatory text.*
</OUTPUT FORMAT>


"""

    response = await single_agent.run(
        full_prompt
    )

    code = extract_code(response.msg.content)
    return code


def generate_environment_from_single_agent(server_name: str,model:str="gpt-4.1-mini", output_dir: str = "temp/agentic/envs") -> str:
    """
    Main function to generate environment from MCP tool definition.
    
    Args:
        server_name: Name of the MCP server
        output_dir: Directory to save generated environment
        
    Returns:
        Path to generated environment file
    """
    
    # Load MCP tool definitions
    from src.core.agents import AgentFactory
    mcp_data = load_server_def(server_name)
    
    # Determine environment name
    env_name = mcp_data.get('metadata', {}).get('name', 'unknown').replace('-', '_').replace(" ", "_")
    
    env_output_dir = output_dir + "/" + env_name
    
    single_agent = AgentFactory.build_from_config(
        DeepResearchAgentConfig.default()
    )
    
    ## Initialize sandbox

    custom_map = {
                "utils/llm.py": "utils",
                "utils/unified_base_env.py": "utils",
                "data/tools/combined_tools.json": "data/tools"
        }
    requirements = ["numpy", "requests","mcp==1.9.4","gym==0.26.2","gymnasium==1.2.0","pytest"]
    
    single_agent.check_sandbox(custom_map,requirements)
    stop = ["</tool_use>\n<analysis>","</multi_tool_use.parallel>\n<analysis>","<analysis>\n<final>"]
    
    # Generate environment code
    env_code = asyncio.run(single_agent_code_gen(single_agent,mcp_data,max_tokens=6144,stop=stop))
    return env_code,env_name
