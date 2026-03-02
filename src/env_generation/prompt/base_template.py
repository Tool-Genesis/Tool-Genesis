"""
Base templates for MCP prompt generation.

This module contains reusable template constants and snippets that can be
imported and used across different prompt builders.
"""

from textwrap import dedent
import os

base_test_env_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "test_utils",
    "unified_agentic_env_test.py",
)
with open(base_test_env_path, "r", encoding="utf-8") as f:
    base_test_env_template = f.read()

apps_test_template_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "apps",
    "test",
    "test_template.py",
)
with open(apps_test_template_path, "r", encoding="utf-8") as f:
    apps_mcp_test_template = f.read()


TESTING_SECTION = dedent(
    f"""
Your tests are **tool contracts**, not tutorials. For every MCP tool, write two tests:
- one **OK** case that verifies the returned payload shape/types and any deterministic fields;
- one **INVALID** case that proves intentional error handling (use `pytest.raises(...)` and assert the message contains a meaningful substring).
"Done" means: in a single pytest run, **every tool has both cases and all tests pass**. Do not claim completion until you explicitly confirm pytest ran and passed.

TEMPLATE TEST CODE TO EXTEND:
```python
{apps_mcp_test_template}
```

"""
)

MCP_TOOL_REGISTRATION = dedent(
    """
    ```python
def _initialize_mcp_server(self):
    if not FastMCP:
        return None
    mcp = FastMCP("your_server_name")
    @mcp.tool()
    def add_two_numbers(a: int, b: int) -> Dict[str, Any]:
        \"\"\"Add two numbers together (dict output contract).\"\"\"
        if not isinstance(a, int) or not isinstance(b, int):
            raise ValueError("a and b must be integers")
        s = a + b
        return {
            "result": s,
            "meta": {
                "a": a,
                "b": b,
                "operation": "add",
                "parity": "even" if (s % 2 == 0) else "odd"
            }
        }
    mcp._env = self
    return mcp

```
Replace <YourClassName> with your actual class name.
    """
)

def build_research_report_response_structure(server_name: str) -> str:
    """Return the response contract JSON for the given server name."""
    return dedent(
        f"""
        {{
            "research_report_path": "<The path of the generated research report for {server_name}>",
        }}
        """
    )

def build_datasets_report_response_structure(server_name: str) -> str:
    """Return the response contract JSON for the given server name."""
    return dedent(
        f"""
        {{
            "env_state_desc_path": "<The path of the generated datasets report for {server_name}>",
            "requirements_path": "<The path of the generated requirements for {server_name}>",
            "python_code_path": "<The path of the generated python code for {server_name}>",
        }}
        """
    )

def build_research_report_w_plab_response_structure(server_name: str) -> str:
    """Return the response contract JSON for the given server name."""
    return dedent(
        f"""
        {{
            "research_report_path": "<The path of the generated research report for {server_name}>",
            "tools_simulations_path": "<The path of the generated tools simulations report for {server_name}>"
        }}
        """
    )

def build_only_env_state_response_structure(server_name: str) -> str:
    """Return the response contract JSON for the given server name."""
    return dedent(
        f"""
        {{
            "env_state_desc_path": "<The path of the generated env state description for {server_name}>",
            "user_profile_path": "<The path of the generated user profile for {server_name}>",
        }}
        """
    )

def build_codegen_response_structure(server_name: str) -> str:
    """Return the response contract JSON for the given server name."""
    return dedent(
        f"""
        {{
            "code_path": "<The path of the generated code for {server_name}>",
        }}
        """
    )
