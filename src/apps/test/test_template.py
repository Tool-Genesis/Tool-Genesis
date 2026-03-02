"""
This is a template file for creating pytest tests for your tools.

How to use this template:
1.  **Copy this file**: Make a copy of this file for your test suite (e.g., `test/test_my_new_feature.py`).

2.  **Configure Server Path**:
    -   Update the `SERVER_ENV_PATH` variable to point to the Python file containing your server environment.
    -   Update the `SERVER_CLASS_NAME` variable with the name of your server class.

3.  **Write Your Tests**:
    -   For each tool you want to test, copy the `test_tool_template` function.
    -   Rename the function to something descriptive (e.g., `test_my_awesome_tool`).
    -   Change the `tool_name` variable to the name of the tool you are testing.
    -   Update the `tool_args` dictionary with the parameters your tool expects.
    -   Modify the `assert` statements to verify that the tool's response is correct.

4.  **Run Pytest**:
    -   Run `pytest` from your terminal in the project root directory. Pytest will automatically discover and run your new tests.
"""
import pytest
import os

from src.apps.test import MCPServerTestClient

# --- Configuration: Set the path to your MCP server environment ---
# TODO: Replace this with the actual path to your server environment file.
# For example: "src/apps/your_server_env.py"
SERVER_ENV_PATH = "YourServerEnvPath"

# TODO: Replace this with the actual class name of your server.
SERVER_CLASS_NAME = "YourServerClassName"


def _is_placeholder_config() -> bool:
    return (
        not SERVER_ENV_PATH
        or SERVER_ENV_PATH == "YourServerEnvPath"
        or not SERVER_CLASS_NAME
        or SERVER_CLASS_NAME == "YourServerClassName"
    )


@pytest.fixture(scope="module")
def mcp_server_test_client() -> MCPServerTestClient:
    """
    A pytest fixture that sets up the MCPServerTestClient for the specified server.
    This fixture is created once per test module.
    """
    if _is_placeholder_config():
        pytest.skip("Template placeholders are not configured. Set SERVER_ENV_PATH and SERVER_CLASS_NAME to run this test.")
    if not os.path.exists(SERVER_ENV_PATH):
        pytest.skip(f"Server environment file not found: {SERVER_ENV_PATH}")

    print(f"\\nSetting up test client for: {SERVER_ENV_PATH}")
    
    # You can customize the initial server specification here.
    initial_spec = {}
    
    client = MCPServerTestClient(
        file_path=SERVER_ENV_PATH,
        class_name=SERVER_CLASS_NAME,
        spec=initial_spec
    )
    
    yield client
    
    # Teardown: close the client after tests are done.
    print("\\nCleaning up MCPServerTestClient.")
    client.close()

# --- Test Case Template ---

"""


"""

@pytest.mark.asyncio
async def test_tool_template(mcp_server_test_client: MCPServerTestClient):
    """
    This is a template for testing a single tool.
    
    How to use this template:
    1.  Copy and paste this function for each tool you want to test.
    2.  Rename the function to reflect the tool being tested (e.g., `test_my_tool_name`).
    3.  Replace 'your_tool_name' with the actual name of the tool.
    4.  Define the arguments for the tool call.
    5.  Update the assertions to validate the expected response from the tool.
    """
    # TODO: Replace with the name of the tool you want to test.
    tool_name = "your_tool_name"
    
    # TODO: Define the arguments for your tool.
    tool_args = {
        "param1": "value1",
        "param2": 123,
    }

    # Call the tool using the test client.
    response = await mcp_server_test_client.call_tool(
        tool_name,
        **tool_args
    )

    # Print the response for debugging purposes.
    print(f"Response from {tool_name}: {response}")

    # TODO: Add assertions to verify the tool's response.
    # These are just examples. You should write assertions that are
    # specific to your tool's expected output.
    assert response is not None
    assert "expected_key" in response
    assert response["param1"] == "value1"
