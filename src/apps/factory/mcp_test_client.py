import os
import sys
import json
from pathlib import Path

# --- Test Setup: Add project root to sys.path ---
# This allows us to import modules from the 'src' directory
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
# --- End Test Setup ---

from src.apps.factory.mcp_server_factory import MCPServerFactory


def _extract_json_from_response(response) -> dict:
    """Extracts a JSON dictionary from a tool response."""
    if isinstance(response, dict):
        return response
    
    # Handle cases where the response is a list containing a dictionary
    if isinstance(response, list) and response and isinstance(response[0], dict):
        return response[0]

    # Handle TextContent or other objects with a 'text' attribute
    if hasattr(response, 'text'):
        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, TypeError):
            pass

    # If it's a list, try to extract from the first element
    if isinstance(response, list) and response and hasattr(response[0], 'text'):
        try:
            return json.loads(response[0].text)
        except (json.JSONDecodeError, TypeError):
            pass

    raise TypeError(f"Unexpected response type: {type(response)}")


class MCPServerTestClient:
    """
    A helper class to bootstrap an MCP Server from a file 
    and provide a client interface for calling its tools within a test.
    This mirrors the logic of mcp_boot.py but for in-process testing.
    """
    def __init__(self, file_path: str, class_name: str, spec: dict = None):
        """
        Initializes the test client by creating an MCP server instance.

        Args:
            file_path (str): Path to the Python file containing the App class.
            class_name (str): The name of the App class in the file.
            spec (dict, optional): The initial spec dictionary for the App. Defaults to {}.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The specified server file does not exist: {file_path}")

        # 1. Use the factory to load the App class from the specified file
        self.factory = MCPServerFactory(app_cls=None, file_path=file_path, class_name=class_name)
        
        # 2. Create the MCP server instance with the provided spec
        # This is the core step that mcp_boot.py performs
        self.mcp_server = self.factory.create(
            spec=spec or {},
            register_atexit_save=False  # We don't want to save state during tests
        )
        
        # 3. The FastMCP instance itself is used to call tools. No separate client object needed.

    async def call_tool(self, tool_name: str, **params) -> dict:
        """
        Calls a tool on the loaded MCP server.

        Args:
            tool_name (str): The name of the tool to call.
            **params: The parameters to pass to the tool.

        Returns:
            dict: The result from the tool execution.
        """
        # The FastMCP instance's call_tool method is a coroutine and must be awaited.
        raw_response = await self.mcp_server.call_tool(tool_name, params)
        return _extract_json_from_response(raw_response)

    def close(self):
        """Placeholder for any cleanup logic if needed in the future."""
        print("\nCleaning up MCPServerTestClient.")
        pass
