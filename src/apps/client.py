import asyncio
from typing import Optional, Union
from contextlib import AsyncExitStack
import httpx

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
import os

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.sessions: dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()

        proxy_url = (
            os.getenv("MCP_CLIENT_PROXY")
            or os.getenv("HTTPS_PROXY")
            or os.getenv("HTTP_PROXY")
        )
        transport = (
            httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url))
            if proxy_url
            else httpx.HTTPTransport()
        )
        custom_client = httpx.Client(transport=transport, timeout=30.0)

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")

        # Create Anthropic client with custom httpx client and API key
        self.anthropic = Anthropic(
            api_key=anthropic_api_key,
            http_client=custom_client,
        )

    async def load_config(self, config_path: str) -> dict:
        """Load server configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file.
        
        Returns:
            dict: The loaded configuration.
        """
        import json
        with open(config_path, 'r') as f:
            return json.load(f)

    async def connect_to_server(self, server_config: Union[str, dict]):
        """Connect to an MCP server
        
        Args:
            server_config: Either a path to local server script (.py/.js), a server config dict, or a path to a JSON config file.
        """
        if isinstance(server_config, str):
            if server_config.endswith('.json'):
                # Load configuration from JSON file
                server_config = await self.load_config(server_config)
            else:
                # Local server script path
                is_python = server_config.endswith('.py')
                is_js = server_config.endswith('.js')
                if not (is_python or is_js):
                    raise ValueError("Server script must be a .py or .js file")
                
                command = "python" if is_python else "node"
                server_params = StdioServerParameters(
                    command=command,
                    args=[server_config],
                    env=None
                )
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()
                response = await session.list_tools()
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])
                self.sessions["default"] = session
                return

        # Remote server config
        if not isinstance(server_config, dict) or 'mcpServers' not in server_config:
            raise ValueError("Invalid server config format")
        
        # Connect to all servers in config
        for server_name, server_spec in server_config['mcpServers'].items():
            server_params = StdioServerParameters(
                command=server_spec['command'],
                args=server_spec['args'],
                env=server_spec.get('env')
            )
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to server '{server_name}' with tools:", [tool.name for tool in tools])
            self.sessions[server_name] = session

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        # Add CoT prompt to encourage step-by-step reasoning
        messages = [
            {
                "role": "user", 
                "content": "Let's solve this step by step. Please explain your reasoning as you go.\n\n" + query
            }
        ]

        # Collect tools from all connected servers
        available_tools = []
        for server_name, session in self.sessions.items():
            response = await session.list_tools()
            server_tools = [{ 
                "name": f"{server_name}_{tool.name}",  # Changed from dot to underscore
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in response.tools]
            available_tools.extend(server_tools)

        final_text = []
        max_turns = 1  # Maximum number of tool calling turns
        current_turn = 0

        while current_turn < max_turns:
            # Get response from Claude
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=messages,
                tools=available_tools
            )

            # Process each content item from Claude's response
            has_tool_calls = False
            for content in response.content:
                if content.type == 'text':
                    final_text.append(content.text)
                    # Add Claude's reasoning to conversation history
                    messages.append({
                        "role": "assistant",
                        "content": content.text
                    })
                elif content.type == 'tool_use':
                    has_tool_calls = True
                    full_tool_name = content.name
                    # Fix: Use underscore instead of dot for splitting since we changed the separator earlier
                    server_name, tool_name = full_tool_name.split('_', 1)
                    tool_args = content.input
                    
                    # Execute tool call on appropriate server
                    session = self.sessions[server_name]
                    result = await session.call_tool(tool_name, tool_args)
                    tool_result = f"[Tool {full_tool_name} returned: {result.content}]"
                    final_text.append(tool_result)

                    # Add tool result to conversation history for Claude to see
                    messages.append({
                        "role": "user",
                        "content": f"The tool {full_tool_name} returned: {result.content}\nPlease continue with the next step."
                    })

            # If no tool calls were made, we're done
            if not has_tool_calls:
                break

            current_turn += 1

        if current_turn == max_turns:
            final_text.append("\n[Reached maximum number of tool calling turns]")

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
