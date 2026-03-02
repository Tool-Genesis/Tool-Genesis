"""
MCP Server Tools Discovery Toolkit

This module provides a toolkit that dynamically discovers and exposes MCP server tools
as FunctionTools for agent interaction. The focus is on making MCP server tools
available to agents in a discoverable and usable format.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union
from loguru import logger

from src.core.toolkits.base import BaseToolkit
from src.core.toolkits.function_tool import FunctionTool
from src.core.toolkits.mcp_function_tool import MCPFunctionTool
from src.core.sandbox.mcp_sandbox_bridge import MCPSandboxBridge
from src.core.sandbox.persistent_sandbox import PersistentEnvironmentSandbox


class MCPServerToolsToolkit(BaseToolkit):
    """
    Toolkit that dynamically discovers and exposes MCP server tools as FunctionTools.
    
    This toolkit discovers all available MCP server tools and returns them as MCPFunctionTool instances.
    It can automatically initialize MCP servers from provided configurations during initialization.
    """
    
    name = "MCPServerToolsToolkit"
    description = "Toolkit for discovering and wrapping MCP server tools"
    
    def __init__(
        self,
        sandbox: Optional[PersistentEnvironmentSandbox] = None,
        server_names: Optional[Union[str, List[str]]] = None,
        init_states: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        timeout: Optional[float] = None,
        concurrent_init: bool = True,
        fail_fast: bool = False,
        registry_path: Optional[str] = None,
        mount_dir: Optional[str] = None
    ):
        """
        Initialize the MCP Server Tools Toolkit.
        
        Args:
            sandbox: Optional PersistentEnvironmentSandbox instance. If None, 
                    a new one will be created.
            server_names: Optional MCP server names to initialize. Can be a single
                         server name string or a list of server names. These names
                         should match entries in the registry.json file.
            init_states: Optional initial states for MCP servers. Can be a single
                        state dict or a list of state dicts. If provided, these
                        will be passed to the MCP servers during initialization.
            timeout: Optional timeout for toolkit operations.
            concurrent_init: Whether to initialize multiple servers concurrently (ignored in sync version).
            fail_fast: Whether to fail fast when initializing multiple servers.
            registry_path: Optional path to the registry.json file. If None, uses default path.
            mount_dir: Optional path to the mount directory.
        """
        super().__init__(timeout=timeout)
        
        if sandbox is None:
            sandbox = PersistentEnvironmentSandbox(mount_dir=mount_dir)
        
        self.bridge = MCPSandboxBridge(sandbox, registry_path=registry_path)
        self.server_names = server_names
        self.registry_path = registry_path
        self.init_states = init_states
        self.concurrent_init = concurrent_init
        self.fail_fast = fail_fast
        self._initialized = False
        
        # Cache for discovered tools and server tools
        self._discovered_tools: Optional[List[FunctionTool]] = None
        self._server_tools_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._tool_info_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._tool_schema_cache: Optional[Dict[str, Any]] = None
    
    def _ensure_servers_initialized(self)->Dict[str, Any]:
        """
        Ensure MCP servers are initialized if server names were provided.
        
        This method is called automatically when get_tools() is invoked
        to ensure servers are ready before tool discovery.
        """
        if self._initialized or not self.server_names:
            return
        
        try:
            logger.info("Initializing MCP servers from provided server names...")
                        
            result = self.bridge.create_mcp_server(
                server_names=self.server_names,
                init_states=self.init_states,                
                registry_path=self.registry_path,
                fail_fast=self.fail_fast
            )
            
            if result.get("success"):
                created_count = result.get("created", 0)
                failed_count = result.get("failed", 0)
                
                if created_count > 0:
                    logger.info(f"Successfully initialized {created_count} MCP servers")
                if failed_count > 0:
                    logger.warning(f"Failed to initialize {failed_count} MCP servers: {result.get('errors', {})}")

            else:
                logger.error(f"Failed to initialize MCP servers: {result.get('error', 'Unknown error')}")
                
            self._initialized = True
            return result
            
        except Exception as e:
            logger.error(f"Exception during MCP server initialization: {e}")
            self._initialized = False  # Mark as initialized to avoid retry loops

    def initialize_servers(self) -> Dict[str, Any]:
        """Public helper to trigger MCP server initialization explicitly."""
        return self._ensure_servers_initialized()

    def cleanup(self) -> None:
        """Cleanup resources held by the toolkit (including sandbox bridge)."""
        if hasattr(self.bridge, "cleanup"):
            self.bridge.cleanup()

    def discover_mcp_server_tools(self, server_name: str) -> Dict[str, Any]:
        """
        Discover tools from a specific MCP server.
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            Dict with discovery results
        """
        try:
            # Check server status first
            status_result = self.bridge.get_mcp_server_status(server_name)
            if status_result.get("status") != "running":
                return {
                    "success": False,
                    "server_name": server_name,
                    "tools": [],
                    "error": f"Server {server_name} is not running"
                }
            
            # Get tools from server
            tools_result = self.bridge.list_mcp_server_tools(server_name)
            
            if tools_result.get("success"):
                tool_names = tools_result.get("tools", [])
                # Convert tool names to tool info dictionaries
                tools = []
                for tool_name in tool_names:
                    tools.append({
                        "name": tool_name,
                        "description": f"MCP tool {tool_name} from server {server_name}",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    })
                
                # Cache the tools
                self._server_tools_cache[server_name] = tools
                
                return {
                    "success": True,
                    "server_name": server_name,
                    "tools": tools,
                    "error": ""
                }
            else:
                return {
                    "success": False,
                    "server_name": server_name,
                    "tools": [],
                    "error": tools_result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error discovering tools from server {server_name}: {e}")
            return {
                "success": False,
                "server_name": server_name,
                "tools": [],
                "error": str(e)
            }

    def discover_all_mcp_tools(self) -> Dict[str, Any]:
        """
        Discover tools from all active MCP servers.
        
        Returns:
            Dict with discovery results from all servers
        """
        try:
            # Get list of active servers
            servers_result = self.bridge.list_mcp_servers()
            
            if not servers_result.get("success"):
                return {
                    "success": False,
                    "servers": {},
                    "error": servers_result.get("error", "Failed to list servers")
                }
            
            servers = servers_result.get("servers", [])
            all_server_tools = {}
            
            for server_info in servers:
                # Extract server name from server info dict
                if isinstance(server_info, dict):
                    server_name = server_info.get("name")
                else:
                    # Fallback for string format
                    server_name = server_info
                
                if server_name:
                    server_result = self.discover_mcp_server_tools(server_name)
                    # Extract just the tools list for the expected format
                    if server_result.get("success"):
                        all_server_tools[server_name] = server_result.get("tools", [])
                    else:
                        all_server_tools[server_name] = []
            
            return {
                "success": True,
                "servers": all_server_tools,
                "error": ""
            }
            
        except Exception as e:
            logger.error(f"Error discovering all MCP tools: {e}")
            return {
                "success": False,
                "servers": {},
                "error": str(e)
            }

    def _sanitize_tool_name(self, server_name: str, tool_name: str) -> str:
        """
        Sanitize tool name for use as function name.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Original tool name
            
        Returns:
            Sanitized function name
        """
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', f"{server_name}_{tool_name}")
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized

    def _create_mcp_tool_wrapper(self, server_name: str, tool_info: Dict[str, Any]) -> callable:
        """
        Create a wrapper function for an MCP tool.
        
        Args:
            server_name: Name of the MCP server
            tool_info: Tool information from MCP server
            
        Returns:
            Wrapper function for the MCP tool
        """
        tool_name = tool_info["name"]
        description = tool_info.get("description", f"MCP tool {tool_name} from server {server_name}")
        input_schema = tool_info.get("inputSchema", {})
        
        # Create sanitized function name
        func_name = self._sanitize_tool_name(server_name, tool_name)
        
        def wrapper_func(**kwargs):
            """MCP tool wrapper function."""
            try:
                result = self.bridge.call_mcp_tool(
                    server_name,
                    tool_name,
                    kwargs
                )
                return result
            except Exception as e:
                logger.error(f"Error calling MCP tool {server_name}.{tool_name}: {e}")
                raise
        
        # Set function metadata
        wrapper_func.__name__ = func_name
        wrapper_func.__qualname__ = func_name
        
        # Build docstring with parameter information
        doc_parts = [description]
        doc_parts.append(f"\nMCP Server: {server_name}")
        doc_parts.append(f"Tool Name: {tool_name}")
        
        # Add parameter documentation
        if input_schema and "properties" in input_schema:
            doc_parts.append("\nParameters:")
            for param_name, param_info in input_schema["properties"].items():
                param_type = param_info.get("type", "unknown")
                param_desc = param_info.get("description", "No description")
                doc_parts.append(f"  {param_name} ({param_type}): {param_desc}")
        
        wrapper_func.__doc__ = "\n".join(doc_parts)
        
        return wrapper_func

    def _load_tool_schemas(self, target_server: Optional[str] = None) -> Dict[str, Any]:
        """
        Load tool schemas from either a provided registry.json or a combined_tools.json fallback.

        Preference order:
        1) If `self.registry_path` is set and points to a valid registry.json, load per-server schema
           from each entry's `json_schema_path`.
        2) Otherwise, fall back to the combined tools file.

        Returns:
            Dict containing all tool schemas indexed by canonical server keys.
            Keys include both the normalized `server_name` (spaces/hyphens -> underscores)
            and the `server_slug` when available, to maximize match success.
        """
        if self._tool_schema_cache is not None:
            if not target_server:
                return self._tool_schema_cache
            tkey = target_server
            tnorm = tkey.replace(" ", "_").replace("-", "_")
            if (
                tkey in self._tool_schema_cache
                or tnorm in self._tool_schema_cache
                or tnorm.lower() in self._tool_schema_cache
            ):
                return self._tool_schema_cache

        # Try registry.json first when provided
        try:
            if self.registry_path and os.path.exists(self.registry_path):
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    reg = json.load(f)
                schema_cache: Dict[str, Any] = dict(self._tool_schema_cache or {})
                if isinstance(reg, dict):
                    if target_server and target_server in reg:
                        payload = reg.get(target_server) or {}
                        try:
                            schema_path = payload.get("json_schema_path")
                            if schema_path and os.path.exists(schema_path):
                                with open(schema_path, 'r', encoding='utf-8') as sf:
                                    schema_obj = json.load(sf)
                                server_key = (schema_obj.get("server_name") or "").replace(" ", "_").replace("-", "_")
                                if server_key:
                                    schema_cache[server_key] = schema_obj
                                    schema_cache[server_key.lower()] = schema_obj
                                schema_cache[target_server] = schema_obj
                                slug = payload.get("server_slug")
                                if isinstance(slug, str) and slug:
                                    schema_cache[slug] = schema_obj
                        except Exception as ie:
                            logger.warning(f"Failed to load schema for '{target_server}': {ie}")
                    elif target_server:
                        for slug, payload in reg.items():
                            if slug == target_server or payload.get("server_slug") == target_server:
                                try:
                                    schema_path = payload.get("json_schema_path")
                                    if not schema_path or not os.path.exists(schema_path):
                                        break
                                    with open(schema_path, 'r', encoding='utf-8') as sf:
                                        schema_obj = json.load(sf)
                                    server_key = (schema_obj.get("server_name") or "").replace(" ", "_").replace("-", "_")
                                    if server_key:
                                        schema_cache[server_key] = schema_obj
                                        schema_cache[server_key.lower()] = schema_obj
                                    schema_cache[target_server] = schema_obj
                                    schema_cache[slug] = schema_obj
                                except Exception as ie:
                                    logger.warning(f"Failed to load schema for '{slug}': {ie}")
                                break
                    else:
                        for slug, payload in reg.items():
                            try:
                                schema_path = payload.get("json_schema_path")
                                if not schema_path or not os.path.exists(schema_path):
                                    continue
                                with open(schema_path, 'r', encoding='utf-8') as sf:
                                    schema_obj = json.load(sf)
                                server_key = (schema_obj.get("server_name") or "").replace(" ", "_").replace("-", "_")
                                if server_key:
                                    schema_cache[server_key] = schema_obj
                                    schema_cache[server_key.lower()] = schema_obj
                                if isinstance(slug, str) and slug:
                                    schema_cache[slug] = schema_obj
                            except Exception as ie:
                                logger.warning(f"Failed to load schema for '{slug}': {ie}")
                                continue
                self._tool_schema_cache = schema_cache
                logger.info(f"Loaded tool schemas from registry for {len(schema_cache)} servers")
                return self._tool_schema_cache
        except Exception as e:
            logger.error(f"Failed to load tool schemas from registry '{self.registry_path}': {e}")
            # continue to fallback

        # Fallback to combined_tools.json (optional, configurable via env)
        try:
            json_path = os.environ.get("MCP_COMBINED_TOOLS_JSON")
            if not json_path:
                logger.warning("MCP_COMBINED_TOOLS_JSON not set; skipping combined tools fallback")
                self._tool_schema_cache = {}
                return self._tool_schema_cache
            if not os.path.exists(json_path):
                logger.warning(f"Tool schema file not found: {json_path}")
                self._tool_schema_cache = {}
                return self._tool_schema_cache
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            schema_cache: Dict[str, Any] = dict(self._tool_schema_cache or {})
            if isinstance(data, dict) and "servers" in data:
                if target_server:
                    tnorm = target_server.replace(" ", "_").replace("-", "_")
                    found = None
                    for server_info in data["servers"]:
                        try:
                            sname = (server_info.get("server_name") or "").replace(" ", "_").replace("-", "_")
                        except Exception:
                            sname = ""
                        if sname == tnorm or sname.lower() == tnorm.lower() or server_info.get("server_name") == target_server:
                            found = server_info
                            break
                    if found:
                        schema_cache[tnorm] = found
                        schema_cache[tnorm.lower()] = found
                        schema_cache[target_server] = found
                else:
                    for server_info in data["servers"]:
                        try:
                            sname = (server_info.get("server_name") or "").replace(" ", "_").replace("-", "_")
                            if sname:
                                schema_cache[sname] = server_info
                                schema_cache[sname.lower()] = server_info
                        except Exception:
                            continue
            self._tool_schema_cache = schema_cache
            logger.info(f"Loaded tool schemas for {len(schema_cache)} servers from combined tools file")
            return self._tool_schema_cache
        except Exception as e:
            logger.error(f"Failed to load tool schemas from combined tools: {e}")
            self._tool_schema_cache = {}
            return self._tool_schema_cache

    def _get_mcp_tool_schema(self, server_name: str, tool_name: str) -> Dict[str, Any]:
        """
        Get tool schema from combined_tools-50.json based on server_name and tool_name.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool
            
        Returns:
            Dict containing the tool schema, or empty dict if not found
        """
        try:
            # Load schemas if not already cached
            schemas = self._load_tool_schemas(server_name)
            
            # Find the server in the schemas
            server_schema = schemas.get(server_name)
            if not server_schema:
                logger.warning(f"No schema found for server: {server_name}")
                return {}
            
            # Find the specific tool in the server's tools
            tools = server_schema.get("tools", [])
            for tool_info in tools:
                if tool_info.get("name") == tool_name:
                    return tool_info
            
            logger.warning(f"No schema found for tool {tool_name} in server {server_name}")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting tool schema for {server_name}.{tool_name}: {e}")
            return {}

    def get_tools(self) -> List[FunctionTool]:
        """
        Get all available MCP server tools as FunctionTool instances.
        
        This method automatically initializes MCP servers if server names were provided
        during toolkit initialization, then discovers and returns all available tools.
        
        Returns:
            List[FunctionTool]: List of MCP tools wrapped as FunctionTool instances
        """
        # Ensure servers are initialized if needed
        self._ensure_servers_initialized()
        
        # Return cached tools if available
        if self._discovered_tools is not None:
            return self._discovered_tools
        
        tools = []
        
        try:
            # Discover tools from all servers
            discovery_result = self.discover_all_mcp_tools()
            
            if discovery_result.get("success"):
                servers_tools = discovery_result.get("servers", {})
                
                for server_name, server_tools in servers_tools.items():
                    for tool_info in server_tools:
                        try:
                            # Create wrapper function for the tool
                            wrapper_func = self._create_mcp_tool_wrapper(server_name, tool_info)
                            mcp_schema = self._get_mcp_tool_schema(server_name, tool_info["name"])
                            # Create MCPFunctionTool instance
                            mcp_tool = MCPFunctionTool(
                                func=wrapper_func,
                                mcp_schema=mcp_schema,
                                server_name=server_name,
                                tool_name=tool_info["name"]
                            )
                            
                            tools.append(mcp_tool)
                            
                        except Exception as e:
                            logger.error(f"Failed to create tool wrapper for {server_name}.{tool_info.get('name', 'unknown')}: {e}")
                            continue
                
                logger.info(f"Successfully discovered {len(tools)} MCP tools from {len(servers_tools)} servers")
            else:
                logger.error(f"Failed to discover MCP tools: {discovery_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Exception during tool discovery: {e}")
        
        # Cache the discovered tools
        self._discovered_tools = tools
        return tools

    def get_server_tools(self, server_name: str) -> List[FunctionTool]:
        """
        Get tools from a specific MCP server.
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            List[FunctionTool]: List of tools from the specified server
        """
        all_tools = self.get_tools()
        return [tool for tool in all_tools if hasattr(tool, 'server_name') and tool.server_name == server_name]

    def refresh_tools(self) -> List[FunctionTool]:
        """
        Refresh the tool cache and rediscover all tools.
        
        Returns:
            List[FunctionTool]: Updated list of tools
        """
        # Clear caches
        self._discovered_tools = None
        self._server_tools_cache.clear()
        self._tool_info_cache.clear()
        self._tool_schema_cache = None
        
        # Rediscover tools
        return self.get_tools()

    def get_tool_info(self, server_name: str = None) -> Dict[str, Any]:
        """
        Get detailed information about available tools.
        
        Args:
            server_name: Optional server name to filter tools
            
        Returns:
            Dict with tool information
        """
        tools = self.get_tools()
        
        if server_name:
            tools = [tool for tool in tools if hasattr(tool, 'server_name') and tool.server_name == server_name]
        
        tool_info = {
            "total_tools": len(tools),
            "servers": {},
            "tools": []
        }
        
        for tool in tools:
            if hasattr(tool, 'server_name'):
                server = tool.server_name
                if server not in tool_info["servers"]:
                    tool_info["servers"][server] = 0
                tool_info["servers"][server] += 1
                
                tool_info["tools"].append({
                    "name": getattr(tool.func, '__name__', 'unknown'),
                    "server": server,
                    "tool_name": getattr(tool, 'tool_name', 'unknown'),
                    "description": getattr(tool.func, '__doc__', 'No description')
                })
        
        return tool_info
