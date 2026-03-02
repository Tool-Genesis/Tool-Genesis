"""
MCP Function Tool

This module provides a specialized FunctionTool class optimized for MCP server tools.
It extends the base FunctionTool with MCP-specific functionality and schema handling.
"""

from typing import Any, Dict, Optional, Callable
import copy
from loguru import logger
import inspect

from src.core.toolkits.function_tool import FunctionTool


class MCPFunctionTool(FunctionTool):
    """
    Specialized FunctionTool for MCP server tools.
    
    This class extends FunctionTool with MCP-specific optimizations:
    - Direct schema support from MCP tool definitions
    - Enhanced error handling for MCP tool calls
    - Better integration with MCP server bridge
    """
    
    def __init__(
        self,
        func: Callable,
        mcp_schema: Optional[Dict[str, Any]] = None,
        server_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        mcp_bridge: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize MCPFunctionTool.
        
        Args:
            func: The MCP tool wrapper function
            mcp_schema: Original MCP tool schema from server
            server_name: Name of the MCP server
            tool_name: Original tool name on MCP server
            mcp_bridge: MCP bridge instance for tool calls
            **kwargs: Additional arguments passed to FunctionTool
        """
        # Store MCP-specific metadata
        self.mcp_schema = mcp_schema or {}
        self.server_name = server_name
        self.tool_name = tool_name
        self._mcp_bridge = mcp_bridge
        
        # Generate OpenAI schema from MCP schema if available
        openai_schema = self._convert_mcp_to_openai_schema() if mcp_schema else None
        
        # Initialize parent with converted schema
        super().__init__(
            func=func,
            openai_tool_schema=openai_schema,
            **kwargs
        )
    
    def _convert_mcp_to_openai_schema(self) -> Dict[str, Any]:
        """
        Convert MCP tool schema to OpenAI tool schema format.
        
        Returns:
            Dict containing OpenAI-compatible tool schema
        """
        try:
            # Use the original tool name from MCP server, not the sanitized wrapper function name
            # This ensures the schema function name matches what Agent expects in _internal_tools
            func_name = self.tool_name or "unknown_tool"
            
            # Get description from MCP schema first, then fall back to function docstring
            description = self.mcp_schema.get("description", "")
            if not description:
                func_doc = getattr(getattr(self, 'func', None), '__doc__', None) or f"MCP tool: {self.tool_name}"
                description = func_doc.split('\n')[0] if func_doc else f"Tool from {self.server_name} server"
            
            # Convert MCP inputSchema to OpenAI parameters format
            parameters = self._convert_mcp_input_schema()
            
            # Build OpenAI function schema
            openai_function_schema = {
                "name": func_name,
                "description": description,
                "strict": True,  # Add strict mode for OpenAI compatibility
                "parameters": parameters
            }
            
            # Wrap in tool schema
            openai_tool_schema = {
                "type": "function",
                "function": openai_function_schema
            }
            
            return openai_tool_schema
            
        except Exception as e:
            logger.warning(f"Failed to convert MCP schema to OpenAI format: {e}")
            # Fall back to default schema generation
            return None
    
    def _convert_mcp_input_schema(self) -> Dict[str, Any]:
        """
        Convert MCP inputSchema to OpenAI parameters format.
        
        Returns:
            Dict containing OpenAI-compatible parameters schema
        """
        try:
            # Extract inputSchema from MCP schema
            input_schema = self.mcp_schema.get("inputSchema", {})
            
            # If no inputSchema found, return empty parameters
            if not input_schema:
                return {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            
            # Convert MCP inputSchema to OpenAI parameters format
            parameters = {
                "type": input_schema.get("type", "object"),
                "properties": input_schema.get("properties", {}),
                "required": input_schema.get("required", []),
                "additionalProperties": input_schema.get("additionalProperties", False)
            }
            normalized = self._normalize_json_schema(copy.deepcopy(parameters))
            return normalized
            
        except Exception as e:
            logger.warning(f"Failed to convert MCP input schema: {e}")
            # Return default empty parameters
            return {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }

    def _normalize_json_schema(self, schema: Any) -> Any:
        if isinstance(schema, dict):
            if isinstance(schema.get("items"), list):
                schema["prefixItems"] = schema["items"]
                schema["items"] = False
            for key, value in list(schema.items()):
                schema[key] = self._normalize_json_schema(value)
            return schema
        if isinstance(schema, list):
            return [self._normalize_json_schema(item) for item in schema]
        return schema
    
    def _remove_titles_from_schema(self, schema: Dict[str, Any]) -> None:
        """
        Recursively remove 'title' fields from schema to avoid OpenAI issues.
        
        Args:
            schema: Schema dictionary to clean
        """
        if isinstance(schema, dict):
            # Remove title if present
            schema.pop("title", None)
            
            # Recursively process nested objects
            for key, value in schema.items():
                if isinstance(value, dict):
                    self._remove_titles_from_schema(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._remove_titles_from_schema(item)
    
    def get_mcp_metadata(self) -> Dict[str, Any]:
        """
        Get MCP-specific metadata for this tool.
        
        Returns:
            Dict containing MCP metadata
        """
        return {
            "server_name": self.server_name,
            "tool_name": self.tool_name,
            "mcp_schema": self.mcp_schema,
            "is_mcp_tool": True
        }
    
    async def async_call(self, *args: Any, **kwargs: Any) -> Any:
        """
        Asynchronously call the MCP tool through the MCP bridge.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Tool execution result
        """
        try:
            # If this is an MCP tool with server and tool names, use MCP bridge
            if self.server_name and self.tool_name and hasattr(self, '_mcp_bridge'):
                logger.debug(f"Calling MCP tool: {self.server_name}.{self.tool_name}")
                
                # Convert args and kwargs to a single arguments dict
                arguments = {}
                
                # Handle positional arguments by mapping to parameter names
                if args:
                    sig = inspect.signature(self.func)
                    param_names = list(sig.parameters.keys())
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            arguments[param_names[i]] = arg
                
                # Add keyword arguments
                arguments.update(kwargs)
                
                # Call through MCP bridge (now synchronous)
                result = self._mcp_bridge.call_mcp_tool(
                    self.server_name, 
                    self.tool_name, 
                    arguments
                )
                
                logger.debug(f"MCP tool call successful: {self.server_name}.{self.tool_name}")
                return result
            
            # Fall back to parent implementation for non-MCP tools
            result = await super().async_call(*args, **kwargs)
            return result
            
        except Exception as e:
            # Enhanced error logging for MCP tools
            logger.error(
                f"MCP tool call failed: {self.server_name}.{self.tool_name} - {e}"
            )
            
            # Re-raise with additional context
            raise RuntimeError(
                f"MCP tool '{self.tool_name}' on server '{self.server_name}' failed: {e}"
            ) from e
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the MCP tool through the MCP bridge (synchronous).
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Tool execution result
        """
        try:
            # If this is an MCP tool with server and tool names, use MCP bridge
            if self.server_name and self.tool_name and hasattr(self, '_mcp_bridge') and self._mcp_bridge:
                logger.debug(f"Calling MCP tool: {self.server_name}.{self.tool_name}")
                
                # Convert args and kwargs to a single arguments dict
                arguments = {}
                
                # Handle positional arguments by mapping to parameter names
                if args:
                    sig = inspect.signature(self.func)
                    param_names = list(sig.parameters.keys())
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            arguments[param_names[i]] = arg
                
                # Add keyword arguments
                arguments.update(kwargs)
                
                # Call through MCP bridge (now synchronous)
                result = self._mcp_bridge.call_mcp_tool(
                    self.server_name, 
                    self.tool_name, 
                    arguments
                )
                
                logger.debug(f"MCP tool call successful: {self.server_name}.{self.tool_name}")
                return result
            
            # Fall back to parent implementation for non-MCP tools
            result = super().__call__(*args, **kwargs)
            return result
            
        except Exception as e:
            # Enhanced error logging for MCP tools
            logger.error(
                f"MCP tool call failed: {self.server_name}.{self.tool_name} - {e}"
            )
            
            # Re-raise with additional context
            raise RuntimeError(
                f"MCP tool '{self.tool_name}' on server '{self.server_name}' failed: {e}"
            ) from e
    
    def get_openai_tool_schema(self) -> Dict[str, Any]:
        """
        Get OpenAI tool schema with MCP-specific enhancements.
        
        Returns:
            Dict containing OpenAI tool schema
        """
        try:
            # If we have MCP schema (including empty dict), use our conversion
            if self.mcp_schema is not None:
                converted_schema = self._convert_mcp_to_openai_schema()
                if converted_schema:
                    # Add MCP-specific metadata to description if available
                    if self.server_name and self.tool_name:
                        function_schema = converted_schema.get("function", {})
                        current_desc = function_schema.get("description", "")
                        
                        # Enhance description with MCP context
                        enhanced_desc = f"{current_desc}\n\nMCP Server: {self.server_name}\nOriginal Tool: {self.tool_name}"
                        function_schema["description"] = enhanced_desc.strip()
                        
                        converted_schema["function"] = function_schema
                    
                    return converted_schema
            
            # Fall back to parent implementation
            schema = super().get_openai_tool_schema()
            
            # Add MCP-specific metadata to description if available
            if self.server_name and self.tool_name:
                function_schema = schema.get("function", {})
                current_desc = function_schema.get("description", "")
                
                # Enhance description with MCP context
                enhanced_desc = f"{current_desc}\n\nMCP Server: {self.server_name}\nOriginal Tool: {self.tool_name}"
                function_schema["description"] = enhanced_desc.strip()
                
                schema["function"] = function_schema
            
            return schema
            
        except Exception as e:
            logger.error(f"Failed to get OpenAI schema for MCP tool: {e}")
            raise
    
    def __repr__(self) -> str:
        """String representation of MCPFunctionTool."""
        return (
            f"MCPFunctionTool(server='{self.server_name}', "
            f"tool='{self.tool_name}', func='{self.func.__name__}')"
        )
