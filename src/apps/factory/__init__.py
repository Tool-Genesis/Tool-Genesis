"""
MCP Server Factory module - Dynamic MCP server creation and management.

This module provides factory pattern implementation for creating MCP servers
without hardcoded __main__ sections.
"""

from .mcp_server_factory import MCPServerFactory

__all__ = ['MCPServerFactory']