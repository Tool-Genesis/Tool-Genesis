import re
import ast
from typing import Any, Dict, Optional, Tuple
import os
import json
from pathlib import Path
from src.core.toolkits.mcp_sandbox_bridge_toolkit import MCPServerToolsToolkit

_JSON_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}

def _compliant_input_schema_only(s: Dict[str, Any]) -> bool:
    """
    Lightweight structural compliance check focusing ONLY on `input_schema`.

    Key rule:
    - Explicitly forbid tools from including `parameters`.

    This is NOT a full JSON Schema / OpenAPI validator.
    It is a strict sanity checker for your benchmark's list_tools-style payload.
    """

    if not isinstance(s, dict):
        return False

    tools = s.get("tools")
    if not isinstance(tools, list) or not tools:
        return False

    for t in tools:
        if not isinstance(t, dict):
            return False

        if not isinstance(t.get("name"), str):
            return False
        if not isinstance(t.get("description"), str):
            return False

        # (1) Explicitly forbid `parameters`
        if "parameters" in t:
            return False

        inp = t.get("input_schema") or t.get("inputSchema")
        if not _is_valid_input_schema(inp):
            return False

    return True


def _is_valid_input_schema(schema: Any) -> bool:
    """Validate a restricted JSON Schema subset suitable for tool input."""

    if not isinstance(schema, dict):
        return False

    tp = schema.get("type")
    if not isinstance(tp, str) or tp not in _JSON_TYPES:
        return False

    # -------- object --------
    if tp == "object":
        if "properties" not in schema or "required" not in schema or "additionalProperties" not in schema:
            return False
        props = schema.get("properties")
        if not isinstance(props, dict):
            return False
        for pk, pv in props.items():
            if not isinstance(pk, str):
                return False
            if not _is_valid_input_schema(pv):
                return False
        req = schema.get("required")
        if not isinstance(req, list):
            return False
        if any(not isinstance(x, str) for x in req):
            return False
        if any(x not in props for x in req):
            return False
        addp = schema.get("additionalProperties")
        if addp is not False:
            return False
        if "items" in schema:
            return False
        return True

    # -------- array --------
    if tp == "array":
        items = schema.get("items")
        if not _is_valid_input_schema(items):
            return False

        # disallow object-specific fields on arrays
        if "properties" in schema:
            return False
        if "required" in schema:
            return False
        if "additionalProperties" in schema:
            return False

        return True

    # -------- primitives / null --------
    # For strictness, disallow structural keywords that don't apply
    if "properties" in schema:
        return False
    if "items" in schema:
        return False
    if "required" in schema:
        return False
    if "additionalProperties" in schema:
        return False

    return True

def _launch_success(server_name: Optional[str], registry_path: Optional[str], attempts: int = 3) -> int:
    if not server_name or not registry_path:
        return 0
    num_success = 0
    for _ in range(max(1, attempts)):
        tk = None
        try:
            tk = MCPServerToolsToolkit(server_names=server_name.strip(), registry_path=registry_path)
            init_result = tk.initialize_servers() or {}
            if not init_result.get("success", False):
                continue
            tools_result = tk.bridge.list_mcp_server_tools(server_name.strip())
            if tools_result.get("success", False):
                cnt = tools_result.get("count")
                if (cnt is not None and cnt > 0) or len(tools_result.get("tools", [])) > 0:
                    num_success += 1
        except Exception:
            continue
        finally:
            try:
                if tk:
                    tk.refresh_tools()
                    tk.cleanup()
            except Exception:
                pass
    return num_success

def l1_protocol_compliance_metrics(schema: Optional[Dict[str, Any]], attempts: int, server_name: Optional[str] = None, registry_path: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    effective_attempts = max(1, attempts)
    if schema is None:
        return False, {
            "compliance": False,
            "server_launch_success": 0,
            "server_launch_attempts": effective_attempts,
        }
    compliant = _compliant_input_schema_only(schema)
    num_success = _launch_success(server_name, registry_path, effective_attempts)
    l1_success = compliant and num_success > 0
    return l1_success, {
        "compliance": compliant,
        "server_launch_success": num_success,
        "server_launch_attempts": effective_attempts,
    }
