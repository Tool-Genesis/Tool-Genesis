"""Validation utilities for environment generation."""

from typing import Dict, Any, Tuple, Optional
import json
import tempfile
import subprocess
import sys
from pathlib import Path


def only_validate(env_code: str, mcp_data: Dict[str, Any]) -> bool:
    """
    Validate environment code without refinement.
    
    Args:
        env_code: The environment code to validate
        mcp_data: MCP server data for validation context
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        # Basic syntax check
        compile(env_code, '<string>', 'exec')
        
        # Check for required components
        required_patterns = [
            'class.*Env',
            'def __init__',
            'def _get_environment_state',
            'def _reset_environment_state'
        ]
        
        for pattern in required_patterns:
            if pattern not in env_code:
                return False
                
        return True
    except Exception:
        return False


def validate_and_refine(env_code: str, mcp_data: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Validate and potentially refine environment code.
    
    Args:
        env_code: The environment code to validate and refine
        mcp_data: MCP server data for validation context
        
    Returns:
        Tuple[str, bool]: (refined_code, success)
    """
    # First try basic validation
    if only_validate(env_code, mcp_data):
        return env_code, True
    
    # If basic validation fails, try to fix common issues
    try:
        # Add missing imports if needed
        if 'import json' not in env_code:
            env_code = 'import json\n' + env_code
            
        if 'from typing import' not in env_code:
            env_code = 'from typing import Dict, Any, List, Optional\n' + env_code
            
        # Try validation again
        if only_validate(env_code, mcp_data):
            return env_code, True
            
    except Exception:
        pass
    
    return env_code, False