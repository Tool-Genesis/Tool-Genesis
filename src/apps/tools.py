from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class ToolDefinition:
    """
    Tool definition structure for MCP tools.
    
    The environment generator will automatically determine the implementation approach:
    - Direct implementation: For tools that can be simulated with simple code logic
      (e.g., mathematical operations, file system operations, data transformations)
    - LLM simulation: For tools requiring external services or complex behavior
      (e.g., web search, maps API, proprietary data access, complex reasoning)
    
    The choice between direct vs LLM simulation should be made by the generator
    based on the tool's complexity and requirements.
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    mcp_implementation: Optional[callable] = None  # MCP tool implementation function