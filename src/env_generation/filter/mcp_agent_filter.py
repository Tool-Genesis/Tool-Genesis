import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.core.agents import ChatAgent
from src.core.models import ModelFactory
from src.core.types import ModelPlatformType, ModelType
from src.core.utils import extract_payload_from_response

class MCPScore(BaseModel):
    score: float = Field(..., description="A float between 0.0 (completely wrong) and 1.0 (perfectly implemented).")
    reason: str = Field(..., description="A brief explanation for your score.")


def initialize_agent() -> ChatAgent:
    """Initializes and returns a ChatAgent."""
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENROUTER,
        model_type=ModelType.OPENROUTER_GPT_4_1_MINI,
    )
    system_message = "You are a senior software engineer. Your task is to evaluate an MCP server implementation against its tool schema definition."
    agent = ChatAgent(system_message=system_message, model=model)
    return agent


def get_tool_schema(server_name: str, combined_tools_path: Path) -> Dict[str, Any]:
    """Extracts the tool schema for a specific server from the combined tools file."""
    with open(combined_tools_path, 'r', encoding='utf-8') as f:
        all_tools = json.load(f)
    for server in all_tools.get("servers", []):
        if server.get("server_name") == server_name:
            return server
    return {}

def agent_filter(items_to_filter: List[Dict[str, str]], combined_tools_path: Path, threshold: float) -> List[Dict[str, Any]]:
    """
    Uses an LLM agent to filter MCP server code based on its completeness against the JSON schema.
    """
    agent = initialize_agent()
    accepted_items = []

    for item in items_to_filter:
        server_name = item["server_name"]
        file_path = Path(item["path"])

        if not file_path.exists():
            logging.warning(f"File not found for agent filtering: {file_path}. Skipping.")
            continue

        code_content = file_path.read_text(encoding="utf-8")
        tool_schema = get_tool_schema(server_name, combined_tools_path)

        if not tool_schema:
            logging.warning(f"Schema not found for server: {server_name}. Skipping.")
            continue

        prompt = f"""
        Please act as a senior software engineer. Your task is to evaluate an MCP server implementation against its tool schema definition.

        **Tool Schema:**
        ```json
        {json.dumps(tool_schema, indent=2)}
        ```

        **Server Code:**
        ```python
        {code_content}
        ```

        **Evaluation Criteria:**
        1.  **Completeness:** Does the code implement all the tools defined in the schema? Are all function names, parameters, and return types matching?
        2.  **Correctness:** Does the implementation logic seem correct and robust? Does it handle potential errors?

        **Output Format:**
        Provide your response as a single JSON object with two keys:
        - `score`: A float between 0.0 (completely wrong) and 1.0 (perfectly implemented).
        - `reason`: A brief explanation for your score.

        **Your JSON response:**
        """

        try:
            response = agent.step(prompt, MCPScore)
            payload = extract_payload_from_response(response, MCPScore)
            
            if hasattr(payload, "model_dump"):
                result = payload.model_dump()
            else:
                result = payload

            score = result.get("score", 0.0)
            reason = result.get("reason", "No reason provided.")

            if score >= threshold:
                item["agent_score"] = score
                item["agent_reason"] = reason
                accepted_items.append(item)
            else:
                logging.info(f"Rejected {server_name} with score {score} (< {threshold}). Reason: {reason}")
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse LLM response for {server_name}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during agent filtering for {server_name}: {e}")

    return accepted_items