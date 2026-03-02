import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class PromptContext:
    research_report: str
    tools_json: dict
    env_state: dict
    interaction_history: list
    tool_call: dict

class BasePrompt(ABC):
    @abstractmethod
    def build(self, context: PromptContext) -> str:
        pass

class SimulationPrompt(BasePrompt):
    def build(self, context: PromptContext) -> str:
        return f"""
You are an office environment simulator. Your task is to generate realistic environment feedback based on a model's actions by simulating the execution of a tool call and returning the result in JSON format.

**Environment Description:**
{context.research_report}

**Available Tools and Action Formats:**
{json.dumps(context.tools_json, indent=2)}

**Current Environment State:**
{json.dumps(context.env_state)}

Please simulate the execution result of this action. You need to:
1. FIRST check if the action is valid according to the "Available Tools and Action Formats" above.
2. If the action is invalid (e.g., wrong tool name, missing or invalid parameters), return failure immediately.
3. If the action is valid, simulate its execution and determine the realistic outcome.
4. Return the appropriate content describing the outcome.

IMPORTANT: Only actions listed in the "Available Tools and Action Formats" section are valid. If the action is not in that list, it should fail. To modify the environment state (`env_state`), the action must explicitly use a file-writing tool (e.g., `file_tool`) to overwrite the `env_state.json` file. Your simulation should reflect the outcome of this file operation.

Response format:
```json
{{
  "success": <boolean>,
  "content": "<string>"
}}
```
Please ensure your response is realistic. Invalid actions should always return success=false.

**Interaction History:**
{json.dumps(context.interaction_history, indent=2)}

**Action to Simulate:**
{json.dumps(context.tool_call, indent=2)}
"""