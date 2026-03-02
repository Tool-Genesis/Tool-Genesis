import json
from loguru import logger
import re
import textwrap
from typing import Any, Callable, Dict, List, Optional, Union
from openai.types.chat.chat_completion import ChatCompletion, Choice

from src.core.messages import (
    ToolCallRequest
)
from src.core.memories import(
    ToolCallingRecord
)

from src.core.toolkits import FunctionTool

def convert_to_function_tool(
    tool: Union[FunctionTool, Callable],
) -> FunctionTool:
    r"""Convert a tool to a FunctionTool from Callable."""
    return tool if isinstance(tool, FunctionTool) else FunctionTool(tool)


def convert_to_schema(
    tool: Union[FunctionTool, Callable, Dict[str, Any]],
) -> Dict[str, Any]:
    r"""Convert a tool to a schema from Callable or FunctionTool."""
    if isinstance(tool, FunctionTool):
        return tool.get_openai_tool_schema()
    elif callable(tool):
        return FunctionTool(tool).get_openai_tool_schema()
    else:
        return tool


def handle_logprobs(choice: Choice) -> Optional[List[Dict[str, Any]]]:
    if choice.logprobs is None:
        return None

    tokens_logprobs = choice.logprobs.content

    if tokens_logprobs is None:
        return None

    return [
        {
            "token": token_logprob.token,
            "logprob": token_logprob.logprob,
            "top_logprobs": [
                (top_logprob.token, top_logprob.logprob)
                for top_logprob in token_logprob.top_logprobs
            ],
        }
        for token_logprob in tokens_logprobs
    ]

def safe_model_dump(obj) -> Dict[str, Any]:
    r"""Safely dump a Pydantic model to a dictionary.

    This method attempts to use the `model_dump` method if available,
    otherwise it falls back to the `dict` method.
    """
    # Check if the `model_dump` method exists (Pydantic v2)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Fallback to `dict()` method (Pydantic v1)
    elif hasattr(obj, "dict"):
        return obj.dict()
    else:
        raise TypeError("The object is not a Pydantic model")


def get_info_dict(
    session_id: Optional[str],
    usage: Optional[Dict[str, int]],
    termination_reasons: List[str],
    num_tokens: int,
    tool_calls: List[ToolCallingRecord],
    external_tool_call_requests: Optional[List[ToolCallRequest]] = None,
) -> Dict[str, Any]:
    r"""Returns a dictionary containing information about the chat session.

    Args:
        session_id (str, optional): The ID of the chat session.
        usage (Dict[str, int], optional): Information about the usage of
            the LLM.
        termination_reasons (List[str]): The reasons for the termination
            of the chat session.
        num_tokens (int): The number of tokens used in the chat session.
        tool_calls (List[ToolCallingRecord]): The list of function
            calling records, containing the information of called tools.
        external_tool_call_requests (Optional[List[ToolCallRequest]]): The
            requests for external tool calls.


    Returns:
        Dict[str, Any]: The chat session information.
    """
    return {
        "id": session_id,
        "usage": usage,
        "termination_reasons": termination_reasons,
        "num_tokens": num_tokens,
        "tool_calls": tool_calls,
        "external_tool_call_requests": external_tool_call_requests,
    }